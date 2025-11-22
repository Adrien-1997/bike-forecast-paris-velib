# service/jobs/train_model.py

# =============================================================================
# Cloud Run training entrypoint (harmonised)
#
# Responsibilities:
# - Download and concatenate daily shards from GCS (or directly download a
#   single Parquet file used as training base).
# - Automatically locate the training module (core/train/tools/flat layouts).
# - Resolve model versioning:
#       read latest.json → compute next version → export as ENV
# - Call the training function (XGBoost-only on the forecast side).
# - Upload the trained model .joblib artifact to GCS as a fallback when the
#   trainer did not self-publish it.
# =============================================================================

from __future__ import annotations
import os, sys, tempfile, re, importlib, inspect, json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Tuple, List, Callable, Dict, Any, Optional

import pandas as pd
from google.cloud import storage
import importlib.util as _iu

# ─────────────────────────────────────────────
# GCS helpers
# ─────────────────────────────────────────────

RE_COMPACT = re.compile(r".*/compact_(\d{4}-\d{2}-\d{2})\.parquet$")


def _parse_gs(uri: str) -> Tuple[str, str]:
    """
    Split a GCS URI (gs://bucket/path) into bucket name and object key.

    Parameters
    ----------
    uri : str
        GCS URI starting with "gs://".

    Returns
    -------
    (str, str)
        Tuple (bucket, key) where key has no trailing slash.

    Raises
    ------
    AssertionError
        If the URI does not start with "gs://".
    """
    assert uri.startswith("gs://"), f"Bad GCS uri: {uri}"
    bkt, key = uri[5:].split("/", 1)
    return bkt, key.rstrip("/")


def _download(cli: storage.Client, src_uri: str, dst_path: Path) -> None:
    """
    Download a GCS object to a local file.

    Parameters
    ----------
    cli : google.cloud.storage.Client
        Storage client instance.
    src_uri : str
        Source GCS URI (gs://bucket/path/to/file).
    dst_path : pathlib.Path
        Local destination path. Parent directories are created if needed.
    """
    bkt, key = _parse_gs(src_uri)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cli.bucket(bkt).blob(key).download_to_filename(str(dst_path))


def _upload_to_gcs(local_path: Path, dst_uri: str) -> str:
    """
    Upload a local file to a GCS object.

    Parameters
    ----------
    local_path : pathlib.Path
        Local path of the file to upload.
    dst_uri : str
        Destination GCS URI (gs://bucket/path/to/file.joblib).

    Returns
    -------
    str
        The GCS URI of the uploaded object.

    Raises
    ------
    FileNotFoundError
        If the local file does not exist.
    """
    if not local_path.exists():
        raise FileNotFoundError(f"Local model not found: {local_path}")
    bkt, key = _parse_gs(dst_uri)
    storage.Client().bucket(bkt).blob(key).upload_from_filename(str(local_path))
    return f"gs://{bkt}/{key}"


def _list_daily(cli: storage.Client, gcs_prefix: str, lookback_days: int) -> List[str]:
    """
    List daily compact shards (compact_YYYY-MM-DD.parquet) to concatenate.

    The function:
    - Lists blobs under the given GCS prefix.
    - Keeps only those matching RE_COMPACT (compact_YYYY-MM-DD.parquet).
    - Filters shards whose day is >= (today_utc - lookback_days).
    - Sorts URIs by date encoded in the filename.

    Parameters
    ----------
    cli : google.cloud.storage.Client
        Storage client instance.
    gcs_prefix : str
        GCS prefix where compact daily shards are stored, e.g.
        "gs://bucket/velib/daily".
    lookback_days : int
        Number of days of history to include when training.

    Returns
    -------
    list of str
        List of GCS URIs for the selected daily shards.

    Raises
    ------
    RuntimeError
        If no shard is found within the lookback window.
    """
    bkt, prefix = _parse_gs(gcs_prefix)
    now = datetime.now(timezone.utc).date()
    min_day = now - timedelta(days=lookback_days)
    uris: List[str] = []

    for blob in cli.list_blobs(bkt, prefix=prefix + "/"):
        m = RE_COMPACT.match(blob.name)
        if not m:
            continue
        day = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        if day >= min_day:
            uris.append(f"gs://{bkt}/{blob.name}")

    # sort by date in the filename
    uris.sort(key=lambda u: RE_COMPACT.match(u).group(1) if RE_COMPACT.match(u) else u)
    if not uris:
        raise RuntimeError(f"Aucun shard daily trouvé sous {gcs_prefix} sur {lookback_days} jours.")
    return uris


def _concat_locally(cli: storage.Client, shard_uris: List[str], out_path: Path) -> Path:
    """
    Download and concatenate daily shards into a single local Parquet file.

    Steps
    -----
    - Download each shard under a temporary folder.
    - Load each shard as a pandas DataFrame.
    - Concatenate all shards row-wise.
    - Write the result to `out_path`.

    Parameters
    ----------
    cli : google.cloud.storage.Client
        Storage client instance.
    shard_uris : list of str
        GCS URIs of daily compact shards to download and merge.
    out_path : pathlib.Path
        Local Parquet file path for the concatenated dataset.

    Returns
    -------
    pathlib.Path
        The `out_path` argument for convenience.
    """
    tmp_dir = out_path.parent / "daily_shards"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dfs: List[pd.DataFrame] = []

    for i, uri in enumerate(shard_uris, 1):
        local_part = tmp_dir / Path(uri).name
        print(f"[train_job] download shard {i}/{len(shard_uris)}: {uri} → {local_part}", flush=True)
        _download(cli, uri, local_part)
        dfs.append(pd.read_parquet(local_part))

    print(f"[train_job] concat {len(dfs)} shards → {out_path}", flush=True)
    full = pd.concat(dfs, axis=0, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)
    return out_path

# ─────────────────────────────────────────────
# Repo root resolution and sys.path patching
# ─────────────────────────────────────────────

def _ensure_repo_on_path() -> Path:
    """
    Ensure the repository root (containing 'service/' or 'train/') is on sys.path.

    Search order
    ------------
    1. If the 'service' or 'train' modules can already be imported, return cwd.
    2. If REPO_ROOT env var is defined and points to a folder that contains
       'service/' or 'train/', insert it into sys.path and use it.
    3. Walk up from the current file location's parents plus a few common roots
       (/app, cwd) and pick the first directory containing 'service/' or
       'train/'.
    4. As a last resort, return cwd and emit a warning.

    Returns
    -------
    pathlib.Path
        The resolved repository root directory.
    """
    if _iu.find_spec("service") is not None or _iu.find_spec("train") is not None:
        return Path.cwd()

    env_root = os.getenv("REPO_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if (p / "service").exists() or (p / "train").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            print(f"[train_job] repo_root from REPO_ROOT: {p}", flush=True)
            return p

    here = Path(__file__).resolve()
    candidates = [here] + list(here.parents) + [Path("/app"), Path.cwd()]
    for c in candidates:
        try:
            if (c / "service").exists() or (c / "train").exists():
                if str(c) not in sys.path:
                    sys.path.insert(0, str(c))
                print(f"[train_job] repo_root resolved: {c}", flush=True)
                return c
        except Exception:
            pass

    print("[train_job][warn] unable to locate repo root containing 'service/' or 'train/'", file=sys.stderr)
    return Path.cwd()

# ─────────────────────────────────────────────
# Versioning helpers (latest.json → bump)
# ─────────────────────────────────────────────

def _read_latest_version(cli: storage.Client, bucket: str, prefix: str) -> Optional[str]:
    """
    Read gs://bucket/prefix/latest.json and return the 'version' field if present.

    Parameters
    ----------
    cli : google.cloud.storage.Client
        Storage client.
    bucket : str
        Bucket name.
    prefix : str
        Prefix under which latest.json lives.

    Returns
    -------
    str or None
        The version string if found and non-empty, otherwise None.
    """
    try:
        blob = cli.bucket(bucket).blob(f"{prefix.rstrip('/')}/latest.json")
        if not blob.exists():
            return None
        data = blob.download_as_bytes()
        obj = json.loads(data.decode("utf-8"))
        v = obj.get("version")
        if isinstance(v, str) and v.strip():
            return v.strip()
        return None
    except Exception as e:
        print(f"[version][warn] unable to read latest.json: {e}")
        return None


def _parse_semver(s: Optional[str]) -> Tuple[int,int,int]:
    """
    Parse 'MAJOR.MINOR[.PATCH]' into (major, minor, patch) integers.

    Parameters
    ----------
    s : str or None
        Semantic version string, e.g. '2.3.1'. When None or invalid,
        falls back to (2,0,0).

    Returns
    -------
    (int, int, int)
        Parsed (major, minor, patch).
    """
    if not s:
        return (2,0,0)
    m = re.match(r"^\s*(\d+)\.(\d+)(?:\.(\d+))?\s*$", s)
    if not m:
        return (2,0,0)
    maj = int(m.group(1)); minr = int(m.group(2)); pat = int(m.group(3) or 0)
    return (maj,minr,pat)


def _bump_version(prev: Optional[str], mode: str = "minor") -> Tuple[str,str]:
    """
    Compute the next version from a previous semantic version.

    Parameters
    ----------
    prev : str or None
        Previous version string (e.g., '2.0.0'). When None or invalid,
        '2.0.0' is used as a base.
    mode : {"major", "minor", "patch"}, default "minor"
        Type of bump to apply.

    Returns
    -------
    (str, str)
        Tuple (base, next) where:
        - base is the resolved current version (e.g. '2.0.0')
        - next is the bumped version (e.g. '2.1.0')
    """
    base_tuple = _parse_semver(prev)
    maj,minr,pat = base_tuple
    mode = (mode or "minor").lower()
    if mode == "major":
        next_v = f"{maj+1}.0.0"
    elif mode == "patch":
        next_v = f"{maj}.{minr}.{pat+1}"
    else:
        next_v = f"{maj}.{minr+1}.0"
    base_v = f"{maj}.{minr}.{pat}"
    return base_v, next_v


def _resolve_and_export_version(cli: storage.Client) -> None:
    """
    Resolve current model version from latest.json and export version env vars.

    Behavior
    --------
    - If MODEL_GCS_BUCKET and MODEL_GCS_PREFIX are set:
        - Read latest.json under that prefix.
        - Parse its 'version' field as the base version.
        - Compute MODEL_NEXT_VERSION by bumping base according to VERSION_BUMP.
    - Otherwise:
        - Fallback to a default base '2.0.0' and bump from there.

    Exported environment variables
    ------------------------------
    MODEL_BASE_VERSION : str
        Resolved current version (or '2.0.0' when none is found).
    MODEL_NEXT_VERSION : str
        Bumped version according to VERSION_BUMP (minor by default).
    """
    bucket = os.environ.get("MODEL_GCS_BUCKET")
    prefix = os.environ.get("MODEL_GCS_PREFIX")
    bump = os.environ.get("VERSION_BUMP", "minor").lower()

    if bucket and prefix:
        prev = _read_latest_version(cli, bucket, prefix)
        base_v, next_v = _bump_version(prev, bump)
    else:
        print("[version][note] MODEL_GCS_BUCKET/PREFIX not set → default versioning used (2.0.0 → 2.1.0)")
        base_v, next_v = _bump_version("2.0.0", bump)

    os.environ["MODEL_BASE_VERSION"] = base_v
    os.environ["MODEL_NEXT_VERSION"] = next_v
    print(f"[version] base='{base_v}'  next='{next_v}'  (bump={bump})")

# ─────────────────────────────────────────────
# Dynamic import of the training callable
# ─────────────────────────────────────────────

def _find_training_callable(module_name: str) -> Callable[..., Any]:
    """
    Discover a valid training entrypoint inside the given module.

    Search order
    ------------
    1. Look for a callable function among:
       ['train', 'train_model', 'train_forecast', 'fit', 'run', 'main'].
    2. Fallback: look for a class with a .fit() method and create a wrapper
       that instantiates the class and calls .fit(**kwargs).

    Parameters
    ----------
    module_name : str
        Fully qualified Python module name (e.g. 'service.core.forecast').

    Returns
    -------
    callable
        A callable object that can be invoked with keyword arguments to train.

    Raises
    ------
    ImportError
        If no valid training function or class is found.
    """
    mod = importlib.import_module(module_name)
    candidates = ["train", "train_model", "train_forecast", "fit", "run", "main"]
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            print(f"[train_job] using training entrypoint: {module_name}.{name}", flush=True)
            return fn
    # fallback: class with .fit()
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and callable(getattr(obj, "fit", None)):
            def _callable_wrapper(**kwargs):
                inst = obj()
                return inst.fit(**kwargs) if kwargs else inst.fit()
            print(f"[train_job] using class entrypoint: {module_name}.{attr}.fit", flush=True)
            return _callable_wrapper
    raise ImportError(f"Aucune fonction d'entraînement trouvée dans {module_name}.")


def _adapt_kwargs_for_signature(fn: Callable[..., Any], base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt base keyword arguments to match the target function signature.

    This helper:
    - Inspects the target callable's signature.
    - Renames a few well-known parameters when needed, for example:
        - horizon_minutes → horizon_bins / horizon_min / horizon / minutes_ahead /
          target_horizon_min
        - lookback_days  → lookback / window_days / days
    - Drops keys that are not supported by the callable unless it accepts
      arbitrary **kwargs.

    Parameters
    ----------
    fn : callable
        Training callable whose signature will be inspected.
    base_kwargs : dict
        Base keyword arguments to adapt (e.g. src, horizon_bins, horizon_minutes,
        lookback_days, out_path, ...).

    Returns
    -------
    dict
        Adapted kwargs dict, safe to use when calling `fn`.
    """
    sig = inspect.signature(fn)
    params = sig.parameters
    km = dict(base_kwargs)

    # map horizon_minutes -> horizon_bins/etc si besoin
    if "horizon_minutes" in km and "horizon_minutes" not in params:
        for alias in ["horizon_bins", "horizon_min", "horizon", "minutes_ahead", "target_horizon_min"]:
            if alias in params:
                km[alias] = km.pop("horizon_minutes")
                break
        else:
            km.pop("horizon_minutes", None)

    if "lookback_days" in km and "lookback_days" not in params:
        for alias in ["lookback", "window_days", "days"]:
            if alias in params:
                km[alias] = km.pop("lookback_days")
                break
        else:
            km.pop("lookback_days", None)

    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_kwargs:
        print(f"[train_job] call with kwargs (var-kwargs accepted): {km}", flush=True)
        return km
    filtered = {k: v for k, v in km.items() if k in params}
    print(f"[train_job] call with kwargs: {filtered}", flush=True)
    return filtered

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> int:
    """
    Cloud Run job entrypoint for model training.

    High-level flow
    ---------------
    1. Read configuration from environment variables:
       - TRAIN_EXPORT_GCS : GCS URI of either:
           - a single training parquet, or
           - a daily prefix containing compact_YYYY-MM-DD.parquet shards.
       - HORIZON_MIN      : forecast horizon in minutes (default: 15).
       - LOOKBACK_DAYS    : number of days to look back when concatenating
                            daily shards (default: 30).
    2. Ensure MODEL_TYPE is set (defaults to 'xgb').
    3. Compute HORIZON_BINS from HORIZON_MIN (5-minute bins).
    4. Resolve model version (latest.json) and export MODEL_BASE_VERSION and
       MODEL_NEXT_VERSION environment variables.
    5. Depending on TRAIN_EXPORT_GCS:
       - If it ends with '.parquet' → download directly to a local Parquet.
       - Else → treat it as a prefix and list+concat daily shards.
    6. Ensure the repository root is on sys.path and resolve the training
       module (service.core.forecast preferred).
    7. Find a valid training callable in the module and adapt kwargs to its
       signature.
    8. Run training and log returned metrics if any.
    9. If the trainer did not self-publish (MODEL_GCS_BUCKET/PREFIX not set),
       upload the resulting model.joblib to MODEL_GCS (or a default path).

    Environment variables
    ---------------------
    TRAIN_EXPORT_GCS : str (required)
        GCS URI pointing either to a training parquet file or to a daily prefix.
    HORIZON_MIN : str, default "15"
        Forecast horizon in minutes.
    LOOKBACK_DAYS : str, default "30"
        Lookback window in days for daily shards.
    MODEL_TYPE : str, default "xgb"
        Model family used by the training script.
    MODEL_GCS_BUCKET, MODEL_GCS_PREFIX : str (optional)
        When both are set, the trainer is expected to self-publish the model
        artifacts under this prefix; the job will then skip the manual upload.
    MODEL_GCS : str, optional
        Fallback GCS URI where to store the model artifact if self-publishing
        is not configured. Defaults to an internal gs:// path.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    # Inputs
    TRAIN_EXPORT_GCS = os.environ["TRAIN_EXPORT_GCS"]  # gs://.../training/*.parquet OR gs://.../daily (prefix)
    HORIZON_MIN = int(os.environ.get("HORIZON_MIN", "15"))
    LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "30"))

    # ✅ Force XGB by default if not set (aligned with forecast.py XGB-only path)
    os.environ.setdefault("MODEL_TYPE", "xgb")

    # Horizon in 5-min bins (horizon_bins is used by the trainer)
    HORIZON_BINS = max(1, HORIZON_MIN // 5)

    tmp_root = Path(tempfile.gettempdir()) / "velib_train"
    local_file = tmp_root / "exports" / "velib_concat.parquet"
    cli = storage.Client()

    # 0) Version resolution (latest.json -> bump) → export ENV
    _resolve_and_export_version(cli)

    # 1) Training dataset: download single Parquet vs concat recent daily shards
    if TRAIN_EXPORT_GCS.endswith(".parquet"):
        print(f"[train_job] download {TRAIN_EXPORT_GCS} → {local_file}", flush=True)
        _download(cli, TRAIN_EXPORT_GCS, local_file)
    else:
        print(f"[train_job] list & concat daily shards from prefix={TRAIN_EXPORT_GCS}", flush=True)
        shard_uris = _list_daily(cli, TRAIN_EXPORT_GCS, LOOKBACK_DAYS)
        _concat_locally(cli, shard_uris, local_file)

    # 2) Ensure repo root is on sys.path before importing training module
    _ensure_repo_on_path()

    # 3) Determine the training module (fixing layout inconsistencies)
    module_name = os.getenv("TRAIN_MODULE")
    if not module_name:
        # order: core → tools → train → flat (compat)
        import service.core.forecast  # recommended
        module_name = "service.core.forecast"

    # 4) Local model output path
    model_out = tmp_root / "model.joblib"
    model_out.parent.mkdir(parents=True, exist_ok=True)

    # 5) Call training
    train_fn = _find_training_callable(module_name)
    base_kwargs = {
        "src": str(local_file),
        # both are passed for robustness; trainer XGB uses horizon_bins
        "horizon_bins": HORIZON_BINS,
        "horizon_minutes": HORIZON_MIN,
        "lookback_days": LOOKBACK_DAYS,
        "out_path": str(model_out),
    }
    call_kwargs = _adapt_kwargs_for_signature(train_fn, base_kwargs)

    print("[train_job] start training", flush=True)
    metrics = train_fn(**call_kwargs)
    if metrics is not None:
        print("[train_job] metrics:", metrics, flush=True)

    # 6) Publishing
    # Case 1: forecast.py self-publishes if MODEL_GCS_BUCKET/PREFIX are present
    has_self_publish = bool(os.environ.get("MODEL_GCS_BUCKET") and os.environ.get("MODEL_GCS_PREFIX"))

    if has_self_publish:
        print("[train_job] trainer self-publishing enabled (MODEL_GCS_BUCKET/MODEL_GCS_PREFIX set) → skip manual upload", flush=True)
    else:
        # Fallback: upload a single “compat” artifact (latest.joblib by default)
        MODEL_GCS = os.environ.get(
            "MODEL_GCS",
            f"gs://velib-forecast-472820_cloudbuild/velib/models/h15/latest.joblib"
        )
        local_model = model_out if model_out.exists() else Path("model.joblib")
        gcs_uri = _upload_to_gcs(local_model, MODEL_GCS)
        print(f"[train_job] model uploaded → {gcs_uri}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
