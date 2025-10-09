# service/tools/train_model.py
# =============================================================================
# Cloud Run training entrypoint
# - Télécharge et concatène les shards daily depuis GCS
# - Localise automatiquement le module d'entraînement (service.tools/train ou train)
# - Appelle train_model (ou équivalent) avec adaptation des kwargs
# - Upload le modèle .joblib vers GCS
# =============================================================================

from __future__ import annotations
import os, sys, tempfile, re, importlib, inspect
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Tuple, List, Callable, Dict, Any

import pandas as pd
from google.cloud import storage
import importlib.util as _iu

# ─────────────────────────────────────────────
# Helpers utilitaires GCS
# ─────────────────────────────────────────────

RE_COMPACT = re.compile(r".*/compact_(\d{4}-\d{2}-\d{2})\.parquet$")

def _parse_gs(uri: str) -> Tuple[str, str]:
    assert uri.startswith("gs://"), f"Bad GCS uri: {uri}"
    bkt, key = uri[5:].split("/", 1)
    return bkt, key.rstrip("/")

def _download(cli: storage.Client, src_uri: str, dst_path: Path) -> None:
    bkt, key = _parse_gs(src_uri)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cli.bucket(bkt).blob(key).download_to_filename(str(dst_path))

def _upload_to_gcs(local_path: Path, dst_uri: str) -> str:
    """Upload a local file to gs://bucket/key and return the gs:// URI."""
    if not local_path.exists():
        raise FileNotFoundError(f"Local model not found: {local_path}")
    bkt, key = _parse_gs(dst_uri)
    storage.Client().bucket(bkt).blob(key).upload_from_filename(str(local_path))
    return f"gs://{bkt}/{key}"

def _list_daily(cli: storage.Client, gcs_prefix: str, lookback_days: int) -> List[str]:
    """Retourne les URIs GCS des shards daily à concaténer, filtrés par lookback."""
    bkt, prefix = _parse_gs(gcs_prefix)
    now = datetime.now(timezone.utc).date()
    min_day = now - timedelta(days=lookback_days)
    uris: List[str] = []

    # client.list_blobs accepte le nom du bucket
    for blob in cli.list_blobs(bkt, prefix=prefix + "/"):
        m = RE_COMPACT.match(blob.name)
        if not m:
            continue
        day = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        if day >= min_day:
            uris.append(f"gs://{bkt}/{blob.name}")

    uris.sort(key=lambda u: RE_COMPACT.match(u).group(1) if RE_COMPACT.match(u) else u)
    if not uris:
        raise RuntimeError(f"Aucun shard daily trouvé sous {gcs_prefix} sur {lookback_days} jours.")
    return uris

def _concat_locally(cli: storage.Client, shard_uris: List[str], out_path: Path) -> Path:
    """Télécharge chaque shard, concatène en un unique parquet local et retourne le chemin."""
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
# Localisation du repo root et correction sys.path
# ─────────────────────────────────────────────

def _ensure_repo_on_path() -> Path:
    """Insère le répertoire racine du repo dans sys.path pour supporter les layouts."""
    # déjà importable ?
    if _iu.find_spec("service") is not None or _iu.find_spec("train") is not None:
        return Path.cwd()

    # REPO_ROOT
    env_root = os.getenv("REPO_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if (p / "service").exists() or (p / "train").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            print(f"[train_job] repo_root from REPO_ROOT: {p}", flush=True)
            return p

    # scan
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
# Import dynamique de la fonction d'entraînement
# ─────────────────────────────────────────────

def _find_training_callable(module_name: str) -> Callable[..., Any]:
    """Cherche une fonction d'entraînement valide dans le module donné."""
    mod = importlib.import_module(module_name)
    candidates = ["train", "train_model", "train_forecast", "fit", "run", "main"]
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            print(f"[train_job] using training entrypoint: {module_name}.{name}", flush=True)
            return fn
    # si aucune fonction, essayer class avec fit()
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type):
            m_fit = getattr(obj, "fit", None)
            if callable(m_fit):
                def _callable_wrapper(**kwargs):
                    inst = obj()
                    return inst.fit(**kwargs) if kwargs else inst.fit()
                print(f"[train_job] using class entrypoint: {module_name}.{attr}.fit", flush=True)
                return _callable_wrapper
    raise ImportError(f"Aucune fonction d'entraînement trouvée dans {module_name}.")

def _adapt_kwargs_for_signature(fn: Callable[..., Any], base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Adapte les kwargs selon la signature de la fonction."""
    sig = inspect.signature(fn)
    params = sig.parameters
    km = dict(base_kwargs)

    # Map horizon_minutes -> (horizon_bins | horizon_min | horizon | minutes_ahead | target_horizon_min)
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
    TRAIN_EXPORT_GCS = os.environ["TRAIN_EXPORT_GCS"]
    HORIZON_MIN = int(os.environ.get("HORIZON_MIN", "15"))
    LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "30"))

    tmp_root = Path(tempfile.gettempdir()) / "velib_train"
    local_file = tmp_root / "exports" / "velib_concat.parquet"
    cli = storage.Client()

    # téléchargement et concaténation
    if TRAIN_EXPORT_GCS.endswith(".parquet"):
        print(f"[train_job] download {TRAIN_EXPORT_GCS} → {local_file}", flush=True)
        _download(cli, TRAIN_EXPORT_GCS, local_file)
    else:
        print(f"[train_job] list & concat daily shards from prefix={TRAIN_EXPORT_GCS}", flush=True)
        shard_uris = _list_daily(cli, TRAIN_EXPORT_GCS, LOOKBACK_DAYS)
        _concat_locally(cli, shard_uris, local_file)

    # Ajouter repo root
    _ensure_repo_on_path()

    # Déterminer le module d’entraînement (ENV prioritaire)
    module_name = os.getenv("TRAIN_MODULE")
    if not module_name:
        # priorité au nouveau module outils
        try:
            import service.tools.forecast  # layout racine, nouveau module
            module_name = "service.tools.forecast"
        except ModuleNotFoundError:
            try:
                import service.core.forecast  # layout racine, ancien module
                module_name = "service.train.forecast"
            except ModuleNotFoundError:
                import train.forecast          # layout aplati
                module_name = "train.forecast"

    # Préparer sortie
    model_out = tmp_root / "model.joblib"
    model_out.parent.mkdir(parents=True, exist_ok=True)
    MODEL_GCS = os.environ.get(
        "MODEL_GCS",
        f"gs://velib-forecast-472820_cloudbuild/velib/models/lgb_nbvelos_T+{HORIZON_MIN}min.joblib"
    )

    # Appeler fonction d’entraînement
    train_fn = _find_training_callable(module_name)
    base_kwargs = {
        "src": str(local_file),
        "out_path": str(model_out),
        "horizon_minutes": HORIZON_MIN,   # sera mappé vers horizon_bins / horizon_min selon la signature
        "lookback_days": LOOKBACK_DAYS,
    }
    call_kwargs = _adapt_kwargs_for_signature(train_fn, base_kwargs)

    print("[train_job] start training", flush=True)
    metrics = train_fn(**call_kwargs)
    print("[train_job] metrics:", metrics, flush=True)

    # Upload modèle
    local_model = model_out if model_out.exists() else Path("model.joblib")
    gcs_uri = _upload_to_gcs(local_model, MODEL_GCS)
    print(f"[train_job] model uploaded → {gcs_uri}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
