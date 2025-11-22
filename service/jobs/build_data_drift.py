# service/jobs/build_data_drift.py

"""
Vélib’ Forecast — Data Drift builder (Schema v1.4)

This job:
- Reads event exports (events_YYYY-MM-DD.parquet) from the exports bucket.
- Normalizes the event schema (timestamps, station_id, numerics).
- Splits data into a **reference** window and a **current** window.
- Computes feature-wise drift metrics:
    - Population Stability Index (PSI)
    - Kolmogorov–Smirnov statistic (KS)
    - Standardized mean and variance deltas
- Computes a global PSI proxy and a daily EMA on `occ_ratio`.
- Produces per-zone drift metrics (PSI by spatial zone) for maps.
- Writes a LATEST-only JSON bundle for the monitoring UI:

  {GCS_MONITORING_PREFIX}/monitoring/data/drift/latest/
    - psi_by_feature.json
    - ks_by_feature.json
    - deltas_by_feature.json
    - psi_global_daily_ema.json
    - summary.json
    - alerts.json
    - bounds.json
    - zones.json
    - features_detected.json

Environment
-----------
Unified env variable names (single source of truth):

- GCS_EXPORTS_PREFIX
    gs://.../velib/exports
    Required. Root prefix containing events_YYYY-MM-DD.parquet.

- GCS_MONITORING_PREFIX
    gs://.../velib[/monitoring]
    Required. Root prefix for monitoring artifacts; '/monitoring' is appended
    automatically if missing.

- MON_TZ
    Monitoring timezone (e.g. "Europe/Paris").
    Default: "Europe/Paris".

- MON_LAST_DAYS
    Current drift window size (in days). Example: 14.

- MON_REF_DAYS
    Reference drift window size (in days). Example: 28.

- FORECAST_HORIZONS
    Comma-separated list of forecast horizons ("15,60").
    Not used in this job, kept for interface consistency.

Drift design principles
-----------------------
- Only **true features** are used for drift (no raw timestamps, no lat/lon,
  no sine/cosine encodings or derived time indicators).
- PSI / KS are computed on daily averages per station.
- Numeric features must have at least 5 valid points in both ref & current
  windows, otherwise the feature is skipped with a log message.
- Global PSI is:
    - PSI on `occ_ratio` if present,
    - otherwise median PSI across features.
- Alerts are raised on global PSI thresholds:
    - >= 0.25 → high
    - >= 0.10 → medium
"""

from __future__ import annotations
import os, re, json, sys
from io import BytesIO
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception as e:
    raise RuntimeError("pyarrow is required") from e

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage is required") from e

SCHEMA_VERSION = "1.4"

# ──────────────────────────────────────────────────────────────────────────────
# ENV — unified names
# ──────────────────────────────────────────────────────────────────────────────

def _env(name: str, default=None):
    """
    Read an environment variable, with a default when missing or empty.

    Parameters
    ----------
    name : str
        Environment variable name.
    default :
        Default value to return when the variable is unset or empty.

    Returns
    -------
    Any
        The environment value (string) or the provided default.
    """
    v = os.environ.get(name)
    return v if (v is not None and v != "") else default


def _env_int(name: str, default: int) -> int:
    """
    Read an integer environment variable with a default.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : int
        Default integer value when parsing fails or variable is missing.

    Returns
    -------
    int
        Parsed integer value.
    """
    try:
        return int(_env(name, default))
    except Exception:
        return default


def _env_list_int(name: str, default_csv: str) -> list[int]:
    """
    Read a comma-separated list of ints from an environment variable.

    Parameters
    ----------
    name : str
        Environment variable name.
    default_csv : str
        Default CSV string to use when the variable is missing.

    Returns
    -------
    list of int
        Parsed list of integers; invalid tokens are silently discarded.
    """
    raw = str(_env(name, default_csv))
    out: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except Exception:
            pass
    return out

GCS_EXPORTS_PREFIX    = _env("GCS_EXPORTS_PREFIX")
GCS_MONITORING_PREFIX = _env("GCS_MONITORING_PREFIX")
MON_TZ                = _env("MON_TZ", "Europe/Paris")
MON_LAST_DAYS         = _env_int("MON_LAST_DAYS", 14)
MON_REF_DAYS          = _env_int("MON_REF_DAYS", 28)
FORECAST_HORIZONS     = _env_list_int("FORECAST_HORIZONS", "15,60")  # (not used here)

# Sanity checks GCS
if not (GCS_EXPORTS_PREFIX and GCS_EXPORTS_PREFIX.startswith("gs://")):
    raise RuntimeError("GCS_EXPORTS_PREFIX missing or invalid")
if not (GCS_MONITORING_PREFIX and GCS_MONITORING_PREFIX.startswith("gs://")):
    raise RuntimeError("GCS_MONITORING_PREFIX missing or invalid")

# ──────────────────────────────────────────────────────────────────────────────
# GCS helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split(gs: str) -> Tuple[str, str]:
    """
    Split a GCS URI `gs://bucket/path` into (bucket, key).

    Parameters
    ----------
    gs : str
        GCS URI starting with "gs://".

    Returns
    -------
    (str, str)
        Tuple (bucket_name, object_key).
    """
    assert gs.startswith("gs://"), f"bad GCS uri: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k.rstrip("/")


def _read_parquet_blob_to_df(blob: "storage.Blob") -> pd.DataFrame:
    """
    Download a Parquet blob from GCS and load it as a pandas DataFrame.

    Parameters
    ----------
    blob : google.cloud.storage.Blob
        Parquet file blob.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the Parquet content.
    """
    buf = BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    tbl = pq.read_table(buf)
    return tbl.to_pandas()


def _list_event_blobs(exports_prefix: str, start_date: datetime, end_date: datetime) -> List["storage.Blob"]:
    """
    List event blobs (events_YYYY-MM-DD.parquet) within a given UTC date window.

    Parameters
    ----------
    exports_prefix : str
        GCS prefix where events_*.parquet are stored (GCS_EXPORTS_PREFIX).
    start_date : datetime
        Inclusive UTC start date.
    end_date : datetime
        Inclusive UTC end date.

    Returns
    -------
    list of google.cloud.storage.Blob
        Sorted list of blobs matching the date constraints.
    """
    bkt, key_prefix = _split(exports_prefix)
    client = storage.Client()
    bucket = client.bucket(bkt)
    blobs = list(client.list_blobs(bucket, prefix=key_prefix.strip("/") + "/"))
    pat = re.compile(r"events_(\d{4}-\d{2}-\d{2})\.parquet$")
    out: List["storage.Blob"] = []
    for bl in blobs:
        m = pat.search(bl.name)
        if not m:
            continue
        d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        if start_date.date() <= d <= end_date.date():
            out.append(bl)
    out.sort(key=lambda b: b.name)
    return out


def _upload_json_gs(obj: dict | list, gs_uri: str):
    """
    Upload a JSON-serializable object as a JSON file to GCS.

    Notes
    -----
    - All floats are sanitized to avoid NaN/Inf (replaced by null).
    - JSON is encoded as UTF-8 and written with compact separators.

    Parameters
    ----------
    obj : dict or list
        JSON-serializable object to upload.
    gs_uri : str
        Destination GCS URI (gs://bucket/path/to/file.json).
    """
    def _san(o):
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_san(v) for v in o]
        if isinstance(o, float):
            return float(o) if np.isfinite(o) else None
        return o
    safe = _san(obj)
    bkt, key = _split(gs_uri)
    data = json.dumps(safe, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    storage.Client().bucket(bkt).blob(key).upload_from_string(data, content_type="application/json")
    print(f"[data.drift] wrote → {gs_uri} ({len(data):,} bytes)")

# ──────────────────────────────────────────────────────────────────────────────
# Event normalization (build_datasets.py schema)
# ──────────────────────────────────────────────────────────────────────────────

KEY_STR = {"station_id", "status", "name"}
KEY_TIME = {"tbin_utc", "ts", "timestamp"}

# Explicit exclusions for drift features
EXCLUDE_EXACT = {
    # identifiers & time-like
    "station_id", "date_local", "dow", "hour", "h", "min", "ts_local",
    # geo
    "lat", "lon",
    # we never use raw timestamps as features
    "tbin_utc", "ts", "timestamp",
}
EXCLUDE_PATTERNS = [
    r".*_(sin|cos)$",
    r"^(hour|minute|dow)(_|$)",
]

def _is_time_or_coord(col: str) -> bool:
    """
    Return True if a column is time-like or coordinate-like and must be excluded.

    Rules
    -----
    - Exclude explicit names (station_id, lat, lon, tbin_utc, etc.).
    - Exclude columns matching EXCLUDE_PATTERNS (e.g., *_sin, *_cos, hour_*).
    """
    if col in EXCLUDE_EXACT:
        return True
    return any(re.match(p, col) for p in EXCLUDE_PATTERNS)


def _ensure_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize event DataFrame and preserve critical types.

    Requirements
    ------------
    - Must contain at least 'tbin_utc' and 'station_id'.

    Transformations
    ---------------
    - 'station_id' → string.
    - 'tbin_utc'   → datetime64[ns] naive (UTC).
    - datetime columns are left untouched.
    - numeric columns are left untouched.
    - other object/bool columns are coerced to numeric when possible.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw event data.

    Returns
    -------
    pandas.DataFrame
        Normalized events DataFrame.
    """
    if not {"tbin_utc", "station_id"}.issubset(df.columns):
        raise RuntimeError("Missing minimal columns: tbin_utc/station_id")

    out = df.copy()

    # station_id as string
    out["station_id"] = out["station_id"].astype("string")

    # tbin_utc as naive UTC datetime
    out["tbin_utc"] = pd.to_datetime(out["tbin_utc"], errors="coerce", utc=True).dt.tz_convert(None)

    # For each column: coerce object/bool to numeric, leave datetime/numeric intact
    for c in out.columns:
        if c in KEY_STR or c in KEY_TIME:
            continue  # station_id/status/name and timestamps untouched
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            continue
        try:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        except Exception:
            pass

    return out


def _to_local(df: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    """
    Add local time columns from UTC tbin_utc.

    Columns added
    -------------
    - date_local : local calendar date
    - dow        : day of week (0=Monday)
    - hour       : local hour
    - ts_local   : localized timestamp

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with 'tbin_utc' column.
    tz : str or None
        Timezone name (e.g., 'Europe/Paris'). If None, remain in UTC.

    Returns
    -------
    pandas.DataFrame
        DataFrame with additional local time columns.
    """
    dt = pd.to_datetime(df["tbin_utc"], errors="coerce", utc=True)
    dt_local = dt.dt.tz_convert(tz) if tz else dt
    return df.assign(date_local=dt_local.dt.date, dow=dt_local.dt.dayofweek, hour=dt_local.dt.hour, ts_local=dt_local)

# ──────────────────────────────────────────────────────────────────────────────
# Drift metrics
# ──────────────────────────────────────────────────────────────────────────────

def _valid_series(x) -> pd.Series:
    """
    Convert input to a 1D numeric Series without NaNs.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    pandas.Series
        Clean numeric series; returns an empty Series when coercion fails.
    """
    try:
        s = pd.to_numeric(pd.Series(x, copy=False), errors="coerce").dropna()
        return s if isinstance(s, pd.Series) else pd.Series([], dtype=float)
    except Exception:
        return pd.Series([], dtype=float)


def _psi_continuous(ref: pd.Series, cur: pd.Series, bins: int = 20, eps: float = 1e-9) -> float:
    """
    Compute a binned Population Stability Index (PSI) between two samples.

    Steps
    -----
    - Clean both samples with _valid_series.
    - Build quantile-based bins from the reference sample.
    - Compute distributions (p_ref, p_cur) across these bins.
    - PSI = sum((p_ref - p_cur) * log(p_ref / p_cur)).

    Parameters
    ----------
    ref : pandas.Series
        Reference sample.
    cur : pandas.Series
        Current sample.
    bins : int, default 20
        Target number of quantile bins.
    eps : float, default 1e-9
        Small value added to probabilities for numerical stability.

    Returns
    -------
    float
        PSI value; NaN when data is insufficient or degenerate.
    """
    a = _valid_series(ref)
    b = _valid_series(cur)
    if len(a) < 5 or len(b) < 5:
        return np.nan
    if a.min() == a.max() or b.min() == b.max():
        return np.nan
    try:
        q = np.unique(np.nanquantile(a, np.linspace(0, 1, bins + 1)))
    except Exception:
        return np.nan
    if q.size < 3:
        return np.nan
    ca, _ = np.histogram(a, bins=q)
    cb, _ = np.histogram(b, bins=q)
    if ca.sum() == 0 or cb.sum() == 0:
        return np.nan
    pa = (ca / ca.sum()).astype(float) + eps
    pb = (cb / cb.sum()).astype(float) + eps
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def _ks_stat(ref: pd.Series, cur: pd.Series) -> float:
    """
    Compute a discretized Kolmogorov–Smirnov statistic between two samples.

    Implementation notes
    --------------------
    - Use combined quantiles from both samples as grid.
    - Approximate CDFs via histograms.
    - Return max absolute difference between the two CDFs.

    Parameters
    ----------
    ref : pandas.Series
        Reference sample.
    cur : pandas.Series
        Current sample.

    Returns
    -------
    float
        KS statistic; NaN when data is insufficient or degenerate.
    """
    a = _valid_series(ref)
    b = _valid_series(cur)
    if len(a) < 5 or len(b) < 5:
        return np.nan
    both = pd.concat([a, b], ignore_index=True)
    if both.empty or both.min() == both.max():
        return np.nan
    try:
        q = np.unique(np.nanquantile(both, np.linspace(0, 1, 201)))
    except Exception:
        return np.nan
    if q.size < 3:
        return np.nan
    ca, _ = np.histogram(a, bins=q)
    cb, _ = np.histogram(b, bins=q)
    if ca.sum() == 0 or cb.sum() == 0:
        return np.nan
    cdfa = np.cumsum(ca) / ca.sum()
    cdfb = np.cumsum(cb) / cb.sum()
    return float(np.max(np.abs(cdfa - cdfb)))


def _delta_mean_var(ref: pd.Series, cur: pd.Series) -> tuple[float, float]:
    """
    Compute standardized mean and variance deltas between two samples.

    Definitions
    -----------
    - dm = (mean_cur - mean_ref) / std_ref
    - dv = (var_cur - var_ref) / var_ref

    Parameters
    ----------
    ref : pandas.Series
        Reference sample.
    cur : pandas.Series
        Current sample.

    Returns
    -------
    (float, float)
        Tuple (delta_mean, delta_var), possibly NaN when data is insufficient.
    """
    a = _valid_series(ref)
    b = _valid_series(cur)
    if len(a) < 5 or len(b) < 5:
        return (np.nan, np.nan)
    if a.std(ddof=1) == 0:
        return (np.nan, np.nan)
    dm = (b.mean() - a.mean()) / (a.std(ddof=1) + 1e-9)
    avar = a.var(ddof=1)
    dv = (b.var(ddof=1) - avar) / (avar + 1e-9)
    return (float(dm), float(dv))


def _split_windows(df: pd.DataFrame, current_days: int, reference_days: int) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Timestamp]]:
    """
    Split events into reference and current time windows.

    Windows
    -------
    - tmax = max(tbin_utc)
    - current window   : [tmax - current_days, tmax]
    - reference window : [tmax - current_days - reference_days, tmax - current_days)

    Parameters
    ----------
    df : pandas.DataFrame
        Normalized events with 'tbin_utc'.
    current_days : int
        Size of the current window (days).
    reference_days : int
        Size of the reference window (days).

    Returns
    -------
    (DataFrame, DataFrame, dict)
        Tuple (ref_df, cur_df, bounds) where bounds contains timestamps:
        - tmax
        - t_cur_start
        - t_ref_start
        - t_ref_end
    """
    if df.empty:
        return df.iloc[0:0].copy(), df.iloc[0:0].copy(), {}
    tmax = df["tbin_utc"].max()
    t_cur_start = tmax - pd.Timedelta(days=current_days)
    t_ref_end = t_cur_start
    t_ref_start = t_ref_end - pd.Timedelta(days=reference_days)
    ref = df[(df["tbin_utc"] >= t_ref_start) & (df["tbin_utc"] < t_ref_end)].copy()
    cur = df[(df["tbin_utc"] >= t_cur_start) & (df["tbin_utc"] <= tmax)].copy()
    bounds = {"tmax": tmax, "t_cur_start": t_cur_start, "t_ref_start": t_ref_start, "t_ref_end": t_ref_end}
    return ref, cur, bounds


def _assign_zone(df: pd.DataFrame) -> pd.Series:
    """
    Compute a generic "zone" identifier per station for spatial drift.

    Strategy
    --------
    1. Prefer a semantic zone field if present:
        - 'arrondissement', 'arr', 'zone', 'district'
    2. Otherwise, derive a pseudo-zone from lat/lon by rounding:
        - lat_rounded = round(lat * 100) / 100
        - lon_rounded = round(lon * 100) / 100
        - zone = "lat_rounded,lon_rounded"

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with optional 'lat' and 'lon' columns.

    Returns
    -------
    pandas.Series
        Series of zone identifiers (object dtype).
    """
    for c in ("arrondissement", "arr", "zone", "district"):
        if c in df.columns:
            return df[c].astype(str)
    lat = pd.to_numeric(df.get("lat"), errors="coerce")
    lon = pd.to_numeric(df.get("lon"), errors="coerce")
    mask = lat.notna() & lon.notna()
    z = pd.Series(index=df.index, dtype=object, name="zone")
    lat_r = (lat[mask] * 100).round() / 100.0
    lon_r = (lon[mask] * 100).round() / 100.0
    z.loc[mask] = lat_r.astype(str) + "," + lon_r.astype(str)
    return z


def _compute_drift(events: pd.DataFrame, current_days: int, reference_days: int, tz: Optional[str]) -> dict:
    """
    Compute data drift metrics for the events dataset.

    Steps
    -----
    1. Add local time columns (date_local, dow, hour, ts_local).
    2. Split into reference and current windows based on MON_REF_DAYS / MON_LAST_DAYS.
    3. Aggregate events to daily averages per station.
    4. Automatically discover numeric candidate features (excluding coord/time).
    5. For each feature:
        - PSI (continuous)
        - KS statistic
        - standardized mean and variance deltas
    6. Compute global PSI (on occ_ratio when available, else median PSI).
    7. Compute a daily EMA on occ_ratio for trend visualization.
    8. Build zone-level PSI metrics (by occ_ratio if available, else bikes).
    9. Build summary and alert payloads.

    Parameters
    ----------
    events : pandas.DataFrame
        Normalized events DataFrame as returned by `_ensure_events`.
    current_days : int
        Number of days in the current window.
    reference_days : int
        Number of days in the reference window.
    tz : str or None
        Timezone for local calendar aggregation.

    Returns
    -------
    dict
        Dictionary with:
        - psi_df, ks_df, deltas_df : feature-wise metrics (DataFrames)
        - psi_daily_ema            : daily EMA DataFrame
        - summary                  : global summary dict
        - alerts                   : list of alert dicts
        - bounds                   : dict with UTC window bounds
        - zones                    : dict with zone-level PSI rows
        - feature_list             : list of numeric feature names used
    """
    df = _to_local(events, tz)
    ref, cur, bounds = _split_windows(df, current_days=current_days, reference_days=reference_days)

    # Aggregation per (date_local, station) — daily averages
    def agg(d: pd.DataFrame):
        # Numeric candidate columns (exclude coord/time here too)
        num_cols = [
            c for c in d.columns
            if c not in {"station_id","tbin_utc","status","name","date_local","dow","hour","ts_local"}
            and pd.api.types.is_numeric_dtype(d[c])
            and not _is_time_or_coord(c)
        ]
        # Keep lat/lon separately for zones map
        keep = ["date_local","station_id"] + num_cols + [c for c in ("lat","lon") if c in d.columns]
        return d[keep].groupby(["date_local","station_id"], dropna=True).mean(numeric_only=True).reset_index()

    ref = agg(ref); cur = agg(cur)

    # Auto features: numeric intersection, excluding coord/time
    common_num = [
        c for c in ref.columns
        if c not in {"date_local","station_id"} and c in cur.columns
        and pd.api.types.is_numeric_dtype(ref[c]) and not _is_time_or_coord(c)
    ]

    rows_psi, rows_ks, rows_delta = [], [], []
    for f in sorted(common_num):
        try:
            rf = _valid_series(ref[f]); cf = _valid_series(cur[f])
            if len(rf) < 5 or len(cf) < 5:
                print(f"[data.drift][info] skip feature '{f}' (insufficient data)")
                continue
            psi = _psi_continuous(rf, cf)
            ks  = _ks_stat(rf, cf)
            dm, dv = _delta_mean_var(rf, cf)
            rows_psi.append({"feature": f, "psi": float(psi) if np.isfinite(psi) else None})
            rows_ks.append({"feature": f, "ks":  float(ks)  if np.isfinite(ks)  else None})
            rows_delta.append({"feature": f,
                               "delta_mean": float(dm) if np.isfinite(dm) else None,
                               "delta_var":  float(dv) if np.isfinite(dv) else None})
        except Exception as e:
            print(f"[data.drift][warn] metrics failed for feature '{f}': {e}")

    psi_df = pd.DataFrame(rows_psi)
    ks_df = pd.DataFrame(rows_ks)
    d_df = pd.DataFrame(rows_delta)

    # Daily EMA on occ_ratio (if available)
    ema_df = pd.DataFrame(columns=["date_local","psi_ema"])
    if "occ_ratio" in events.columns:
        by_day = df.groupby("date_local")["occ_ratio"].apply(
            lambda s: float(np.nanmean(pd.to_numeric(s, errors='coerce')))
        ).reset_index()
        by_day = by_day.sort_values("date_local")
        alpha = 2 / (7 + 1.0)  # 7-day EMA
        ema, last = [], None
        for _, r in by_day.iterrows():
            x = r["occ_ratio"]
            if pd.isna(x):
                ema.append(np.nan); continue
            last = x if last is None else (alpha * x + (1 - alpha) * last)
            ema.append(last)
        ema_df = pd.DataFrame({"date_local": by_day["date_local"].astype(str), "psi_ema": ema})

    # Global PSI
    psi_global = None
    if not psi_df.empty:
        if "occ_ratio" in list(psi_df["feature"]):
            v = psi_df.loc[psi_df["feature"]=="occ_ratio","psi"].values[0]
            psi_global = float(v) if v is not None and np.isfinite(v) else None
        else:
            v = np.nanmedian([p for p in psi_df["psi"] if p is not None])
            psi_global = float(v) if np.isfinite(v) else None

    alerts = []
    if psi_global is not None:
        if psi_global >= 0.25:
            alerts.append({"level": "high", "code": "psi_global_high", "text": f"High global PSI ({psi_global:.3f})"})
        elif psi_global >= 0.10:
            alerts.append({"level": "medium", "code": "psi_global_medium", "text": f"Moderate global PSI ({psi_global:.3f})"})

    summary = {
        "psi_global": psi_global,
        "top_feature": (psi_df.sort_values("psi", ascending=False).iloc[0]["feature"] if not psi_df.empty else None),
        "top_feature_psi": (float(psi_df.sort_values("psi", ascending=False).iloc[0]["psi"]) if not psi_df.empty and psi_df.sort_values("psi", ascending=False).iloc[0]["psi"] is not None else None),
    }

    # Zone-level PSI (occ_ratio if available, else bikes) — robust
    zones_doc = {"rows": []}
    try:
        rows = []
        if {"lat", "lon"}.issubset(df.columns):
            ref_z = ref.assign(zone=_assign_zone(ref))
            cur_z = cur.assign(zone=_assign_zone(cur))
            for z, rsub in ref_z.groupby("zone", dropna=True):
                csub = cur_z[cur_z["zone"] == z]
                if csub.empty:
                    continue
                metric = None
                for m in ("occ_ratio", "bikes"):
                    if m in rsub.columns and m in csub.columns:
                        metric = m
                        break
                if not metric:
                    continue

                rvals = _valid_series(rsub[metric])
                cvals = _valid_series(csub[metric])
                if len(rvals) < 5 or len(cvals) < 5:
                    continue

                psi = _psi_continuous(rvals, cvals)
                lat = float(rsub["lat"].median()) if "lat" in rsub.columns and rsub["lat"].notna().any() else None
                lon = float(rsub["lon"].median()) if "lon" in rsub.columns and rsub["lon"].notna().any() else None
                zname = None if (z is None or (isinstance(z, float) and np.isnan(z))) else str(z)
                rows.append({"zone": zname, "psi": float(psi) if np.isfinite(psi) else None, "lat": lat, "lon": lon})
        zones_doc = {"rows": rows}
    except Exception as e:
        print(f"[data.drift] zones PSI failed: {e}")

    return {
        "psi_df": psi_df, "ks_df": ks_df, "deltas_df": d_df,
        "psi_daily_ema": ema_df,
        "summary": summary, "alerts": alerts,
        "bounds": {
            "tmax_utc": pd.Timestamp(bounds.get("tmax")).tz_localize("UTC").isoformat().replace("+00:00","Z") if bounds else None,
            "cur_start_utc": pd.Timestamp(bounds.get("t_cur_start")).tz_localize("UTC").isoformat().replace("+00:00","Z") if bounds else None,
            "ref_start_utc": pd.Timestamp(bounds.get("t_ref_start")).tz_localize("UTC").isoformat().replace("+00:00","Z") if bounds else None,
            "ref_end_utc": pd.Timestamp(bounds.get("t_ref_end")).tz_localize("UTC").isoformat().replace("+00:00","Z") if bounds else None,
        },
        "zones": zones_doc,
        "feature_list": sorted(common_num),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    """
    CLI entrypoint for the data drift monitoring job.

    Steps
    -----
    1. Determine the UTC reading window:
       days = max(MON_LAST_DAYS, MON_REF_DAYS, 7).
    2. List and read all events_YYYY-MM-DD.parquet in that window.
    3. Normalize events with `_ensure_events`.
    4. Compute drift metrics via `_compute_drift`.
    5. Normalize monitoring prefix (ensure a `/monitoring` segment).
    6. Write a LATEST-only JSON bundle under:

       {GCS_MONITORING_PREFIX}/monitoring/data/drift/latest

    Returns
    -------
    int
        Exit code (0 on success, 2 on fatal error).
    """
    # Reading window on GCS = >= max(MON_LAST_DAYS, MON_REF_DAYS, 7)
    now = datetime.now(timezone.utc)
    window_days = max(int(MON_LAST_DAYS), int(MON_REF_DAYS), 7)
    start = (now - timedelta(days=window_days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"[data.drift] window UTC: {start.date()} → {now.date()} (days={window_days})")

    blobs = _list_event_blobs(GCS_EXPORTS_PREFIX, start, now)
    if not blobs:
        print("[data.drift] no event blobs in window — nothing to do")
        return 0

    frames: List[pd.DataFrame] = []
    for bl in blobs:
        print(f"[read] {bl.name}")
        try:
            df = _read_parquet_blob_to_df(bl)
            frames.append(df)
        except Exception as e:
            print(f"[warn] failed to read {bl.name}: {e}")

    if not frames:
        print("[data.drift] no readable data — nothing to do")
        return 0

    ev = pd.concat(frames, ignore_index=True)
    if ev.empty:
        print("[data.drift] events empty — nothing to do")
        return 0

    ev_norm = _ensure_events(ev)
    res = _compute_drift(ev_norm, current_days=int(MON_LAST_DAYS), reference_days=int(MON_REF_DAYS), tz=MON_TZ)

    # Normalize monitoring prefix (append /monitoring if absent)
    mon_base = GCS_MONITORING_PREFIX.rstrip("/")
    if not mon_base.endswith("/monitoring"):
        mon_base = mon_base + "/monitoring"
    out_prefix = f"{mon_base}/data/drift/latest"

    files = {
        "psi_by_feature.json":       res["psi_df"].to_dict(orient="records"),
        "ks_by_feature.json":        res["ks_df"].to_dict(orient="records"),
        "deltas_by_feature.json":    res["deltas_df"].to_dict(orient="records"),
        "psi_global_daily_ema.json": res["psi_daily_ema"].to_dict(orient="records"),
        "summary.json":              {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"), **res["summary"]},
        "alerts.json":               res["alerts"],
        "bounds.json":               res["bounds"],
        "zones.json":                res["zones"],
        "features_detected.json":    res["feature_list"],
    }
    for fname, payload in files.items():
        _upload_json_gs(payload, f"{out_prefix}/{fname}")

    print(f"[data.drift] done → {out_prefix}/ (LATEST only)")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[data.drift][fatal] {e}", file=sys.stderr)
        sys.exit(2)
