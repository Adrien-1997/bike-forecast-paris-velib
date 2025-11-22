#  build_serving_forecast.py

"""
Vélib’ Forecast — Serving Forecast job (short-horizon, 4h rolling window).

Role
----
This job builds a 4-hour (5-minute bins) feature snapshot and, optionally,
produces short-horizon forecasts (e.g. 15 min, 60 min) using pre-trained
models stored on GCS.

Inputs
------
GCS (raw ingestion parquet, 5-minute bins)
    GCS_RAW_PREFIX/date=YYYY-MM-DD/hour=HH/*.parquet

The raw schema is aligned with the ingestion job and must contain at least the
columns defined in `service.core.time_features.BASE_COLUMNS`.

Models (per horizon, on GCS)
    GCS_MODEL_URI_T15   = .../models/h15/latest.joblib
    GCS_MODEL_URI_T60   = .../models/h60/latest.joblib
    (future horizons can be added similarly)

Feature window
--------------
- Time resolution: 5 minutes (BIN_MINUTES = 5)
- Window size:     WINDOW_HOURS (default 4 hours)
- Internally:      WINDOW_BINS = max(window in bins, LAG_MAX_BINS+1)
                   where LAG_MAX_BINS = 48 (max lag used in features)

At runtime:
    now_utc                : wall-clock or fixed via NOW_UTC_ISO
    end_tbin_aware (UTC)   : now_utc floored to nearest 5-min
    start_tbin_aware (UTC) : end_tbin_aware - (WINDOW_BINS-1)*5min

Forecasts
---------
For each configured horizon (e.g. 15, 60 minutes):
- Build features per station at `tbin_latest` (last bin in window).
- Call `predict_from_features_df` (training core) to obtain predictions.
- Realign station_id / metadata and enforce a clean JSON shape.
- Upload:

    {SERVING_FORECAST_PREFIX}/h{H}/latest.json

Example:
    gs://.../serving/forecast/h15/latest.json

    {
      "generated_at": "2025-11-01T17:55:00Z",
      "horizon_min": 15,
      "data": [ {...}, {...} ]
    }

Environment
-----------
Required:
    GCS_RAW_PREFIX
    SERVING_FORECAST_PREFIX
    FORECAST_HORIZONS        e.g. "15,60"
    GCS_MODEL_URI_T15        (for 15-min horizon)
    GCS_MODEL_URI_T60        (for 60-min horizon)

Optional:
    WINDOW_HOURS             (default 4)
    WITH_FORECAST            (default "1"; "0"/"false" disables inference)
    NOW_UTC_ISO              (fixed clock in ISO for tests)

Notes
-----
- This job is stateless: no local cache or persistent state is kept.
- All timestamps are treated in UTC, converted to naive pandas datetime for
  feature building, and pushed back to ISO strings at JSON export time.
"""

from __future__ import annotations
import os, sys, shutil, json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict
import importlib.util as _iu

import numpy as np
import pandas as pd
from google.cloud import storage

# ─────────────────────────────────────────────
#  Repo root auto-resolution (service/ or train/)
# ─────────────────────────────────────────────
def _ensure_repo_root():
    """
    Ensure that the repo root (containing `service/` or `train/`) is on sys.path.

    This makes the module robust to different execution layouts:
    - local `python -m service.jobs.build_serving_forecast`
    - Docker / Cloud Run job with `WORKDIR /app`
    - CI / build environments where the working directory is not the repo root.

    Strategy
    --------
    1. If `service` or `train` is already importable → do nothing.
    2. Otherwise, walk upwards from the current file and:
         - if a directory containing `service/` or `train/` is found,
           prepend it to `sys.path`.
    3. As a last resort, if `/app` exists, prepend `/app` to `sys.path`.
    """
    if _iu.find_spec("service") is not None or _iu.find_spec("train") is not None:
        return
    here = Path(__file__).resolve()
    for c in [here] + list(here.parents) + [Path("/app"), Path.cwd()]:
        if (c / "service").exists() or (c / "train").exists():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            print(f"[serving_forecast] repo_root={c}")
            return
    if Path("/app").exists() and "/app" not in sys.path:
        sys.path.insert(0, "/app")
        print("[serving_forecast] fallback /app")

_ensure_repo_root()

# ─────────────────────────────────────────────
#  Imports training/forecast (multi-layout safe)
# ─────────────────────────────────────────────
try:
    from service.core.cal_features import add_time_features
    from service.core.time_features import BASE_COLUMNS as TRAIN_BASE_COLUMNS
    from service.core.forecast import predict_from_features_df
except ModuleNotFoundError:
    # Fallback import attempts kept for historical layouts or packaging issues.
    try:
        from service.core.cal_features import add_time_features
        from service.core.time_features import BASE_COLUMNS as TRAIN_BASE_COLUMNS
        from service.core.forecast import predict_from_features_df
    except ModuleNotFoundError:
        from service.core.cal_features import add_time_features
        from service.core.time_features import BASE_COLUMNS as TRAIN_BASE_COLUMNS
        from service.core.forecast import predict_from_features_df

# ─────────────────────────────────────────────
#  ENV config
# ─────────────────────────────────────────────
# Raw data (bronze, 5-minute ingestion)
RAW_PREFIX = os.environ["GCS_RAW_PREFIX"]

# Time resolution & feature window
BIN_MINUTES = 5
WINDOW_HOURS = int(os.environ.get("WINDOW_HOURS", "4"))
LAG_MAX_BINS = 48  # max lag used in features
# Ensure we always have enough history to compute all lags:
WINDOW_BINS = max(WINDOW_HOURS * 60 // BIN_MINUTES, LAG_MAX_BINS + 1)

# Forecasting toggle & outputs
WITH_FORECAST = str(os.environ.get("WITH_FORECAST", "1")).lower() in ("1", "true", "yes")
SERVING_FORECAST_PREFIX = os.environ.get("SERVING_FORECAST_PREFIX")
FORECAST_HORIZONS = os.environ.get("FORECAST_HORIZONS", "15,60")

# Per-horizon model URIs (pre-trained models on GCS)
GCS_MODEL_URI_T15 = os.environ.get("GCS_MODEL_URI_T15")
GCS_MODEL_URI_T60 = os.environ.get("GCS_MODEL_URI_T60")

# ─────────────────────────────────────────────
#  Time utils
# ─────────────────────────────────────────────
def _floor_5min(dt: datetime) -> datetime:
    """
    Floor a timezone-aware datetime to the previous 5-minute boundary.

    Example
    -------
    12:07:23 → 12:05:00
    12:10:00 → 12:10:00
    """
    m = (dt.minute // 5) * 5
    return dt.replace(minute=m, second=0, microsecond=0)

def _iter_hours(start: datetime, end: datetime):
    """
    Iterate over whole hours between two datetimes (inclusive).

    Each yielded datetime is truncated to the hour (minute/second/microsecond=0).

    Used to build the hourly partition path:
      bronze/date=YYYY-MM-DD/hour=HH/
    """
    cur = start.replace(minute=0, second=0, microsecond=0)
    last = end.replace(minute=0, second=0, microsecond=0)
    while cur <= last:
        yield cur
        cur += timedelta(hours=1)

# ─────────────────────────────────────────────
#  GCS utils
# ─────────────────────────────────────────────
def _parse_gs(uri: str) -> Tuple[str, str]:
    """
    Split a GCS URI of the form `gs://bucket/path` into (bucket, key).

    Raises
    ------
    AssertionError
        If the URI does not start with `gs://`.
    """
    assert uri.startswith("gs://"), f"bad GCS uri: {uri}"
    bkt, key = uri[5:].split("/", 1)
    return bkt, key

def _upload_bytes(cli: storage.Client, data: bytes, dest_uri: str, content_type: str = "application/json") -> None:
    """
    Upload a bytes payload to the given GCS URI using the provided client.

    Parameters
    ----------
    cli : google.cloud.storage.Client
        GCS client instance.
    data : bytes
        Raw bytes to upload.
    dest_uri : str
        Destination GCS URI (gs://bucket/path).
    content_type : str
        MIME type (default "application/json").
    """
    bkt, key = _parse_gs(dest_uri)
    cli.bucket(bkt).blob(key).upload_from_string(data, content_type=content_type)

def _list_raw_files_for_window(cli: storage.Client, start: datetime, end: datetime) -> List[str]:
    """
    List all raw parquet files in the ingestion (bronze) bucket for a time window.

    The ingestion layout is:

        {RAW_PREFIX}/date=YYYY-MM-DD/hour=HH/...

    For each hour between `start` and `end` (inclusive), this function:
      - derives the corresponding date/hour partition prefix,
      - lists blobs under that prefix,
      - keeps only `.parquet` files.

    Returns
    -------
    list[str]
        Sorted list of `gs://bucket/path` URIs.
    """
    bkt, pfx_root = _parse_gs(RAW_PREFIX)
    out: List[str] = []
    for h in _iter_hours(start, end):
        day = h.strftime("%Y-%m-%d"); hh = h.strftime("%H")
        prefix = f"{pfx_root}/date={day}/hour={hh}/"
        for b in cli.list_blobs(bkt, prefix=prefix):
            if b.name.endswith(".parquet"):
                out.append(f"gs://{bkt}/{b.name}")
    return sorted(out)

def _download_gs_files(cli: storage.Client, uris: List[str], dest_dir: Path) -> List[Path]:
    """
    Download a list of GCS URIs locally to a working directory.

    Parameters
    ----------
    cli : google.cloud.storage.Client
        GCS client instance.
    uris : list[str]
        GCS URIs to download.
    dest_dir : pathlib.Path
        Local directory where files will be written.

    Returns
    -------
    list[pathlib.Path]
        List of local paths corresponding to the downloaded files.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for uri in uris:
        bkt, key = _parse_gs(uri)
        local = dest_dir / Path(key).name
        cli.bucket(bkt).blob(key).download_to_filename(str(local))
        paths.append(local)
    return paths

# ─────────────────────────────────────────────
#  IO / normalization
# ─────────────────────────────────────────────
BASE_COLS = list(TRAIN_BASE_COLUMNS)  # aligned with training

def _to_naive_utc(series: pd.Series) -> pd.Series:
    """
    Convert a timestamp-like Series to naive UTC datetimes.

    Steps
    -----
    1. Parse as UTC-aware datetime (errors coerced to NaT).
    2. Convert to UTC timezone.
    3. Drop tzinfo to obtain naive datetime64[ns].
    """
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    return dt.dt.tz_convert("UTC").dt.tz_localize(None)

def _read_concat_parquets(files: List[Path]) -> pd.DataFrame:
    """
    Read multiple parquet files and return a single, normalized DataFrame.

    Responsibilities
    ----------------
    - Read all files (skip corrupted ones with a warning).
    - Normalize timestamps (`ts_utc`, `tbin_utc`) to naive UTC.
    - Coerce core numeric columns (bikes/capacity/mechanical/ebike, weather).
    - Normalize text columns (`status`, `name`) to pandas string dtype.
    - Build a robust `station_id` column from:
        station_id or stationcode (if station_id missing/empty).
    - De-duplicate rows per (station_id, tbin_utc) with ts_utc as tie-breaker
      (last sample wins).
    - Ensure all training base columns (`BASE_COLS`) exist in the final
      dataframe, filled with NA when missing.

    Returns
    -------
    pandas.DataFrame
        DataFrame restricted to `BASE_COLS` columns.
    """
    dfs = []
    for p in files:
        try:
            df = pd.read_parquet(p)
            dfs.append(df)
        except Exception as e:
            print(f"[read] skip {p.name}: {e}")
    if not dfs:
        return pd.DataFrame(columns=BASE_COLS)

    df = pd.concat(dfs, ignore_index=True)

    # time typing/normalization
    if "ts_utc" in df.columns:
        df["ts_utc"] = _to_naive_utc(df["ts_utc"])
    if "tbin_utc" in df.columns:
        df["tbin_utc"] = _to_naive_utc(df["tbin_utc"])

    # numerics (do NOT touch station_id)
    for c in ["bikes", "capacity", "mechanical", "ebike"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["lat", "lon", "temp_C", "precip_mm", "wind_mps"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # text
    if "status" in df.columns:
        df["status"] = df["status"].astype("string")
    if "name" in df.columns:
        df["name"] = df["name"].astype("string")

    # robust station_id
    if "station_id" not in df.columns:
        if "stationcode" in df.columns:
            df["station_id"] = df["stationcode"].astype("string")
        else:
            df["station_id"] = pd.NA
    df["station_id"] = df["station_id"].astype("string")
    if "stationcode" in df.columns:
        sc = df["stationcode"].astype("string")
        m_empty = df["station_id"].isna() | (df["station_id"].str.strip() == "")
        df.loc[m_empty, "station_id"] = sc

    # de-dup
    if set(["station_id", "tbin_utc", "ts_utc"]).issubset(df.columns):
        df = df.sort_values(["station_id", "tbin_utc", "ts_utc"])
        df = df.groupby(["station_id", "tbin_utc"], as_index=False, dropna=True).tail(1).reset_index(drop=True)

    # ensure expected columns
    for c in BASE_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    return df[BASE_COLS]

# ─────────────────────────────────────────────
#  Feature building helpers (per-station)
# ─────────────────────────────────────────────
def _build_one_station_full(st_df: pd.DataFrame) -> pd.Series:
    """
    Build a compact, training-like feature vector for a single station.

    Assumes `st_df` contains the full historical window (≥ 48 bins) for a
    single station, sorted by `tbin_utc`.

    The resulting series includes:
      - identification / current bin:
          station_id, tbin_latest, capacity_bin, occ_ratio_bin
      - bike composition:
          mechanical, ebike
      - static / geo:
          lat, lon
      - weather:
          temp_C, precip_mm, wind_mps, and their lag-1 values
      - historical signals:
          lag_bikes_{1,2,3,6,12,24,48}
          roll_mean_{3,6,12}, roll_std_{3,6,12}
          trend_nb_12b, trend_occ_12b (slope per 5-min over last 12 bins)
      - calendar features (UTC and Paris):
          hour, minute, dow, month, is_weekend,
          hod_sin/hod_cos, dow_sin/dow_cos,
          paris_hour, paris_dow, paris_is_we

    Notes
    -----
    - All trends use a leak-safe window: they are computed on data strictly
      before the latest bin.
    - Weather lag-1 features are based on the series shifted by one bin.
    """
    st_df = st_df.sort_values("tbin_utc").copy()

    cur = st_df.iloc[-1]
    st_id = cur.get("station_id", np.nan)
    tbin_latest = pd.to_datetime(cur["tbin_utc"], errors="coerce")

    bikes = pd.to_numeric(st_df["bikes"], errors="coerce")
    capacity = pd.to_numeric(st_df["capacity"], errors="coerce")

    # lags
    def lag_last(s: pd.Series, n: int):
        """
        Return the value at position -1 of `s` shifted by n steps, if available.
        Otherwise returns NaN.
        """
        return s.shift(n).iloc[-1] if len(s) >= (n + 1) else np.nan
    lag_set = (1, 2, 3, 6, 12, 24, 48)
    lag_vals = {f"lag_bikes_{L}": lag_last(bikes, L) for L in lag_set}

    # rollings (avoid leak → shift by 1)
    roll_vals: Dict[str, float] = {}
    for W in (3, 6, 12):
        s_shift = bikes.shift(1)
        roll_vals[f"roll_mean_{W}"] = s_shift.rolling(W, min_periods=max(1, W // 2)).mean().iloc[-1]
        roll_vals[f"roll_std_{W}"]  = s_shift.rolling(W, min_periods=max(1, W // 2)).std().iloc[-1]

    # trends (12 bins)
    def _slope_per_5m(ts: pd.Series, y: pd.Series) -> float:
        """
        Compute the linear slope of y as a function of time expressed in 5-min bins.

        Parameters
        ----------
        ts : pandas.Series
            Time series (datetime-like).
        y : pandas.Series
            Numeric series.

        Returns
        -------
        float
            Slope per 5-minute step. NaN if not enough points or variance.
        """
        t = pd.to_datetime(ts, errors="coerce")
        m = t.notna() & y.notna()
        if m.sum() < 2: return np.nan
        x = t[m].astype("datetime64[s]").astype("int64").astype(np.float64) / (BIN_MINUTES * 60.0)
        yy = y[m].astype(float).to_numpy()
        vx = np.var(x)
        if vx == 0: return np.nan
        return float(np.cov(x, yy, ddof=0)[0, 1] / vx)

    win_nb  = st_df.tail(13).iloc[:-1]
    win_occ = st_df.tail(13).iloc[:-1]
    trend_nb_12b  = _slope_per_5m(win_nb["tbin_utc"],  win_nb["bikes"]) if "bikes" in win_nb.columns else np.nan
    trend_occ_12b = _slope_per_5m(
        win_occ["tbin_utc"],
        (win_occ["bikes"] / win_occ["capacity"].where(win_occ["capacity"] > 0))
    ) if {"bikes","capacity"}.issubset(win_occ.columns) else np.nan

    # ratios & weather (lag 1)
    occ_ratio_bin = (cur.get("bikes", np.nan) / cur.get("capacity", np.nan)) if pd.notna(cur.get("capacity", np.nan)) and cur.get("capacity", 0) > 0 else np.nan
    temp_C    = pd.to_numeric(st_df["temp_C"], errors="coerce")
    precip_mm = pd.to_numeric(st_df["precip_mm"], errors="coerce")
    wind_mps  = pd.to_numeric(st_df["wind_mps"], errors="coerce")
    temp_C_lag1    = lag_last(temp_C, 1)
    precip_mm_lag1 = lag_last(precip_mm, 1)
    wind_mps_lag1  = lag_last(wind_mps, 1)

    # calendar features (UTC + Paris)
    cal = pd.DataFrame({"tbin_utc": [tbin_latest]})
    cal = add_time_features(cal, ts_col="tbin_utc", add_paris_derived=True).iloc[0].to_dict()

    status = cur.get("status", None)

    out = {
        "station_id":      st_id,
        "tbin_latest":     tbin_latest,
        "capacity_bin":    cur.get("capacity", np.nan),
        "occ_ratio_bin":   occ_ratio_bin,
        "occ_ratio":       occ_ratio_bin,
        "mechanical":      cur.get("mechanical", np.nan),
        "ebike":           cur.get("ebike", np.nan),
        "lat":             cur.get("lat", np.nan),
        "lon":             cur.get("lon", np.nan),
        "temp_C":          cur.get("temp_C", np.nan),
        "precip_mm":       cur.get("precip_mm", np.nan),
        "wind_mps":        cur.get("wind_mps", np.nan),
        "temp_C_lag1":     temp_C_lag1,
        "precip_mm_lag1":  precip_mm_lag1,
        "wind_mps_lag1":   wind_mps_lag1,
        "trend_nb_12b":    trend_nb_12b,
        "trend_occ_12b":   trend_occ_12b,
        "status":          status,     # encoded to status_code later
        # calendar
        "hour":        cal.get("hour"),
        "minute":      cal.get("minute"),
        "dow":         cal.get("dow"),
        "month":       cal.get("month"),
        "is_weekend":  cal.get("is_weekend"),
        "hod_sin":     cal.get("hod_sin"),
        "hod_cos":     cal.get("hod_cos"),
        "dow_sin":     cal.get("dow_sin"),
        "dow_cos":     cal.get("dow_cos"),
        "paris_hour":  cal.get("paris_hour"),
        "paris_dow":   cal.get("paris_dow"),
        "paris_is_we": cal.get("paris_is_we"),
    }
    out.update(lag_vals)
    out.update(roll_vals)
    return pd.Series(out)

def _build_features(df: pd.DataFrame, start_tbin: datetime, end_tbin: datetime) -> pd.DataFrame:
    """
    Build one feature row per station for the feature window [start_tbin, end_tbin].

    Steps
    -----
    1. Filter the raw dataframe on the time window and keep only needed columns.
    2. Group by `station_id` and call `_build_one_station_full` on each group.
    3. Enforce robust station_id (string) and datetime typing.
    4. Encode `status_code` as an integer (categorical encoding of `status`).

    Parameters
    ----------
    df : pandas.DataFrame
        Raw, normalized ingestion dataframe (bronze).
    start_tbin : datetime
        Lower bound (naive UTC) of the historical window.
    end_tbin : datetime
        Upper bound (naive UTC) of the historical window.

    Returns
    -------
    pandas.DataFrame
        Feature dataframe with one row per station.
    """
    # select historical window to compute features
    m = (df["tbin_utc"] >= start_tbin) & (df["tbin_utc"] <= end_tbin)
    cols_needed = [
        "tbin_utc","station_id","bikes","capacity","mechanical","ebike","status",
        "lat","lon","temp_C","precip_mm","wind_mps"
    ]
    dfw = df.loc[m, cols_needed].dropna(subset=["station_id","tbin_utc"]).copy()
    if dfw.empty:
        return pd.DataFrame()

    # Important: set station_id from groupby key (not from column values possibly altered)
    def _build_with_key(g: pd.DataFrame) -> pd.Series:
        out = _build_one_station_full(g)
        out["station_id"] = str(g.name) if g.name is not None else pd.NA
        return out

    try:
        feats = (
            dfw.groupby("station_id", dropna=True, group_keys=False)
               .apply(_build_with_key, include_groups=False)  # pandas ≥ 2.2
               .reset_index(drop=True)
        )
    except TypeError:
        # Backward-compatible path for older pandas versions.
        feats = (
            dfw.groupby("station_id", dropna=True, group_keys=False)
               .apply(_build_with_key)
               .reset_index(drop=True)
        )

    # final typing
    feats["station_id"]  = feats["station_id"].astype("string")
    feats["tbin_latest"] = pd.to_datetime(feats["tbin_latest"], errors="coerce")

    # encode status_code
    if "status" in feats.columns:
        cats = sorted([s for s in feats["status"].dropna().unique()])
        status_map = {s: i for i, s in enumerate(cats)}
        feats["status_code"] = feats["status"].map(status_map).astype("Int64")
        feats = feats.drop(columns=["status"])

    try:
        print("[features] sample station_id:", feats["station_id"].head(3).tolist())
    except Exception:
        pass

    return feats

# ─────────────────────────────────────────────
#  JSON sanitization
# ─────────────────────────────────────────────
def _to_jsonable(v):
    """
    Convert a scalar value to a JSON-friendly representation.

    Rules
    -----
    - None → None
    - NaN / NA → None
    - numpy integer → int
    - numpy float   → float
    - pandas.Timestamp / datetime →
         ISO8601 string in UTC (Z suffix)
    - everything else → left as-is
    """
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        dt = v
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    return v

def _records_jsonable(df: pd.DataFrame) -> list[dict]:
    """
    Convert a dataframe to a list of JSON-safe records.

    - Datetime columns are converted to ISO strings in UTC with "Z" suffix.
    - All scalar values are passed through `_to_jsonable`.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    list[dict]
        List of JSON-serializable dictionaries.
    """
    if df.empty:
        return []
    df2 = df.copy()
    for c in df2.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        s = pd.to_datetime(df2[c], utc=True, errors="coerce")
        df2[c] = s.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    recs = df2.to_dict(orient="records")
    return [{k: _to_jsonable(v) for k, v in r.items()} for r in recs]

# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main() -> int:
    """
    CLI entrypoint for the Serving Forecast job.

    Pipeline
    --------
    1. Resolve `now_utc`:
         - if NOW_UTC_ISO present → parse as fixed UTC datetime,
         - else use current UTC time.
    2. Compute feature window:
         - end_tbin_aware = floor_5min(now_utc)
         - start_tbin_aware = end_tbin_aware - (WINDOW_BINS - 1)*5min
    3. List all raw parquet files under GCS_RAW_PREFIX for the window.
         - If none found → exit(0).
    4. Download these parquets to /tmp, read & normalize them.
    5. Build one feature row per station using `_build_features`.
         - If no features → exit(0).
    6. If WITH_FORECAST is disabled → stop after feature computation.
    7. For each horizon in FORECAST_HORIZONS:
         - resolve the corresponding model URI (GCS_MODEL_URI_T{H}),
         - call `predict_from_features_df` (training core),
         - align station_id and metadata, add horizon/model metadata,
         - coerce bikes_pred_int from bikes_pred if needed,
         - convert to JSON-safe records.
    8. Upload per-horizon JSON bundles to:
         SERVING_FORECAST_PREFIX/h{H}/latest.json

    Returns
    -------
    int
        0 on success (even if some horizons are skipped or empty).
    """
    # 0) now / feature window
    if "NOW_UTC_ISO" in os.environ:
        now_utc = datetime.fromisoformat(os.environ["NOW_UTC_ISO"].replace("Z", "+00:00")).astimezone(timezone.utc)
    else:
        now_utc = datetime.now(timezone.utc)

    end_tbin_aware = _floor_5min(now_utc)
    start_tbin_aware = end_tbin_aware - timedelta(minutes=BIN_MINUTES * (WINDOW_BINS - 1))
    end_naive   = end_tbin_aware.replace(tzinfo=None)
    start_naive = start_tbin_aware.replace(tzinfo=None)

    print(f"[features_4h][cfg] RAW={RAW_PREFIX} WIN_H={WINDOW_HOURS} BINS={WINDOW_BINS}", flush=True)
    print(f"[features_4h] window UTC: {start_tbin_aware.isoformat()} → {end_tbin_aware.isoformat()} (inclusive)", flush=True)

    cli = storage.Client()

    # 1) list + 2) download
    gcs_files = _list_raw_files_for_window(cli, start_tbin_aware, end_tbin_aware)
    print(f"[features_4h] gcs files found = {len(gcs_files)}", flush=True)
    if not gcs_files:
        print("[features_4h] no raw files in window — exit 0", flush=True)
        return 0

    work = Path("/tmp/features_4h_raw"); shutil.rmtree(work, ignore_errors=True)
    local_files = _download_gs_files(cli, gcs_files, work)
    print(f"[features_4h] local files = {len(local_files)}", flush=True)

    # 3) read + features
    df = _read_concat_parquets(local_files)
    feats = _build_features(df, start_naive, end_naive)
    print(f"[features_4h] features rows={len(feats):,}", flush=True)
    if feats.empty:
        print("[features_4h] no features → no forecast", flush=True)
        return 0

    # 4) inference → per-horizon JSONs
    if not WITH_FORECAST:
        print("[forecast] WITH_FORECAST disabled — nothing to do", flush=True)
        return 0
    if not SERVING_FORECAST_PREFIX:
        raise RuntimeError("SERVING_FORECAST_PREFIX is required to write latest.json")

    def _model_uri_for(hmin: int) -> str | None:
        """
        Return the configured model URI for a given horizon in minutes.

        Currently supported:
          - 15 → GCS_MODEL_URI_T15
          - 60 → GCS_MODEL_URI_T60

        Future horizons can be added by extending this mapping.
        """
        if hmin == 15 and GCS_MODEL_URI_T15: return GCS_MODEL_URI_T15
        if hmin == 60 and GCS_MODEL_URI_T60: return GCS_MODEL_URI_T60
        # future horizons can be added here (e.g., 120 → GCS_MODEL_URI_T120)
        return None  # horizon not configured → skip

    horizons_min = [int(x.strip()) for x in FORECAST_HORIZONS.split(",") if x.strip()]
    consolidated: Dict[str, list] = {}
    generated_at = end_tbin_aware.isoformat().replace("+00:00", "Z")

    for hmin in horizons_min:
        uri = _model_uri_for(hmin)
        if not uri:
            print(f"[forecast][skip] no model configured for h={hmin} — skipping")
            consolidated[str(hmin)] = []
            continue

        # Core inference: calls into training pipeline utilities.
        preds = predict_from_features_df(
            feats_df=feats,
            model_uri=uri,
            horizon_bins=max(1, hmin // 5),  # 15→3 bins, 60→12 bins
            model_alias=None,
        )

        if preds.empty:
            print(f"[forecast] empty preds for h={hmin}")
            consolidated[str(hmin)] = []
            continue

        # positional realignment (safety)
        preds = preds.reset_index(drop=True).copy()
        feats_idx = feats.reset_index(drop=True).copy()

        # ensure station_id comes from feats (string)
        preds["station_id"] = feats_idx["station_id"].astype("string")

        # include useful fields for UI
        if "tbin_latest" not in preds.columns:
            preds["tbin_latest"] = feats_idx["tbin_latest"].values
        if "capacity_bin" not in preds.columns and "capacity_bin" in feats_idx.columns:
            preds["capacity_bin"] = feats_idx["capacity_bin"].values

        # meta fields
        preds["horizon_min"] = hmin
        preds["pred_ts_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if "model_version" not in preds.columns:
            try:
                preds["model_version"] = uri.rsplit("/", 1)[-1].replace(".joblib", "")
            except Exception:
                preds["model_version"] = f"model_h{hmin}"

        # integer bike prediction if only float present
        if "bikes_pred_int" not in preds.columns and "bikes_pred" in preds.columns:
            preds["bikes_pred_int"] = (
                np.rint(pd.to_numeric(preds["bikes_pred"], errors="coerce"))
                .clip(lower=0)
                .astype("Int64")
            )

        try:
            print("[forecast][sample]",
                  preds[["station_id","bikes_pred","bikes_pred_int"]].head(3).to_dict("records"))
        except Exception:
            pass

        consolidated[str(hmin)] = _records_jsonable(preds)

    # 5) upload per horizon: serving/h15/latest.json, serving/h60/latest.json
    for hmin, recs in consolidated.items():
        sub_bundle = {
            "generated_at": generated_at,
            "horizon_min": int(hmin),
            "data": recs,
        }
        dest_uri = f"{SERVING_FORECAST_PREFIX.rstrip('/')}/h{hmin}/latest.json"
        _upload_bytes(cli, json.dumps(sub_bundle, ensure_ascii=False).encode("utf-8"), dest_uri)
        print(f"[forecast] uploaded → {dest_uri}", flush=True)

    print(f"[forecast] done: {len(consolidated)} horizons uploaded", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
