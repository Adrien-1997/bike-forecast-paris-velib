# service/core/time_features.py
# =============================================================================
# Build the training frame from 5-min Parquet snapshots (already weather-joined):
# - strict schema
# - per-bin dedupe
# - target y_nb at +horizon_bins (5 min per bin)
# - bike lags, rolling stats, simple weather lags
# - calendar + sinusoidal features (UTC + Paris)
#
# Aligned with the notebook (velib-lgbm.ipynb):
#   - lag_bins: (1, 2, 3, 6, 12, 24, 36, 48)
#   - rolling windows: (3, 6, 12, 24, 36, 48) on bikes with shift(1)
#   - weather lags: 1 for (temp_C, precip_mm, wind_mps)
#   - occ_ratio = bikes / capacity (>0)
# =============================================================================

"""
Time-based feature engineering for Vélib' 5-minute snapshots.

This module builds the **purely temporal** (non-spatial) training frame used
for forecasting models. It assumes that:

- raw data come as 5-minute Parquet snapshots where weather is **already joined**
  (columns temp_C, precip_mm, wind_mps),
- the schema is (or will be coerced to) `BASE_COLUMNS`,
- station_id is treated as a string at feature-building time.

High-level pipeline
-------------------
1. I/O
   - `_read_many_parquets` reads one/many parquet files, normalises columns,
     and guarantees that all `BASE_COLUMNS` exist.
   - `_coerce_types` enforces a stable dtype regime (timestamps, numeric, strings).
   - `_dedupe_per_bin` keeps the **latest** record per `(station_id, tbin_utc)`.

2. Feature engineering
   - `_add_target_and_lags` adds:
       * target y_nb at +horizon_bins (5 minutes x horizon_bins),
       * bike lags for several horizons,
       * rolling mean/std with shift(1) to avoid leakage,
       * simple weather lags (L=1),
       * occupancy ratio,
       * calendar + sinusoidal features via `add_time_features`,
       * ordinal encoding of status → status_code.
   - `build_training_frame` orchestrates everything and returns:
       (full_df, X, y, feat_cols).

Public API
----------
- `build_training_frame(src, start_date=None, end_date=None, horizon_bins=3)`
    → (full_df, X, y, feat_cols)

The contract with the training code is:
- X is a float32 DataFrame containing **only** columns listed in `feat_cols`,
- y is the float32 Series `y_nb`,
- full_df is the enriched DataFrame (including target, features, and raw cols),
- per-bin deduplication and date filtering have already been applied.
"""

from __future__ import annotations
import os, glob
from typing import Iterable, Tuple, List, Optional
import pandas as pd

# Prefer relative import; fallback to service.train.* for older layouts
try:
    from .cal_features import add_time_features
except Exception:
    from service.train.cal_features import add_time_features  # type: ignore

# Columns expected from the ingestion + weather join pipeline.
BASE_COLUMNS = [
    "ts_utc", "tbin_utc", "station_id", "bikes", "capacity", "mechanical", "ebike",
    "status", "lat", "lon", "name", "temp_C", "precip_mm", "wind_mps"
]

# ───────────────────────── I/O ─────────────────────────

def _read_many_parquets(path_or_glob: str) -> pd.DataFrame:
    """
    Read one or multiple Parquet files and normalise to `BASE_COLUMNS`.

    Parameters
    ----------
    path_or_glob : str
        - Path to a single `.parquet` file,
        - Path to a directory containing `.parquet` files,
        - Glob pattern like `"path/to/*.parquet"`.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame with at least `BASE_COLUMNS`:
        any missing base column is created and filled with NA,
        and the final column order is exactly `BASE_COLUMNS`.

    Notes
    -----
    - Any file that fails to load is skipped with a warning.
    - If no file can be read, an empty DataFrame with `BASE_COLUMNS`
      is returned.
    """
    paths: List[str] = []
    if os.path.isdir(path_or_glob):
        paths = sorted(glob.glob(os.path.join(path_or_glob, "*.parquet")))
    else:
        if "*" in path_or_glob or "?" in path_or_glob:
            paths = sorted(glob.glob(path_or_glob))
        else:
            paths = [path_or_glob]

    if not paths:
        return pd.DataFrame(columns=BASE_COLUMNS)

    dfs = []
    for p in paths:
        try:
            dfs.append(pd.read_parquet(p))
        except Exception as e:
            print(f"[features][warn] failed to read parquet: {p} → {e}")
    if not dfs:
        return pd.DataFrame(columns=BASE_COLUMNS)

    out = pd.concat(dfs, ignore_index=True, sort=False)

    # ensure schema presence
    for c in BASE_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    return out[BASE_COLUMNS]

# ───────────────────────── Cleaning ─────────────────────────

def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce base columns to stable types (aligned with the notebook).

    Transformations
    ---------------
    - `ts_utc`, `tbin_utc`:
        * converted to timezone-aware UTC datetimes,
        * then converted to **naive** UTC datetimes via `tz_convert(None)`.
    - `station_id`:
        * cast to pandas `string` dtype (more robust than object).
    - `bikes`, `capacity`, `mechanical`, `ebike`:
        * cast to numeric via `pd.to_numeric(errors="coerce")`.
    - `lat`, `lon`, `temp_C`, `precip_mm`, `wind_mps`:
        * same numeric coercion.
    - `status`, `name`:
        * cast to `string`.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw input DataFrame.

    Returns
    -------
    pandas.DataFrame
        Same DataFrame with normalised dtypes.
    """
    df["ts_utc"]   = pd.to_datetime(df["ts_utc"],   utc=True, errors="coerce").dt.tz_convert(None)
    df["tbin_utc"] = pd.to_datetime(df["tbin_utc"], utc=True, errors="coerce").dt.tz_convert(None)

    # Keep station_id as string for training frame (as in the notebook)
    df["station_id"] = df["station_id"].astype("string")

    for c in ["bikes", "capacity", "mechanical", "ebike"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["lat", "lon", "temp_C", "precip_mm", "wind_mps"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["status"] = df["status"].astype("string")
    df["name"]   = df["name"].astype("string")
    return df


def _dedupe_per_bin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the **latest** record per (station_id, tbin_utc) using ts_utc.

    Strategy
    --------
    - Sort by (station_id, tbin_utc, ts_utc) ascending.
    - For each (station_id, tbin_utc) group, retain the `tail(1)` record
      (i.e. the one with the most recent `ts_utc`).
    - Reset the index for a clean downstream use.

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame where multiple snapshots may exist per bin.

    Returns
    -------
    pandas.DataFrame
        Deduplicated frame with at most one row per (station_id, tbin_utc).
    """
    df = df.sort_values(["station_id", "tbin_utc", "ts_utc"], ascending=[True, True, True])
    dedup = df.groupby(["station_id", "tbin_utc"], as_index=False).tail(1)
    return dedup.reset_index(drop=True)

# ───────────────────────── Feature engineering ─────────────────────────

def _add_target_and_lags(
    df: pd.DataFrame,
    horizon_bins: int = 3,
    lag_bins: Iterable[int] = (1, 2, 3, 6, 12, 24, 36, 48),
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Enrich the frame with the **target** and time-based features.

    Adds
    ----
    - `y_nb` :
        Target = bikes at +`horizon_bins` 5-minute bins
        (i.e. horizon_bins * 5 minutes ahead).
    - `lag_bikes_L` :
        For each L in `lag_bins`, bikes lagged by L bins.
    - `roll_mean_W`, `roll_std_W` :
        For each rolling window W in (3, 6, 12, 24, 36, 48) bins:
        rolling mean / std of bikes, where we first apply `shift(1)` to
        avoid any leakage on the current target time.
    - `occ_ratio` :
        bikes / capacity, only for `capacity > 0` (otherwise NaN).
    - simple weather lags (L=1 bin):
        `temp_C_lag1`, `precip_mm_lag1`, `wind_mps_lag1`.
    - calendar features:
        via `add_time_features(df, ts_col="tbin_utc", add_paris_derived=True)`.
    - `status_code` :
        ordinal encoding of the categorical `status` (sorted unique values).

    Parameters
    ----------
    df : pandas.DataFrame
        Clean and deduplicated frame (types already coerced).
    horizon_bins : int, default 3
        Forecast horizon in 5-minute bins (3 → 15 minutes, 12 → 1 hour, etc.).
    lag_bins : Iterable[int]
        Collection of lag sizes (in bins) to use for bike counts.

    Returns
    -------
    (df, feat_cols)
        df : pandas.DataFrame
            Same DataFrame, enriched with the target and all features.
        feat_cols : list[str]
            Ordered list of feature column names (excluding the target).
    """
    df = df.sort_values(["station_id", "tbin_utc"])

    # Target
    df["y_nb"] = df.groupby("station_id", group_keys=False)["bikes"].shift(-horizon_bins)

    # Bike lags
    for L in lag_bins:
        df[f"lag_bikes_{L}"] = df.groupby("station_id", group_keys=False)["bikes"].shift(L)

    # Rolling windows with shift(1) to avoid leakage
    rolling_windows = (3, 6, 12, 24, 36, 48)
    for W in rolling_windows:
        df[f"roll_mean_{W}"] = (
            df.groupby("station_id", group_keys=False)["bikes"]
              .apply(lambda s: s.shift(1).rolling(W, min_periods=max(1, W//2)).mean())
        )
        df[f"roll_std_{W}"] = (
            df.groupby("station_id", group_keys=False)["bikes"]
              .apply(lambda s: s.shift(1).rolling(W, min_periods=max(1, W//2)).std())
        )

    # Occupancy ratio
    df["occ_ratio"] = df["bikes"] / df["capacity"].where(df["capacity"] > 0)

    # Weather lags (L=1)
    for c in ("temp_C", "precip_mm", "wind_mps"):
        df[f"{c}_lag1"] = df.groupby("station_id", group_keys=False)[c].shift(1)

    # Calendar/time features
    add_time_features(df, ts_col="tbin_utc", add_paris_derived=True)

    # Categorical status → ordinal code (stable)
    status_map = {s: i for i, s in enumerate(sorted(df["status"].dropna().unique()))}
    df["status_code"] = df["status"].map(status_map).astype("Int64")

    # Final feature columns (exactly what the notebook expects)
    feat_cols = [
        "capacity", "mechanical", "ebike",
        "lat", "lon",
        "temp_C", "precip_mm", "wind_mps",
        "occ_ratio",
        *(f"lag_bikes_{L}" for L in lag_bins),
        *(f"roll_mean_{W}" for W in rolling_windows),
        *(f"roll_std_{W}" for W in rolling_windows),
        "temp_C_lag1", "precip_mm_lag1", "wind_mps_lag1",
        "hour", "minute", "dow", "month", "is_weekend",
        "hod_sin", "hod_cos", "dow_sin", "dow_cos",
        "paris_hour", "paris_dow", "paris_is_we",
        "status_code",
    ]
    return df, feat_cols

# ───────────────────────── Public API ─────────────────────────

def build_training_frame(
    src: str,
    start_date: Optional[str] = None,
    end_date: Optional[str]   = None,
    horizon_bins: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """
    Build the full **time-feature** training frame from 5-minute Parquet snapshots.

    Steps
    -----
    1. Load raw data
       - `_read_many_parquets(src)` collects all snapshots and normalises them
         to `BASE_COLUMNS`.
    2. Clean / normalise
       - `_coerce_types` ensures stable dtypes.
       - `_dedupe_per_bin` keeps the last record per `(station_id, tbin_utc)`.
    3. Date filtering (optional, UTC)
       - If `start_date` is provided, keep rows where `tbin_utc >= start_date`.
       - If `end_date` is provided, keep rows where
         `tbin_utc <= end_date + 1 day - 5 minutes`.
    4. Feature engineering
       - `_add_target_and_lags` adds the target y_nb and all temporal features.
    5. Build model frame
       - `df_model` = subset of df with:
           ["station_id", "tbin_utc", "bikes", "y_nb"] + feat_cols,
         after dropping rows where `y_nb` is NA.
       - X = df_model[feat_cols].astype("float32").
       - y = df_model["y_nb"].astype("float32").

    Parameters
    ----------
    src : str
        File path, directory or glob pattern pointing to 5-minute Parquet snapshots.
    start_date : str | None, default None
        Lower bound (inclusive) on `tbin_utc`, as "YYYY-MM-DD".
    end_date : str | None, default None
        Upper bound (inclusive) on `tbin_utc`, as "YYYY-MM-DD".
        Internally converted to `end_date + 1 day - 5 minutes`.
    horizon_bins : int, default 3
        Forecast horizon in 5-minute bins (passed down to `_add_target_and_lags`).

    Returns
    -------
    (full_df, X, y, feat_cols)
        full_df : pandas.DataFrame
            Enriched frame (raw columns + y_nb + all features).
        X : pandas.DataFrame
            Feature matrix ready for model training (float32).
        y : pandas.Series
            Target series y_nb (float32).
        feat_cols : list[str]
            Ordered list of feature column names matching X's columns.

    Notes
    -----
    - If no data can be read, returns:
        (empty_df, empty_df, empty_series, []).
    - This function does **not** perform any train/validation split; it is
      intentionally generic so that different training strategies can be
      layered on top.
    """
    df = _read_many_parquets(src)
    if df.empty:
        return df, pd.DataFrame(), pd.Series(dtype="float64"), []

    df = _coerce_types(df)
    df = _dedupe_per_bin(df)

    if start_date:
        df = df[df["tbin_utc"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["tbin_utc"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)]

    df, feat_cols = _add_target_and_lags(df, horizon_bins=horizon_bins)

    # Drop NA on target
    used_cols = ["station_id", "tbin_utc", "bikes", "y_nb"] + feat_cols
    df_model = df[used_cols].dropna(subset=["y_nb"])

    X = df_model[feat_cols].astype("float32").copy()
    y = df_model["y_nb"].astype("float32").copy()

    return df, X, y, feat_cols
