# service/core/features.py
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

from __future__ import annotations
import os, glob
from typing import Iterable, Tuple, List, Optional
import pandas as pd

# Prefer relative import; fallback to service.train.* for older layouts
try:
    from .cal_features import add_time_features
except Exception:
    from service.train.cal_features import add_time_features  # type: ignore

BASE_COLUMNS = [
    "ts_utc","tbin_utc","station_id","bikes","capacity","mechanical","ebike",
    "status","lat","lon","name","temp_C","precip_mm","wind_mps"
]

# ───────────────────────── I/O ─────────────────────────

def _read_many_parquets(path_or_glob: str) -> pd.DataFrame:
    """Read one file, a glob (*.parquet), or all *.parquet in a directory."""
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
    """Coerce base columns to stable types (aligned with the notebook)."""
    df["ts_utc"]   = pd.to_datetime(df["ts_utc"],   utc=True, errors="coerce").dt.tz_convert(None)
    df["tbin_utc"] = pd.to_datetime(df["tbin_utc"], utc=True, errors="coerce").dt.tz_convert(None)

    # Keep station_id as string for training frame (as in the notebook)
    df["station_id"] = df["station_id"].astype("string")

    for c in ["bikes","capacity","mechanical","ebike"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["lat","lon","temp_C","precip_mm","wind_mps"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["status"] = df["status"].astype("string")
    df["name"]   = df["name"].astype("string")
    return df


def _dedupe_per_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest record per (station_id, tbin_utc) using ts_utc."""
    df = df.sort_values(["station_id","tbin_utc","ts_utc"], ascending=[True, True, True])
    dedup = df.groupby(["station_id","tbin_utc"], as_index=False).tail(1)
    return dedup.reset_index(drop=True)

# ───────────────────────── Feature engineering ─────────────────────────

def _add_target_and_lags(
    df: pd.DataFrame,
    horizon_bins: int = 3,
    lag_bins: Iterable[int] = (1, 2, 3, 6, 12, 24, 36, 48),
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adds:
      - target y_nb = bikes at +horizon_bins
      - bike lags over `lag_bins`
      - rolling mean/std over windows (3,6,12,24,36,48) with shift(1)
      - occ_ratio
      - weather lag(1) for temp_C / precip_mm / wind_mps
      - calendar features (UTC + Paris) via add_time_features()
    """
    df = df.sort_values(["station_id","tbin_utc"])

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
    for c in ("temp_C","precip_mm","wind_mps"):
        df[f"{c}_lag1"] = df.groupby("station_id", group_keys=False)[c].shift(1)

    # Calendar/time features
    add_time_features(df, ts_col="tbin_utc", add_paris_derived=True)

    # Categorical status → ordinal code (stable)
    status_map = {s: i for i, s in enumerate(sorted(df["status"].dropna().unique()))}
    df["status_code"] = df["status"].map(status_map).astype("Int64")

    # Final feature columns (exactly what the notebook expects)
    feat_cols = [
        "capacity","mechanical","ebike",
        "lat","lon",
        "temp_C","precip_mm","wind_mps",
        "occ_ratio",
        *(f"lag_bikes_{L}" for L in lag_bins),
        *(f"roll_mean_{W}" for W in rolling_windows),
        *(f"roll_std_{W}" for W in rolling_windows),
        "temp_C_lag1","precip_mm_lag1","wind_mps_lag1",
        "hour","minute","dow","month","is_weekend",
        "hod_sin","hod_cos","dow_sin","dow_cos",
        "paris_hour","paris_dow","paris_is_we",
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
    Build the training frame:
      - load + type + per-bin dedupe
      - optional UTC date filter [start_date, end_date]
      - add target/lag/rolling/weather/calendar features
      - return (full_df, X, y, feat_cols)
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
    used_cols = ["station_id","tbin_utc","bikes","y_nb"] + feat_cols
    df_model = df[used_cols].dropna(subset=["y_nb"])

    X = df_model[feat_cols].astype("float32").copy()
    y = df_model["y_nb"].astype("float32").copy()

    return df, X, y, feat_cols
