# service/core/spatial_features.py
# =============================================================================
# Build the training frame (spatial stack) from 5-min Parquet snapshots:
# - strict schema + per-bin dedupe (identiques à time_features)
# - target y_nb at +horizon_bins (5 min per bin)
# - base time features (lags, rolling, weather lags, calendar, status_code)
# - spatial statics (distance centre, grille, KMeans, TE capacité)
# - spatial dynamics (KNN des voisins : moyennes de bikes à t-L bins)
#
# Retourne (full_df, X, y, feat_cols) — même API que time_features.build_training_frame
# =============================================================================

from __future__ import annotations
import os, glob
from typing import Iterable, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Import du calendrier identique à time_features
try:
    from .cal_features import add_time_features
except Exception:
    from service.train.cal_features import add_time_features  # type: ignore

# ───────────────────────── Schéma de base ─────────────────────────
BASE_COLUMNS = [
    "ts_utc","tbin_utc","station_id","bikes","capacity","mechanical","ebike",
    "status","lat","lon","name","temp_C","precip_mm","wind_mps"
]

# ───────────────────────── I/O (identiques) ─────────────────────────
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
            print(f"[spatial_features][warn] failed to read parquet: {p} → {e}")
    if not dfs:
        return pd.DataFrame(columns=BASE_COLUMNS)

    out = pd.concat(dfs, ignore_index=True, sort=False)

    for c in BASE_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    return out[BASE_COLUMNS]

def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce base columns to stable types (aligned with time_features)."""
    df["ts_utc"]   = pd.to_datetime(df["ts_utc"],   utc=True, errors="coerce").dt.tz_convert(None)
    df["tbin_utc"] = pd.to_datetime(df["tbin_utc"], utc=True, errors="coerce").dt.tz_convert(None)

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

# ───────────────────────── Time features de base ─────────────────────────
def _add_target_and_lags(
    df: pd.DataFrame,
    horizon_bins: int = 3,
    lag_bins: Iterable[int] = (1, 2, 3, 6, 12, 24, 36, 48),
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aligne exactement les features de time_features :
      - y_nb à +horizon_bins
      - lags bikes (lag_bins)
      - rolling mean/std (3,6,12,24,36,48) avec shift(1)
      - occ_ratio
      - weather lag(1) pour temp_C / precip_mm / wind_mps
      - calendrier via add_time_features()
      - status_code ordinal
    """
    df = df.sort_values(["station_id","tbin_utc"]).copy()

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

    # Categorical status → ordinal code
    status_map = {s: i for i, s in enumerate(sorted(df["status"].dropna().unique()))}
    df["status_code"] = df["status"].map(status_map).astype("Int64")

    # Base feature list (exactement la même qu'en time_features)
    feat_cols_base = [
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
    return df, feat_cols_base

# ───────────────────────── Spatial statics ─────────────────────────
def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

def build_station_static(
    df: pd.DataFrame, *,
    center=(48.853, 2.349),
    grid_m=300,
    k_clusters=30,
    random_state=42
) -> pd.DataFrame:
    """
    Retourne un DF unique par station avec features spatiales statiques.
    """
    cols_needed = {"station_id", "lat", "lon", "capacity"}
    missing = cols_needed - set(df.columns)
    if missing:
        raise KeyError(f"build_station_static: colonnes manquantes: {missing}")

    if "tbin_utc" in df.columns:
        st = (df.sort_values(["station_id","tbin_utc"])
                .groupby("station_id", as_index=False).tail(1))[["station_id","lat","lon","capacity"]].drop_duplicates()
    else:
        st = df[["station_id","lat","lon","capacity"]].drop_duplicates()

    st["station_id"] = st["station_id"].astype("string")

    # distance au centre
    st["dist_center_km"] = _haversine_km(st["lat"], st["lon"], center[0], center[1]).astype("float32")

    # grille ~300 m (approx Paris)
    lat_deg_per_m = 1.0 / 111_000.0
    lon_deg_per_m = 1.0 / 75_000.0
    st["grid_x"] = np.floor(st["lon"] / (grid_m * lon_deg_per_m)).astype("int32")
    st["grid_y"] = np.floor(st["lat"] / (grid_m * lat_deg_per_m)).astype("int32")

    # KMeans spatial (+ capacité)
    km = KMeans(n_clusters=k_clusters, n_init=10, random_state=random_state)
    st["kmeans_cluster"] = km.fit_predict(st[["lat","lon","capacity"]]).astype("int32")

    # target encoding statique (capacités moyennes par cluster)
    te = st.groupby("kmeans_cluster", as_index=False)["capacity"].mean().rename(columns={"capacity":"te_cap_mean"})
    st = st.merge(te, on="kmeans_cluster", how="left")
    st["te_cap_mean"] = st["te_cap_mean"].astype("float32")

    return st[["station_id","lat","lon","dist_center_km","grid_x","grid_y","kmeans_cluster","te_cap_mean"]]

# ───────────────────────── Spatial dynamics (KNN) ─────────────────────────
def compute_knn_map(stations_df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Retourne (station_id -> k voisins + distances).
    """
    required = {"station_id","lat","lon"}
    missing = required - set(stations_df.columns)
    if missing:
        raise KeyError(f"compute_knn_map: stations_df missing columns: {missing}")
    stations = stations_df[["station_id","lat","lon"]].drop_duplicates().copy()
    stations["station_id"] = stations["station_id"].astype("string")
    S = stations.shape[0]
    lat = stations["lat"].to_numpy()
    lon = stations["lon"].to_numpy()

    # matrice distances (SxS)
    lat_mat1 = np.repeat(lat[:,None], S, axis=1)
    lon_mat1 = np.repeat(lon[:,None], S, axis=1)
    lat_mat2 = lat_mat1.T
    lon_mat2 = lon_mat1.T
    D = _haversine_km(lat_mat1, lon_mat1, lat_mat2, lon_mat2)
    np.fill_diagonal(D, np.inf)
    kk = min(k, max(1, S-1))
    idx = np.argpartition(D, kth=kk-1, axis=1)[:, :kk]

    rows = []
    for i in range(S):
        sid = stations.iloc[i]["station_id"]
        neigh_idx = idx[i]
        for rank, j in enumerate(neigh_idx, 1):
            rows.append((sid, int(rank), stations.iloc[j]["station_id"], float(D[i, j])))
    knn = pd.DataFrame(rows, columns=["station_id","rank","neighbor_id","dist_km"])
    return knn

# ───────────────────────── Assemblage spatial ─────────────────────────
def _add_features_spatial(df: pd.DataFrame, horizon_bins: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construit les features temporelles + SPATIALES (statiques + KNN dynamiques).
    """
    base_df, base_cols = _add_target_and_lags(df, horizon_bins=horizon_bins)

    # Statiques
    st_static_src = base_df[["station_id","lat","lon","capacity"]].drop_duplicates()
    st_static = build_station_static(st_static_src, center=(48.853, 2.349),
                                     grid_m=300, k_clusters=30, random_state=42)
    # merge sans dupliquer lat/lon
    enhanced = base_df.merge(st_static.drop(columns=["lat","lon"]), on="station_id", how="left")

    # Dynamiques KNN (k=5)
    knn = compute_knn_map(stations_df=st_static[["station_id","lat","lon"]], k=5)
    base = enhanced[["tbin_utc","station_id","bikes"]].copy()
    base["station_id"] = base["station_id"].astype("string")

    for L in (1,3,12):  # t-L (bins de 5min)
        neigh = knn.merge(base.rename(columns={"station_id":"neighbor_id"}), on="neighbor_id", how="left")
        neigh["tbin_join"] = pd.to_datetime(neigh["tbin_utc"], errors="coerce") + pd.Timedelta(minutes=5 * L)
        tmp = (neigh[["station_id","tbin_join","bikes"]]
               .rename(columns={"tbin_join":"tbin_utc", "bikes":f"knn_mean_bikes_lag{L}_tmp"}))
        tmp = tmp.groupby(["station_id","tbin_utc"], as_index=False)[f"knn_mean_bikes_lag{L}_tmp"].mean()
        enhanced = enhanced.merge(tmp, on=["station_id","tbin_utc"], how="left")
        enhanced.rename(columns={f"knn_mean_bikes_lag{L}_tmp":f"knn_mean_bikes_lag{L}"}, inplace=True)

    # Liste finale des features (base + spatial)
    feat_cols = (
        base_cols
        + ["dist_center_km","grid_x","grid_y","kmeans_cluster","te_cap_mean"]
        + [f"knn_mean_bikes_lag{L}" for L in (1,3,12)]
    )

    # Types cohérents
    for c in ["dist_center_km","te_cap_mean","grid_x","grid_y","kmeans_cluster",
              "knn_mean_bikes_lag1","knn_mean_bikes_lag3","knn_mean_bikes_lag12"]:
        enhanced[c] = pd.to_numeric(enhanced[c], errors="coerce")

    return enhanced, feat_cols

# ───────────────────────── Public API ─────────────────────────
def build_training_frame_spatial(
    src: str,
    start_date: Optional[str] = None,
    end_date: Optional[str]   = None,
    horizon_bins: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """
    Build (spatial stack):
      - load + type + per-bin dedupe
      - optional UTC date filter [start_date, end_date]
      - add base time features + spatial statics + KNN dynamics
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

    df2, feat_cols = _add_features_spatial(df, horizon_bins=horizon_bins)

    # Drop NA sur la target
    used_cols = ["station_id","tbin_utc","bikes","y_nb"] + feat_cols
    df_model = df2[used_cols].dropna(subset=["y_nb"])

    X = df_model[feat_cols].astype("float32").copy()
    y = df_model["y_nb"].astype("float32").copy()

    return df2, X, y, feat_cols
