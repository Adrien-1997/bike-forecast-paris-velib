# features.py
# =============================================================================
# Lecture des parquets 5 min (snapshot déjà joint à la météo),
# construction de la table d'entraînement avec cible y_nb (bikes horizon),
# lags/rollings et features calendaires.
#
# Schéma attendu (tous parquets doivent contenir ces colonnes) :
# ['ts_utc','tbin_utc','station_id','bikes','capacity','mechanical','ebike',
#  'status','lat','lon','name','temp_C','precip_mm','wind_mps']
#
# Utilisation rapide:
#   from features import build_training_frame
#   df, X, y, feat_cols = build_training_frame("data_local/daily/*.parquet", horizon_bins=3)
# =============================================================================

from __future__ import annotations
import os, glob
import pandas as pd
from typing import Iterable, Tuple, List, Optional
from cal_features import add_time_features

BASE_COLUMNS = [
    "ts_utc","tbin_utc","station_id","bikes","capacity","mechanical","ebike",
    "status","lat","lon","name","temp_C","precip_mm","wind_mps"
]

def _read_many_parquets(path_or_glob: str) -> pd.DataFrame:
    """
    Lit un fichier, un glob (*.parquet) ou un dossier (concat de *.parquet).
    Retourne un DataFrame concaténé.
    """
    paths: List[str] = []
    if os.path.isdir(path_or_glob):
        paths = sorted(glob.glob(os.path.join(path_or_glob, "*.parquet")))
    else:
        # fichier unique OU glob
        if "*" in path_or_glob or "?" in path_or_glob:
            paths = sorted(glob.glob(path_or_glob))
        else:
            paths = [path_or_glob]

    if not paths:
        return pd.DataFrame(columns=BASE_COLUMNS)

    dfs = []
    for p in paths:
        try:
            df = pd.read_parquet(p)
            dfs.append(df)
        except Exception as e:
            print(f"[features][warn] lecture parquet échouée: {p} → {e}")
    if not dfs:
        return pd.DataFrame(columns=BASE_COLUMNS)
    out = pd.concat(dfs, ignore_index=True)
    # force présence colonnes
    for c in BASE_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    return out[BASE_COLUMNS]

def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df["ts_utc"]   = pd.to_datetime(df["ts_utc"],   utc=True, errors="coerce").dt.tz_convert(None)
    df["tbin_utc"] = pd.to_datetime(df["tbin_utc"], utc=True, errors="coerce").dt.tz_convert(None)
    for c in ["station_id","bikes","capacity","mechanical","ebike"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["lat","lon","temp_C","precip_mm","wind_mps"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["status"] = df["status"].astype("string")
    df["name"]   = df["name"].astype("string")
    return df

def _dedupe_per_bin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Plusieurs lignes possibles par (station_id, tbin_utc). On garde la dernière
    en se basant sur ts_utc (max). Si ts_utc manquant, on garde la première.
    """
    df = df.sort_values(["station_id","tbin_utc","ts_utc"], ascending=[True, True, True])
    # on prend la dernière par groupe
    dedup = df.groupby(["station_id","tbin_utc"], as_index=False).tail(1)
    return dedup.reset_index(drop=True)

def _add_target_and_lags(
    df: pd.DataFrame,
    horizon_bins: int = 3,
    lag_bins: Iterable[int] = (1,2,3,6,12),
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Cible y_nb = bikes à +horizon_bins (par station).
    Ajoute des lags et des rollings simples.
    """
    df = df.sort_values(["station_id","tbin_utc"])
    # cible
    df["y_nb"] = df.groupby("station_id", group_keys=False)["bikes"].shift(-horizon_bins)

    # lags
    for L in lag_bins:
        df[f"lag_bikes_{L}"] = df.groupby("station_id", group_keys=False)["bikes"].shift(L)

    # rollings (utilisent les lags existants pour éviter leak)
    # fenetres en bins: 3 (15min), 6 (30min), 12 (1h)
    for W in (3,6,12):
        df[f"roll_mean_{W}"] = (
            df.groupby("station_id", group_keys=False)["bikes"]
              .apply(lambda s: s.shift(1).rolling(W, min_periods=max(1, W//2)).mean())
        )
        df[f"roll_std_{W}"] = (
            df.groupby("station_id", group_keys=False)["bikes"]
              .apply(lambda s: s.shift(1).rolling(W, min_periods=max(1, W//2)).std())
        )

    # ratios simples
    df["occ_ratio"] = df["bikes"] / df["capacity"].where(df["capacity"] > 0)

    # features météo (déjà présentes) : temp_C, precip_mm, wind_mps — on crée aussi des lags météo
    for L in (1,):
        for c in ("temp_C","precip_mm","wind_mps"):
            df[f"{c}_lag{L}"] = df.groupby("station_id", group_keys=False)[c].shift(L)

    # features calendaires
    add_time_features(df, ts_col="tbin_utc", add_paris_derived=True)

    # liste des colonnes features (on exclut l'ID, la cible, et les champs textuels lat/lon/name si tu veux)
    feat_cols = [
        # base niveau station + état instantané
        "capacity","mechanical","ebike","status",
        "lat","lon",
        "temp_C","precip_mm","wind_mps",
        "occ_ratio",
        # lags + rollings
        *(f"lag_bikes_{L}" for L in lag_bins),
        "roll_mean_3","roll_mean_6","roll_mean_12",
        "roll_std_3","roll_std_6","roll_std_12",
        # météo lag
        "temp_C_lag1","precip_mm_lag1","wind_mps_lag1",
        # temps
        "hour","minute","dow","month","is_weekend",
        "hod_sin","hod_cos","dow_sin","dow_cos",
        "paris_hour","paris_dow","paris_is_we",
    ]

    # cast status (string) -> catégorie ordinale simple (éviter OneHot si modèle sklearn robuste)
    # si modèle gère string (HGBR), on peut laisser tel quel ; sinon convertir
    # Ici: on garde la string et laissera le modèle HGBR ignorer (il ne supporte pas string).
    # Donc on encode rapidement :
    status_map = {s:i for i, s in enumerate(sorted(df["status"].dropna().unique()))}
    df["status_code"] = df["status"].map(status_map).astype("Int64")
    feat_cols = ["status_code" if c=="status" else c for c in feat_cols]

    return df, feat_cols

def build_training_frame(
    src: str,
    start_date: Optional[str] = None,
    end_date: Optional[str]   = None,
    horizon_bins: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """
    Construit le DF d'entraînement à partir des parquets 5min:
      - charge + typage + dédoublonnage par (station_id, tbin_utc)
      - filtre date [start_date, end_date] (UTC, sur tbin_utc)
      - ajoute cible y_nb (horizon_bins), lags/rollings, météo lag, features calendaires
    Retourne: (df_full, X, y, feat_cols)
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

    # drop NA sur cible et features essentielles
    used_cols = ["station_id","tbin_utc","bikes","y_nb"] + feat_cols
    df_model = df[used_cols].dropna(subset=["y_nb"])

    X = df_model[feat_cols].copy()
    y = df_model["y_nb"].astype("float32").copy()

    return df, X, y, feat_cols
