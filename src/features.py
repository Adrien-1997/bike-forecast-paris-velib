# src/features.py
from __future__ import annotations

import pandas as pd
import numpy as np

from src.cal_features import add_calendar_features, feature_cols as calfeat_cols
from src.utils_io import get_export_path

def _load_base_5min() -> pd.DataFrame:
    """
    Charge le parquet agrégé au pas 5 min (docs/exports/velib.parquet via get_export_path)
    et normalise les timestamps (UTC naïfs).
    """
    try:
        path = get_export_path("velib.parquet")
        df = pd.read_parquet(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            "[features] Dataset introuvable (velib.parquet). "
            "Lance d'abord le workflow 'velib-ingest' ou assure-toi que le fichier est dispo sur HF."
        )
    if df.empty:
        raise ValueError("[features] Parquet vide.")
    df["tbin_utc"] = pd.to_datetime(df["tbin_utc"], utc=True).dt.tz_localize(None)
    if "hour_utc" in df.columns:
        df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True).dt.tz_localize(None)
    else:
        df["hour_utc"] = df["tbin_utc"].dt.floor("h")
    return df


def build_training_frame(horizon_minutes: int = 15, lookback_days: int = 30):
    """
    Construit le dataset d'entraînement à partir du parquet 5 min.
    - Cible: y_nb = nb vélos à +horizon_minutes (ex: 15 → +3 bins)
    - Lags/rollings/trends basés sur des bins de 5 min
    """
    if horizon_minutes % 5 != 0 or horizon_minutes < 5:
        raise ValueError("horizon_minutes must be a multiple of 5 and >= 5.")

    base = _load_base_5min()

    # Fenêtre lookback calée sur la donnée (jamais avant tmin)
    tmin = pd.to_datetime(base["tbin_utc"]).min()
    tmax = pd.to_datetime(base["tbin_utc"]).max()
    binsize = pd.Timedelta(minutes=5)
    cutoff = max(tmax.floor(binsize) - pd.Timedelta(days=lookback_days), tmin)
    base = base[base["tbin_utc"] >= cutoff].copy()

    # Tri
    base = base.sort_values(["stationcode", "tbin_utc"])

    # Cible: +N bins (N = horizon_minutes / 5)
    bins_h = max(1, int(round(horizon_minutes / 5)))
    base["y_nb"] = base.groupby("stationcode", group_keys=False)["nb_velos_bin"].shift(-bins_h)

    # Exiger un minimum de contexte par station (souple)
    # besoin ≈ horizon + lags max (~48) ; on reste raisonnable
    min_rows = max(bins_h + 12, 12)
    vc = base.groupby("stationcode")["tbin_utc"].transform("count")
    base = base[vc >= min_rows].copy()

    # Lags & rollings (bins de 5 min): 1,3,6,12,24,48 => 5,15,30,60,120,240 min
    def add_lags_rollings(dfg: pd.DataFrame) -> pd.DataFrame:
        g = dfg.copy()
        lag_bins = (1, 3, 6, 12, 24, 48)
        for b in lag_bins:
            g[f"lag_nb_{b}b"]  = g["nb_velos_bin"].shift(b)
            g[f"lag_occ_{b}b"] = g["occ_ratio_bin"].shift(b)
        # Moyennes glissantes ~1h et ~2h
        g["roll_nb_12b"]  = g["nb_velos_bin"].rolling(12, min_periods=1).mean()
        g["roll_nb_24b"]  = g["nb_velos_bin"].rolling(24, min_periods=1).mean()
        g["roll_occ_12b"] = g["occ_ratio_bin"].rolling(12, min_periods=1).mean()
        g["roll_occ_24b"] = g["occ_ratio_bin"].rolling(24, min_periods=1).mean()
        # Trend ~1h
        g["trend_nb_12b"]  = (g["nb_velos_bin"] - g["nb_velos_bin"].shift(12)) / 12.0
        g["trend_occ_12b"] = (g["occ_ratio_bin"] - g["occ_ratio_bin"].shift(12)) / 12.0
        return g

    # groupby.apply sans FutureWarning (compat pandas récents)
    try:
        base = base.groupby("stationcode", group_keys=False).apply(
            add_lags_rollings, include_groups=False  # pandas >= 2.2
        )
    except TypeError:
        base = base.groupby("stationcode", group_keys=False, as_index=False).apply(
            lambda g: add_lags_rollings(g.drop(columns=[], errors="ignore"))
        )

    # Features calendaires (Europe/Paris) — inclut minute/min5_idx si tbin_utc présent
    base = add_calendar_features(base, tz="Europe/Paris")

    # Colonnes numériques
    num_cols = [
        "nb_velos_bin","nb_bornes_bin","capacity_bin","occ_ratio_bin",
        "temp_C","precip_mm","wind_mps",
        *[f"lag_nb_{b}b" for b in (1,3,6,12,24,48)],
        *[f"lag_occ_{b}b" for b in (1,3,6,12,24,48)],
        "roll_nb_12b","roll_nb_24b","roll_occ_12b","roll_occ_24b",
        "trend_nb_12b","trend_occ_12b",
        *calfeat_cols(base),
    ]
    seen = set(); num_cols = [c for c in num_cols if not (c in seen or seen.add(c))]
    for c in num_cols:
        if c not in base.columns:
            base[c] = 0.0

    base[num_cols] = (
        base[num_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
    )

    # Enlever lignes sans cible
    base = base.dropna(subset=["y_nb"]).reset_index(drop=True)
    base["y_nb"] = base["y_nb"].astype("float32")

    # Fallback si vide: rejouer sur tout le parquet
    if base.empty:
        base_all = _load_base_5min().sort_values(["stationcode","tbin_utc"]).copy()
        base_all["y_nb"] = base_all.groupby("stationcode", group_keys=False)["nb_velos_bin"].shift(-bins_h)
        vc = base_all.groupby("stationcode")["tbin_utc"].transform("count")
        base_all = base_all[vc >= min_rows].dropna(subset=["y_nb"])
        if not base_all.empty:
            base_all = base_all.groupby("stationcode", group_keys=False, as_index=False).apply(
                lambda g: add_lags_rollings(g.drop(columns=[], errors="ignore"))
            )
            base_all = add_calendar_features(base_all, tz="Europe/Paris")
            for c in num_cols:
                if c not in base_all.columns:
                    base_all[c] = 0.0
            base_all[num_cols] = (
                base_all[num_cols]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .astype("float32")
            )
            base_all = base_all.reset_index(drop=True)
            return base_all, num_cols

    return base, num_cols


def prepare_live_features(df_live: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    out = df_live.copy()
    for c in feat_cols:
        if c not in out.columns:
            out[c] = 0.0
    out = out[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    return out
