# src/features.py
from __future__ import annotations

import pandas as pd
import numpy as np
from src.aggregate import occupancy_15min
from src.cal_features import add_calendar_features, feature_cols as calfeat_cols


def build_training_frame(horizon_minutes: int = 60, lookback_days: int = 30):
    """
    Dataset d'entraînement à partir de l'agrégat 15min (exports/velib.parquet).
    - Cible: y_nb = nb vélos à +horizon_minutes (ex: 60 → +4 bins)
    - Lags/rollings en nombre de bins (15 min chacun)
    """
    base = occupancy_15min(with_weather=True)
    if base is None or base.empty:
        return pd.DataFrame(), []

    # Fenêtre lookback
    cutoff = pd.Timestamp.utcnow().floor("15min") - pd.Timedelta(days=lookback_days)
    base = base[base["tbin_utc"] >= cutoff.tz_localize(None)].copy()

    # Tri
    base = base.sort_values(["stationcode", "tbin_utc"])

    # Cible: +N bins (N = horizon_minutes / 15)
    bins_h = max(1, int(round(horizon_minutes / 15)))
    base["y_nb"] = base.groupby("stationcode")["nb_velos_bin"].shift(-bins_h)

    # Lags & rollings (en bins)
    def add_lags_rollings(dfg: pd.DataFrame) -> pd.DataFrame:
        g = dfg.copy()
        lag_bins = (1, 2, 3, 4, 8, 16)  # 15, 30, 45, 60, 120, 240 min
        for b in lag_bins:
            g[f"lag_nb_{b}b"]  = g["nb_velos_bin"].shift(b)
            g[f"lag_occ_{b}b"] = g["occ_ratio_bin"].shift(b)

        # Rollings (moyennes)
        g["roll_nb_4b"]   = g["nb_velos_bin"].rolling(4, min_periods=1).mean()     # 1h
        g["roll_nb_8b"]   = g["nb_velos_bin"].rolling(8, min_periods=1).mean()     # 2h
        g["roll_occ_4b"]  = g["occ_ratio_bin"].rolling(4, min_periods=1).mean()
        g["roll_occ_8b"]  = g["occ_ratio_bin"].rolling(8, min_periods=1).mean()

        # Tendance locale (pente approx sur 1h)
        g["trend_nb_4b"]  = (g["nb_velos_bin"] - g["nb_velos_bin"].shift(4)) / 4.0
        g["trend_occ_4b"] = (g["occ_ratio_bin"] - g["occ_ratio_bin"].shift(4)) / 4.0
        return g

    base = base.groupby("stationcode", group_keys=False).apply(add_lags_rollings)

    # Features calendaires (Europe/Paris) basées sur hour_utc
    base = add_calendar_features(base, tz="Europe/Paris")

    # Numériques
    num_cols = [
        "nb_velos_bin","nb_bornes_bin","capacity_bin","occ_ratio_bin",
        "temp_C","precip_mm","wind_mps",
        *[f"lag_nb_{b}b" for b in (1,2,3,4,8,16)],
        *[f"lag_occ_{b}b" for b in (1,2,3,4,8,16)],
        "roll_nb_4b","roll_nb_8b","roll_occ_4b","roll_occ_8b",
        "trend_nb_4b","trend_occ_4b",
        *calfeat_cols(base),  # contient aussi l'heure/dow + météo si présente
    ]

    # dédoublonnage en conservant l'ordre
    seen = set()
    num_cols = [c for c in num_cols if not (c in seen or seen.add(c))]

    # colonnes manquantes -> 0.0
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

    base = base.dropna(subset=["y_nb"]).reset_index(drop=True)
    base["y_nb"] = base["y_nb"].astype("float32")
    return base, num_cols


def prepare_live_features(df_live: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    out = df_live.copy()
    for c in feat_cols:
        if c not in out.columns:
            out[c] = 0.0
    out = out[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    return out
