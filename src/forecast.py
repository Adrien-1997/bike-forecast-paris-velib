# src/forecast.py
from __future__ import annotations

import os
from pathlib import Path
import joblib
import pandas as pd
from lightgbm import LGBMRegressor

from src.features import build_training_frame, _load_base_15min

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "lgb_nbvelos_T+60min.joblib"


def _print_dataset_span():
    df = _load_base_15min()
    tmin = pd.to_datetime(df["tbin_utc"]).min()
    tmax = pd.to_datetime(df["tbin_utc"]).max()
    nrow = len(df)
    nsta = df["stationcode"].nunique()
    print(f"[train] parquet span: {tmin} -> {tmax} | rows={nrow} | stations={nsta}")
    return tmin, tmax, nrow, nsta


def _try_build(horizon_minutes: int, lookback_days: int):
    try:
        df, cols = build_training_frame(horizon_minutes=horizon_minutes, lookback_days=lookback_days)
        return df, cols
    except Exception as e:
        print(f"[train] build failed for lookback={lookback_days}: {e}")
        return pd.DataFrame(), []


def train(horizon_minutes: int = 60, lookback_days: int = 30):
    """
    Entraîne LGBM pour prédire nb vélos à T+60 min (bins 15 min).
    Stratégie robuste : essaie plusieurs fenêtres si la première est vide.
    """
    _print_dataset_span()

    # Essais de fenêtres (de la plus courte à la plus large, puis 'all')
    candidates = [lookback_days, 60, 90, 120, -1]  # -1 => tout le parquet
    X = y = feat_cols = None

    for lb in candidates:
        if lb == -1:
            # fallback "tout le parquet"
            df_all, cols_all = _try_build(horizon_minutes, lookback_days=3650)
            if not df_all.empty:
                feat_cols = cols_all
                X = df_all[feat_cols]
                y = df_all["y_nb"]
                print(f"[train] using full parquet (rows={len(df_all)})")
                break
        else:
            df, cols = _try_build(horizon_minutes, lookback_days=lb)
            if not df.empty:
                feat_cols = cols
                X = df[feat_cols]
                y = df["y_nb"]
                print(f"[train] using lookback_days={lb} (rows={len(df)})")
                break

    if X is None or len(X) == 0:
        raise RuntimeError(
            "[train] Impossible de constituer un dataset d'entraînement non vide. "
            "Vérifie que le workflow 'velib-ingest' publie bien docs/exports/velib.parquet "
            "avec suffisamment d'historique."
        )

    # Split temporel simple (80/20)
    # On garde l'ordre temporel tel quel
    split = int(len(X) * 0.8)
    X_tr, X_va = X.iloc[:split], X.iloc[split:]
    y_tr, y_va = y.iloc[:split], y.iloc[split:]

    params = dict(
        n_estimators=1000,
        learning_rate=0.08,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
    model = LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="l2",            # RMSE²
        verbose=100,
        callbacks=None,
    )

    # Évaluation rapide
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    y_pred = model.predict(X_va)
    mae = mean_absolute_error(y_va, y_pred)
    rmse = mean_squared_error(y_va, y_pred, squared=False)
    print(f"[train] MAE={mae:.3f} | RMSE={rmse:.3f} on {len(y_va)} samples")

    # Sauvegarde artefact
    joblib.dump(
        dict(model=model, feat_cols=feat_cols, horizon_minutes=horizon_minutes),
        MODEL_PATH
    )
    print(f"[train] saved → {MODEL_PATH.resolve()}")
    return {"mae": mae, "rmse": rmse, "n_valid": int(len(y_va))}
