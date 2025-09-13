# src/forecast.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

from src.features import build_training_frame, prepare_live_features


def _temporal_split(df: pd.DataFrame, feat_cols: List[str], valid_frac: float = 0.2):
    if "tbin_utc" not in df.columns:
        raise ValueError("Le dataset doit contenir 'tbin_utc' (15 min) pour un split temporel.")
    df_sorted = df.sort_values(["tbin_utc", "stationcode"]).reset_index(drop=True)
    n = len(df_sorted)
    cut = max(1, int(n * (1 - valid_frac)))
    X_train = df_sorted.iloc[:cut][feat_cols]
    y_train = df_sorted.iloc[:cut]["y_nb"].astype(float)
    X_valid = df_sorted.iloc[cut:][feat_cols]
    y_valid = df_sorted.iloc[cut:]["y_nb"].astype(float)
    return X_train, y_train, X_valid, y_valid


def train(
    horizon_minutes: int = 60,
    lookback_days: int = 30,
    model_dir: str | Path = "models",
    valid_frac: float = 0.2,
    params: Dict | None = None,
) -> Path:
    df, feat_cols = build_training_frame(horizon_minutes=horizon_minutes, lookback_days=lookback_days)
    if df is None or df.empty or not feat_cols:
        raise ValueError("[train] Dataset vide — lance d'abord ingestion/aggregate, ou augmente lookback_days.")

    X_train, y_train, X_valid, y_valid = _temporal_split(df, feat_cols, valid_frac=valid_frac)
    if X_train.empty or X_valid.empty:
        raise ValueError("[train] Train/valid vides après split — ajuste valid_frac ou lookback_days.")

    default_params = dict(
        objective="regression",
        metric=["l1", "l2"],
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=50,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        seed=42,
        verbose=-1,
    )
    if params:
        default_params.update(params)

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=list(X_train.columns))
    valid_set = lgb.Dataset(X_valid, label=y_valid, feature_name=list(X_valid.columns))

    model = lgb.train(
        default_params,
        train_set,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)],
    )

    yhat_valid = model.predict(X_valid, num_iteration=model.best_iteration)
    mae = float(np.mean(np.abs(yhat_valid - y_valid)))
    rmse = float(np.sqrt(np.mean((yhat_valid - y_valid) ** 2)))
    print(f"[train] valid MAE={mae:.3f} | RMSE={rmse:.3f} | it={model.best_iteration}")

    model_dir = Path(model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / f"lgb_nbvelos_T+{int(horizon_minutes)}min.joblib"
    bundle = {
        "model": model,
        "features": list(X_train.columns),
        "horizon_minutes": int(horizon_minutes),
        "lookback_days": int(lookback_days),
        "valid_frac": float(valid_frac),
        "metrics": {"mae_valid": mae, "rmse_valid": rmse},
        "version": "v3.0-15min",
    }
    joblib.dump(bundle, out_path)
    print(f"[train] OK → {out_path} (features={len(bundle['features'])})")
    return out_path


def load_model_bundle(horizon_minutes: int = 60, model_dir: str | Path = "models"):
    path = Path(model_dir) / f"lgb_nbvelos_T+{int(horizon_minutes)}min.joblib"
    artefact = joblib.load(path)
    if isinstance(artefact, dict):
        return artefact["model"], artefact.get("features", [])
    return artefact, getattr(artefact, "feature_name_", [])


def predict_from_bundle(df_live: pd.DataFrame, horizon_minutes: int = 60, model_dir: str | Path = "models"):
    model, feats = load_model_bundle(horizon_minutes=horizon_minutes, model_dir=model_dir)
    if not feats:
        raise ValueError("Le bundle chargé ne contient pas la liste de features.")
    X = prepare_live_features(df_live, feats)
    best_it = getattr(model, "best_iteration", None)
    y = model.predict(X, num_iteration=best_it)
    return np.asarray(y, dtype=float)


if __name__ == "__main__":
    out = train(horizon_minutes=60, lookback_days=30)
    print(f"[forecast.__main__] modèle sauvegardé → {out}")
