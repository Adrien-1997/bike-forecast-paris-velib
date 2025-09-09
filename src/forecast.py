# src/forecast.py
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb

from src.features import build_training_frame

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train(horizon_hours: int = 1, lookback_days: int = 90, num_boost_round: int = 1500):
    """
    Entraîne un LightGBM global pour prédire y_nb (nb vélos à t+h) par station.
    Sauve un bundle joblib: models/lgb_nbvelos_T+{h}h.joblib
    """
    df, feat_cols = build_training_frame(horizon_hours=horizon_hours, lookback_days=lookback_days)

    # Features
    X = df[feat_cols].copy()
    # on ne veut pas de 'hour_utc' comme feature brute
    if "hour_utc" in X.columns:
        X.drop(columns=["hour_utc"], inplace=True)

    # Catégories
    categorical = []
    if "stationcode" in X.columns:
        X["stationcode"] = X["stationcode"].astype("category")
        categorical.append(X.columns.get_loc("stationcode"))

    # Cible
    y = df["y_nb"].astype(float).values

    dtrain = lgb.Dataset(X, label=y, categorical_feature=categorical or None, free_raw_data=False)

    params = dict(
        objective="regression",
        metric=["l1", "l2"],
        learning_rate=0.05,
        num_leaves=64,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=50,
        seed=42,
        verbosity=-1,
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain],
        valid_names=["train"],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)],
    )

    bundle = {
        "model": model,
        "feat_cols": X.columns.tolist(),
        "categorical_idx": categorical,
        "horizon_hours": horizon_hours,
    }
    out_path = MODELS_DIR / f"lgb_nbvelos_T+{horizon_hours}h.joblib"
    joblib.dump(bundle, out_path)
    print(f"[forecast] saved → {out_path}")
    return model


def predict(horizon_hours: int = 1, lookback_days: int = 90) -> pd.DataFrame:
    """
    Charge le modèle T+h et prédit sur le dernier frame de features.
    Retourne stationcode, hour_utc, capacity_hour, y_nb_pred, occ_ratio_pred.
    """
    bundle_path = MODELS_DIR / f"lgb_nbvelos_T+{horizon_hours}h.joblib"
    if not bundle_path.exists():
        # entraîne à la volée si nécessaire
        train(horizon_hours=horizon_hours, lookback_days=lookback_days)

    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    feat_cols = bundle["feat_cols"]

    df, _ = build_training_frame(horizon_hours=horizon_hours, lookback_days=lookback_days)

    X = df[feat_cols].copy()
    if "stationcode" in X.columns:
        X["stationcode"] = X["stationcode"].astype("category")

    yhat = model.predict(X)

    out = df[["stationcode", "hour_utc", "capacity_hour"]].copy()
    out["y_nb_pred"] = np.clip(yhat, 0, out["capacity_hour"].fillna(np.inf))
    out["occ_ratio_pred"] = (out["y_nb_pred"] / out["capacity_hour"]).where(out["capacity_hour"] > 0)
    return out


if __name__ == "__main__":
    for h in (1, 3, 6):
        train(h)
