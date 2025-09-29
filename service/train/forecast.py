# service/train/forecast.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json

import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import lightgbm as lgb

from .features import build_training_frame, _load_base_5min

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def _print_dataset_span():
    df = _load_base_5min()
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

def train(horizon_minutes: int = 15, lookback_days: int = 30):
    """
    Train LGBM to predict nb bikes at T+horizon_minutes (5-min bins).
    Robust strategy: try several lookback windows; fallback to full parquet.
    """
    _print_dataset_span()

    candidates = [lookback_days, 60, 90, 120, -1]  # -1 => full parquet
    X = y = feat_cols = None

    for lb in candidates:
        if lb == -1:
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
            "[train] No non-empty training dataset. "
            "Ensure tools.export_training_base produced exports/velib.parquet with enough history."
        )

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
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )

    y_true = y_va.values if hasattr(y_va, "values") else np.asarray(y_va)
    y_pred = np.asarray(model.predict(X_va))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    print(f"[train] MAE={mae:.3f} | RMSE={rmse:.3f} on {len(y_true)} samples")

    out_path = MODELS_DIR / f"lgb_nbvelos_T+{horizon_minutes}min.joblib"
    joblib.dump(
        {"model": model, "feat_cols": feat_cols, "horizon_minutes": horizon_minutes},
        out_path
    )
    if (not out_path.exists()) or (out_path.stat().st_size == 0):
        raise RuntimeError(f"[train] Save failed → {out_path}")
    print(f"[train] saved → {out_path.resolve()}")

    # --- write metrics.json next to the model ---
    metrics = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "horizon_minutes": int(horizon_minutes),
        "rows_training": int(len(X_tr)),
        "rows_validation": int(len(y_va)),
        "mae": mae,
        "rmse": rmse,
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[train] wrote metrics → {(MODELS_DIR / 'metrics.json').resolve()}")

    return {"mae": mae, "rmse": rmse, "n_valid": int(len(y_true)), "model_path": str(out_path)}

def load_model_bundle(horizon_minutes: int = 15, model_dir: str | Path = MODELS_DIR):
    """
    Load bundle (model, feat_cols, horizon_minutes) from models/.
    Returns (model, feat_cols).
    """
    path = Path(model_dir) / f"lgb_nbvelos_T+{horizon_minutes}min.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    bundle = joblib.load(path)
    model = bundle.get("model")
    feat_cols = bundle.get("feat_cols", [])
    return model, feat_cols

if __name__ == "__main__":
    # lance un entraînement baseline (15 min d’horizon, 30j lookback)
    train(horizon_minutes=15, lookback_days=30)
