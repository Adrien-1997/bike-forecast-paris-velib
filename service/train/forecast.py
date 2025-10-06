# train/forecast.py
# =============================================================================
# Training + inference utilities for Vélib' forecasting (layout aplati).
#
# ✅ API clé pour la pipeline: predict_from_features_df(feats_df, model_uri, horizon_bins)
#    - Prend le DataFrame de features déjà construit en RAM
#    - Charge le modèle (chemin local OU URI GCS gs://)
#    - Aligne les colonnes et renvoie un DataFrame de prévisions propre
#
# Ce fichier est tolérant au layout:
#   - racine/service/train/...  (imports via service.train.*)
#   - aplati/train/...          (imports via train.*)
# =============================================================================

from __future__ import annotations

import argparse
import io
import os
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# ---- Optional sklearn imports for the CLI train() path ----
try:
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    _SKLEARN_AVAILABLE = False


# ============================================================================
# Utilities
# ============================================================================

def _utc_now_naive() -> datetime:
    """Current UTC as tz-naive datetime (pour écrire parquet/JSON proprement)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

def _is_gcs_uri(uri: str) -> bool:
    return isinstance(uri, str) and uri.startswith("gs://")

def _load_bytes_from_gcs(uri: str) -> bytes:
    """Charge un blob GCS en mémoire (bytes)."""
    from google.cloud import storage
    assert _is_gcs_uri(uri), f"Not a GCS URI: {uri}"
    bucket, key = uri[5:].split("/", 1)
    cli = storage.Client()
    blob = cli.bucket(bucket).blob(key)
    return blob.download_as_bytes()

def _load_joblib_from_uri(uri: str):
    """Charge un artefact joblib depuis un chemin local ou gs://."""
    if _is_gcs_uri(uri):
        data = _load_bytes_from_gcs(uri)
        with io.BytesIO(data) as buf:
            return joblib.load(buf)
    return joblib.load(uri)

def _ensure_columns(df: pd.DataFrame, cols: Iterable[str], fill: float = 0.0) -> pd.DataFrame:
    """S'assure que toutes les colonnes existent (créées avec une valeur par défaut)."""
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df

def _select_in_order(df: pd.DataFrame, ordered_cols: Iterable[str]) -> pd.DataFrame:
    """Renvoie une vue avec colonnes dans l'ordre attendu par le modèle."""
    return df[[c for c in ordered_cols]]

def _clip_and_round(y: np.ndarray, cap: np.ndarray | float | int) -> Tuple[np.ndarray, np.ndarray]:
    """Borne les prédictions à [0, capacity] et renvoie (float, entier arrondi)."""
    y = np.asarray(y, dtype=np.float32)
    cap_arr = np.asarray(cap) if not np.isscalar(cap) else np.full_like(y, float(cap))
    y_clipped = np.clip(y, 0.0, cap_arr.astype(np.float32))
    y_int = np.rint(y_clipped).astype(np.int16)
    return y_clipped, y_int

def _derive_model_version_from_uri(uri: str) -> str:
    """Petit helper pour extraire une version lisible depuis le nom de fichier."""
    base = uri.rstrip("/").split("/")[-1]
    return os.path.splitext(base)[0]


# ============================================================================
# In-pipeline inference API
# ============================================================================

def predict_from_features_df(
    feats_df: pd.DataFrame,
    model_uri: str,
    horizon_bins: Optional[int] = None,
    model_alias: Optional[str] = None,
) -> pd.DataFrame:
    """
    Inference directe à partir d'un DataFrame de features en mémoire.

    Exige dans feats_df au minimum:
      - 'station_id', 'tbin_latest', 'capacity_bin'
      - et toutes les colonnes listées dans 'feat_cols' du modèle
    """
    if feats_df is None or len(feats_df) == 0:
        return pd.DataFrame(
            columns=[
                "station_id",
                "tbin_latest",
                "horizon_min",
                "bikes_pred",
                "bikes_pred_int",
                "capacity_bin",
                "pred_ts_utc",
                "target_ts_utc",
                "model_version",
            ]
        )

    pack = _load_joblib_from_uri(model_uri)
    if not isinstance(pack, dict) or "model" not in pack or "feat_cols" not in pack:
        raise ValueError("Invalid model pack. Expect dict with keys: 'model', 'feat_cols', 'horizon_bins'.")

    model = pack["model"]
    feat_cols: List[str] = list(pack["feat_cols"])
    baked_hz = int(pack.get("horizon_bins", 3))
    hz_bins = int(horizon_bins) if horizon_bins is not None else baked_hz
    hz_min = int(hz_bins * 5)

    # checks
    for col in ["station_id", "tbin_latest", "capacity_bin"]:
        if col not in feats_df.columns:
            raise KeyError(f"Missing required column in features: '{col}'")

    # align features
    X_df = feats_df.copy()
    _ensure_columns(X_df, feat_cols, fill=0.0)
    X = _select_in_order(X_df, feat_cols).astype(np.float32)

    # predict
    y_hat = model.predict(X)

    # clamp & round
    cap = feats_df["capacity_bin"].to_numpy()
    y_clip, y_int = _clip_and_round(y_hat, cap)

    # timestamps
    tbin_latest = pd.to_datetime(feats_df["tbin_latest"], errors="coerce")
    target_ts = tbin_latest + pd.to_timedelta(hz_min, unit="m")

    out = pd.DataFrame(
        {
            "station_id": feats_df["station_id"].astype("Int64"),
            "tbin_latest": tbin_latest,
            "horizon_min": np.int16(hz_min),
            "bikes_pred": y_clip.astype(np.float32),
            "bikes_pred_int": y_int.astype(np.int16),
            "capacity_bin": feats_df["capacity_bin"].astype("Int64"),
            "pred_ts_utc": _utc_now_naive(),
            "target_ts_utc": target_ts,
            "model_version": (model_alias or _derive_model_version_from_uri(model_uri)),
        }
    )

    return out[
        [
            "station_id",
            "tbin_latest",
            "horizon_min",
            "bikes_pred",
            "bikes_pred_int",
            "capacity_bin",
            "pred_ts_utc",
            "target_ts_utc",
            "model_version",
        ]
    ]


# ============================================================================
# Training & offline predict CLI (facultatif)
# ============================================================================

def _require_sklearn():
    if not _SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for training/predict CLI. Install it in this environment.")

def _import_build_training_frame():
    """
    Import tolérant: d'abord layout aplati train.*, sinon service.train.*
    """
    try:
        from train.features import build_training_frame  # type: ignore
        return build_training_frame
    except ModuleNotFoundError:
        pass
    try:
        from service.train.features import build_training_frame  # type: ignore
        return build_training_frame
    except ModuleNotFoundError as e:
        raise RuntimeError("Missing 'train.features.build_training_frame' (or 'service.train.features').") from e

def train_model(src: str, horizon_bins: int = 3, out_path: str = "model.joblib"):
    """
    Entraîne un HistGradientBoostingRegressor de base à partir des snapshots 5 min.
    """
    _require_sklearn()
    build_training_frame = _import_build_training_frame()

    df, X, y, feat_cols = build_training_frame(src, horizon_bins=horizon_bins)
    if X.empty:
        raise RuntimeError("No training data (X is empty). Check your parquet source.")

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=None,
        max_iter=300,
        learning_rate=0.06,
        l2_regularization=0.0,
        random_state=42,
    )
    model.fit(X, y)

    # simple temporal hold-out: last 10%
    n = len(X)
    idx_split = int(n * 0.9)
    if 0 < idx_split < n:
        yhat = model.predict(X[idx_split:])
        mae = mean_absolute_error(y[idx_split:], yhat)
        rmse = mean_squared_error(y[idx_split:], yhat, squared=False)
        print(f"[train] hold-out MAE={mae:.3f} RMSE={rmse:.3f} (n={n-idx_split})")

    joblib.dump(
        {"model": model, "feat_cols": list(feat_cols), "horizon_bins": int(horizon_bins)},
        out_path,
    )
    print(f"[train] model saved → {out_path}")

def _import_training_utils_for_offline():
    """
    Import tolérant des utilitaires utilisés par l'offline predict.
    """
    # layout aplati
    try:
        from train.features import (  # type: ignore
            _read_many_parquets,
            _coerce_types,
            _dedupe_per_bin,
            _add_target_and_lags,
        )
        return _read_many_parquets, _coerce_types, _dedupe_per_bin, _add_target_and_lags
    except ModuleNotFoundError:
        pass
    # layout service
    try:
        from service.train.features import (  # type: ignore
            _read_many_parquets,
            _coerce_types,
            _dedupe_per_bin,
            _add_target_and_lags,
        )
        return _read_many_parquets, _coerce_types, _dedupe_per_bin, _add_target_and_lags
    except ModuleNotFoundError as e:
        raise RuntimeError("Missing required utilities in 'train.features' (or 'service.train.features').") from e

def _latest_feature_frame_for_predict(src: str, horizon_bins: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construit le dernier frame de features par station (offline), en lisant les parquets 5 min.
    """
    _read_many_parquets, _coerce_types, _dedupe_per_bin, _add_target_and_lags = _import_training_utils_for_offline()

    df = _read_many_parquets(src)
    if df.empty:
        return pd.DataFrame(), []

    df = _coerce_types(df)
    df = _dedupe_per_bin(df)

    # dernier bin observé par station
    last = (
        df.sort_values(["station_id", "tbin_utc"])
        .groupby("station_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    # ~12 bins d'historique pour lags/rollings
    hist = df.merge(last[["station_id", "tbin_utc"]], on="station_id", how="inner", suffixes=("", "_last"))
    hist = hist[hist["tbin_utc"] <= hist["tbin_utc_last"]]
    hist = hist[hist["tbin_utc"] >= hist["tbin_utc_last"] - pd.Timedelta(minutes=60)]  # 12 bins

    hist2, feat_cols = _add_target_and_lags(hist, horizon_bins=horizon_bins)

    Xstar = (
        hist2.sort_values(["station_id", "tbin_utc"])
        .groupby("station_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return Xstar, list(feat_cols)

def predict_latest_offline(src: str, model_path: str) -> pd.DataFrame:
    """
    Prédiction offline qui LIT les sources et reconstruit elle-même les features.
    """
    _require_sklearn()

    pack = joblib.load(model_path)
    model = pack["model"]
    feat_cols = list(pack["feat_cols"])
    horizon_bins = int(pack.get("horizon_bins", 3))

    Xstar, _ = _latest_feature_frame_for_predict(src, horizon_bins)
    if Xstar.empty:
        print("[predict] no base to predict.")
        return pd.DataFrame()

    # meta requises
    if "station_id" not in Xstar.columns:
        raise KeyError("Offline predict: missing 'station_id' in Xstar.")
    if "tbin_latest" not in Xstar.columns:
        Xstar["tbin_latest"] = pd.to_datetime(Xstar.get("tbin_utc"), errors="coerce")

    # align features
    for c in feat_cols:
        if c not in Xstar.columns:
            Xstar[c] = 0.0
    X = Xstar[feat_cols].astype(np.float32)

    yhat = model.predict(X)
    cap = (Xstar["capacity_bin"] if "capacity_bin" in Xstar.columns else Xstar.get("capacity", 0)).to_numpy()
    y_clip, y_int = _clip_and_round(yhat, cap)

    # timestamps
    tbin_latest = pd.to_datetime(Xstar["tbin_latest"], errors="coerce")
    hz_min = int(horizon_bins * 5)
    target_ts = tbin_latest + pd.to_timedelta(hz_min, unit="m")

    out = pd.DataFrame(
        {
            "station_id": Xstar["station_id"].astype("Int64"),
            "tbin_latest": tbin_latest,
            "horizon_min": np.int16(hz_min),
            "bikes_pred": y_clip.astype(np.float32),
            "bikes_pred_int": y_int.astype(np.int16),
            "capacity_bin": (Xstar["capacity_bin"] if "capacity_bin" in Xstar.columns else cap).astype("Int64"),
            "pred_ts_utc": _utc_now_naive(),
            "target_ts_utc": target_ts,
            "model_version": _derive_model_version_from_uri(model_path),
        }
    )
    return out


# ============================================================================
# CLI
# ============================================================================

def _cli():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_tr = sub.add_parser("train", help="Train a model")
    ap_tr.add_argument("--src", required=True, help="file/glob/folder of 5-min parquet snapshots")
    ap_tr.add_argument("--horizon", type=int, default=3, help="forecast horizon in 5-min bins")
    ap_tr.add_argument("--out", default="model.joblib", help="output joblib path")

    ap_pr = sub.add_parser("predict", help="Predict on the latest observed bin per station (offline, reads sources)")
    ap_pr.add_argument("--src", required=True, help="file/glob/folder of 5-min parquet snapshots")
    ap_pr.add_argument("--horizon", type=int, default=None, help="(optional) override horizon")
    ap_pr.add_argument("--model", required=True, help="trained model joblib")

    args = ap.parse_args()

    if args.cmd == "train":
        train_model(args.src, horizon_bins=args.horizon, out_path=args.out)

    elif args.cmd == "predict":
        if args.horizon is not None:
            pack = joblib.load(args.model)
            pack["horizon_bins"] = int(args.horizon)
            joblib.dump(pack, args.model)
        preds = predict_latest_offline(args.src, args.model)
        if not preds.empty:
            print(preds.head(12).to_string(index=False))

if __name__ == "__main__":
    _cli()
