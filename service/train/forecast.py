# forecast.py
# =============================================================================
# Entraînement + prédiction simple sur la base des features construites
# dans features.py (modèle scikit-learn HistGradientBoostingRegressor).
#
# Exemples CLI :
#   # train
#   python -m forecast train --src "data_local/daily/*.parquet" --horizon 3 --out model.joblib
#
#   # predict sur le dernier bin observé de chaque station
#   python -m forecast predict --src "data_local/daily/*.parquet" --horizon 3 --model model.joblib
# =============================================================================

from __future__ import annotations
import argparse, joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import build_training_frame, _read_many_parquets, _coerce_types, _dedupe_per_bin, _add_target_and_lags

def train_model(src: str, horizon_bins: int = 3, out_path: str = "model.joblib"):
    df, X, y, feat_cols = build_training_frame(src, horizon_bins=horizon_bins)
    if X.empty:
        raise RuntimeError("Pas de données d'entraînement (X vide). Vérifie la source parquet.")

    # Modèle robuste, rapide, sans preprocessing lourd
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=None,
        max_iter=300,
        learning_rate=0.06,
        l2_regularization=0.0,
        random_state=42
    )
    model.fit(X, y)

    # score simple en hold-out temporel: split dernier 10%
    n = len(X)
    idx_split = int(n * 0.9)
    if idx_split > 0 and idx_split < n:
        yhat = model.predict(X[idx_split:])
        mae = mean_absolute_error(y[idx_split:], yhat)
        rmse = mean_squared_error(y[idx_split:], yhat, squared=False)
        print(f"[train] hold-out MAE={mae:.3f} RMSE={rmse:.3f} (n={n-idx_split})")

    joblib.dump({"model": model, "feat_cols": feat_cols, "horizon_bins": horizon_bins}, out_path)
    print(f"[train] modèle sauvegardé → {out_path}")

def _latest_feature_frame_for_predict(src: str, horizon_bins: int) -> pd.DataFrame:
    """
    Construit la matrice X* pour prédire y_nb au prochain horizon sur le
    DERNIER tbin_utc disponible pour chaque station (one-shot forecast).
    """
    df = _read_many_parquets(src)
    if df.empty:
        return pd.DataFrame()

    df = _coerce_types(df)
    df = _dedupe_per_bin(df)

    # garder uniquement le dernier tbin par station
    last = (df.sort_values(["station_id","tbin_utc"])
              .groupby("station_id", as_index=False).tail(1)
              .reset_index(drop=True))

    # recréer lags/rollings/cales pour ce point (il faut un peu d'historique,
    # donc on prend aussi les 12 bins précédents pour calculer les lags/rollings)
    hist = (df
            .merge(last[["station_id","tbin_utc"]], on="station_id", how="inner", suffixes=("","_last"))
           )
    hist = hist[hist["tbin_utc"] <= hist["tbin_utc_last"]]
    hist = hist[hist["tbin_utc"] >= hist["tbin_utc_last"] - pd.Timedelta(minutes=60)]  # 12 bins

    # recalcul features avec cible future (qu'on ignorera)
    hist2, feat_cols = _add_target_and_lags(hist, horizon_bins=horizon_bins)
    # reprendre uniquement la dernière ligne par station (celle à tbin_utc_last)
    Xstar = (hist2.sort_values(["station_id","tbin_utc"])
                  .groupby("station_id", as_index=False).tail(1)
                  .reset_index(drop=True))
    return Xstar[feat_cols + ["station_id","tbin_utc"]], feat_cols

def predict_latest(src: str, model_path: str):
    pack = joblib.load(model_path)
    model = pack["model"]
    feat_cols = pack["feat_cols"]
    horizon_bins = pack["horizon_bins"]

    Xstar, fcols = _latest_feature_frame_for_predict(src, horizon_bins)
    if Xstar.empty:
        print("[predict] pas de base pour prédire.")
        return pd.DataFrame()

    # aligner colonnes (sécurité)
    for c in feat_cols:
        if c not in Xstar.columns:
            Xstar[c] = 0
    X = Xstar[feat_cols]
    yhat = model.predict(X)
    out = Xstar[["station_id","tbin_utc"]].copy()
    out["y_hat"] = yhat
    out["horizon_bins"] = horizon_bins
    print(f"[predict] prédictions: {len(out)} stations, horizon={horizon_bins} bins")
    return out

# --------------------- CLI ---------------------

def _cli():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_tr = sub.add_parser("train", help="Entraîner un modèle")
    ap_tr.add_argument("--src", required=True, help="fichier/glob/dossier de parquets 5min")
    ap_tr.add_argument("--horizon", type=int, default=3, help="horizon en bins (5min)")
    ap_tr.add_argument("--out", default="model.joblib", help="output joblib")

    ap_pr = sub.add_parser("predict", help="Prédire sur le dernier bin observé de chaque station")
    ap_pr.add_argument("--src", required=True, help="fichier/glob/dossier de parquets 5min")
    ap_pr.add_argument("--horizon", type=int, default=None, help="(optionnel) override horizon")
    ap_pr.add_argument("--model", required=True, help="modele joblib")

    args = ap.parse_args()

    if args.cmd == "train":
        train_model(args.src, horizon_bins=args.horizon, out_path=args.out)
    elif args.cmd == "predict":
        # si --horizon fourni, on ignore celui du modèle pour reconstruire les features
        if args.horizon is not None:
            pack = joblib.load(args.model)
            pack["horizon_bins"] = int(args.horizon)
            joblib.dump(pack, args.model)
        preds = predict_latest(args.src, args.model)
        if not preds.empty:
            print(preds.head(12).to_string(index=False))

if __name__ == "__main__":
    _cli()
