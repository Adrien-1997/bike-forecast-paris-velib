# tools/make_backtest.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

from src.features import build_training_frame
from src.forecast import train, predict  # réutilise ton entrainement global

DOCS = ROOT / "docs"
FIGS = DOCS / "assets" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
OUT  = DOCS / "forecast.md"

def _backtest(h=1, n_splits=5, min_train_weeks=2):
    # Frame complète (lookback déjà géré par build_training_frame)
    df, feat_cols = build_training_frame(horizon_hours=h, lookback_days=90)
    df = df.sort_values("hour_utc").reset_index(drop=True)

    # On garde uniquement colonnes features du modèle
    # et convertit stationcode en category si présent
    X = df[feat_cols].copy()
    if "hour_utc" in X: X = X.drop(columns=["hour_utc"])
    if "stationcode" in X.columns:
        X["stationcode"] = X["stationcode"].astype("category")
    y = df["y_nb"].astype(float).values
    ts  = pd.to_datetime(df["hour_utc"])  # pour reporter les dates

    # TimeSeriesSplit “expanding window”
    tss = TimeSeriesSplit(n_splits=n_splits, test_size=None)
    out_rows = []
    y_all, yhat_all = [], []

    # Entraînement LightGBM à chaque split
    for i, (train_idx, test_idx) in enumerate(tss.split(X)):
        # sécurité : impose un minimum de train (sinon split trop tôt)
        if (ts.iloc[train_idx].max() - ts.iloc[train_idx].min()).days < min_train_weeks*7:
            continue

        # (ré)entraîne un modèle global sur ce train
        # on réutilise src.forecast.train pour rester cohérent,
        # mais ici on fait un fit in-memory simple pour la vitesse
        import lightgbm as lgb
        cat_idx = [X.columns.get_loc("stationcode")] if "stationcode" in X.columns else None
        dtrain = lgb.Dataset(X.iloc[train_idx], label=y[train_idx],
                             categorical_feature=cat_idx, free_raw_data=False)
        params = dict(objective="regression", metric=["l1","l2"], learning_rate=0.05,
                      num_leaves=64, feature_fraction=0.9, bagging_fraction=0.8,
                      bagging_freq=1, min_data_in_leaf=50, seed=42, verbosity=-1)
        model = lgb.train(params, dtrain, num_boost_round=800)

        yhat = model.predict(X.iloc[test_idx])
        yt   = y[test_idx]
        mae  = mean_absolute_error(yt, yhat)
        rmse = mean_squared_error(yt, yhat, squared=False)
        med  = median_absolute_error(yt, yhat)

        out_rows.append({
            "fold": i+1,
            "test_start": ts.iloc[test_idx].min(),
            "test_end":   ts.iloc[test_idx].max(),
            "n_test": len(test_idx),
            "MAE": mae, "RMSE": rmse, "MedAE": med
        })
        y_all.append(yt); yhat_all.append(yhat)

    if not out_rows:
        raise SystemExit("Backtest: pas assez de données pour créer des splits.")

    metrics = pd.DataFrame(out_rows)
    y_all   = np.concatenate(y_all); yhat_all = np.concatenate(yhat_all)
    overall = {
        "MAE": float(np.mean(np.abs(y_all - yhat_all))),
        "RMSE": float(np.sqrt(np.mean((y_all - yhat_all)**2))),
        "MedAE": float(np.median(np.abs(y_all - yhat_all)))
    }
    return metrics, overall

def _plot_metrics_table(df: pd.DataFrame, h: int):
    # rend un PNG simple de la table
    fig = plt.figure(figsize=(8, 1 + 0.35*len(df)))
    plt.axis('off')
    cols = ["fold","test_start","test_end","n_test","MAE","RMSE","MedAE"]
    tbl = plt.table(cellText=df[cols].values,
                    colLabels=cols, loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    tbl.scale(1, 1.2)
    fig_name = f"backtest_metrics_T+{h}h.png"
    plt.tight_layout(); plt.savefig(FIGS/fig_name, dpi=160); plt.close()
    return fig_name

def _plot_pred_vs_true_sample(h=1):
    # utilise predict(h) et l’agrégat observé pour la dernière fenêtre
    dfp = predict(h)
    dfp["hour_utc"] = pd.to_datetime(dfp["hour_utc"])
    try:
        obs = pd.read_parquet("exports/velib_hourly.parquet")
    except Exception:
        return None
    obs["hour_utc"] = pd.to_datetime(obs["hour_utc"])

    last_ts = dfp["hour_utc"].max()
    stations = (dfp[dfp["hour_utc"]==last_ts]
                .sort_values("y_nb_pred").head(3)["stationcode"].tolist())

    md_blocks=[]
    for sc in stations:
        o = obs[obs["stationcode"]==sc].sort_values("hour_utc")
        p = dfp[dfp["stationcode"]==sc].sort_values("hour_utc")
        m = pd.merge(o[["hour_utc","nb_velos_hour"]], p[["hour_utc","y_nb_pred"]],
                     on="hour_utc", how="inner")
        if m.empty: continue
        fig = FIGS / f"obs_pred_{sc}_T+{h}h.png"
        plt.figure(figsize=(9,4))
        plt.plot(m["hour_utc"], m["nb_velos_hour"], label="observé")
        plt.plot(m["hour_utc"], m["y_nb_pred"], linestyle="--", label=f"prédit T+{h}h")
        plt.title(f"Station {sc} — Observé vs Prédit")
        plt.xlabel("UTC"); plt.ylabel("nb vélos"); plt.legend()
        plt.tight_layout(); plt.savefig(fig, dpi=140); plt.close()
        md_blocks.append(f"### Station `{sc}`\n\n![obs vs pred](assets/figs/{fig.name})\n")
    return "\n".join(md_blocks)

def main():
    # garantit un modèle T+1h
    if not (ROOT/"models"/"lgb_nbvelos_T+1h.joblib").exists():
        train(1)

    # Backtest
    metrics, overall = _backtest(h=1, n_splits=5)
    table_png = _plot_metrics_table(metrics, h=1)
    obs_pred_md = _plot_pred_vs_true_sample(h=1)

    # Top-10 risque
    df_pred = predict(1)
    df_pred["hour_utc"] = pd.to_datetime(df_pred["hour_utc"])
    last_ts = df_pred["hour_utc"].max()
    snap = (df_pred[df_pred["hour_utc"]==last_ts]
            .sort_values("y_nb_pred").head(10))

    # Feature importance (gain) déjà entraînée dans models/
    bundle = joblib.load(ROOT/"models"/"lgb_nbvelos_T+1h.joblib")
    model, feat_cols = bundle["model"], bundle["feat_cols"]
    imp = pd.DataFrame({"feature": feat_cols,
                        "gain": model.feature_importance(importance_type="gain")})\
             .sort_values("gain", ascending=False).head(20)
    fi_png = f"feat_importance_T+1h.png"
    plt.figure(figsize=(8,6)); plt.barh(imp["feature"][::-1], imp["gain"][::-1])
    plt.title("Feature importance (gain) — T+1h"); plt.tight_layout()
    plt.savefig(FIGS/fi_png, dpi=150); plt.close()

    # Écrit la page
    lines = [
        "# Prévisions",
        f"**Échéance la plus récente** : `{last_ts}` (UTC)",
        "",
        "## Top-10 stations à risque (faible nb vélos prévu T+1h)",
        "",
        "| station | y_nb_pred | occ_ratio_pred |",
        "|---|---:|---:|",
    ]
    for r in snap.itertuples():
        occ = f"{r.occ_ratio_pred:.2f}" if pd.notna(r.occ_ratio_pred) else ""
        lines.append(f"| `{r.stationcode}` | {int(round(r.y_nb_pred))} | {occ} |")

    lines += [
        "",
        "## Backtest temporel (expanding window)",
        f"![backtest](assets/figs/{table_png})",
        "",
        f"**Global** — MAE: **{overall['MAE']:.2f}** vélos • RMSE: **{overall['RMSE']:.2f}** • MedAE: **{overall['MedAE']:.2f}**",
        "",
        "## Observé vs. prédit (échantillon)",
        obs_pred_md or "_Séries insuffisantes pour tracer_",
        "",
        "## Importance des variables",
        f"![importance](assets/figs/{fi_png})",
        "",
        "> Modèle : LightGBM global (catégorie `stationcode`), features calendrier (Europe/Paris), lags/rollings, météo, Fourier 24h/7j.",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("[backtest] OK → docs/forecast.md + figs/")

if __name__ == "__main__":
    main()
