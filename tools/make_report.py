# tools/make_forecast_page.py
import sys
from pathlib import Path

# --- ensure src/ est bien importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from src.forecast import train, predict
from src.features import build_training_frame

DOCS = ROOT / "docs"
FIGS = DOCS / "assets" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
OUT = DOCS / "forecast.md"


def _plot_obs_vs_pred(stations, horizon=1):
    hourly = pd.read_parquet("exports/velib_hourly.parquet")
    hourly["hour_utc"] = pd.to_datetime(hourly["hour_utc"])
    dfp = predict(horizon)
    dfp["hour_utc"] = pd.to_datetime(dfp["hour_utc"])

    md_blocks = []
    for sc in stations:
        obs = hourly[hourly["stationcode"] == sc].sort_values("hour_utc")
        pred = dfp[dfp["stationcode"] == sc].sort_values("hour_utc")
        if obs.empty or pred.empty:
            continue
        dfm = pd.merge(
            obs[["hour_utc", "nb_velos_hour"]],
            pred[["hour_utc", "y_nb_pred"]],
            on="hour_utc",
            how="inner",
        )
        if dfm.empty:
            continue

        fig = FIGS / f"obs_pred_{sc}_T+{horizon}h.png"
        plt.figure(figsize=(9, 4))
        plt.plot(dfm["hour_utc"], dfm["nb_velos_hour"], label="observé (nb_velos_hour)")
        plt.plot(dfm["hour_utc"], dfm["y_nb_pred"], linestyle="--", label=f"prédit T+{horizon}h")
        plt.title(f"Station {sc} — Observé vs Prédit")
        plt.xlabel("UTC")
        plt.ylabel("nb vélos")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig, dpi=140)
        plt.close()
        md_blocks.append(f"### Station `{sc}`\n\n![obs vs pred](assets/figs/{fig.name})\n")
    return "\n".join(md_blocks)


def _plot_feature_importance(h=1):
    bundle = joblib.load(Path("models") / f"lgb_nbvelos_T+{h}h.joblib")
    model, feat_cols = bundle["model"], bundle["feat_cols"]
    imp = pd.DataFrame(
        {"feature": feat_cols, "gain": model.feature_importance(importance_type="gain")}
    )
    imp = imp.sort_values("gain", ascending=False).head(25)
    fig = FIGS / f"feat_importance_T+{h}h.png"
    plt.figure(figsize=(8, 6))
    plt.barh(imp["feature"][::-1], imp["gain"][::-1])
    plt.title(f"Feature importance (gain) — T+{h}h")
    plt.tight_layout()
    plt.savefig(fig, dpi=140)
    plt.close()
    return fig.name


def _plot_residuals(h=1):
    df, feat = build_training_frame(horizon_hours=h)
    y = df["y_nb"].astype(float).values
    X = df[feat].copy()
    if "stationcode" in X.columns:
        X["stationcode"] = X["stationcode"].astype("category")
    bundle = joblib.load(Path("models") / f"lgb_nbvelos_T+{h}h.joblib")
    model, feat_cols = bundle["model"], bundle["feat_cols"]
    yhat = model.predict(X[feat_cols])
    resid = y - yhat
    fig = FIGS / f"residuals_T+{h}h.png"
    plt.figure(figsize=(8, 4))
    plt.hist(resid, bins=40)
    plt.title(f"Distribution des résidus — T+{h}h")
    plt.xlabel("y - yhat")
    plt.ylabel("freq")
    plt.tight_layout()
    plt.savefig(fig, dpi=140)
    plt.close()
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))
    return fig.name, mae, rmse


def main():
    # entraîne si absent
    if not Path("models/lgb_nbvelos_T+1h.joblib").exists():
        train(1)

    df_pred = predict(1)
    df_pred["hour_utc"] = pd.to_datetime(df_pred["hour_utc"])
    last_ts = df_pred["hour_utc"].max()
    snap = df_pred[df_pred["hour_utc"] == last_ts].copy()
    snap = snap.sort_values("y_nb_pred").head(10)

    stations = list(snap["stationcode"].head(3).unique())
    obs_pred_md = _plot_obs_vs_pred(stations, horizon=1)
    fi_png = _plot_feature_importance(1)
    res_png, mae, rmse = _plot_residuals(1)

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
        "## Observé vs Prédit (échantillon)",
        "",
        obs_pred_md or "_Aucune série suffisante pour tracer._",
        "",
        "## Qualité (in-sample, ordre de grandeur)",
        f"- MAE ≈ **{mae:.2f}** vélos — RMSE ≈ **{rmse:.2f}** vélos",
        f"![residuals](assets/figs/{res_png})",
        "",
        "## Importance des variables",
        f"![importance](assets/figs/{fi_png})",
        "",
        "> Remarque : métriques in-sample, à raffiner avec TimeSeriesSplit.",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("[forecast page] OK → docs/forecast.md")


if __name__ == "__main__":
    main()
