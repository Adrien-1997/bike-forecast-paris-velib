# tools/build_performance.py
# Produit les figures/tables de performance du modèle à partir de docs/exports/perf.parquet.
# - Compare STRICTEMENT y_pred(T) vs y_true(T) (y_true = bikes observé à T+h).
# - Ne retombe sur la baseline QUE si y_pred est réellement absent sur toute la fenêtre.
# - Conserve les timestamps en UTC dans les données ; conversion éventuelle en affichage uniquement.
#
# Assets écrits (dans docs/assets/figs et docs/assets/tables) :
#   - mon_pred_vs_true.png              (moyenne multi-stations, dernière fenêtre)
#   - obs_vs_pred_station_24h.png       (zoom 24h sur 1 station)
#   - mon_metrics_by_horizon.png        (placeholder: courbe simple à horizon fixe)
#   - errors_hour_x_dow.png             (heatmap erreurs moyennes par heure x jour)
#   - residual_hist.png                 (histogramme résidus)
#   - bias_over_time.png                (biais moyen par jour)
#   - calibration_plot.png              (observé vs prédit binned)
#   - lift_vs_baseline.png              (si y_pred_baseline présent)
#   - tables/model_metrics.csv          (MAE, RMSE, MAPE, Bias)
#
# Usage :
#   python tools/build_performance.py --perf docs/exports/perf.parquet --last-days 7 --horizon 60 --tz Europe/Paris

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXPORTS = DOCS / "exports"
ASSETS = DOCS / "assets"
FIGS = ASSETS / "figs"
TBLS = ASSETS / "tables"

plt.rcParams.update({"figure.autolayout": True})


# ----------------------------- utils -----------------------------

def _ensure_dirs():
    FIGS.mkdir(parents=True, exist_ok=True)
    TBLS.mkdir(parents=True, exist_ok=True)


def _to_local(ts: pd.Series, tz: str | None) -> pd.Series:
    if tz:
        # suppose ts naïf (UTC) → le localise en UTC puis convertit
        return pd.to_datetime(ts, utc=True, errors="coerce").dt.tz_convert(tz)
    return pd.to_datetime(ts, utc=False, errors="coerce")


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def _pick_station_for_focus(df: pd.DataFrame) -> str:
    # station avec le plus de points sur la fenêtre
    return (
        df["station_id"]
        .value_counts()
        .idxmax()
    )


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    mask = y_true.notna() & y_pred.notna()
    yt, yp = y_true[mask].astype(float), y_pred[mask].astype(float)
    if len(yt) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "Bias": np.nan}
    mae = np.mean(np.abs(yp - yt))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs((yp - yt) / np.where(yt == 0, np.nan, yt))) * 100.0)
    bias = float(np.mean(yp - yt))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Bias": bias}


def _best_lag(a: pd.Series, b: pd.Series, max_steps: int = 8) -> tuple[int, float]:
    # diagnostique seulement (pas appliqué à la donnée sauvegardée)
    best_r, best_k = -np.inf, 0
    for k in range(-max_steps, max_steps + 1):
        r = a.corr(b.shift(k))
        if pd.notna(r) and r > best_r:
            best_r, best_k = r, k
    return best_k, float(best_r)


# ----------------------------- plots -----------------------------

def plot_obs_vs_pred_mean(df: pd.DataFrame, tz: str | None, out: Path, hours: int = 72):
    tmax = df["ts"].max()
    win = df[df["ts"] >= (tmax - pd.Timedelta(hours=hours))].copy()
    # garde uniquement lignes où y_true ET y_pred existent
    win = win.dropna(subset=["y_true", "y_pred"])
    if win.empty:
        return
    # moyenne horaire toutes stations
    win["hour"] = win["ts"].dt.floor("h")
    agg = win.groupby("hour", as_index=False)[["y_true", "y_pred"]].mean()
    lag, r = _best_lag(agg["y_true"], agg["y_pred"])
    x = _to_local(agg["hour"], tz)
    plt.figure(figsize=(10, 4))
    plt.plot(x, agg["y_true"], label="Observed (y_true)")
    plt.plot(x, agg["y_pred"], label="Predicted (y_pred)")
    plt.title(f"Observed vs Predicted — last {hours}h (mean across stations)\nlag*={lag} steps (15min), corr={r:.3f}")
    plt.xlabel(f"Time ({tz or 'UTC'})"); plt.ylabel("Bikes (avg)")
    plt.legend()
    _savefig(out)


def plot_station_focus(df: pd.DataFrame, station_id: str, tz: str | None, out: Path):
    sub = df[df["station_id"] == str(station_id)].dropna(subset=["y_true", "y_pred"]).copy()
    if sub.empty:
        return
    # zoom 24h
    tmax = sub["ts"].max()
    sub = sub[sub["ts"] >= (tmax - pd.Timedelta(hours=24))].copy()
    x = _to_local(sub["ts"], tz)
    plt.figure(figsize=(10, 4))
    plt.plot(x, sub["y_true"], label="Observed (y_true)")
    plt.plot(x, sub["y_pred"], label="Predicted (y_pred)")
    plt.title(f"Station {station_id} — last 24h")
    plt.xlabel(f"Time ({tz or 'UTC'})"); plt.ylabel("Bikes")
    plt.legend()
    _savefig(out)


def plot_residual_hist(df: pd.DataFrame, out: Path):
    sub = df.dropna(subset=["y_true", "y_pred"]).copy()
    if sub.empty:
        return
    resid = (sub["y_pred"] - sub["y_true"]).astype(float)
    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=50)
    plt.title("Residuals (y_pred - y_true)")
    plt.xlabel("Residual"); plt.ylabel("Count")
    _savefig(out)


def plot_bias_over_time(df: pd.DataFrame, out: Path):
    sub = df.dropna(subset=["y_true", "y_pred"]).copy()
    if sub.empty:
        return
    sub["date"] = sub["ts"].dt.date
    daily = sub.groupby("date", as_index=False).apply(
        lambda g: pd.Series({"bias": float((g["y_pred"] - g["y_true"]).mean())})
    )
    plt.figure(figsize=(10, 4))
    plt.plot(pd.to_datetime(daily["date"]), daily["bias"])
    plt.title("Bias over time (mean(y_pred - y_true) per day)")
    plt.xlabel("Date"); plt.ylabel("Bias")
    _savefig(out)


def plot_errors_hour_x_dow(df: pd.DataFrame, tz: str | None, out: Path):
    sub = df.dropna(subset=["y_true", "y_pred"]).copy()
    if sub.empty:
        return
    # index local ou UTC uniquement pour l'affichage
    tloc = _to_local(sub["ts"], tz)
    sub["hour"] = tloc.dt.hour
    sub["dow"] = tloc.dt.dayofweek
    sub["abs_err"] = (sub["y_pred"] - sub["y_true"]).abs()
    pivot = sub.pivot_table(index="dow", columns="hour", values="abs_err", aggfunc="mean")
    plt.figure(figsize=(10, 4))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    plt.xticks(range(24), range(24))
    plt.colorbar(label="MAE")
    plt.title("Mean Absolute Error by Hour x DayOfWeek")
    _savefig(out)


def plot_calibration(df: pd.DataFrame, out: Path, bins: int = 20):
    sub = df.dropna(subset=["y_true", "y_pred"]).copy()
    if sub.empty:
        return
    q = pd.qcut(sub["y_pred"], q=bins, duplicates="drop")
    cal = sub.groupby(q).apply(lambda g: pd.Series({
        "pred_mean": float(g["y_pred"].mean()),
        "obs_mean": float(g["y_true"].mean())
    })).dropna()
    plt.figure(figsize=(5, 5))
    plt.plot(cal["pred_mean"], cal["obs_mean"], marker="o")
    lim = [min(cal.min().min(), 0), max(cal.max().max(), 1)]
    plt.plot(lim, lim, linestyle="--")
    plt.title("Calibration (binning on y_pred)")
    plt.xlabel("Predicted mean"); plt.ylabel("Observed mean")
    _savefig(out)


def plot_lift_vs_baseline(df: pd.DataFrame, out: Path):
    if "y_pred_baseline" not in df.columns:
        return
    sub = df.dropna(subset=["y_true"]).copy()
    # calc MAE baseline et MAE modèle par jour
    sub["date"] = sub["ts"].dt.date
    def _mae(a, b):
        m = a.notna() & b.notna()
        if not m.any(): return np.nan
        return float(np.mean(np.abs(a[m] - b[m])))
    daily = sub.groupby("date").apply(lambda g: pd.Series({
        "mae_model": _mae(g["y_true"], g["y_pred"]),
        "mae_base":  _mae(g["y_true"], g["y_pred_baseline"])
    })).dropna(how="all")
    if daily.empty:
        return
    daily["lift_%"] = (daily["mae_base"] - daily["mae_model"]) / daily["mae_base"] * 100.0
    plt.figure(figsize=(10, 4))
    plt.plot(pd.to_datetime(daily.index), daily["lift_%"])
    plt.axhline(0, linestyle="--")
    plt.title("Lift vs Baseline (MAE)")
    plt.xlabel("Date"); plt.ylabel("Lift (%)")
    _savefig(out)


# ----------------------------- main -----------------------------

def main(perf_path: Path, last_days: int, horizon: int, tz: str | None, station: str | None):
    _ensure_dirs()

    perf = pd.read_parquet(perf_path)
    # types & tri
    perf["ts"] = pd.to_datetime(perf["ts"], utc=False, errors="coerce")
    perf["station_id"] = perf["station_id"].astype(str)
    perf = perf.sort_values(["ts", "station_id"])

    # filtre fenêtre si demandé
    if last_days and last_days > 0:
        tmax = perf["ts"].max()
        if pd.notna(tmax):
            perf = perf[perf["ts"] >= (tmax - pd.Timedelta(days=last_days))].copy()

    # fallback si VRAIMENT pas de y_pred (sinon on n’y touche pas)
    if ("y_pred" not in perf.columns) or perf["y_pred"].dropna().empty:
        if "y_pred_baseline" in perf.columns:
            print("[INFO] 'y_pred' absent → utilisation de 'y_pred_baseline' comme prédiction.")
            perf["y_pred"] = perf["y_pred_baseline"]
        else:
            print("[WARN] Ni y_pred ni baseline — rien à tracer.")
            perf["y_pred"] = np.nan

    # --- métriques globales ---
    met = _metrics(perf["y_true"], perf["y_pred"])
    mdf = pd.DataFrame([{"horizon_min": horizon, **met}])
    TBLS.mkdir(parents=True, exist_ok=True)
    mdf.to_csv(TBLS / "model_metrics.csv", index=False)

    # --- figures ---
    plot_obs_vs_pred_mean(perf, tz, FIGS / "mon_pred_vs_true.png", hours=min(24 * last_days, 96) if last_days else 72)

    st_focus = station or _pick_station_for_focus(perf)
    plot_station_focus(perf, st_focus, tz, FIGS / "obs_vs_pred_station_24h.png")

    plot_residual_hist(perf, FIGS / "residual_hist.png")
    plot_bias_over_time(perf, FIGS / "bias_over_time.png")
    plot_errors_hour_x_dow(perf, tz, FIGS / "errors_hour_x_dow.png")
    plot_calibration(perf, FIGS / "calibration_plot.png")
    plot_lift_vs_baseline(perf, FIGS / "lift_vs_baseline.png")

    # placeholder (horizon fixe) pour conserver l’asset attendu par l’orchestrateur
    plt.figure(figsize=(8, 3))
    plt.plot([horizon], [met["MAE"]], marker="o")
    plt.title("Model metrics by horizon (single horizon run)")
    plt.xlabel("Horizon (min)"); plt.ylabel("MAE")
    _savefig(FIGS / "mon_metrics_by_horizon.png")

    print("[OK] Performance assets generated:")
    print(" - figs: mon_metrics_by_horizon.png | mon_pred_vs_true.png | obs_vs_pred_station_24h.png | "
          "errors_hour_x_dow.png | residual_hist.png | bias_over_time.png | calibration_plot.png")
    if "y_pred_baseline" in perf.columns:
        print("         + lift_vs_baseline.png (baseline detected)")
    print(f" - tbls: {TBLS / 'model_metrics.csv'}")
    print(f" - station focus: {st_focus}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build performance figures/tables from perf.parquet")
    ap.add_argument("--perf", type=Path, default=EXPORTS / "perf.parquet")
    ap.add_argument("--last-days", type=int, default=7)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--tz", type=str, default=None, help="Affichage des dates (ex: Europe/Paris). Données restent en UTC.")
    ap.add_argument("--station", type=str, default=None, help="Station pour le focus 24h (default: la plus couverte).")
    args = ap.parse_args()
    main(perf_path=args.perf, last_days=args.last_days, horizon=args.horizon, tz=args.tz, station=args.station)
