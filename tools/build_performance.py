# tools/build_performance.py
# Performance des prévisions : métriques, obs vs préd, heatmaps d'erreurs, calibration, biais.
# - lit perf.(parquet|csv) si fourni ; sinon construit depuis docs/exports/velib.parquet via datasets.load_normalized()
# - par défaut, horizon=60 min, fenêtre last_days=7
#
# Sorties (docs/assets):
#   tables/model_metrics.csv
#   figs/mon_metrics_by_horizon.png
#   figs/mon_pred_vs_true.png
#   figs/obs_vs_pred_station_24h.png
#   figs/errors_hour_x_dow.png
#   figs/residual_hist.png
#   figs/bias_over_time.png
#   figs/calibration_plot.png
#   (optionnel) figs/lift_vs_baseline.png si baseline dispo
#
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_normalized

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIGS = DOCS / "assets" / "figs"
TABLES = DOCS / "assets" / "tables"


def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    ensure_dir(path)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def read_perf(perf_path: Optional[Path], horizon: int, last_days: Optional[int]) -> pd.DataFrame:
    """
    Charge perf [ts, station_id, y_true, y_pred, horizon_min].
    - Si perf_path existe, on le lit.
    - Sinon, on construit depuis velib.parquet.
    """
    if perf_path and Path(perf_path).exists():
        if str(perf_path).lower().endswith(".parquet"):
            perf = pd.read_parquet(perf_path)
        else:
            perf = pd.read_csv(perf_path)
    else:
        _, perf, _, _, _ = load_normalized(
            DOCS / "exports" / "velib.parquet",
            horizon_minutes=horizon,
            last_days=last_days,
        )

    need = {"ts", "station_id", "y_true", "y_pred"}
    miss = need - set(perf.columns)
    if miss:
        raise ValueError(f"perf missing columns: {miss}")

    perf["ts"] = pd.to_datetime(perf["ts"], utc=False, errors="coerce")
    perf = perf.sort_values(["station_id", "ts"]).reset_index(drop=True)

    # horizon column
    if "horizon_min" not in perf.columns:
        perf["horizon_min"] = horizon

    # Numériques safe
    perf["y_true"] = pd.to_numeric(perf["y_true"], errors="coerce")
    perf["y_pred"] = pd.to_numeric(perf["y_pred"], errors="coerce")

    return perf


def metrics_by_horizon(perf: pd.DataFrame, out_csv: Path, out_fig: Path) -> pd.DataFrame:
    rows = []
    for h, g in perf.groupby("horizon_min"):
        e = g["y_pred"] - g["y_true"]
        mae = float(np.nanmean(np.abs(e)))
        rmse = float(np.sqrt(np.nanmean(e**2)))
        mape = float(np.nanmean(np.abs(e) / np.maximum(g["y_true"], 1e-6)))
        smape = float(np.nanmean(2 * np.abs(e) / (np.abs(g["y_true"]) + np.abs(g["y_pred"]) + 1e-6)))
        rows.append({"horizon": int(h), "MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape})
    met = pd.DataFrame(rows).sort_values("horizon")
    ensure_dir(out_csv); met.to_csv(out_csv, index=False)

    # Figure
    plt.figure(figsize=(8, 4))
    x = met["horizon"].astype(str)
    plt.plot(x, met["MAE"], marker="o", label="MAE")
    plt.plot(x, met["RMSE"], marker="s", label="RMSE")
    plt.title("Metrics by horizon")
    plt.xlabel("Horizon (min)"); plt.ylabel("Error")
    plt.legend()
    save_fig(out_fig)

    return met


def plot_obs_vs_pred(perf: pd.DataFrame, days: int, out_fig: Path) -> None:
    recent = perf[perf["ts"] >= (perf["ts"].max() - pd.Timedelta(days=days))].copy()
    recent["hour"] = recent["ts"].dt.floor("h")
    agg = recent.groupby("hour")[["y_true", "y_pred"]].mean().reset_index()

    plt.figure(figsize=(10, 4))
    plt.plot(agg["hour"], agg["y_true"], label="Observed (avg)")
    plt.plot(agg["hour"], agg["y_pred"], label="Predicted (avg)")
    plt.title(f"Observed vs Predicted — last {days} days (mean across stations)")
    plt.xlabel("Time"); plt.ylabel("Bikes (avg)")
    plt.legend()
    save_fig(out_fig)


def plot_station_focus(perf: pd.DataFrame, station: Optional[str], hours: int, out_fig: Path) -> str:
    recent = perf[perf["ts"] >= (perf["ts"].max() - pd.Timedelta(hours=hours))].copy()
    if station is None:
        # station la plus échantillonnée
        station = recent["station_id"].value_counts().idxmax()
    s = recent[recent["station_id"] == str(station)].copy()
    if s.empty:
        return str(station)

    plt.figure(figsize=(10, 4))
    plt.plot(s["ts"], s["y_true"], label="Observed")
    plt.plot(s["ts"], s["y_pred"], label="Predicted")
    plt.title(f"Observed vs Predicted — last {hours}h (station {station})")
    plt.xlabel("Time"); plt.ylabel("Bikes")
    plt.legend()
    save_fig(out_fig)
    return str(station)


def plot_errors_hour_dow(perf: pd.DataFrame, out_fig: Path) -> None:
    df = perf.copy()
    df["err"] = (df["y_pred"] - df["y_true"]).abs()
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.weekday
    mat = df.groupby(["dow", "hour"])["err"].mean().unstack(fill_value=np.nan)

    plt.figure(figsize=(9, 4.8))
    try:
        arr = mat.to_numpy(dtype=float)
    except Exception:
        arr = mat.values.astype(float, copy=False)
    if arr.size == 0 or np.all(np.isnan(arr)):
        arr = np.zeros((1, 1), dtype=float)

    im = plt.imshow(arr, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks(ticks=range(7), labels=["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"])
    plt.xticks(ticks=range(24), labels=[str(h) for h in range(24)])
    plt.title("MAE heatmap — hour × weekday")
    plt.xlabel("Hour"); plt.ylabel("Day of week")
    save_fig(out_fig)


def plot_residual_hist(perf: pd.DataFrame, out_fig: Path) -> None:
    e = perf["y_pred"] - perf["y_true"]
    plt.figure(figsize=(8, 4))
    plt.hist(e.dropna(), bins=40)
    plt.title("Residuals distribution (y_pred - y_true)")
    plt.xlabel("Residual"); plt.ylabel("Count")
    save_fig(out_fig)


def plot_bias_over_time(perf: pd.DataFrame, out_fig: Path) -> None:
    df = perf.copy()
    df["hour"] = df["ts"].dt.floor("h")
    bias = (df.assign(err=(df["y_pred"] - df["y_true"]))
              .groupby("hour")["err"].mean().reset_index())
    plt.figure(figsize=(10, 4))
    plt.plot(bias["hour"], bias["err"])
    plt.axhline(0, linestyle="--")
    plt.title("Bias over time — hourly mean error")
    plt.xlabel("Time"); plt.ylabel("Mean error")
    save_fig(out_fig)


def plot_calibration(perf: pd.DataFrame, out_fig: Path, n_bins: int = 20) -> None:
    df = perf.dropna(subset=["y_true", "y_pred"]).copy()

    # Cas dégénérés
    if df.empty or df["y_pred"].nunique(dropna=True) < 2:
        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("Calibration (variation insuffisante)")
        plt.xlabel("Predicted"); plt.ylabel("Observed")
        save_fig(out_fig)
        return

    # Binning par quantiles (réduit si peu de valeurs distinctes)
    q = min(n_bins, max(2, int(df["y_pred"].nunique(dropna=True))))
    df["bin"] = pd.qcut(df["y_pred"], q=q, duplicates="drop")

    # Moyennes par bin (observed=False pour compatibilité pandas)
    cal = df.groupby("bin", observed=False)[["y_pred", "y_true"]].mean().reset_index(drop=True)

    # Conversion float NA-safe
    yp = pd.to_numeric(cal["y_pred"], errors="coerce").to_numpy(dtype=float)
    yt = pd.to_numeric(cal["y_true"], errors="coerce").to_numpy(dtype=float)

    # Si tout est NaN après conversion
    if yp.size == 0 or yt.size == 0 or np.all(np.isnan(yp)) or np.all(np.isnan(yt)):
        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("Calibration (données insuffisantes)")
        plt.xlabel("Predicted"); plt.ylabel("Observed")
        save_fig(out_fig)
        return

    lo = float(min(np.nanmin(yp), np.nanmin(yt)))
    hi = float(max(np.nanmax(yp), np.nanmax(yt)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = 0.0, max(1.0, float(np.nanmax(yt)))

    xs = np.linspace(lo, hi, 100)

    plt.figure(figsize=(5, 5))
    plt.scatter(yp, yt, s=20)
    plt.plot(xs, xs, linestyle="--")
    plt.title("Calibration (mean true vs mean predicted per bin)")
    plt.xlabel("Predicted"); plt.ylabel("Observed")
    save_fig(out_fig)


def plot_lift_vs_baseline(perf: pd.DataFrame, baseline_col: str, out_fig: Path) -> bool:
    if baseline_col not in perf.columns:
        return False
    df = perf.dropna(subset=["y_true", "y_pred", baseline_col]).copy()
    err_model = (df["y_pred"] - df["y_true"]).abs()
    err_base = (df[baseline_col] - df["y_true"]).abs()
    lift = 1.0 - (err_model.mean() / (err_base.mean() + 1e-6))
    # courbe par déciles de difficulté (ici proxysé par y_true)
    df["q"] = pd.qcut(df["y_true"], q=10, duplicates="drop")
    g = df.groupby("q")[["y_true", "y_pred", baseline_col]].mean().reset_index(drop=True)
    e_m = (g["y_pred"] - g["y_true"]).abs()
    e_b = (g[baseline_col] - g["y_true"]).abs()
    plt.figure(figsize=(7,4))
    plt.plot(range(len(e_m)), e_m, marker="o", label="Model MAE")
    plt.plot(range(len(e_b)), e_b, marker="s", label="Baseline MAE")
    plt.title(f"Lift vs baseline (overall gain ≈ {lift*100:.1f}%)")
    plt.xlabel("True (deciles)"); plt.ylabel("MAE")
    plt.legend()
    save_fig(out_fig)
    return True


def main():
    ap = argparse.ArgumentParser(description="Build forecast performance assets (figs + tables).")
    ap.add_argument("--perf", type=Path, default=DOCS / "exports" / "perf.parquet",
                    help="Chemin vers perf.(parquet|csv). S'il n'existe pas, on utilisera velib.parquet via datasets.load_normalized().")
    ap.add_argument("--horizon", type=int, default=60, help="Horizon (minutes) if we need to rebuild from velib.parquet")
    ap.add_argument("--last-days", type=int, default=7, help="Fenêtre temporelle pour les figures temporelles (obs vs préd).")
    ap.add_argument("--station", type=str, default=None, help="Station à mettre en avant (24h)")
    ap.add_argument("--baseline-col", type=str, default="y_pred_baseline", help="Nom de la colonne baseline (si présent)")
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    perf = read_perf(args.perf, horizon=args.horizon, last_days=args.last_days)

    # 1) Métriques
    met = metrics_by_horizon(perf, TABLES / "model_metrics.csv", FIGS / "mon_metrics_by_horizon.png")

    # 2) Obs vs Préd (agrégé) sur last_days
    plot_obs_vs_pred(perf, days=args.last_days, out_fig=FIGS / "mon_pred_vs_true.png")

    # 3) Focus station 24h
    chosen = plot_station_focus(perf, station=args.station, hours=24, out_fig=FIGS / "obs_vs_pred_station_24h.png")

    # 4) Heatmap des erreurs (MAE) heure × jour
    plot_errors_hour_dow(perf, FIGS / "errors_hour_x_dow.png")

    # 5) Distribution des résidus
    plot_residual_hist(perf, FIGS / "residual_hist.png")

    # 6) Biais moyen horaire
    plot_bias_over_time(perf, FIGS / "bias_over_time.png")

    # 7) Calibration
    plot_calibration(perf, FIGS / "calibration_plot.png")

    # 8) (Optionnel) Lift vs baseline si une colonne baseline est dispo
    had_lift = plot_lift_vs_baseline(perf, args.baseline_col, FIGS / "lift_vs_baseline.png")

    print("[OK] Performance assets generated:")
    print(" - figs: mon_metrics_by_horizon.png | mon_pred_vs_true.png | obs_vs_pred_station_24h.png | errors_hour_x_dow.png | residual_hist.png | bias_over_time.png | calibration_plot.png")
    if had_lift:
        print("         + lift_vs_baseline.png (baseline detected)")
    print(f" - tbls: {TABLES / 'model_metrics.csv'}")
    print(f" - station focus: {chosen}")
    print("Tips: --station <id> pour forcer la station focus, --baseline-col <col> si vous avez une baseline explicite")


if __name__ == "__main__":
    main()