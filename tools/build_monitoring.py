# tools/build_monitoring.py
# Génère des indicateurs de monitoring :
#  - mon_data_health.png      : complétude & volumes
#  - mon_psi.png              : Population Stability Index (features clés)
#  - mon_feature_importance.png: importance proxy (corrélations simples)
#  - mon_error_trend.png      : MAE quotidien (y_pred vs y_true)
#  - tables: data_health.csv | psi_features.csv | feature_importance_proxy.csv | daily_error.csv
#
# Règles:
# - Les données restent en UTC; tz optionnel pour affichage.
# - La performance compare STRICTEMENT y_pred(T) vs y_true(T).
# - Si y_pred est manquant sur toute la fenêtre, fallback baseline.

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # <- backend non interactif pour CI/headless
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXPORTS = DOCS / "exports"
ASSETS = DOCS / "assets"
FIGS = ASSETS / "figs"
TBLS = ASSETS / "tables"

plt.rcParams.update({"figure.autolayout": True})


# ------------------------- utils -------------------------

def _ensure_dirs():
    FIGS.mkdir(parents=True, exist_ok=True)
    TBLS.mkdir(parents=True, exist_ok=True)


def _to_local(ts: pd.Series, tz: str | None) -> pd.Series:
    if tz:
        return pd.to_datetime(ts, utc=True, errors="coerce").dt.tz_convert(tz)
    return pd.to_datetime(ts, utc=False, errors="coerce")


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def _psi(reference: pd.Series, current: pd.Series, bins: int = 20) -> float:
    """Population Stability Index simple (mêmes bornes sur ref & cur)."""
    ref = pd.to_numeric(reference, errors="coerce").dropna().astype(float)
    cur = pd.to_numeric(current, errors="coerce").dropna().astype(float)
    if len(ref) < 10 or len(cur) < 10:
        return np.nan
    qs = np.quantile(ref, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    ref_hist, _ = np.histogram(ref, bins=qs)
    cur_hist, _ = np.histogram(cur, bins=qs)
    ref_p = np.where(ref_hist == 0, 1e-6, ref_hist) / max(ref_hist.sum(), 1)
    cur_p = np.where(cur_hist == 0, 1e-6, cur_hist) / max(cur_hist.sum(), 1)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def _mae(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if not m.any(): return np.nan
    return float(np.mean(np.abs(a[m] - b[m])))


# ------------------------- plots -------------------------

def plot_data_health(events: pd.DataFrame, perf: pd.DataFrame, tz: str | None, out: Path):
    # Completeness (y_true, y_pred) et volume par jour
    p = perf.copy()
    p["date"] = _to_local(p["ts"], tz).dt.date
    agg = p.groupby("date").agg(
        rows=("ts", "size"),
        y_true_cov=("y_true", lambda s: float(s.notna().mean()) * 100.0),
        y_pred_cov=("y_pred", lambda s: float(s.notna().mean()) * 100.0)
    ).reset_index()

    plt.figure(figsize=(10, 4))
    ax1 = plt.gca()
    ax1.plot(pd.to_datetime(agg["date"]), agg["rows"], label="Rows (perf)")
    ax1.set_ylabel("Rows")
    ax2 = ax1.twinx()
    ax2.plot(pd.to_datetime(agg["date"]), agg["y_true_cov"], linestyle="--", label="y_true %")
    ax2.plot(pd.to_datetime(agg["date"]), agg["y_pred_cov"], linestyle="--", label="y_pred %")
    ax2.set_ylabel("Coverage (%)")
    ax1.set_title("Data health (volume & coverage)")
    ax1.set_xlabel(f"Date ({tz or 'UTC'})")
    ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
    _savefig(out)

    agg.to_csv(TBLS / "data_health.csv", index=False)


def plot_psi_features(events: pd.DataFrame, perf: pd.DataFrame, last_days: int, out_fig: Path, out_tbl: Path):
    # Fenêtre ref = 'last_days * 2 à last_days' ; current = 'last_days'
    p = perf.copy()
    p = p.sort_values("ts")
    tmax = p["ts"].max()
    if pd.isna(tmax):
        return
    cur_start = tmax - pd.Timedelta(days=last_days)
    ref_start = tmax - pd.Timedelta(days=2 * last_days)
    ref = p[(p["ts"] >= ref_start) & (p["ts"] < cur_start)]
    cur = p[p["ts"] >= cur_start]

    # features candidates (si présentes)
    cand = [
        "nb_velos_bin","nb_bornes_bin","capacity_bin","occ_ratio_bin",
        "temp_C","precip_mm","wind_mps"
    ]
    # si events a lat/lon/name, on les ignore (non numériques / stationnaires)
    # on essaie aussi quelques features dérivées éventuelles en perf
    cand = [c for c in cand if c in events.columns or c in perf.columns]

    rows = []
    for c in cand:
        r = _psi(ref.get(c, pd.Series(dtype=float)), cur.get(c, pd.Series(dtype=float)))
        rows.append({"feature": c, "psi": r})
    df_psi = pd.DataFrame(rows).sort_values("psi", ascending=False)
    df_psi.to_csv(out_tbl, index=False)

    plt.figure(figsize=(8, 4))
    plt.barh(df_psi["feature"], df_psi["psi"])
    plt.gca().invert_yaxis()
    plt.xlabel("PSI (higher = bigger shift)")
    plt.title(f"Feature Stability (PSI) — ref=[{2*last_days}-{last_days}]d vs cur=[{last_days}]d")
    _savefig(out_fig)


def plot_feature_importance_proxy(perf: pd.DataFrame, out_fig: Path, out_tbl: Path):
    # Proxy très simple : |corr(y_pred, feature)| sur la fenêtre (quand dispo)
    p = perf.copy()
    # features candidates plausibles dans perf
    cand = [c for c in p.columns if c.startswith(("lag_", "roll_", "trend_", "temp_", "precip_", "wind_", "occ_"))]
    rows = []
    for c in cand[:80]:  # limite raisonnable
        try:
            corr = float(p[[c, "y_pred"]].dropna().corr().iloc[0, 1])
        except Exception:
            corr = np.nan
        rows.append({"feature": c, "abs_corr_with_y_pred": abs(corr) if pd.notna(corr) else np.nan})
    df_imp = pd.DataFrame(rows).dropna().sort_values("abs_corr_with_y_pred", ascending=False).head(30)
    df_imp.to_csv(out_tbl, index=False)

    plt.figure(figsize=(8, 6))
    plt.barh(df_imp["feature"], df_imp["abs_corr_with_y_pred"])
    plt.gca().invert_yaxis()
    plt.xlabel("|corr(y_pred, feature)|")
    plt.title("Feature importance proxy (correlations)")
    _savefig(out_fig)


def plot_error_trend(perf: pd.DataFrame, tz: str | None, out_fig: Path, out_tbl: Path):
    p = perf.dropna(subset=["y_true"]).copy()
    # fallback baseline si y_pred absent
    if "y_pred" not in p.columns or p["y_pred"].dropna().empty:
        if "y_pred_baseline" in p.columns:
            print("[INFO] 'y_pred' absent → utilisation de 'y_pred_baseline' pour le monitoring.")
            p["y_pred"] = p["y_pred_baseline"]
        else:
            p["y_pred"] = np.nan

    p["date"] = _to_local(p["ts"], tz).dt.date
    daily = p.groupby("date").apply(lambda g: pd.Series({
        "MAE": _mae(g["y_true"], g["y_pred"]),
        "RMSE": float(np.sqrt(np.mean((g["y_pred"].astype(float) - g["y_true"].astype(float))**2)))
    })).reset_index()

    daily.to_csv(out_tbl, index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(pd.to_datetime(daily["date"]), daily["MAE"], label="MAE")
    plt.plot(pd.to_datetime(daily["date"]), daily["RMSE"], label="RMSE")
    plt.title("Daily error trend")
    plt.xlabel(f"Date ({tz or 'UTC'})"); plt.ylabel("Error")
    plt.legend()
    _savefig(out_fig)


# ------------------------- main -------------------------

def main(events_path: Path, perf_path: Path, last_days: int, tz: str | None):
    _ensure_dirs()

    events = pd.read_parquet(events_path)
    perf = pd.read_parquet(perf_path)

    # types
    for df in (events, perf):
        df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
        if "station_id" in df.columns:
            df["station_id"] = df["station_id"].astype(str)

    # fenêtre
    if last_days and last_days > 0:
        tmax = perf["ts"].max()
        if pd.notna(tmax):
            perf = perf[perf["ts"] >= (tmax - pd.Timedelta(days=last_days))].copy()
            events = events[events["ts"] >= (tmax - pd.Timedelta(days=last_days))].copy()

    # figures/tables
    plot_data_health(events, perf, tz, FIGS / "mon_data_health.png")
    plot_psi_features(events, perf, last_days, FIGS / "mon_psi.png", TBLS / "psi_features.csv")
    plot_feature_importance_proxy(perf, FIGS / "mon_feature_importance.png", TBLS / "feature_importance_proxy.csv")
    plot_error_trend(perf, tz, FIGS / "mon_error_trend.png", TBLS / "daily_error.csv")

    print("[OK] Monitoring assets generated:")
    print(" - figs: mon_data_health.png | mon_psi.png | mon_feature_importance.png | mon_error_trend.png")
    print(" - tbls: data_health.csv | psi_features.csv | feature_importance_proxy.csv | daily_error.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build monitoring indicators from events/perf")
    ap.add_argument("--events", type=Path, default=EXPORTS / "events.parquet")
    ap.add_argument("--perf", type=Path, default=EXPORTS / "perf.parquet")
    ap.add_argument("--last-days", type=int, default=7)
    ap.add_argument("--tz", type=str, default=None, help="Affichage (ex: Europe/Paris). Données restent en UTC.")
    args = ap.parse_args()
    main(events_path=args.events, perf_path=args.perf, last_days=args.last_days, tz=args.tz)
