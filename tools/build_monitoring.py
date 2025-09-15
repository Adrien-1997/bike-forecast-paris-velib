# tools/build_monitoring.py
# Rapport Monitoring (Data & Modèle) — pro & détaillé
# Garde la structure existante et ajoute :
# - Tables: global_metrics, error_by_horizon, residuals_summary, calibration, feature_importance_proxy...
# - Visuels: global metrics bar, error by horizon line, residual histogram, calibration curve,
#            feature importance (corr & MI si dispo), data health, PSI (si dispo), trend d'erreur.
#
# Hypothèses douces:
# - docs/exports/perf.parquet : colonnes attendues au minimum: ts, y_true, y_pred (ou y_pred_baseline)
#   Optionnels: station_id, horizon_min, features_* (ou noms de colonnes de features)
# - docs/exports/events.parquet : optionnel (pour meta)
#
# Sorties:
# - docs/assets/tables/*.csv
# - docs/assets/figs/*.png

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")   # headless/CI
import matplotlib.pyplot as plt

# sklearn est optionnel (mutual information)
try:
    from sklearn.feature_selection import mutual_info_regression
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXPORTS = DOCS / "exports"
ASSETS = DOCS / "assets"
FIGS = ASSETS / "figs"
TABLES = ASSETS / "tables"

PERF_PARQ = EXPORTS / "perf.parquet"
EVENTS_PARQ = EXPORTS / "events.parquet"  # optionnel

plt.rcParams.update({
    "figure.autolayout": True,
    "axes.grid": True,
})

# --------------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------------

def _ensure_dirs():
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

def _safe_savefig(path: Path, dpi: int = 150):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi)
    plt.close()

def _coerce_dt(s) -> pd.Series:
    return pd.to_datetime(s, utc=False, errors="coerce")

def _has_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns for c in cols)

def _fmt_pct(x: float) -> float:
    try:
        return float(x) * 100.0
    except Exception:
        return np.nan

# --------------------------------------------------------------------------------------
# Chargement & Préparation
# --------------------------------------------------------------------------------------

def load_perf(perf_path: Path) -> pd.DataFrame:
    try:
        perf = pd.read_parquet(perf_path)
    except Exception as e:
        raise RuntimeError(f"Impossible de lire {perf_path}: {e}")

    # Colonnes minimales
    if "ts" not in perf.columns:
        raise RuntimeError("La colonne 'ts' est requise dans perf.parquet")
    if "y_true" not in perf.columns:
        raise RuntimeError("La colonne 'y_true' est requise dans perf.parquet")

    # y_pred fallback baseline
    if "y_pred" not in perf.columns or perf["y_pred"].dropna().empty:
        if "y_pred" not in perf.columns:
            perf["y_pred"] = np.nan
        if "y_pred_baseline" in perf.columns:
            perf["y_pred"] = perf["y_pred"].fillna(perf["y_pred_baseline"])

    perf["ts"] = _coerce_dt(perf["ts"])
    if "station_id" in perf.columns:
        perf["station_id"] = perf["station_id"].astype(str)

    # clamp raisonnable si capacity dispo (optionnel)
    if "capacity" in perf.columns:
        perf["y_true"] = perf["y_true"].clip(lower=0, upper=perf["capacity"])
        perf["y_pred"] = perf["y_pred"].clip(lower=0, upper=perf["capacity"])

    return perf

# --------------------------------------------------------------------------------------
# Métriques
# --------------------------------------------------------------------------------------

def compute_global_metrics(df: pd.DataFrame) -> Dict[str, float]:
    sub = df.dropna(subset=["y_true", "y_pred"]).copy()
    if sub.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan, "Coverage": 0.0}

    y = sub["y_true"].astype(float).values
    yhat = sub["y_pred"].astype(float).values
    err = yhat - y
    mae = np.mean(np.abs(err))
    rmse = float(np.sqrt(np.mean(err**2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs(err) / np.where(y == 0, np.nan, np.abs(y)))
    if np.allclose(np.var(y), 0.0, atol=1e-12):
        r2 = np.nan
    else:
        r2 = 1.0 - (np.sum(err**2) / np.sum((y - np.mean(y))**2))

    cov = sub["y_pred"].notna().mean()

    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape), "R2": float(r2), "Coverage": float(cov)}

def compute_error_by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    if "horizon_min" not in df.columns:
        return pd.DataFrame()
    sub = df.dropna(subset=["y_true", "y_pred", "horizon_min"]).copy()
    if sub.empty:
        return pd.DataFrame()
    sub["abs_err"] = (sub["y_pred"] - sub["y_true"]).abs()
    g = sub.groupby("horizon_min")["abs_err"].agg(["mean", "median", "count"]).reset_index()
    g.rename(columns={"mean": "MAE", "median": "MedAE", "count": "N"}, inplace=True)
    return g.sort_values("horizon_min")

def compute_residuals_summary(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["y_true", "y_pred"]).copy()
    if sub.empty:
        return pd.DataFrame()
    sub["residual"] = (sub["y_pred"] - sub["y_true"]).astype(float)
    return pd.DataFrame({
        "mean": [sub["residual"].mean()],
        "std":  [sub["residual"].std()],
        "p10":  [sub["residual"].quantile(0.10)],
        "p50":  [sub["residual"].quantile(0.50)],
        "p90":  [sub["residual"].quantile(0.90)],
        "n":    [len(sub)],
    })

def compute_daily_error(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["ts", "y_true", "y_pred"]).copy()
    if sub.empty:
        return pd.DataFrame()
    sub["date"] = sub["ts"].dt.date
    sub["abs_err"] = (sub["y_pred"] - sub["y_true"]).abs()
    return sub.groupby("date")["abs_err"].mean().reset_index(name="MAE")

def compute_calibration_table(df: pd.DataFrame, q: int = 10) -> pd.DataFrame:
    sub = df.dropna(subset=["y_true", "y_pred"]).copy()
    if sub.empty:
        return pd.DataFrame()
    try:
        sub["bin"] = pd.qcut(sub["y_pred"], q=q, duplicates="drop")
    except Exception:
        return pd.DataFrame()
    tab = sub.groupby("bin", observed=False).agg(
        y_pred_mean=("y_pred", "mean"),
        y_true_mean=("y_true", "mean"),
        n=("y_true", "size")
    ).reset_index(drop=False)
    return tab

# --------------------------------------------------------------------------------------
# Feature importance proxy
# --------------------------------------------------------------------------------------

def detect_feature_columns(df: pd.DataFrame) -> List[str]:
    # Heuristique: toutes colonnes numériques non cibles et non meta évidentes
    exclude = {"y_true", "y_pred", "y_pred_baseline", "ts", "station_id", "horizon_min", "capacity"}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num_cols if c not in exclude]
    # aussi autoriser certaines colonnes catégorielles encodées
    return feat_cols

def compute_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["y_true"]).copy()
    if sub.empty:
        return pd.DataFrame(columns=["feature","abs_corr_y_true","abs_corr_residual","mi_y_true","score"])

    features = detect_feature_columns(sub)
    if not features:
        return pd.DataFrame(columns=["feature","abs_corr_y_true","abs_corr_residual","mi_y_true","score"])

    out = []
    # Corrélation (abs) avec y_true et avec le résidu (si y_pred dispo)
    if "y_pred" in sub.columns and sub["y_pred"].notna().any():
        sub["residual"] = (sub["y_pred"] - sub["y_true"]).astype(float)
    else:
        sub["residual"] = np.nan

    for f in features:
        try:
            c_true = float(pd.to_numeric(sub[f], errors="coerce").corr(pd.to_numeric(sub["y_true"], errors="coerce")))
        except Exception:
            c_true = np.nan
        try:
            c_res = float(pd.to_numeric(sub[f], errors="coerce").corr(pd.to_numeric(sub["residual"], errors="coerce")))
        except Exception:
            c_res = np.nan
        out.append({"feature": f, "abs_corr_y_true": abs(c_true) if pd.notna(c_true) else np.nan,
                    "abs_corr_residual": abs(c_res) if pd.notna(c_res) else np.nan})

    imp = pd.DataFrame(out)

    # Mutual Information (optionnelle)
    if _HAS_SKLEARN:
        try:
            X = sub[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
            y = pd.to_numeric(sub["y_true"], errors="coerce").fillna(0.0).values
            mi = mutual_info_regression(X, y, random_state=0)
            imp["mi_y_true"] = mi
        except Exception:
            imp["mi_y_true"] = np.nan
    else:
        imp["mi_y_true"] = np.nan

    # Score agrégé simple pour le tri
    imp["score"] = imp[["abs_corr_y_true", "abs_corr_residual", "mi_y_true"]].fillna(0.0).sum(axis=1)
    return imp.sort_values("score", ascending=False).reset_index(drop=True)

# --------------------------------------------------------------------------------------
# Data Health & PSI (si disponible)
# --------------------------------------------------------------------------------------

def compute_data_health(df: pd.DataFrame) -> pd.DataFrame:
    # Part de manquants par colonne, part d'out-of-range basique si capacity.
    out = []
    for c in ["y_true", "y_pred"]:
        if c in df.columns:
            na = df[c].isna().mean()
            row = {"metric": f"{c}_missing_ratio", "value": float(na)}
            out.append(row)
    if "capacity" in df.columns:
        for c in ["y_true", "y_pred"]:
            if c in df.columns:
                out_range = ( (df[c] < 0) | (df[c] > df["capacity"]) ).mean()
                out.append({"metric": f"{c}_out_of_range_ratio", "value": float(out_range)})
    return pd.DataFrame(out)

def compute_psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    # Population Stability Index simple
    try:
        r, _ = np.histogram(ref.dropna(), bins=bins)
        c, _ = np.histogram(cur.dropna(), bins=bins)
        r = r / (r.sum() + 1e-12)
        c = c / (c.sum() + 1e-12)
        psi = np.sum((c - r) * np.log((c + 1e-12) / (r + 1e-12)))
        return float(psi)
    except Exception:
        return np.nan

def compute_psi_features(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.copy()
    if sub.empty or "ts" not in sub.columns:
        return pd.DataFrame(columns=["feature", "psi"])  # ← colonnes explicites

    sub = sub.sort_values("ts")
    n = len(sub)
    if n < 50:
        return pd.DataFrame(columns=["feature", "psi"])  # ← colonnes explicites

    mid = n // 2
    ref = sub.iloc[:mid]
    cur = sub.iloc[mid:]

    feats = detect_feature_columns(sub)
    if not feats:
        return pd.DataFrame(columns=["feature", "psi"])  # ← rien à calculer

    out = []
    for f in feats:
        try:
            psi = compute_psi(pd.to_numeric(ref[f], errors="coerce"),
                              pd.to_numeric(cur[f], errors="coerce"))
        except Exception:
            psi = np.nan
        out.append({"feature": f, "psi": psi})

    if not out:
        return pd.DataFrame(columns=["feature", "psi"])  # ← sécurité

    df_out = pd.DataFrame(out, columns=["feature", "psi"])
    return df_out.sort_values("psi", ascending=False)


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------

def plot_global_metrics_bar(metrics: Dict[str, float], out: Path):
    keys = ["MAE", "RMSE", "MAPE", "R2", "Coverage"]
    vals = [metrics.get(k, np.nan) for k in keys]
    plt.figure(figsize=(7, 3.2))
    plt.bar(keys, vals)
    plt.title("Métriques globales du modèle")
    plt.ylabel("Valeur")
    _safe_savefig(out)

def plot_error_by_horizon(df: pd.DataFrame, out: Path):
    if df.empty:
        return
    plt.figure(figsize=(8, 3))
    plt.plot(df["horizon_min"], df["MAE"])
    if "MedAE" in df.columns:
        plt.plot(df["horizon_min"], df["MedAE"])
        plt.legend(["MAE", "MedAE"])
    else:
        plt.legend(["MAE"])
    plt.title("Erreur par horizon (minutes)")
    plt.xlabel("Horizon (min)"); plt.ylabel("Erreur (vélos)")
    _safe_savefig(out)

def plot_residual_histogram(df: pd.DataFrame, out: Path):
    sub = df.dropna(subset=["y_true", "y_pred"]).copy()
    if sub.empty:
        return
    resid = (sub["y_pred"] - sub["y_true"]).astype(float)
    plt.figure(figsize=(7, 3))
    plt.hist(resid, bins=40)
    mu, sig = float(resid.mean()), float(resid.std())
    plt.title(f"Résidus (prédit − observé) — μ={mu:.2f}, σ={sig:.2f}")
    plt.xlabel("Écart (vélos)"); plt.ylabel("Occurrences")
    _safe_savefig(out)

def plot_calibration_curve(tab: pd.DataFrame, out: Path):
    if tab.empty:
        return
    plt.figure(figsize=(6, 6))
    plt.plot(tab["y_pred_mean"], tab["y_true_mean"])
    # diagonale parfaite
    mn = float(min(tab["y_pred_mean"].min(), tab["y_true_mean"].min()))
    mx = float(max(tab["y_pred_mean"].max(), tab["y_true_mean"].max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.title("Calibration (moyenne par quantile de y_pred)")
    plt.xlabel("y_pred (moy. par quantile)")
    plt.ylabel("y_true (moy. par quantile)")
    _safe_savefig(out)

def plot_feature_importance(imp: pd.DataFrame, out: Path, top_k: int = 20):
    if imp.empty:
        return
    df = imp.head(top_k).copy()
    # barres sur score, annotations corr/mi
    plt.figure(figsize=(9, max(3, 0.35*len(df))))
    plt.barh(df["feature"], df["score"])
    plt.gca().invert_yaxis()
    plt.title("Importance des features (proxy)")
    plt.xlabel("Score (|corr(y)| + |corr(resid)| + MI)")
    _safe_savefig(out)

def plot_daily_error(trend: pd.DataFrame, out: Path):
    if trend.empty:
        return
    plt.figure(figsize=(9, 3))
    plt.plot(pd.to_datetime(trend["date"]), trend["MAE"])
    plt.title("Tendance d'erreur (MAE quotidienne)")
    plt.xlabel("Date"); plt.ylabel("MAE (vélos)")
    _safe_savefig(out)

def plot_data_health(health: pd.DataFrame, out: Path):
    if health.empty:
        return
    plt.figure(figsize=(8, 3))
    plt.bar(health["metric"], health["value"])
    plt.title("Data health — ratios")
    plt.ylabel("Part")
    plt.xticks(rotation=30, ha="right")
    _safe_savefig(out)

def plot_psi_features(psi: pd.DataFrame, out: Path, top_k: int = 20):
    if psi.empty:
        return
    df = psi.head(top_k)
    plt.figure(figsize=(9, max(3, 0.35*len(df))))
    plt.barh(df["feature"], df["psi"])
    plt.gca().invert_yaxis()
    plt.title("PSI par feature (début vs fin de période)")
    plt.xlabel("PSI")
    _safe_savefig(out)

# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

def run(perf_path: Path = PERF_PARQ) -> int:
    _ensure_dirs()
    perf = load_perf(perf_path)

    # 1) Métriques globales
    gmetrics = compute_global_metrics(perf)
    pd.DataFrame([gmetrics]).to_csv(TABLES / "global_metrics.csv", index=False)
    plot_global_metrics_bar(gmetrics, FIGS / "mon_global_metrics.png")

    # 2) Erreur par horizon
    err_h = compute_error_by_horizon(perf)
    if not err_h.empty:
        err_h.to_csv(TABLES / "error_by_horizon.csv", index=False)
        plot_error_by_horizon(err_h, FIGS / "mon_error_by_horizon.png")

    # 3) Résidus
    res_sum = compute_residuals_summary(perf)
    if not res_sum.empty:
        res_sum.to_csv(TABLES / "residuals_summary.csv", index=False)
    plot_residual_histogram(perf, FIGS / "mon_residual_hist.png")

    # 4) Calibration
    calib = compute_calibration_table(perf, q=10)
    if not calib.empty:
        calib.to_csv(TABLES / "calibration_table.csv", index=False)
        plot_calibration_curve(calib, FIGS / "mon_calibration.png")

    # 5) Tendance d’erreur quotidienne
    trend = compute_daily_error(perf)
    if not trend.empty:
        trend.to_csv(TABLES / "daily_error.csv", index=False)
        plot_daily_error(trend, FIGS / "mon_error_trend.png")

    # 6) Data health
    health = compute_data_health(perf)
    if not health.empty:
        health.to_csv(TABLES / "data_health.csv", index=False)
        plot_data_health(health, FIGS / "mon_data_health.png")

    # 7) PSI (début vs fin) — features numériques
    psi = compute_psi_features(perf)
    if not psi.empty:
        psi.to_csv(TABLES / "psi_features.csv", index=False)
        plot_psi_features(psi, FIGS / "mon_psi.png")

    # 8) Importance de features (corr + MI si dispo)
    imp = compute_feature_importance(perf)
    if not imp.empty:
        imp.to_csv(TABLES / "feature_importance_proxy.csv", index=False)
        plot_feature_importance(imp, FIGS / "mon_feature_importance.png")

    print("[OK] Monitoring model report generated in assets/figs and assets/tables")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build detailed monitoring report (model + data)")
    ap.add_argument("--perf", type=Path, default=PERF_PARQ)
    args = ap.parse_args()
    raise SystemExit(run(perf_path=args.perf))
