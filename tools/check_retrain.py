# tools/check_retrain.py
# Décide s'il faut (re)entraîner le modèle en se basant sur :
#  - la dérive d'erreur (MAE récent vs MAE de référence),
#  - la dérive de population (PSI de quelques features si disponibles).
#
# Il exploite en priorité les tables déjà produites par le pipeline :
#   docs/assets/tables/daily_error.csv  (build_monitoring.py)
#   docs/assets/tables/psi_features.csv (build_monitoring.py)
#
# S'il ne les trouve pas, il retombe sur un calcul direct depuis perf.parquet.
#
# Usage:
#   python tools/check_retrain.py --perf docs/exports/perf.parquet --last-days 14 \
#       --mae-lift-th 0.15 --psi-th 0.2
#
# Sortie:
#   - Imprime les métriques et une recommandation binaire : RETRAIN = yes/no

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXPORTS = DOCS / "exports"
ASSETS = DOCS / "assets"
TBLS = ASSETS / "tables"

def _read_daily_error() -> pd.DataFrame | None:
    p = TBLS / "daily_error.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
            # compat: colonnes "date","MAE","RMSE"
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date")
        except Exception:
            return None
    return None

def _read_psi() -> pd.DataFrame | None:
    p = TBLS / "psi_features.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
            # compat: colonnes "feature","psi"
            return df
        except Exception:
            return None
    return None

def _compute_daily_error_from_perf(perf_path: Path, last_days: int) -> pd.DataFrame:
    perf = pd.read_parquet(perf_path)
    perf["ts"] = pd.to_datetime(perf["ts"], utc=False, errors="coerce")
    # fallback baseline si y_pred absent
    if ("y_pred" not in perf.columns) or perf["y_pred"].dropna().empty:
        if "y_pred_baseline" in perf.columns:
            perf["y_pred"] = perf["y_pred_baseline"]
        else:
            perf["y_pred"] = np.nan
    # fenêtre 2*last_days
    tmax = perf["ts"].max()
    if pd.notna(tmax) and last_days and last_days > 0:
        perf = perf[perf["ts"] >= (tmax - pd.Timedelta(days=2 * last_days))].copy()
    # daily MAE
    perf["date"] = perf["ts"].dt.date
    def _mae(g):
        m = g["y_true"].notna() & g["y_pred"].notna()
        return float(np.mean(np.abs(g.loc[m, "y_pred"] - g.loc[m, "y_true"]))) if m.any() else np.nan
    daily = (perf.groupby("date").apply(_mae)
                  .rename("MAE").reset_index())
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date")

def _psi_simple(ref: pd.Series, cur: pd.Series, bins: int = 20) -> float:
    ref = pd.to_numeric(ref, errors="coerce").dropna().astype(float)
    cur = pd.to_numeric(cur, errors="coerce").dropna().astype(float)
    if len(ref) < 10 or len(cur) < 10:
        return np.nan
    qs = np.quantile(ref, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    ref_hist, _ = np.histogram(ref, bins=qs)
    cur_hist, _ = np.histogram(cur, bins=qs)
    ref_p = np.where(ref_hist == 0, 1e-6, ref_hist) / max(ref_hist.sum(), 1)
    cur_p = np.where(cur_hist == 0, 1e-6, cur_hist) / max(cur_hist.sum(), 1)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))

def _compute_psi_from_perf(perf_path: Path, last_days: int) -> pd.DataFrame:
    perf = pd.read_parquet(perf_path)
    perf["ts"] = pd.to_datetime(perf["ts"], utc=False, errors="coerce")
    tmax = perf["ts"].max()
    if pd.isna(tmax):
        return pd.DataFrame(columns=["feature","psi"])
    cur_start = tmax - pd.Timedelta(days=last_days)
    ref_start = tmax - pd.Timedelta(days=2 * last_days)
    ref = perf[(perf["ts"] >= ref_start) & (perf["ts"] < cur_start)].copy()
    cur = perf[perf["ts"] >= cur_start].copy()

    cand = [c for c in perf.columns if c.startswith(("lag_", "roll_", "trend_", "temp_", "precip_", "wind_", "occ_", "capacity", "nb_velos", "nb_bornes"))]
    rows = []
    for c in cand[:40]:  # limite raisonnable
        try:
            psi = _psi_simple(ref[c], cur[c])
        except Exception:
            psi = np.nan
        rows.append({"feature": c, "psi": psi})
    df = pd.DataFrame(rows).dropna().sort_values("psi", ascending=False)
    return df

def main(perf_path: Path, last_days: int, mae_lift_th: float, psi_th: float):
    # 1) MAE récent vs ref
    daily = _read_daily_error()
    if daily is None:
        daily = _compute_daily_error_from_perf(perf_path, last_days=last_days)

    if daily.empty:
        print("[check_retrain] daily_error vide — décision impossible → RETRAIN = no")
        return

    tmax = daily["date"].max()
    cur = daily[daily["date"] >= (tmax - pd.Timedelta(days=last_days))]["MAE"]
    ref = daily[(daily["date"] >= (tmax - pd.Timedelta(days=2 * last_days))) & (daily["date"] < (tmax - pd.Timedelta(days=last_days)))]["MAE"]

    mae_cur = float(cur.mean()) if len(cur) else np.nan
    mae_ref = float(ref.mean()) if len(ref) else np.nan

    if np.isnan(mae_cur) or np.isnan(mae_ref) or mae_ref == 0:
        mae_lift = np.nan
    else:
        mae_lift = (mae_cur - mae_ref) / mae_ref  # >0 si pire

    # 2) PSI
    psi_tbl = _read_psi()
    if psi_tbl is None or psi_tbl.empty:
        psi_tbl = _compute_psi_from_perf(perf_path, last_days=last_days)
    max_psi = float(psi_tbl["psi"].max()) if not psi_tbl.empty else np.nan

    # 3) décision
    trigger_mae = (not np.isnan(mae_lift)) and (mae_lift >= mae_lift_th)
    trigger_psi = (not np.isnan(max_psi)) and (max_psi >= psi_th)
    retrain = trigger_mae or trigger_psi

    print("[check_retrain] --- Diagnostics ---")
    print(f" Window last_days      : {last_days} d")
    print(f" MAE ref / cur         : {mae_ref:.3f} / {mae_cur:.3f}")
    print(f" MAE relative lift     : {mae_lift:.2%}  (threshold {mae_lift_th:.0%})")
    print(f" Max PSI               : {max_psi:.3f}   (threshold {psi_th:.3f})")
    print(f" Triggers (MAE/PSI)    : {trigger_mae} / {trigger_psi}")
    print(f" RETRAIN               : {'yes' if retrain else 'no'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Decide whether to retrain the model based on drift & error.")
    ap.add_argument("--perf", type=Path, default=EXPORTS / "perf.parquet")
    ap.add_argument("--last-days", type=int, default=14, help="Window length for recent vs reference comparison")
    ap.add_argument("--mae-lift-th", type=float, default=0.15, help="Relative MAE increase threshold (e.g., 0.15 = +15%)")
    ap.add_argument("--psi-th", type=float, default=0.2, help="PSI threshold (0.1: slight, 0.2: medium, 0.3+: large)")
    args = ap.parse_args()
    main(perf_path=args.perf, last_days=args.last_days, mae_lift_th=args.mae_lift_th, psi_th=args.psi_th)
