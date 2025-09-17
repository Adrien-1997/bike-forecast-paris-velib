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
#   - Imprime les métriques + une recommandation binaire : RETRAIN = yes/no
#   - Termine par une ligne JSON compacte (utile pour `... | tee check.json`)

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXPORTS = DOCS / "exports"
ASSETS = DOCS / "assets"
TBLS = ASSETS / "tables"


def _typed_empty_psi() -> pd.DataFrame:
    return pd.DataFrame({"feature": pd.Series(dtype="object"),
                         "psi": pd.Series(dtype="float64")})


def _read_daily_error() -> pd.DataFrame | None:
    p = TBLS / "daily_error.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
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
            if "psi" not in df.columns:
                return _typed_empty_psi()
            return df[["feature", "psi"]]
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

    def _mae(g: pd.DataFrame) -> float:
        m = g["y_true"].notna() & g["y_pred"].notna()
        return float(np.mean(np.abs(g.loc[m, "y_pred"] - g.loc[m, "y_true"]))) if m.any() else np.nan

    # Use include_groups=False when available to avoid FutureWarning
    try:
        daily = (
            perf.groupby("date", group_keys=False)
            .apply(_mae, include_groups=False)  # pandas >= 2.2
            .rename("MAE")
            .reset_index()
        )
    except TypeError:
        # Older pandas
        daily = (
            perf.groupby("date", group_keys=False)
            .apply(_mae)
            .rename("MAE")
            .reset_index()
        )

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
        return _typed_empty_psi()

    cur_start = tmax - pd.Timedelta(days=last_days)
    ref_start = tmax - pd.Timedelta(days=2 * last_days)
    ref = perf[(perf["ts"] >= ref_start) & (perf["ts"] < cur_start)].copy()
    cur = perf[perf["ts"] >= cur_start].copy()

    cand = [
        c for c in perf.columns
        if c.startswith(("lag_", "roll_", "trend_", "temp_", "precip_", "wind_", "occ_", "capacity", "nb_velos", "nb_bornes"))
    ]
    if not cand:
        return _typed_empty_psi()

    rows = []
    for c in cand[:40]:  # limite raisonnable
        try:
            psi_val = _psi_simple(ref.get(c, pd.Series(dtype="float64")),
                                  cur.get(c, pd.Series(dtype="float64")))
        except Exception:
            psi_val = np.nan
        rows.append({"feature": c, "psi": psi_val})

    df = pd.DataFrame(rows, columns=["feature", "psi"])
    if "psi" not in df.columns:
        return _typed_empty_psi()

    df = df.dropna(subset=["psi"])
    if df.empty:
        return _typed_empty_psi()

    return df.sort_values("psi", ascending=False)


def main(perf_path: Path, last_days: int, mae_lift_th: float, psi_th: float):
    # 1) MAE récent vs ref
    daily = _read_daily_error()
    if daily is None:
        daily = _compute_daily_error_from_perf(perf_path, last_days=last_days)

    if daily.empty:
        print("[check_retrain] daily_error vide — décision impossible → RETRAIN = no")
        # JSON minimal pour CI
        out = {
            "retrain": False,
            "mae_ref": None,
            "mae_cur": None,
            "mae_lift": None,
            "max_psi": None,
            "last_days": last_days,
        }
        print(json.dumps(out, ensure_ascii=False))
        return

    tmax = daily["date"].max()
    cur = daily[daily["date"] >= (tmax - pd.Timedelta(days=last_days))]["MAE"]
    ref = daily[(daily["date"] >= (tmax - pd.Timedelta(days=2 * last_days))) &
                (daily["date"] < (tmax - pd.Timedelta(days=last_days)))]["MAE"]

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
    max_psi = float(psi_tbl["psi"].max()) if (psi_tbl is not None and not psi_tbl.empty) else np.nan

    # 3) décision
    trigger_mae = (not np.isnan(mae_lift)) and (mae_lift >= mae_lift_th)
    trigger_psi = (not np.isnan(max_psi)) and (max_psi >= psi_th)
    retrain = bool(trigger_mae or trigger_psi)

    print("[check_retrain] --- Diagnostics ---")
    print(f" Window last_days      : {last_days} d")
    print(f" MAE ref / cur         : {mae_ref:.3f} / {mae_cur:.3f}")
    print(f" MAE relative lift     : {mae_lift:.2%}  (threshold {mae_lift_th:.0%})")
    print(f" Max PSI               : {max_psi:.3f}   (threshold {psi_th:.3f})")
    print(f" Triggers (MAE/PSI)    : {trigger_mae} / {trigger_psi}")
    print(f" RETRAIN               : {'yes' if retrain else 'no'}")

    # JSON compact pour CI (tee → check.json)
    out = {
        "retrain": retrain,
        "mae_ref": None if np.isnan(mae_ref) else round(mae_ref, 6),
        "mae_cur": None if np.isnan(mae_cur) else round(mae_cur, 6),
        "mae_lift": None if np.isnan(mae_lift) else round(float(mae_lift), 6),
        "max_psi": None if np.isnan(max_psi) else round(max_psi, 6),
        "last_days": last_days,
        "thresholds": {"mae_lift": mae_lift_th, "psi": psi_th},
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Decide whether to retrain the model based on drift & error.")
    ap.add_argument("--perf", type=Path, default=EXPORTS / "perf.parquet")
    ap.add_argument("--last-days", type=int, default=14, help="Window length for recent vs reference comparison")
    ap.add_argument("--mae-lift-th", type=float, default=0.15, help="Relative MAE increase threshold (e.g., 0.15 = +15%)")
    ap.add_argument("--psi-th", type=float, default=0.2, help="PSI threshold (0.1: slight, 0.2: medium, 0.3+: large)")
    args = ap.parse_args()
    main(perf_path=args.perf, last_days=args.last_days, mae_lift_th=args.mae_lift_th, psi_th=args.psi_th)
