# tools/check_retrain.py
# Décideur de ré-entraînement basé sur PSI & performance récente.
# Sortie JSON NaN-safe pour GitHub Actions (jq).
from __future__ import annotations

import os, json, sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
TABLES = DOCS / "assets" / "tables"
EXPORTS = DOCS / "exports"

def read_csv(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None
    return None

def nanfloat(x):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return None

def main():
    # Seuils via env
    th_psi = nanfloat(os.getenv("THRESH_PSI", "0.20")) or 0.20
    th_mae_pct = nanfloat(os.getenv("THRESH_MAE_PCT", "1.20")) or 1.20

    # 1) PSI
    psi_df = read_csv(TABLES / "psi_features.csv")
    max_psi = None
    if psi_df is not None and not psi_df.empty and "psi" in psi_df.columns:
        try:
            max_psi = float(pd.to_numeric(psi_df["psi"], errors="coerce").max())
        except Exception:
            max_psi = None

    # 2) MAE dernier jour
    mae_24h = None
    daily_err = read_csv(TABLES / "daily_error.csv")
    if daily_err is not None and not daily_err.empty and "err" in daily_err.columns:
        try:
            row = daily_err.sort_values("day").tail(1)
            mae_24h = float(row["err"].iloc[0])
        except Exception:
            mae_24h = None
    else:
        # fallback: calcul depuis perf.parquet sur 24h
        perf = None
        try:
            pparq = EXPORTS / "perf.parquet"
            pcsv  = EXPORTS / "perf.csv"
            if pparq.exists():
                perf = pd.read_parquet(pparq)
            elif pcsv.exists():
                perf = pd.read_csv(pcsv)
            if perf is not None and not perf.empty:
                perf["ts"] = pd.to_datetime(perf["ts"], utc=False, errors="coerce")
                recent = perf[perf["ts"] >= (perf["ts"].max() - pd.Timedelta(hours=24))]
                e = (pd.to_numeric(recent["y_pred"], errors="coerce") -
                     pd.to_numeric(recent["y_true"], errors="coerce")).abs()
                mae_24h = float(e.mean())
        except Exception:
            mae_24h = None

    # 3) Baseline
    mae_baseline = None
    base_file = EXPORTS / "baseline.json"
    if base_file.exists():
        try:
            with open(base_file, "r", encoding="utf-8") as f:
                j = json.load(f)
            mae_baseline = nanfloat(j.get("mae_valid"))
        except Exception:
            mae_baseline = None

    # sinon, baseline via colonne perf
    if mae_baseline is None:
        try:
            if 'perf' not in locals() or perf is None:
                if (EXPORTS / "perf.parquet").exists():
                    perf = pd.read_parquet(EXPORTS / "perf.parquet")
                elif (EXPORTS / "perf.csv").exists():
                    perf = pd.read_csv(EXPORTS / "perf.csv")
                else:
                    perf = None
            if perf is not None and "y_pred_baseline" in perf.columns:
                perf["ts"] = pd.to_datetime(perf["ts"], utc=False, errors="coerce")
                recent = perf[perf["ts"] >= (perf["ts"].max() - pd.Timedelta(hours=24))]
                e_base = (pd.to_numeric(recent["y_pred_baseline"], errors="coerce") -
                          pd.to_numeric(recent["y_true"], errors="coerce")).abs()
                mae_baseline = float(e_base.mean())
        except Exception:
            mae_baseline = None

    # fallback ultime : égalité -> pas d’alerte perf
    if mae_baseline is None and mae_24h is not None:
        mae_baseline = float(mae_24h)

    # 4) Décision
    reasons = []
    need_retrain = False

    if max_psi is not None and max_psi >= th_psi:
        reasons.append("psi>=threshold")
        need_retrain = True

    if mae_24h is not None and mae_baseline is not None and mae_baseline > 0:
        ratio = mae_24h / mae_baseline
        if ratio >= th_mae_pct and not np.isclose(ratio, 1.0):
            reasons.append("mae_ratio>=threshold")
            need_retrain = True

    def to_json_num(x):
        try:
            v = float(x)
        except Exception:
            return None
        return v if np.isfinite(v) else None

    out = {
        "need_retrain": bool(need_retrain),
        "max_psi": to_json_num(max_psi),
        "mae_24h": to_json_num(mae_24h),
        "mae_baseline": to_json_num(mae_baseline),
        "reasons": reasons,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    print(json.dumps(out, ensure_ascii=False, allow_nan=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())
