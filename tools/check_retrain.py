# tools/check_retrain.py
# Décideur de ré-entraînement : compat "ancienne" (metrics.json) + "nouvelle" (tables CSV)
from __future__ import annotations
import os, json, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXPORTS = DOCS / "exports"
TABLES = DOCS / "assets" / "tables"

METRICS_JSON  = EXPORTS / "metrics.json"   # ancien
BASELINE_JSON = EXPORTS / "baseline.json"  # ancien + nouveau (utilisé après retrain)

# Seuils (ENV override)
THRESH_PSI     = float(os.getenv("THRESH_PSI", "0.20"))  # PSI >= 0.20 => drift fort
THRESH_MAE_PCT = float(os.getenv("THRESH_MAE_PCT", "1.20"))  # +20% vs baseline

def _read_json(p: Path) -> dict:
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _read_csv(p: Path) -> pd.DataFrame | None:
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        return None
    return None

def _nanfloat(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None

def compute_from_old_style() -> tuple[float|None, float|None, float|None]:
    """
    Retourne (max_psi, mae_24h, mae_baseline) à partir de metrics.json/baseline.json si présents.
    """
    met = _read_json(METRICS_JSON)
    base = _read_json(BASELINE_JSON)

    # PSI
    psi = met.get("psi") or {}
    max_psi = None
    if isinstance(psi, dict) and psi:
        try:
            max_psi = max([float(v) for v in psi.values() if isinstance(v, (int, float))])
        except Exception:
            max_psi = None

    # MAE 24h
    metrics = met.get("metrics") or {}
    mae_24h = _nanfloat(metrics.get("mae_24h"))

    # Baseline
    mae_baseline = _nanfloat(base.get("mae_valid"))

    return max_psi, mae_24h, mae_baseline

def compute_from_new_style() -> tuple[float|None, float|None, float|None]:
    """
    Fallback : lit assets/tables (psi_features.csv, daily_error.csv) et perf.(parquet|csv)
    """
    # PSI
    psidf = _read_csv(TABLES / "psi_features.csv")
    max_psi = None
    if psidf is not None and not psidf.empty and "psi" in psidf.columns:
        try:
            max_psi = float(pd.to_numeric(psidf["psi"], errors="coerce").max())
            if not np.isfinite(max_psi): max_psi = None
        except Exception:
            max_psi = None

    # MAE 24h
    mae_24h = None
    derr = _read_csv(TABLES / "daily_error.csv")
    if derr is not None and not derr.empty and "err" in derr.columns:
        try:
            row = derr.sort_values("day").tail(1)
            mae_24h = float(row["err"].iloc[0])
        except Exception:
            mae_24h = None
    if mae_24h is None:
        # fallback ultime : calcul depuis perf.(parquet|csv)
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

    # Baseline (json ou colonne perf)
    mae_baseline = _nanfloat((_read_json(BASELINE_JSON) or {}).get("mae_valid"))
    if mae_baseline is None:
        try:
            if perf is None:
                if (EXPORTS / "perf.parquet").exists():
                    perf = pd.read_parquet(EXPORTS / "perf.parquet")
                elif (EXPORTS / "perf.csv").exists():
                    perf = pd.read_csv(EXPORTS / "perf.csv")
            if perf is not None and "y_pred_baseline" in perf.columns:
                perf["ts"] = pd.to_datetime(perf["ts"], utc=False, errors="coerce")
                recent = perf[perf["ts"] >= (perf["ts"].max() - pd.Timedelta(hours=24))]
                e_base = (pd.to_numeric(recent["y_pred_baseline"], errors="coerce") -
                          pd.to_numeric(recent["y_true"], errors="coerce")).abs()
                mae_baseline = float(e_base.mean())
        except Exception:
            mae_baseline = None

    return max_psi, mae_24h, mae_baseline

def main():
    # 1) essaie l'ancien format
    max_psi, mae_24h, mae_baseline = compute_from_old_style()

    # 2) fallback nouveau format si manquants
    if max_psi is None or mae_24h is None or mae_baseline is None:
        nmax_psi, nmae_24h, nmae_baseline = compute_from_new_style()
        max_psi      = nmax_psi      if max_psi is None else max_psi
        mae_24h      = nmae_24h      if mae_24h is None else mae_24h
        mae_baseline = nmae_baseline if mae_baseline is None else mae_baseline

    # 3) décisions
    reasons = []
    need_drift = (max_psi is not None and max_psi >= THRESH_PSI)
    if need_drift: reasons.append("psi>=threshold")

    need_perf = False
    if (mae_24h is not None and mae_baseline is not None and mae_baseline > 0):
        ratio = mae_24h / mae_baseline
        if ratio >= THRESH_MAE_PCT and not np.isclose(ratio, 1.0):
            need_perf = True
            reasons.append("mae_ratio>=threshold")

    need_retrain = bool(need_drift or need_perf)

    def jnum(x):
        try:
            v = float(x)
            return v if np.isfinite(v) else None
        except Exception:
            return None

    out = {
        "need_retrain": need_retrain,
        "reason": {"psi_over": need_drift, "perf_over": need_perf},
        "reasons": reasons,
        "max_psi": jnum(max_psi),
        "mae_24h": jnum(mae_24h),
        "mae_baseline": jnum(mae_baseline),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    print(json.dumps(out, ensure_ascii=False, allow_nan=False, indent=2))

    # GitHub Outputs (si on veut récupérer sans jq)
    gho = os.getenv("GITHUB_OUTPUT")
    if gho:
        with open(gho, "a", encoding="utf-8") as f:
            f.write(f"need_retrain={'true' if need_retrain else 'false'}\n")
            f.write(f"max_psi={jnum(max_psi)}\n")
            f.write(f"mae_24h={jnum(mae_24h)}\n")
            f.write(f"mae_baseline={jnum(mae_baseline)}\n")

if __name__ == "__main__":
    sys.exit(main())
