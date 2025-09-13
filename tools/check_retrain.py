# tools/check_retrain.py
from __future__ import annotations
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "docs" / "exports"
METRICS = EXPORTS / "metrics.json"
BASELINE = EXPORTS / "baseline.json"

# Seuils (surchargables par variables d'env dans Actions)
THRESH_PSI = float(os.getenv("THRESH_PSI", "0.20"))
THRESH_MAE_PCT = float(os.getenv("THRESH_MAE_PCT", "1.20"))  # 1.20 = +20%

def _load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    met = _load_json(METRICS)
    psi = met.get("psi", {}) or {}
    metrics = (met.get("metrics") or {})
    mae_24h = float(metrics.get("mae_24h", "nan"))

    base = _load_json(BASELINE)
    mae_base = base.get("mae_valid")  # stocké après train
    need_drift = False
    need_perf = False

    # Règle 1: PSI
    if psi:
        max_psi = max([v for v in psi.values() if isinstance(v, (int,float))], default=0.0)
        need_drift = (max_psi >= THRESH_PSI)
    else:
        max_psi = float("nan")

    # Règle 2: Perf vs baseline (si dispo)
    if isinstance(mae_base, (int,float)) and isinstance(mae_24h, (int,float)):
        need_perf = (mae_24h >= mae_base * THRESH_MAE_PCT)

    need_retrain = bool(need_drift or need_perf)

    # Sortie JSON + GitHub Outputs
    out = {
        "need_retrain": need_retrain,
        "reason": {
            "psi_over": need_drift,
            "perf_over": need_perf
        },
        "max_psi": max_psi,
        "mae_24h": mae_24h,
        "mae_baseline": mae_base
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    gho = os.getenv("GITHUB_OUTPUT")
    if gho:
        with open(gho, "a", encoding="utf-8") as f:
            f.write(f"need_retrain={'true' if need_retrain else 'false'}\n")
            f.write(f"max_psi={max_psi}\n")
            f.write(f"mae_24h={mae_24h}\n")
            f.write(f"mae_baseline={mae_base}\n")

if __name__ == "__main__":
    main()
