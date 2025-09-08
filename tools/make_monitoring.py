import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd, numpy as np, pathlib
from src.monitoring import drift_report
from src.eval import backtest_24h

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXP  = ROOT / "exports"
DOCS = ROOT / "docs"

def _load(name):
    p_parq = EXP / f"{name}.parquet"
    p_csv  = EXP / f"{name}.csv"
    if p_parq.exists():
        return pd.read_parquet(p_parq)
    return pd.read_csv(p_csv)

def main():
    DOCS.mkdir(exist_ok=True)
    hourly = _load("velib_hourly")
    hourly["hour_utc"] = pd.to_datetime(hourly["hour_utc"], errors="coerce")

    # --- DRIFT (7j vs 30j) ---
    dr = drift_report(hourly)

    # --- BACKTEST 24h ---
    metrics, pairs = backtest_24h(hourly)

    md = []
    md += ["# Monitoring"]
    md += ["", "## Drift (7 jours vs. 30 jours précédents)"]
    if not dr.empty:
        try:
            md += [dr.round({"psi":3,"base_mean":3,"curr_mean":3,"base_std":3,"curr_std":3}).to_markdown(index=False)]
        except Exception:
            md += ["<div>" + dr.to_html(index=False, border=0) + "</div>"]
        md += ["", "_PSI_: <0.10 OK • 0.10–0.25 Attention • >0.25 Alerte_", ""]
    else:
        md += ["(Pas assez de données pour calculer le drift.)", ""]

    md += ["## Performance (Backtest 24h)"]
    if metrics:
        md += [
            f"- **MAE**: {metrics['mae']:.3f} • **RMSE**: {metrics['rmse']:.3f}",
            f"- **Rupture vélos** — Precision: {metrics['bike_precision']:.2f} • Recall: {metrics['bike_recall']:.2f} • F1: {metrics['bike_f1']:.2f}",
            f"- **Rupture bornes** — Precision: {metrics['dock_precision']:.2f} • Recall: {metrics['dock_recall']:.2f} • F1: {metrics['dock_f1']:.2f}",
            f"- Paires horodatées comparées: {metrics['n_pairs']} • Stations: {metrics['n_stations']}",
            ""
        ]
    else:
        md += ["(Pas assez de données pour le backtest.)", ""]

    (DOCS / "monitoring.md").write_text("\n".join(md), encoding="utf-8")
    print("OK — docs/monitoring.md mis à jour.")

if __name__ == "__main__":
    main()
