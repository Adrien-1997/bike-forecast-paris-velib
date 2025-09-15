# tools/generate_monitoring.py
# Orchestrateur du pipeline :
#   1) datasets.py           -> events.parquet / perf.parquet
#   2) build_usage.py        -> assets usage
#   3) apply_model.py        -> injecte y_pred (aligné T) dans perf.parquet
#   4) build_performance.py  -> assets perf (y_pred vs y_true)
#   5) build_monitoring.py   -> assets monitoring
#   6) build_station_profiles.py -> pages/figs par station
#
# Exemple :
#   python tools/generate_monitoring.py --input docs/exports/velib.parquet \
#     --horizon 60 --lookback-days 14 --last-days 7 --tz Europe/Paris --clusters 6 \
#     --hours 48 --select 12 --by volatility
#
# Notes :
# - Les données restent en UTC ; --tz n’affecte que l’affichage des figures.
# - L’offset T+h → T est corrigé dans apply_model.py (ne rien changer ici).
# - Ajoute des logs lisibles et stoppe au premier échec (subprocess.check_call).

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
DOCS = ROOT / "docs"
EXPORTS = DOCS / "exports"

EVENTS = EXPORTS / "events.parquet"
PERF = EXPORTS / "perf.parquet"


def run(cmd: list[str | Path]) -> None:
    print("[RUN]", " ".join(map(str, cmd)))
    subprocess.check_call(list(map(str, cmd)))


def main():
    ap = argparse.ArgumentParser(description="Orchestrate data → usage/perf/monitoring/stations")
    # Entrées / sorties
    ap.add_argument("--input", type=Path, default=EXPORTS / "velib.parquet",
                    help="Source normalisée (parquet/csv) à convertir en events/perf")
    ap.add_argument("--out-events", type=Path, default=EVENTS)
    ap.add_argument("--out-perf", type=Path, default=PERF)

    # Paramètres communs
    ap.add_argument("--horizon", type=int, default=60, help="Horizon minutes (doit matcher le modèle)")
    ap.add_argument("--lookback-days", type=int, default=14, help="Fenêtre de reconstruction des features (apply_model)")
    ap.add_argument("--last-days", type=int, default=7, help="Fenêtre d’analyse pour usage/perf/monitoring")
    ap.add_argument("--tz", type=str, default=None, help="Affichage (ex: Europe/Paris). Données restent en UTC.")

    # Usage
    ap.add_argument("--clusters", type=int, default=6, help="Nb de clusters KMeans (si sklearn dispo)")

    # Stations
    ap.add_argument("--hours", type=int, default=48, help="Fenêtre d’affichage pour figures par station")
    ap.add_argument("--select", type=int, default=12, help="Nombre de stations à profiler")
    ap.add_argument("--by", type=str, default="volatility", choices=["volatility", "coverage", "count"],
                    help="Critère de sélection des stations")

    # Options utilitaires
    ap.add_argument("--lag-steps", type=int, default=0,
                    help="Décalage éventuel à appliquer à une y_pred d’entrée dans datasets (rare)")
    ap.add_argument("--as-csv", action="store_true", help="Écrire events/perf en CSV au lieu de Parquet (debug)")

    args = ap.parse_args()

    # 1) datasets
    run([
        sys.executable, TOOLS / "datasets.py",
        "--input", args.input,
        "--horizon", str(args.horizon),
        "--out-events", args.out_events,
        "--out-perf", args.out_perf,
        "--lag-steps", str(args.lag_steps),
        *(["--as-csv"] if args.as_csv else [])
    ])

    # 2) usage
    run([
        sys.executable, TOOLS / "build_usage.py",
        "--events", args.out_events,
        "--last-days", str(args.last_days),
        *(["--tz", args.tz] if args.tz else []),
        "--clusters", str(args.clusters),
    ])

    # 3) inference → inject y_pred
    run([
        sys.executable, TOOLS / "apply_model.py",
        "--horizon", str(args.horizon),
        "--lookback-days", str(args.lookback_days),
    ])

    # 4) performance
    run([
        sys.executable, TOOLS / "build_performance.py",
        "--perf", args.out_perf,
        "--last-days", str(args.last_days),
        *(["--tz", args.tz] if args.tz else []),
        "--horizon", str(args.horizon),
    ])

    # 5) monitoring
    run([
        sys.executable, TOOLS / "build_monitoring.py",
        "--events", args.out_events,
        "--perf", args.out_perf,
        "--last-days", str(args.last_days),
        *(["--tz", args.tz] if args.tz else []),
    ])

    # 6) stations
    run([
        sys.executable, TOOLS / "build_station_profiles.py",
        "--events", args.out_events,
        "--perf", args.out_perf,
        "--last-days", str(args.last_days),
        "--hours", str(args.hours),
        "--select", str(args.select),
        "--by", args.by,
        *(["--tz", args.tz] if args.tz else []),
    ])

    print("[DONE] Full monitoring pipeline completed.")


if __name__ == "__main__":
    main()
