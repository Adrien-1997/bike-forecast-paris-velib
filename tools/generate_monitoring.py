# tools/generate_monitoring.py
# Orchestrateur du pipeline :
#   1) datasets.py           -> events.parquet / perf.parquet
#   2) build_usage.py        -> assets usage
#   3) apply_model.py        -> injecte y_pred (aligné T) dans perf.parquet
#   4) build_performance.py  -> assets perf (y_pred vs y_true)
#   5) build_monitoring.py   -> assets monitoring
#   6) build_station_profiles.py -> pages/figs par station (OPTIONNEL, n’échoue pas le pipeline)

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


def run_optional(cmd: list[str | Path]) -> None:
    """Exécute une étape non bloquante : log l’erreur mais ne casse pas le pipeline."""
    print("[RUN-OPTIONAL]", " ".join(map(str, cmd)))
    res = subprocess.run(list(map(str, cmd)), capture_output=True, text=True)
    if res.returncode != 0:
        print("[WARN] optional step failed (ignored):", " ".join(map(str, cmd)))
        if res.stdout:
            print("[STDOUT]\n", res.stdout)
        if res.stderr:
            print("[STDERR]\n", res.stderr)


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

    # 6) stations (OPTIONNELLE — ne doit pas casser le pipeline)
    run_optional([
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
