# tools/generate_monitoring.py
# Wrapper CI : génère tous les assets & pages du site (usage, perf, monitoring, stations)
# - Idempotent, robuste, verbosité claire pour GitHub Actions.
from __future__ import annotations

import argparse
import sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXPORTS = DOCS / "exports"
SCRIPTS = ROOT / "tools"

def run(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)

def ensure_dirs() -> None:
    (DOCS / "assets" / "figs").mkdir(parents=True, exist_ok=True)
    (DOCS / "assets" / "tables").mkdir(parents=True, exist_ok=True)
    (DOCS / "assets" / "maps").mkdir(parents=True, exist_ok=True)
    (DOCS / "exports" / "auto").mkdir(parents=True, exist_ok=True)
    (DOCS / "stations").mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Build all monitoring assets & pages.")
    ap.add_argument("--input", type=Path, default=EXPORTS / "velib.parquet", help="Données source (velib.parquet|csv)")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--usage-days", type=int, default=7)
    ap.add_argument("--perf-days", type=int, default=7)
    ap.add_argument("--mon-days", type=int, default=14)
    ap.add_argument("--station", type=str, default=None)
    ap.add_argument("--profiles", type=int, default=12, help="Nombre de fiches stations")
    ap.add_argument("--profiles-hours", type=int, default=48)
    ap.add_argument("--profiles-by", type=str, default="volatility", choices=["volatility","error","tension","random"])
    args = ap.parse_args()

    ensure_dirs()

    # 1) Normalisation datasets (events/perf)
    run([sys.executable, str(SCRIPTS / "datasets.py"),
         "--input", str(args.input),
         "--horizon", str(args.horizon)])

    # 2) Usage
    run([sys.executable, str(SCRIPTS / "build_usage.py"),
         "--events", str(EXPORTS / "events.parquet"),
         "--last-days", str(args.usage_days)])

    # 3) Performance
    cmd_perf = [sys.executable, str(SCRIPTS / "build_performance.py"),
                "--perf", str(EXPORTS / "perf.parquet"),
                "--horizon", str(args.horizon),
                "--last-days", str(args.perf_days)]
    if args.station:
        cmd_perf += ["--station", args.station]
    run(cmd_perf)

    # 4) Monitoring
    run([sys.executable, str(SCRIPTS / "build_monitoring.py"),
         "--events", str(EXPORTS / "events.parquet"),
         "--perf", str(EXPORTS / "perf.parquet"),
         "--horizon", str(args.horizon),
         "--last-days", str(args.mon_days)])

    # 5) Insights auto (usage/perf/monitoring)
    run([sys.executable, str(SCRIPTS / "orchestrate_reports.py")])

    # 6) Profils de stations
    run([sys.executable, str(SCRIPTS / "build_station_profiles.py"),
         "--events", str(EXPORTS / "events.parquet"),
         "--perf", str(EXPORTS / "perf.parquet"),
         "--last-days", str(args.usage_days),
         "--hours", str(args.profiles_hours),
         "--select", str(args.profiles),
         "--by", args.profiles_by])

    print("[OK] generate_monitoring: all assets & pages built.", flush=True)

if __name__ == "__main__":
    main()
