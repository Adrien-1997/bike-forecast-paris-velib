# tools/build_monitoring.py
# Orchestrator for building the documentation site with a "1 Python file = 1 page" layout.
# It prepares the canonical exports (events/perf), optionally injects model predictions,
# then calls each page-builder script in a clear, deterministic order.
#
# Pages (files you will create next):
#   Network
#     - tools/build_network_overview.py
#     - tools/build_network_stations.py
#     - tools/build_network_dynamics.py
#   Model
#     - tools/build_model_performance.py        (or reuse existing build_performance.py)
#     - tools/build_model_pipeline.py
#     - tools/build_model_explainability.py
#   Monitoring
#     - tools/build_monitoring_data_health.py
#     - tools/build_monitoring_drift.py
#     - tools/build_monitoring_model_health.py
#   Data
#     - tools/build_data_exports.py
#     - tools/build_data_dictionary.py
#     - tools/build_data_methodology.py
#
# Existing helpers you already have:
#   - tools/datasets.py          → produces docs/exports/events.parquet and docs/exports/perf.parquet
#   - tools/apply_model.py       → injects y_pred aligned at T into perf.parquet (optional if no model)
#
# Usage examples:
#   python tools/build_monitoring.py
#   python tools/build_monitoring.py --pages network.overview,network.stations,model.performance
#   python tools/build_monitoring.py --tz Europe/Paris --last-days 7 --current-days 7 --reference-days 28
#   python tools/build_monitoring.py --skip-apply-model   # if you don't want to inject y_pred
#
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
DOCS = ROOT / "docs"
EXPORTS = DOCS / "exports"
EVENTS = EXPORTS / "events.parquet"
PERF = EXPORTS / "perf.parquet"


def _as_str_list(x) -> List[str]:
    return list(map(str, x))


def run(cmd: List[object]) -> None:
    """Run a mandatory step; raise on non-zero exit."""
    cmd = _as_str_list(cmd)
    print("[RUN]", " ".join(cmd))
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",     # ← au lieu de cp1252
        errors="replace"
    )
    if res.stdout:
        print(res.stdout.rstrip())
    if res.returncode != 0:
        if res.stderr:
            print(res.stderr.rstrip())
        raise SystemExit(f"[FATAL] step failed: {' '.join(cmd)} (code={res.returncode})")


def run_optional(cmd: List[object]) -> None:
    """Run a non-blocking step; log failure but continue."""
    cmd = _as_str_list(cmd)
    print("[RUN-OPTIONAL]", " ".join(cmd))
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",     # ← idem ici
        errors="replace"
    )
    if res.stdout:
        print(res.stdout.rstrip())
    if res.returncode != 0:
        print(f"[WARN] optional step failed (ignored): {' '.join(cmd)} (code={res.returncode})")
        if res.stderr:
            print(res.stderr.rstrip())


def _assert_model_written(perf_path: Path) -> None:
    """Hard check that model predictions were written into y_pred (not baseline)."""
    if not Path(perf_path).exists():
        raise SystemExit(f"[FATAL] perf file missing: {perf_path}")
    perf = pd.read_parquet(perf_path)

    if "y_pred" not in perf.columns:
        raise SystemExit("[FATAL] `y_pred` column missing after apply_model.")

    cov = float(perf["y_pred"].notna().mean() * 100)
    print(f"[CHECK] model coverage in y_pred: {cov:.2f}%")
    if cov == 0:
        raise SystemExit("[FATAL] 0% model predictions written. Check alignment/horizon/features.")

    if "y_pred_baseline" in perf.columns:
        both = perf[["y_pred", "y_pred_baseline"]].dropna()
        if len(both) > 0:
            eq_pct = float(
                np.isclose(
                    both["y_pred"].to_numpy(),
                    both["y_pred_baseline"].to_numpy(),
                    atol=1e-6,
                ).mean()
                * 100.0
            )
            print(f"[CHECK] y_pred equals baseline (isclose): {eq_pct:.2f}% "
                  f"(should be << 100% if model differs).")


def main():
    ap = argparse.ArgumentParser(description="Build all site pages (1 Python file per page).")
    # Inputs / Outputs
    ap.add_argument("--input", type=Path, default=EXPORTS / "velib.parquet",
                    help="Chemin local vers velib.parquet (source agrégée). "
                         "Si absent, datasets.py lira via fallback local→Hugging Face (velib.parquet).")
    ap.add_argument("--out-events", type=Path, default=EVENTS)
    ap.add_argument("--out-perf", type=Path, default=PERF)

    # Global params
    ap.add_argument("--horizon", type=int, default=60, help="Forecasting horizon in minutes.")
    ap.add_argument("--lookback-days", type=int, default=14, help="History window for feature building (apply_model).")
    ap.add_argument("--last-days", type=int, default=7, help="Analysis window for most pages (days).")
    ap.add_argument("--current-days", type=int, default=7, help="Current window (drift/health).")
    ap.add_argument("--reference-days", type=int, default=28, help="Reference window (drift).")
    ap.add_argument("--tz", type=str, default=None, help="Display timezone (data remains UTC).")

    # Options specific to some pages
    ap.add_argument("--clusters", type=int, default=4, help="Number of KMeans clusters for station behavior.")
    ap.add_argument("--hours", type=int, default=48, help="Window for station-level example plots (if used).")
    ap.add_argument("--select", type=int, default=12, help="How many stations to profile (if used).")
    ap.add_argument("--by", type=str, default="volatility", choices=["volatility", "coverage", "count"],
                    help="Selection criterion for stations (if used).")

    # Orchestration
    ap.add_argument("--pages", type=str, default="all",
                    help=("Comma-separated page keys. Use 'all' or choose from: "
                          "network.overview,network.stations,network.dynamics,"
                          "model.performance,model.pipeline,model.explainability,"
                          "monitoring.data_health,monitoring.drift,monitoring.model_health,"
                          "data.exports,data.dictionary,data.methodology"))
    ap.add_argument("--skip-apply-model", action="store_true",
                    help="Skip apply_model.py (no y_pred injection).")
    ap.add_argument("--lag-steps", type=int, default=0,
                    help="Rare: shift to apply to an incoming y_pred inside datasets (debug).")
    ap.add_argument("--as-csv", action="store_true",
                    help="Write events/perf as CSV instead of Parquet (debug).")

    args = ap.parse_args()

    # --------- Pre-requisites (always run) ----------
    # 1) datasets → events/perf
    run([
        sys.executable, TOOLS / "datasets.py",
        "--input", args.input,
        "--horizon", str(args.horizon),
        "--out-events", args.out_events,
        "--out-perf", args.out_perf,
        "--lag-steps", str(args.lag_steps),
        *(["--as-csv"] if args.as_csv else [])
    ])

    # 2) apply_model → inject y_pred (mandatory when not skipped)
    if not args.skip_apply_model:
        run([
            sys.executable, TOOLS / "apply_model.py",
            "--events", args.out_events,
            "--perf", args.out_perf,
            "--horizon", str(args.horizon),
            "--lookback-days", str(args.lookback_days),
            *(["--tz", args.tz] if args.tz else []),
        ])
        _assert_model_written(args.out_perf)
    else:
        print("[SKIP] apply_model requested -> perf keeps baseline only.")

    # --------- Page registry ----------
    # Each page maps to a callable that returns the subprocess command (list)
    registry: Dict[str, callable] = {
        # --- Network ---
        "network.overview": lambda: [
            sys.executable, TOOLS / "build_network_overview.py",
            "--events", args.out_events,
            "--last-days", str(args.last_days),
            *(["--tz", args.tz] if args.tz else []),
        ],
        "network.stations": lambda: [
            sys.executable, TOOLS / "build_network_stations.py",
            "--events", args.out_events,
            "--last-days", str(args.last_days),
            "--clusters", str(args.clusters),
            "--hours", str(args.hours),
            "--select", str(args.select),
            "--by", args.by,
            *(["--tz", args.tz] if args.tz else []),
        ],
        "network.dynamics": lambda: [
            sys.executable, TOOLS / "build_network_dynamics.py",
            "--events", args.out_events,
            "--last-days", str(args.last_days),
            *(["--tz", args.tz] if args.tz else []),
        ],

        # --- Model ---
        "model.performance": lambda: [
            sys.executable, TOOLS / "build_model_performance.py",
            "--perf", args.out_perf,
            "--last-days", str(args.last_days),
            "--horizon", str(args.horizon),
            *(["--tz", args.tz] if args.tz else []),
        ],
        "model.pipeline": lambda: [
            sys.executable, TOOLS / "build_model_pipeline.py",
            "--events", args.out_events,
            "--perf", args.out_perf,
            "--horizon", str(args.horizon),
        ],
        "model.explainability": lambda: [
            sys.executable, TOOLS / "build_model_explainability.py",
            "--perf", args.out_perf,
            "--last-days", str(args.last_days),
            *(["--tz", args.tz] if args.tz else []),
        ],

        # --- Monitoring ---
        "monitoring.data_health": lambda: [
            sys.executable, TOOLS / "build_monitoring_data_health.py",
            "--events", args.out_events,
            "--current-days", str(args.current_days),
            *(["--tz", args.tz] if args.tz else []),
        ],
        "monitoring.drift": lambda: [
            sys.executable, TOOLS / "build_monitoring_drift.py",
            "--events", args.out_events,
            "--current-days", str(args.current_days),
            "--reference-days", str(args.reference_days),
            "--perf", args.out_perf,                 # ← ajouter
            *(["--tz", args.tz] if args.tz else []),
        ],
        "monitoring.model_health": lambda: [
            sys.executable, TOOLS / "build_monitoring_model_health.py",
            "--perf", args.out_perf,
            "--last-days", str(args.last_days),
            "--horizon", str(args.horizon),          # ← ajouter
            *(["--tz", args.tz] if args.tz else []),
        ],
        # --- Data ---
        "data.exports": lambda: [
            sys.executable, TOOLS / "build_data_exports.py",
            "--exports-dir", EXPORTS,
        ],
        "data.dictionary": lambda: [
            sys.executable, TOOLS / "build_data_dictionary.py",
            "--events", args.out_events,
            "--perf", args.out_perf,
        ],
        "data.methodology": lambda: [
            sys.executable, TOOLS / "build_data_methodology.py",
            "--events", args.out_events,
            "--perf", args.out_perf,
            # "--horizon", str(args.horizon),      # ← retirer si non utilisé
        ],
    }

    # Default full order
    full_order = [
        "network.overview",
        "network.stations",
        "network.dynamics",
        "model.performance",
        "model.pipeline",
        "model.explainability",
        "monitoring.data_health",
        "monitoring.drift",
        "monitoring.model_health",
        "data.exports",
        "data.dictionary",
        "data.methodology",
    ]

    if args.pages.strip().lower() == "all":
        selection = full_order
    else:
        requested = [p.strip() for p in args.pages.split(",") if p.strip()]
        # keep the requested order; validate keys
        unknown = [p for p in requested if p not in registry]
        if unknown:
            valid = ", ".join(registry.keys())
            raise SystemExit(f"[ERROR] Unknown page key(s): {unknown}.\nValid values: {valid}")
        selection = requested

    print("[INFO] Pages to build:", ", ".join(selection))

    for key in selection:
        builder = registry[key]
        cmd = builder()
        # Make page execution mandatory to catch issues early
        run(cmd)

    print("[DONE] Site build succeeded.")


if __name__ == "__main__":
    main()
