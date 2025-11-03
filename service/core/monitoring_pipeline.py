# =============================================================================
#  Vélib’ Forecast — Monitoring Pipeline (v2)
# =============================================================================
#  File: service/core/monitoring_pipeline.py
#  Purpose:
#     Central orchestrator for the Monitoring stack. Sequentially runs the
#     analysis jobs that build all JSON artifacts consumed by the Monitoring UI:
#
#       ├── data_health           → data coverage, freshness, missingness
#       ├── data_drift            → feature / target drift analysis
#       ├── model_performance     → daily performance (MAE/RMSE/…)
#       ├── model_explainability  → feature importance, calibration, residuals
#       ├── network_overview      → network-wide KPIs and heatmaps
#       ├── network_dynamics      → hourly evolution, tension, volatility
#       ├── network_stations      → per-station profiles and summaries
#       └── intro                 → metadata (dates, versions, links)
#
#  Output structure (GCS):
#       gs://<project>_cloudbuild/velib/monitoring/
#           ├── data/health/latest/*.json
#           ├── data/drift/latest/*.json
#           ├── model/performance/latest/h15/*.json
#           ├── model/explainability/latest/h60/*.json
#           ├── network/overview/latest/*.json
#           ├── network/dynamics/latest/*.json
#           ├── network/stations/latest/*.json
#           └── intro/latest/intro.json
#
#  Each job writes atomic JSON bundles using the schema:
#      {
#          "schema_version": "1.x",
#          "generated_at": "2025-11-02T08:15:00Z",
#          "meta": {...},
#          "data": [...]
#      }
#
# =============================================================================
#  USAGE
# -----------------------------------------------------------------------------
#   python -m service.core.monitoring_pipeline
#
#   Optional CLI flags:
#     --steps=STEP1,STEP2,...   → Override steps order (default = DEFAULT_STEPS)
#
#   Example:
#     python -m service.core.monitoring_pipeline --steps=data_health,model_performance
#
# =============================================================================
#  ENVIRONMENT VARIABLES
# -----------------------------------------------------------------------------
#  ─── Required ───────────────────────────────────────────────────────────────
#   GCS_EXPORTS_PREFIX     gs://.../velib/exports      (read events_*.parquet, perf_*.parquet)
#   GCS_MONITORING_PREFIX  gs://.../velib/monitoring   (write monitoring JSON)
#
#  ─── Recommended ────────────────────────────────────────────────────────────
#   FORECAST_HORIZONS      15,60
#   MODEL_URI_15           gs://.../models/h15/latest.joblib
#   MODEL_URI_60           gs://.../models/h60/latest.joblib
#   TZ_APP                 Europe/Paris
#
#  ─── Optional (pipeline control) ────────────────────────────────────────────
#   STEPS                  Override step list (comma-separated)
#   DAY                    Target day (YYYY-MM-DD) propagated to jobs that use it
#   CONTINUE_ON_ERROR      1|0  → continue even if a step fails (default=0)
#   DRY_RUN                1|0  → print steps without execution
#   PYTHON_BIN             Custom Python binary (default = sys.executable)
#   GCS_LOCK               gs://.../velib/locks/monitoring.lock (prevent concurrency)
#
#  ─── Optional (time windows / thresholds per job) ───────────────────────────
#   # Network overview
#   OVERVIEW_TZ, OVERVIEW_LAST_DAYS, OVERVIEW_REF_DAYS
#
#   # Network dynamics
#   DYNAMICS_TZ, DYNAMICS_LAST_DAYS, DYNAMICS_PENURY_THRESH, DYNAMICS_SATURATION_THRESH
#   (fallback: PENURY_THRESH, SATURATION_THRESH)
#
#   # Network stations (clustering/options)
#   NETWORK_WINDOW_DAYS, NETWORK_MIN_BINS_KEEP, NETWORK_K, NETWORK_K_MIN, NETWORK_K_MAX
#
#   # Model performance
#   PERF_TZ, PERF_LAST_DAYS, PERF_HORIZONS, PERF_RESID_BINS, PERF_TOP_STATIONS,
#   PERF_CLUSTERS_CSV, PERF_TS_MIN_POINTS
#
#   # Data health
#   DATA_HEALTH_TZ, DATA_HEALTH_LAST_DAYS, DATA_HEALTH_DAY
#
#   # Data drift
#   DRIFT_TZ, DRIFT_WINDOW_DAYS, DRIFT_FEATURES, DRIFT_BINS
#
#   # Intro (URIs if you want to override defaults)
#   INTRO_OVERVIEW_KPIS_URI, INTRO_HEALTH_KPIS_URI, INTRO_DRIFT_SUMMARY_URI,
#   INTRO_MODEL_LATEST_H15_URI, INTRO_MODEL_LATEST_H60_URI,
#   INTRO_FORECAST_H15_URI, INTRO_FORECAST_H60_URI, INTRO_OUT_PREFIX
#
# =============================================================================
#  NOTES
# -----------------------------------------------------------------------------
#   • Steps are executed sequentially; if one fails and CONTINUE_ON_ERROR=0,
#     the pipeline stops immediately.
#   • Jobs overwrite “latest” and may also publish timestamped bundles.
#   • No job modifies datasets or models — only monitoring outputs.
#   • Designed to run daily via Cloud Run Job or Cloud Scheduler.
# =============================================================================

from __future__ import annotations
import os, sys, shlex, contextlib
from dataclasses import dataclass
from subprocess import run, CalledProcessError
from typing import List, Dict, Optional

try:
    from google.cloud import storage  # type: ignore
except Exception:
    storage = None  # lock GCS optionnel

# ──────────────────────────────────────────────────────────────────────────────
# Chaîne MONITORING unifiée (datasets/compact exclus par défaut)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_STEPS = [
    "data_health",
    "data_drift",
    "model_performance",
    "model_explainability",
    "network_overview",
    "network_dynamics",
    "network_stations",
    "intro",
]

# Mapping step -> module exécuté
MODULES: Dict[str, str] = {
    # (Hors monitoring – disponibles mais non exécutés par défaut)
    "compact_daily":        "service.jobs.compact_daily",
    "datasets":             "service.jobs.build_datasets",

    # Monitoring — data
    "data_health":          "service.jobs.build_data_health",
    "data_drift":           "service.jobs.build_data_drift",

    # Monitoring — model
    "model_performance":    "service.jobs.build_model_performance",
    "model_explainability": "service.jobs.build_model_explainability",

    # Monitoring — network
    "network_overview":     "service.jobs.build_network_overview",
    "network_dynamics":     "service.jobs.build_network_dynamics",
    "network_stations":     "service.jobs.build_network_stations",

    # Monitoring — meta
    "intro":                "service.jobs.build_monitoring_intro",
}

@dataclass
class Cfg:
    steps: List[str]
    dry_run: bool
    continue_on_error: bool
    python_bin: str
    gcs_lock: Optional[str]

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v if (v is not None and v != "") else default

def _parse_steps(arg_line: Optional[str]) -> List[str]:
    """Priorité CLI (--steps=...), puis env STEPS; sinon DEFAULT_STEPS."""
    cli = None
    if arg_line:
        for a in arg_line.split():
            if a.startswith("--steps="):
                cli = a.split("=", 1)[1]
    env = _env("STEPS")
    raw = cli or env
    if not raw:
        return list(DEFAULT_STEPS)
    parts = [s.strip() for s in raw.split(",") if s.strip()]

    # Validation
    out: List[str] = []
    for s in parts:
        if s not in MODULES:
            valid = ", ".join(MODULES.keys())
            raise ValueError(f"unknown step '{s}'. Valid: {valid}")
        out.append(s)
    return out

def _acquire_lock(uri: Optional[str]):
    if not uri:
        print("[lock] disabled (no GCS_LOCK)")
        return None
    if storage is None:
        print("[lock][warn] google-cloud-storage not installed → skipping lock")
        return None
    assert uri.startswith("gs://"), f"invalid GCS_LOCK: {uri}"
    bkt, key = uri[5:].split("/", 1)
    blob = storage.Client().bucket(bkt).blob(key)
    try:
        blob.upload_from_string(b"", if_generation_match=0)
        print(f"[lock] acquired → {uri}")
        return blob
    except Exception:
        print("[lock] busy (or create error) → exit 0")
        return None

def _release_lock(blob):
    if not blob:
        return
    with contextlib.suppress(Exception):
        blob.delete()
        print("[lock] released")

def _run_module(python_bin: str, module: str) -> int:
    cmd = [python_bin, "-m", module]
    print("[run]", " ".join(shlex.quote(x) for x in cmd))
    try:
        run(cmd, check=True)
        return 0
    except CalledProcessError as e:
        print(f"[run][error] {module} → returncode={e.returncode}", file=sys.stderr)
        return e.returncode
    except Exception as e:
        print(f"[run][error] {module} → {e}", file=sys.stderr)
        return 1

def _echo_env(keys: List[str], title: str = "[env]") -> None:
    pairs = []
    for k in keys:
        v = os.environ.get(k)
        if v is not None:
            pairs.append(f"{k}={v}")
    if pairs:
        print(f"{title} " + " ".join(pairs))

# ──────────────────────────────────────────────────────────────────────────────
# Par-défaut “fenêtres jours” : 14j partout, OVERVIEW à 28j
# (ne force rien si l’utilisateur a déjà posé la variable)
# ──────────────────────────────────────────────────────────────────────────────
def _set_default(name: str, value: str) -> None:
    if not os.environ.get(name):
        os.environ[name] = value

def _apply_default_windows() -> None:
    # Data Health
    _set_default("DATA_HEALTH_LAST_DAYS", "14")     # si ton job lit LAST_DAYS
    _set_default("DATA_HEALTH_CURRENT_DAYS", "14")  # compat avec ton implémentation actuelle

    # Data Drift (v1.4: MON_* pris en priorité)
    _set_default("MON_CURRENT_DAYS", "14")
    _set_default("MON_REFERENCE_DAYS", "14")
    # Compat alternative
    _set_default("DRIFT_CURRENT_DAYS", "14")
    _set_default("DRIFT_REFERENCE_DAYS", "14")

    # Model Performance
    _set_default("PERF_LAST_DAYS", "14")

    # Model Explainability (fenêtre lookback)
    _set_default("LOOKBACK_DAYS", "14")

    # Network Dynamics
    _set_default("DYNAMICS_LAST_DAYS", "14")

    # Network Stations (fenêtre clustering / profils)
    _set_default("NETWORK_WINDOW_DAYS", "14")

    # Network Overview → EXCEPTION: 28 jours
    _set_default("OVERVIEW_LAST_DAYS", "28")
    # (facultatif) référence utilisée dans l’overview si besoin
    _set_default("OVERVIEW_REF_DAYS", "28")

def main(argv: List[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    arg_line = " ".join(argv[1:]) if len(argv) > 1 else None
    try:
        steps = _parse_steps(arg_line)
    except Exception as e:
        print(f"[cfg][error] {e}", file=sys.stderr)
        return 2

    cfg = Cfg(
        steps=steps,
        dry_run=(_env("DRY_RUN", "0") in ("1","true","True")),
        continue_on_error=(_env("CONTINUE_ON_ERROR", "0") in ("1","true","True")),
        python_bin=_env("PYTHON_BIN", sys.executable) or sys.executable,
        gcs_lock=_env("GCS_LOCK"),  # ex: gs://bucket/velib/locks/monitoring.lock
    )

    # Applique nos défauts 14j partout / 28j pour overview (sans écraser l'existant)
    _apply_default_windows()

    # Echo utile (sans secrets) — tronc commun
    _echo_env([
        "GCS_EXPORTS_PREFIX",
        "GCS_MONITORING_PREFIX",
        "FORECAST_HORIZONS",
        "MODEL_URI_15", "MODEL_URI_60",
        "TZ_APP",
        "DAY", "STEPS",
    ])

    # Echo détaillé par domaine (fenêtres, seuils, TZ, options)
    _echo_env([
        # Overview
        "OVERVIEW_TZ", "OVERVIEW_LAST_DAYS", "OVERVIEW_REF_DAYS",
        # Dynamics
        "DYNAMICS_TZ", "DYNAMICS_LAST_DAYS",
        "DYNAMICS_PENURY_THRESH", "DYNAMICS_SATURATION_THRESH",
        "PENURY_THRESH", "SATURATION_THRESH",
        # Stations / clustering
        "NETWORK_WINDOW_DAYS", "NETWORK_MIN_BINS_KEEP",
        "NETWORK_K", "NETWORK_K_MIN", "NETWORK_K_MAX",
        # Performance
        "PERF_TZ", "PERF_LAST_DAYS", "PERF_HORIZONS",
        "PERF_RESID_BINS", "PERF_TOP_STATIONS",
        "PERF_CLUSTERS_CSV", "PERF_TS_MIN_POINTS",
        # Data health
        "DATA_HEALTH_TZ", "DATA_HEALTH_LAST_DAYS", "DATA_HEALTH_CURRENT_DAYS", "DATA_HEALTH_DAY",
        # Drift
        "MON_CURRENT_DAYS", "MON_REFERENCE_DAYS",
        "DRIFT_TZ", "DRIFT_WINDOW_DAYS", "DRIFT_FEATURES", "DRIFT_BINS",
        # Intro links (optional overrides)
        "INTRO_OVERVIEW_KPIS_URI", "INTRO_HEALTH_KPIS_URI", "INTRO_DRIFT_SUMMARY_URI",
        "INTRO_MODEL_LATEST_H15_URI", "INTRO_MODEL_LATEST_H60_URI",
        "INTRO_FORECAST_H15_URI", "INTRO_FORECAST_H60_URI", "INTRO_OUT_PREFIX",
    ], title="[env:jobs]")

    # lock
    lock = _acquire_lock(cfg.gcs_lock)
    if cfg.gcs_lock and lock is None:
        # lock demandé mais pas obtenu
        return 0

    try:
        print("[orchestrator] steps:", ", ".join(cfg.steps))
        if cfg.dry_run:
            print("[orchestrator] DRY_RUN=1 → no execution")
            return 0

        exit_code = 0
        for s in cfg.steps:
            module = MODULES[s]
            rc = _run_module(cfg.python_bin, module)
            if rc != 0:
                exit_code = rc if exit_code == 0 else exit_code
                if not cfg.continue_on_error:
                    print("[orchestrator] abort on first error (CONTINUE_ON_ERROR=0)")
                    return exit_code
        return exit_code
    finally:
        _release_lock(lock)

if __name__ == "__main__":
    sys.exit(main())
