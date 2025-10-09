# service/core/monitoring_pipeline.py
from __future__ import annotations
import os, sys, shlex, contextlib
from dataclasses import dataclass
from subprocess import run, CalledProcessError
from typing import List, Dict, Optional

try:
    from google.cloud import storage  # type: ignore
except Exception:
    storage = None  # lock GCS optionnel

# Ordre par défaut
DEFAULT_STEPS = [
    "build_datasets",
    "data_health",          # ← NEW: inventaire daily (JSON + CSV/GCS)
    "model_perf",
    "network_dynamics",
    "network_stations",
    "drift",
    "docs_dictionary",
    "docs_methodology",
    "docs_exports",
    "manifest",
]

# Mapping step -> module exécuté
MODULES: Dict[str, str] = {
    "compact_daily":       "service.jobs.compact_daily",
    "build_datasets":      "service.jobs.build_datasets",
    "data_health":         "service.jobs.export_data_health",            # ← NEW
    "model_perf":          "service.jobs.build_monitoring_model_health",
    "network_dynamics":    "service.jobs.build_network_dynamics",
    "network_stations":    "service.jobs.build_network_stations",
    "drift":               "service.jobs.build_monitoring_drift",
    "docs_dictionary":     "service.jobs.build_data_dictionary",
    "docs_methodology":    "service.jobs.build_data_methodology",
    "docs_exports":        "service.jobs.build_data_exports",
    "manifest":            "service.jobs.build_monitoring",
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
    # priorité CLI (--steps=...) puis env STEPS; sinon DEFAULT_STEPS
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
    # validation
    out: List[str] = []
    for s in parts:
        if s not in MODULES:
            raise ValueError(f"unknown step '{s}'. Valid: {', '.join(MODULES)}")
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

    # echo utile (sans secrets)
    echo_keys = [
        "GCS_RAW_PREFIX","GCS_DAILY_PREFIX","GCS_MONTHLY_PREFIX",
        "GCS_EXPORTS_PREFIX","GCS_MONITORING_PREFIX","SERVING_FORECAST_PREFIX",
        "FORECAST_HORIZONS","NETWORK_WINDOW_DAYS","EVENTS_WINDOW_DAYS","PERF_WINDOW_DAYS",
        "DRIFT_FEATURES","DRIFT_BINS","ANCHOR_DAY","DAY","STEPS"
    ]
    echo = " ".join(f"{k}={os.environ.get(k)}" for k in echo_keys if os.environ.get(k) is not None)
    if echo:
        print(f"[env] {echo}")

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
