# service/core/utils_gcs.py
from __future__ import annotations
import os, sys, contextlib
from subprocess import run, CalledProcessError
from typing import List, Tuple, Dict
from datetime import datetime, timezone, timedelta

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage is required for job locking") from e

# ============================================================================
# ENV attendu
# ----------------------------------------------------------------------------
# - JOB :
#     Ingestion & parquet-first
#       • ingest
#       • compact_daily
#       • compact_monthly
#       • build_serving_forecast        (alias: build_features_4h)
#
#     Training / exports
#       • train_model
#       • export_training_base
#
#     Monitoring (JSON artifacts)
#       • monitoring_pipeline           (alias: monitoring)
#
# - GCS_LOCK            : gs://.../velib/locks/job.lock  (optionnel)
# - PYTHON_BIN          : binaire python (optionnel, défaut: "python")
# - DRY_RUN             : 1|0 (affiche la commande sans l'exécuter)
#
# Aides au calcul de date (pour compact_daily / compact_monthly) :
# - DAY                 : YYYY-MM-DD (prioritaire si présent)
# - DAY_OFFSET          : entier (ex: -1 → veille UTC ; +1 → demain UTC). Ignoré si DAY est défini.
# ============================================================================

JOB        = (os.environ.get("JOB", "ingest") or "ingest").strip()
PYTHON_BIN = os.environ.get("PYTHON_BIN") or "python"
LOCK_URI   = os.environ.get("GCS_LOCK")
DRY_RUN    = os.environ.get("DRY_RUN") in ("1", "true", "True")


def parse_gs(uri: str) -> Tuple[str, str]:
    assert uri.startswith("gs://"), f"invalid GCS URI: {uri}"
    bkt, path = uri[5:].split("/", 1)
    return bkt, path


def lock_blob(client: storage.Client, lock_uri: str | None):
    """Best-effort lock via objet GCS vide (if_generation_match=0)."""
    if not lock_uri:
        return None
    bkt, key = parse_gs(lock_uri)
    blob = client.bucket(bkt).blob(key)
    try:
        blob.upload_from_string(b"", if_generation_match=0)
        print(f"[lock] acquired → {lock_uri}")
        return blob
    except Exception:
        # busy or cannot create
        return None


def _dispatch_command(job: str) -> List[str]:
    """
    Construit la commande Python **par module** (-m), arbo service.jobs.* ou service.core.* :
      - service.jobs.ingest
      - service.jobs.compact_daily
      - service.jobs.compact_monthly
      - service.jobs.build_serving_forecast
      - service.jobs.train_model
      - service.jobs.export_training_base
      - service.jobs.export_training_base_to_gcs
      - service.core.monitoring_pipeline            ← orchestration monitoring

    Alias compat:
      - monitoring         → monitoring_pipeline
    """
    alias: Dict[str, str] = {
        "monitoring": "monitoring_pipeline",
    }
    normalized = alias.get(job, job)

    modules = {
        # parquet-first
        "ingest":                       "service.jobs.ingest",
        "compact_daily":                "service.jobs.compact_daily",
        "compact_monthly":              "service.jobs.compact_monthly",
        "build_serving_forecast":       "service.jobs.build_serving_forecast",

        # training / exports
        "train_model":                  "service.jobs.train_model",
        "export_training_base":         "service.jobs.export_training_base",
        "export_training_base_to_gcs":  "service.jobs.export_training_base_to_gcs",

        # monitoring
        "monitoring_pipeline":          "service.core.monitoring_pipeline",
    }
    if normalized not in modules:
        raise ValueError(f"unknown JOB={job}")
    return [PYTHON_BIN, "-m", modules[normalized]]


def _maybe_set_day_env(job: str):
    """
    Pour compact_* : si DAY n'est pas fourni, on le calcule.
    Priorité:
      1) DAY déjà défini → ne rien faire
      2) DAY_OFFSET (entier) → today_utc + offset
      3) par défaut → veille UTC (today_utc - 1)
    """
    if job not in ("compact_daily", "compact_monthly"):
        return

    if os.environ.get("DAY"):
        return

    offset_env = os.environ.get("DAY_OFFSET")
    try:
        if offset_env is not None and offset_env.strip() != "":
            offset = int(offset_env)
        else:
            offset = -1  # veille UTC par défaut
    except Exception:
        offset = -1

    day = (datetime.now(timezone.utc) + timedelta(days=offset)).strftime("%Y-%m-%d")
    os.environ["DAY"] = day
    print(f"[job] DAY not provided → computed from offset={offset}: DAY={day} (UTC)")


def main() -> int:
    client = storage.Client()

    # 0) Calcul auto de DAY si pertinent
    _maybe_set_day_env(JOB)

    # 1) Lock optionnel
    lock = lock_blob(client, LOCK_URI)
    if LOCK_URI and lock is None:
        print(f"[lock] busy or create error → {LOCK_URI} → exit 0")
        return 0
    elif not LOCK_URI:
        print("[lock] disabled (no GCS_LOCK)")

    try:
        # 2) Construire la commande
        try:
            cmd = _dispatch_command(JOB)
        except Exception as e:
            print(f"[job] error while building command: {e}", file=sys.stderr)
            return 2

        # Echo utile debug (sans secrets)
        keys_to_echo = [
            "JOB", "PYTHONPATH",
            # parquet-first
            "GCS_RAW_PREFIX", "GCS_DAILY_PREFIX", "GCS_MONTHLY_PREFIX",
            "GCS_SERVING_PREFIX", "SERVING_FORECAST_PREFIX", "GCS_EXPORTS_PREFIX",
            # models / forecast
            "GCS_MODEL_URI_T15", "GCS_MODEL_URI_T60",
            "FORECAST_HORIZONS", "WINDOW_HOURS", "WITH_FORECAST", "NOW_UTC_ISO",
            # monitoring roots
            "GCS_MONITORING_PREFIX",
            # time helpers
            "DAY", "DAY_OFFSET",
            # training (si utilisé)
            "MODEL_TYPE", "HORIZON_BINS",
            # monitoring pipeline knobs
            "STEPS", "CONTINUE_ON_ERROR", "DRY_RUN",
            "MODEL_URI_15", "MODEL_URI_60", "TZ_APP",
            "MON_LAST_DAYS", "MON_REF_DAYS",
            "OVERVIEW_LAST_DAYS", "DYNAMICS_LAST_DAYS", "NETWORK_WINDOW_DAYS",
            "PERF_LAST_DAYS", "LOOKBACK_DAYS",
        ]
        echo = " ".join(
            f"{k}={os.environ.get(k)}"
            for k in keys_to_echo
            if os.environ.get(k) is not None
        )
        if echo:
            print(f"[env] {echo}")

        # 3) Exécution (ou dry-run)
        print("[job] run:", " ".join(cmd))
        if DRY_RUN:
            print("[job] DRY_RUN=1 → skipping execution")
            return 0

        run(cmd, check=True)
        return 0

    except CalledProcessError as e:
        print(f"[job] failed: {e}", file=sys.stderr)
        return e.returncode
    except Exception as e:
        print(f"[job] error: {e}", file=sys.stderr)
        return 1
    finally:
        if LOCK_URI and lock is not None:
            with contextlib.suppress(Exception):
                lock.delete()
                print("[lock] released")


if __name__ == "__main__":
    sys.exit(main())
