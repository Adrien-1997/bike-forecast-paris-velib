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

# === ENV attendu ===
# - JOB : ingest | compact_daily | compact_monthly | latest_7d | build_serving_forecast
#         | build_features_4h (alias) | build_latest (alias)
#         | train_model | export_training_base | export_training_base_to_gcs
#         | export_data_health | push_hf
# - GCS_LOCK           : gs://.../velib/locks/job.lock  (optionnel)
# - PYTHON_BIN         : binaire python (optionnel, défaut: "python")
#
# Aides au calcul de date (pour compact_daily / compact_monthly) :
# - DAY                : YYYY-MM-DD (prioritaire si présent)
# - DAY_OFFSET         : entier (ex: -1 → veille UTC ; +1 → demain UTC). Ignoré si DAY est défini.

JOB        = (os.environ.get("JOB", "ingest") or "ingest").strip()
PYTHON_BIN = os.environ.get("PYTHON_BIN") or "python"
LOCK_URI   = os.environ.get("GCS_LOCK")

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
        return blob
    except Exception:
        return None

def _dispatch_command(job: str) -> List[str]:
    """
    Construit la commande Python **par module** (-m), nouvelle arbo:
      - jobs.ingest
      - jobs.compact_daily
      - jobs.compact_monthly
      - jobs.latest_7d
      - jobs.build_serving_forecast            (nouveau nom)
      - jobs.train_model
      - jobs.export_training_base
      - jobs.export_training_base_to_gcs
      - jobs.export_data_health
      - jobs.push_hf
    Compat alias:
      - build_features_4h → jobs.build_serving_forecast
      - build_latest      → jobs.latest_7d
    """
    alias: Dict[str, str] = {
        "build_features_4h": "build_serving_forecast",
        "build_latest": "latest_7d",
    }
    normalized = alias.get(job, job)

    modules = {
        "ingest":                       "jobs.ingest",
        "compact_daily":                "jobs.compact_daily",
        "compact_monthly":              "jobs.compact_monthly",
        "latest_7d":                    "jobs.latest_7d",
        "build_serving_forecast":       "jobs.build_serving_forecast",
        "train_model":                  "jobs.train_model",
        "export_training_base":         "jobs.export_training_base",
        "export_training_base_to_gcs":  "jobs.export_training_base_to_gcs",
        "export_data_health":           "jobs.export_data_health",
        "push_hf":                      "jobs.push_hf",
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
        # l'utilisateur a déjà fixé DAY explicitement
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
        print("[lock] busy (or error acquiring lock) → exit 0")
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
            "JOB","PYTHONPATH",
            "GCS_RAW_PREFIX","GCS_DAILY_PREFIX","GCS_MONTHLY_PREFIX",
            "GCS_SERVING_PREFIX","SERVING_FORECAST_PREFIX",
            "DAY","DAY_OFFSET"
        ]
        echo = " ".join(f"{k}={os.environ.get(k)}" for k in keys_to_echo if os.environ.get(k) is not None)
        if echo:
            print(f"[env] {echo}")

        print("[job] run:", " ".join(cmd))
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
