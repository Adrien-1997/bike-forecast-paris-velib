# tools/gcs_job.py
import os, sys, contextlib
from subprocess import run, CalledProcessError
from datetime import datetime, timezone, timedelta
from google.cloud import storage

# Env attendues (selon job)
# - GCS_DB_URI         : gs://.../velib/db/reporting/velib_reporting.duckdb   (jobs analytics)
# - GCS_DB_DAILY       : gs://.../velib/db/daily                               (compact/build_daily)
# - GCS_RAW_PREFIX     : gs://.../velib/bronze                                 (ingest/compact/features)
# - GCS_SERVING_PREFIX : gs://.../velib/serving/features_4h                    (build_features_4h)
# - GCS_LOCK           : gs://.../velib/locks/job.lock (optionnel)
# - DB_LOCAL           : ex: /tmp/velib_reporting.duckdb
# - JOB                : ingest | compact_daily | build_daily | build_monthly | build_latest | build_windows | dim_station | build_features_4h
# - DAY / MONTH        : paramètres optionnels

BUCKET_URI = os.environ.get("GCS_DB_URI")            # peut être None (selon JOB)
LOCK_URI   = os.environ.get("GCS_LOCK")              # optionnel
DB_LOCAL   = os.environ.get("DB_LOCAL", "/tmp/velib.duckdb")

JOB   = os.environ.get("JOB", "ingest")
DAY   = os.environ.get("DAY")
MONTH = os.environ.get("MONTH")

def needs_db(job: str) -> bool:
    """Retourne True si le job nécessite la DB reporting centralisée (GCS_DB_URI)."""
    return job in {
        "build_daily",         # écrit/maj reporting
        "build_monthly",       # (si tu l'ajoutes plus tard)
        "build_windows",       # écrit/maj reporting
        "dim_station",         # écrit/maj reporting
        # "build_latest",
        # "dedup",
    }

def parse_gs(uri: str):
    assert uri.startswith("gs://")
    rest = uri[5:]
    bkt, path = rest.split("/", 1)
    return bkt, path

def lock_blob(client, lock_uri):
    """Crée un lock GCS (objet vide) avec if_generation_match=0. None si absent/non acquis."""
    if not lock_uri:
        return None
    bkt, key = parse_gs(lock_uri)
    blob = client.bucket(bkt).blob(key)
    try:
        blob.upload_from_string(b"", if_generation_match=0)
        return blob
    except Exception:
        return None

def main():
    client = storage.Client()

    # 1) Lock optionnel
    lock = lock_blob(client, LOCK_URI)
    if LOCK_URI and lock is None:
        print("[lock] busy (or error acquiring lock) → exit 0")
        return 0
    elif not LOCK_URI:
        print("[lock] disabled (no GCS_LOCK)")

    db_blob = None

    try:
        # 2) Download DB reporting si nécessaire
        if needs_db(JOB):
            if not BUCKET_URI:
                raise RuntimeError("GCS_DB_URI required for this job")
            bkt, key = parse_gs(BUCKET_URI)
            db_blob = client.bucket(bkt).blob(key)
            if db_blob.exists(client):
                os.makedirs(os.path.dirname(DB_LOCAL) or "/", exist_ok=True)
                db_blob.download_to_filename(DB_LOCAL)
                print("[db] downloaded remote → local")
            else:
                print("[db] remote missing, will create fresh local if script writes it")

        # 3) Dispatch job
        if JOB == "ingest":
            cmd = ["python", "-m", "pipeline.ingest"]

        elif JOB == "compact_daily":
            cmd = ["python", "-m", "pipeline.compact_daily"]  # produit un shard daily

        elif JOB == "build_daily":
            # par défaut J-1 (shard produit par compact_daily)
            day = DAY or (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
            cmd = ["python", "-m", "pipeline.build_daily", "--day", day]

        elif JOB == "build_monthly":
            month = MONTH or datetime.now(timezone.utc).strftime("%Y-%m")
            cmd = ["python", "-m", "pipeline.build_monthly", "--month", month]

        elif JOB == "build_windows":
            # par défaut J-1 comme ancre de fenêtre (7j)
            day = DAY or (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
            cmd = ["python", "-m", "pipeline.build_windows", "--day", day]

        elif JOB == "dim_station":
            cmd = ["python", "-m", "pipeline.dim_station"]

        elif JOB == "build_features_4h":
            # mini magasin de features — ne nécessite pas la DB reporting
            cmd = ["python", "-m", "pipeline.build_features_4h"]

        elif JOB == "build_latest":
            # fichier unique “latest” direct depuis bronze (pas reporting)
            cmd = ["python", "-m", "pipeline.build_latest"]

        elif JOB == "train_model":
            cmd = ["python", "-m", "tools.train_model"]
        else:
            print(f"[job] unknown JOB={JOB}", file=sys.stderr)
            return 2

        print("[job] run:", " ".join(cmd))
        run(cmd, check=True)

        # 4) Upload DB reporting si nécessaire
        if needs_db(JOB) and db_blob is not None and os.path.exists(DB_LOCAL):
            db_blob.upload_from_filename(DB_LOCAL)
            print("[db] uploaded local → remote")

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
