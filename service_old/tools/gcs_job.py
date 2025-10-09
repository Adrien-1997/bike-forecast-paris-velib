# tools/gcs_job.py
from __future__ import annotations
import os, sys, contextlib
from subprocess import run, CalledProcessError
from typing import List, Tuple
from google.cloud import storage

# === ENV attendu ===
# - JOB : ingest | compact_daily | compact_monthly | build_latest_7d | build_features_4h | train_model
# - GCS_RAW_PREFIX     : gs://.../velib/bronze
# - GCS_SERVING_PREFIX : gs://.../velib/serving/features_4h   (si build_features_4h)
# - GCS_LOCK           : gs://.../velib/locks/job.lock        (optionnel)
# - PYTHON_BIN         : binaire python (optionnel, défaut: "python")

JOB        = (os.environ.get("JOB", "ingest") or "ingest").strip()
PYTHON_BIN = os.environ.get("PYTHON_BIN") or "python"
LOCK_URI   = os.environ.get("GCS_LOCK")

def parse_gs(uri: str) -> Tuple[str, str]:
    assert uri.startswith("gs://"), f"invalid GCS URI: {uri}"
    rest = uri[5:]
    bkt, path = rest.split("/", 1)
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

def _file(rel: str) -> str:
    """Résout un chemin de script sous /app et vérifie son existence."""
    path = rel if os.path.isabs(rel) else os.path.join("/app", rel)
    if not os.path.exists(path):
        raise FileNotFoundError(f"script introuvable: {path}")
    return path

def _dispatch_command(job: str) -> List[str]:
    """
    Construit la commande Python **par fichier** (pas de -m).
    Jobs supportés :
      - ingest                -> pipeline/ingest.py
      - compact_daily         -> pipeline/compact_daily.py
      - compact_monthly       -> pipeline/compact_monthly.py
      - build_latest          -> pipeline/latest_7d.py   (nom historique)
      - build_features_4h     -> pipeline/build_features_4h.py
      - train_model           -> tools/train_model.py
    """
    paths = {
        "ingest":            _file("pipeline/ingest.py"),
        "compact_daily":     _file("pipeline/compact_daily.py"),
        "compact_monthly":   _file("pipeline/compact_monthly.py"),
        "build_latest":      _file("pipeline/latest_7d.py"),
        "build_features_4h": _file("pipeline/build_features_4h.py"),
        "train_model":       _file("tools/train_model.py"),
    }
    if job not in paths:
        raise ValueError(f"unknown JOB={job}")
    return [PYTHON_BIN, paths[job]]

def main() -> int:
    client = storage.Client()

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
        keys_to_echo = ["JOB","PYTHONPATH","GCS_RAW_PREFIX","GCS_SERVING_PREFIX"]
        echo = " ".join(f"{k}={os.environ.get(k)}" for k in keys_to_echo if os.environ.get(k) is not None)
        if echo:
            print(f"[env] {echo}")

        print("[job] run:", " ".join(cmd))
        run(cmd, check=True)
        return 0

    except FileNotFoundError as e:
        print(f"[job] error: {e}", file=sys.stderr)
        return 1
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
