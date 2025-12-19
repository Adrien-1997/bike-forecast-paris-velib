# api/routes/monitoring/data_freshness.py

"""Monitoring — Data Freshness endpoint.

Ce module expose l’endpoint `/monitoring/data/freshness` qui sert le JSON
produit par le job `build_data_health.py` / pipeline de fraîcheur :

Arborescence cible (LATEST only)
--------------------------------
<MONITORING_BASE>/monitoring/data/freshness/latest.json

En mode local, ces chemins sont mappés sur :

    <DATA_ROOT>/monitoring/data/freshness/latest.json

Principe :
- le backend ne calcule rien : il proxy un JSON déjà construit par les jobs,
- l’URI logique est toujours dérivée de `GCS_MONITORING_PREFIX` (env),
- la lecture se fait soit :
    * en **local** (STORAGE_BACKEND=local) sous `DATA_ROOT/monitoring/...`,
    * soit via **GCS** (STORAGE_BACKEND=gcs, mode historique),
- les JSON sont "sanitisés" : NaN / ±Inf → null.
"""

from __future__ import annotations

import os
import json
import math
from pathlib import Path
from typing import Tuple, Dict, Any

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

# google.cloud.storage est optionnel : utilisé seulement en mode GCS
try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover - mode local ou lib manquante
    storage = None

# Préfixe global de ce module
router = APIRouter(prefix="/monitoring/data")

# ──────────────────────────────────────────────────────────────
# Backend de stockage (GCS vs local)
# ──────────────────────────────────────────────────────────────

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "gcs").lower()
USE_LOCAL = STORAGE_BACKEND == "local"

# Repo root ≈ <repo>/ (valeur par défaut, mais DATA_ROOT est overridable)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("DATA_ROOT", REPO_ROOT / "data"))

# Préfixe GCS commun pour le mapping local :
#   gs://.../velib/monitoring/... → data/monitoring/...
GCS_LOCAL_ROOT_PREFIX = (
    os.getenv(
        "GCS_LOCAL_ROOT_PREFIX",
        "gs://velib-forecast-472820_cloudbuild/velib/",
    ).rstrip("/")
    + "/"
)

router = APIRouter(prefix="/monitoring/data", tags=["monitoring-data"])

def _split_gs(uri: str) -> Tuple[str, str]:
    """Découpe une URI `gs://bucket/key` en `(bucket, key)`."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Bad GCS URI: {uri}")
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def _local_path_from_gs(uri: str) -> Path:
    """Mappe une URI GCS velib/… vers un fichier local sous DATA_ROOT.

    Exemple :
        gs://.../velib/monitoring/data/freshness/latest.json
        → DATA_ROOT / "monitoring/data/freshness/latest.json"
    """
    if not uri.startswith(GCS_LOCAL_ROOT_PREFIX):
        raise ValueError(
            f"Cannot map GCS URI to local path:\n  uri={uri}\n  prefix={GCS_LOCAL_ROOT_PREFIX}"
        )
    rel = uri[len(GCS_LOCAL_ROOT_PREFIX) :]  # "monitoring/data/freshness/latest.json"
    path = DATA_ROOT / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _read_json_any(gs_uri: str) -> Dict[str, Any]:
    """Lit un JSON soit en local (DATA_ROOT), soit sur GCS, selon STORAGE_BACKEND."""
    if USE_LOCAL:
        path = _local_path_from_gs(gs_uri)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # pragma: no cover - erreurs disque
            raise ValueError(f"Invalid JSON in local file {path}: {e}") from e

    # Mode GCS (historique)
    if storage is None:
        raise RuntimeError("google-cloud-storage requis en mode GCS")

    bkt, key = _split_gs(gs_uri)
    client = storage.Client()
    bucket = client.bucket(bkt)
    blob = bucket.blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"GCS blob not found: {gs_uri}")
    data = blob.download_as_bytes()
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Invalid JSON in {gs_uri}: {e}") from e


def _mon_prefix_or_500() -> str:
    """Retourne `GCS_MONITORING_PREFIX` normalisé ou lève une 500."""
    mon_raw = os.environ.get("GCS_MONITORING_PREFIX", "")
    mon = mon_raw.strip().strip("'\"").rstrip("/")
    if not mon.startswith("gs://"):
        print(f"[data_freshness] GCS_MONITORING_PREFIX invalide. Lu='{mon_raw}' nettoyé='{mon}'")
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX manquant ou invalide")
    print(f"[data_freshness] GCS_MONITORING_PREFIX='{mon}' (backend={STORAGE_BACKEND})")
    return mon


def _json_sanitize(obj):
    """Remplace NaN/±Inf par None pour produire un JSON valide."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _proxy_json(gs_uri: str, request: Request, ttl: int = 60) -> JSONResponse:
    """Lit un JSON (local ou GCS) et le renvoie avec en-têtes HTTP adaptés."""
    try:
        payload = _read_json_any(gs_uri)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document non trouvé")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur lecture backend: {e}")

    safe = _json_sanitize(payload)
    resp = JSONResponse(safe)
    resp.headers["Cache-Control"] = f"public, max-age={max(0, int(ttl))}"
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    return resp


# ──────────────────────────────────────────────────────────────
# ROUTE UNIQUE /monitoring/data/freshness
# ──────────────────────────────────────────────────────────────

@router.get("/freshness")
def get_data_freshness(request: Request):
    prefix = _mon_prefix_or_500()
    base = f"{prefix}/monitoring" if not prefix.endswith("/monitoring") else prefix
    uri = f"{base}/data/freshness/latest/freshness.json"
    return _proxy_json(uri, request, ttl=30)


@router.get("/freshness/manifest")
def get_data_freshness_manifest(request: Request):
    prefix = _mon_prefix_or_500()
    base = f"{prefix}/monitoring" if not prefix.endswith("/monitoring") else prefix
    uri = f"{base}/data/freshness/latest/manifest.json"
    return _proxy_json(uri, request, ttl=30)

