# api/routes/network_dynamics.py

"""Monitoring — Network Dynamics endpoints.

Ce module expose les endpoints :

    /monitoring/network/dynamics/...

Ils servent les artefacts JSON produits par le job
`build_network_dynamics.py`, typiquement organisés comme :

    <GCS_MONITORING_PREFIX>/monitoring/network/dynamics/<latest|TS>/
        heatmaps_profiles.json
        hourly_pen_sat.json
        episodes.json
        by_zone.json
        tension_by_station.json
        regularity_today.json

En mode "maquette locale" (STORAGE_BACKEND=local), ces chemins sont mappés sur :

    <DATA_ROOT>/monitoring/network/dynamics/<latest|TS>/*.json

Principe :
- on garde des URI de type `gs://.../velib/...` dans le code,
- mais on les lit soit via GCS (STORAGE_BACKEND=gcs),
  soit sur disque (STORAGE_BACKEND=local).
"""

from __future__ import annotations

import os
import json
import re
import math
from typing import Optional, Literal, Tuple, Dict, Any
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

# google.cloud.storage est optionnel : uniquement utilisé en mode GCS
try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover - local mode / libs manquantes
    storage = None

router = APIRouter(prefix="/monitoring/network")

# ──────────────────────────────────────────────────────────────────────────────
# Backend de stockage (GCS vs local)
# ──────────────────────────────────────────────────────────────────────────────

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "gcs").lower()
USE_LOCAL = STORAGE_BACKEND == "local"

# Repo root ≈ <repo>/
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


def _split_gs(uri: str) -> Tuple[str, str]:
    """Découpe une URI `gs://bucket/key` en `(bucket, key)`."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Bad GCS URI: {uri}")
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def _local_path_from_gs(uri: str) -> Path:
    """Mappe une URI GCS velib/… vers un fichier local sous DATA_ROOT.

    Exemple :
        gs://.../velib/monitoring/network/dynamics/latest/episodes.json
        → DATA_ROOT / "monitoring/network/dynamics/latest/episodes.json"
    """
    if not uri.startswith(GCS_LOCAL_ROOT_PREFIX):
        raise ValueError(
        f"Cannot map GCS URI to local path:\n  uri={uri}\n  prefix={GCS_LOCAL_ROOT_PREFIX}"
        )
    rel = uri[len(GCS_LOCAL_ROOT_PREFIX) :]  # "monitoring/network/dynamics/..."
    path = DATA_ROOT / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _read_json_any(gs_uri: str) -> Dict[str, Any]:
    """Lit un JSON soit en local (DATA_ROOT), soit sur GCS."""
    if USE_LOCAL:
        # Mode maquette locale : lecture disque
        path = _local_path_from_gs(gs_uri)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # pragma: no cover - erreurs disque
            raise ValueError(f"Invalid JSON in local file {path}: {e}") from e

    # Mode GCS classique
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
        print(f"[network_dynamics] GCS_MONITORING_PREFIX invalide. Lu='{mon_raw}' nettoyé='{mon}'")
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX manquant ou invalide")
    print(f"[network_dynamics] GCS_MONITORING_PREFIX='{mon}' (backend={STORAGE_BACKEND})")
    return mon


# Timestamp "version" attendu pour le time-travel: 2025-10-23T14-30-00Z
_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$")


def _sanitize_at(at: Optional[str]) -> str:
    """Normalise le paramètre `at` (time-travel) en nom de dossier."""
    if not at or at.strip().lower() in ("", "latest"):
        return "latest"
    s = at.strip()
    if not _AT_RE.match(s):
        return "latest"
    return s


def _json_sanitize(obj):
    """Remplace NaN/±Inf par None pour produire un JSON valide."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _proxy_json(gs_uri: str, request: Request, ttl: int = 120) -> JSONResponse:
    """Lit un JSON (local ou GCS) et le proxifie avec des headers de cache."""
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

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints DYNAMICS
# ──────────────────────────────────────────────────────────────────────────────

DocNameDynamics = Literal[
    "heatmaps_profiles",
    "hourly_pen_sat",
    "episodes",
    "by_zone",
    "tension_by_station",
    "regularity_today",
]

_TTL_BY_DOC = {
    "heatmaps_profiles": 300,
    "hourly_pen_sat":    120,
    "episodes":          180,
    "by_zone":           300,
    "tension_by_station":180,
    "regularity_today":  120,
}


@router.get("/dynamics/available")
def network_dynamics_available():
    docs = [
        "heatmaps_profiles",
        "hourly_pen_sat",
        "episodes",
        "by_zone",
        "tension_by_station",
        "regularity_today",
    ]
    return {
        "docs": docs,
        "time_travel": "utiliser ?at=YYYY-MM-DDTHH-MM-SSZ ou sans param pour latest"
    }


@router.get("/dynamics/{doc}")
def network_dynamics_doc(doc: DocNameDynamics, request: Request, at: Optional[str] = None):
    mon = _mon_prefix_or_500()  # ex: gs://.../velib
    folder = _sanitize_at(at)   # 'latest' ou timestamp normalisé

    base = f"{mon}/monitoring" if not mon.endswith("/monitoring") else mon
    gs_uri = f"{base}/network/dynamics/{folder}/{doc}.json"

    ttl = _TTL_BY_DOC.get(doc, 120)
    return _proxy_json(gs_uri, request, ttl=ttl)
