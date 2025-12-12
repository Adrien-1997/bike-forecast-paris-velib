# api/routes/monitoring/data_health.py

"""Monitoring — Data Health endpoints.

Ce module expose les endpoints `/monitoring/data/health/...` qui servent les
artefacts JSON produits par le job `build_data_health.py`.

Arborescence cible (LATEST only)
--------------------------------
<GCS_MONITORING_PREFIX>/monitoring/data/health/latest/
    kpis.json
    station_health.json
    coverage_by_hour.json
    alerts.json
    anomalies/
        manifest.json
        flat.json
        duplicates.json
        missing.json

En mode local (STORAGE_BACKEND=local), ces chemins sont mappés sur :

    <DATA_ROOT>/monitoring/data/health/latest/...

Principes :
- le backend **ne calcule rien** : il proxifie des JSON pré-calculés,
- `GCS_MONITORING_PREFIX` reste la source de vérité logique (`gs://...`),
- la lecture se fait soit via GCS (STORAGE_BACKEND=gcs), soit sur disque
  (STORAGE_BACKEND=local + DATA_ROOT),
- les floats non finis (NaN / ±Inf) sont remplacés par `null`.
"""

from __future__ import annotations

import os
import json
import math
from pathlib import Path
from typing import Literal, Tuple, Any, Optional, List, Dict

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

# google.cloud.storage est optionnel : utilisé seulement en mode GCS
try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover - mode local ou lib manquante
    storage = None

# ───────────────────────────── Router ─────────────────────────────
# Prefix "/monitoring/data", puis sous-chemin "health/..."
router = APIRouter(prefix="/monitoring/data")

# ───────────────────────────── Backend (local vs GCS) ─────────────────────────────

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "gcs").lower()
USE_LOCAL = STORAGE_BACKEND == "local"

# Repo root ≈ <repo>/ (par défaut)
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
        gs://.../velib/monitoring/data/health/latest/kpis.json
        → DATA_ROOT / "monitoring/data/health/latest/kpis.json"
    """
    if not uri.startswith(GCS_LOCAL_ROOT_PREFIX):
        raise ValueError(
            f"Cannot map GCS URI to local path:\n  uri={uri}\n  prefix={GCS_LOCAL_ROOT_PREFIX}"
        )
    rel = uri[len(GCS_LOCAL_ROOT_PREFIX) :]  # "monitoring/data/health/latest/..."
    path = DATA_ROOT / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _read_json_any(gs_uri: str) -> Any:
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
    return json.loads(data.decode("utf-8"))


def _mon_prefix_or_500() -> str:
    """Lit `GCS_MONITORING_PREFIX` et valide qu'il s'agit d'un `gs://...`."""
    raw = os.environ.get("GCS_MONITORING_PREFIX", "")
    mon = raw.strip().strip("'\"").rstrip("/")
    if not mon.startswith("gs://"):
        print(f"[data.health.router] GCS_MONITORING_PREFIX invalide: '{raw}' → '{mon}' (backend={STORAGE_BACKEND})")
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX manquant ou invalide")
    return mon


def _mon_base_or_500() -> str:
    """Retourne le préfixe `.../monitoring` en s'appuyant sur GCS_MONITORING_PREFIX."""
    mon = _mon_prefix_or_500()
    return f"{mon}/monitoring" if not mon.endswith("/monitoring") else mon


def _sanitize(obj):
    """Remplace les floats non finis (NaN / ±Inf) par `None` pour le JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _proxy_json(gs_uri: str, request: Request, ttl: int = 120) -> JSONResponse:
    """Lit un JSON (local ou GCS) et le proxifie avec des headers de cache.

    Comportement :
    - si le fichier n’existe pas → 204 + body `{}` (pas d’erreur),
    - si une autre erreur survient → 502 (erreur de lecture / parsing),
    - sinon → JSONResponse avec contenu "sanitisé".
    """
    try:
        payload = _read_json_any(gs_uri)
    except FileNotFoundError:
        resp = JSONResponse({}, status_code=204)
        resp.headers["Cache-Control"] = f"public, max-age={ttl}"
        resp.headers.setdefault("Access-Control-Allow-Origin", "*")
        return resp
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur lecture backend: {e}")

    safe = _sanitize(payload)
    resp = JSONResponse(safe)
    resp.headers["Cache-Control"] = f"public, max-age={ttl}"
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    return resp


def _proxy_file(rel: str, request: Request, ttl: int) -> JSONResponse:
    """Proxy utilitaire sur un chemin relatif depuis `data/health/latest/`."""
    base = _mon_base_or_500()
    gs_uri = f"{base}/data/health/latest/{rel.lstrip('/')}"
    return _proxy_json(gs_uri, request, ttl=ttl)


# ───────────────────────────── ROUTES ─────────────────────────────

DocName = Literal[
    "kpis",
    "station_health",
    "coverage_by_hour",
    "anomalies",            # fusion des shards (compat ascendante)
    "alerts",
    "anomalies_manifest",   # manifest.json
    "anomalies_flat",       # flat.json
    "anomalies_duplicates", # duplicates.json
    "anomalies_missing",    # missing.json
]

_TTL: Dict[str, int] = {
    "kpis": 60,
    "station_health": 300,
    "coverage_by_hour": 300,
    "anomalies": 120,
    "alerts": 60,
    "anomalies_manifest": 120,
    "anomalies_flat": 120,
    "anomalies_duplicates": 120,
    "anomalies_missing": 120,
}


@router.get("/health/available")
def available_docs():
    """Petit index des documents `data health` disponibles (latest only)."""
    return {
        "docs": list(_TTL.keys()),
        "time_travel": "non — latest-only",
    }


@router.get("/health/anomalies_manifest")
def anomalies_manifest(request: Request):
    """Proxy direct de `anomalies/manifest.json` (latest)."""
    return _proxy_file("anomalies/manifest.json", request, _TTL["anomalies_manifest"])


@router.get("/health/anomalies_flat")
def anomalies_flat(request: Request):
    """Proxy direct de `anomalies/flat.json` (latest)."""
    return _proxy_file("anomalies/flat.json", request, _TTL["anomalies_flat"])


@router.get("/health/anomalies_duplicates")
def anomalies_duplicates(request: Request):
    """Proxy direct de `anomalies/duplicates.json` (latest)."""
    return _proxy_file("anomalies/duplicates.json", request, _TTL["anomalies_duplicates"])


@router.get("/health/anomalies_missing")
def anomalies_missing(request: Request):
    """Proxy direct de `anomalies/missing.json` (latest)."""
    return _proxy_file("anomalies/missing.json", request, _TTL["anomalies_missing"])


@router.get("/health/{doc}")
def get_doc(doc: DocName, request: Request):
    """
    Proxy JSON pour les artefacts *data health*.

    Endpoints :
      /monitoring/data/health/kpis
      /monitoring/data/health/station_health
      /monitoring/data/health/coverage_by_hour
      /monitoring/data/health/alerts
      /monitoring/data/health/anomalies
      /monitoring/data/health/anomalies_manifest
      /monitoring/data/health/anomalies_flat
      /monitoring/data/health/anomalies_duplicates
      /monitoring/data/health/anomalies_missing
    """

    # Compat ascendante : fusionner les shards si /anomalies est demandé
    if doc == "anomalies":
        base = _mon_base_or_500()
        parts: List[Any] = []
        for name in ("flat", "duplicates", "missing"):
            uri = f"{base}/data/health/latest/anomalies/{name}.json"
            try:
                parts.append(_read_json_any(uri))
            except FileNotFoundError:
                continue

        merged: List[Any] = []
        for p in parts:
            if isinstance(p, list):
                merged.extend(p)

        resp = JSONResponse(_sanitize(merged))
        resp.headers["Cache-Control"] = f"public, max-age={_TTL['anomalies']}"
        resp.headers.setdefault("Access-Control-Allow-Origin", "*")
        return resp

    # Cas standard (kpis, station_health, coverage_by_hour, alerts)
    if doc in ("kpis", "station_health", "coverage_by_hour", "alerts"):
        return _proxy_file(f"{doc}.json", request, _TTL.get(doc, 120))

    # Nouveaux docs (si atteints via {doc})
    if doc == "anomalies_manifest":
        return anomalies_manifest(request)
    if doc == "anomalies_flat":
        return anomalies_flat(request)
    if doc == "anomalies_duplicates":
        return anomalies_duplicates(request)
    if doc == "anomalies_missing":
        return anomalies_missing(request)

    raise HTTPException(status_code=404, detail=f"Unknown doc '{doc}'")
