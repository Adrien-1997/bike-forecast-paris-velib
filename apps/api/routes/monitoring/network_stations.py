# apps/api/routes/monitoring/network_stations.py
from __future__ import annotations
import os
import json
import re
import math
from typing import Optional, Literal, Tuple

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage requis côté API") from e


router = APIRouter(prefix="/monitoring/network")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers GCS (mêmes patterns que network_overview.py)
# ──────────────────────────────────────────────────────────────────────────────

def _split_gs(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Bad GCS URI: {uri}")
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def _gcs_read_json(gs_uri: str) -> dict:
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
    mon_raw = os.environ.get("GCS_MONITORING_PREFIX", "")
    mon = mon_raw.strip().strip("'\"").rstrip("/")
    if not mon.startswith("gs://"):
        print(f"[network_stations] GCS_MONITORING_PREFIX invalide. Lu='{mon_raw}' nettoyé='{mon}'")
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX manquant ou invalide")
    print(f"[network_stations] GCS_MONITORING_PREFIX='{mon}'")
    return mon


_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$")

def _sanitize_at(at: Optional[str]) -> str:
    if not at or at.strip().lower() in ("", "latest"):
        return "latest"
    s = at.strip()
    if not _AT_RE.match(s):
        # force 'latest' si format non reconnu (évite path traversal)
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
    try:
        payload = _gcs_read_json(gs_uri)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document non trouvé")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur lecture GCS: {e}")

    # Sanitize (évite ValueError: Out of range float values are not JSON compliant)
    safe = _json_sanitize(payload)

    resp = JSONResponse(safe)
    # Cache côté client / CDN
    resp.headers["Cache-Control"] = f"public, max-age={max(0, int(ttl))}"
    # CORS permissif si derrière un proxy (optionnel)
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    return resp


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints STATIONS
# ──────────────────────────────────────────────────────────────────────────────

DocNameStations = Literal["kpis", "centroids", "pca_scatter", "pca_circle", "stats7"]

_TTL_BY_DOC = {
    "kpis": 60,
    "centroids": 300,
    "pca_scatter": 300,
    "pca_circle": 300,
    "stats7": 120,
}

@router.get("/stations/available")
def network_stations_available():
    """Index des documents stations + rappel du time-travel."""
    docs = ["kpis", "centroids", "pca_scatter", "pca_circle", "stats7"]
    return {
        "docs": docs,
        "time_travel": "utiliser ?at=YYYY-MM-DDTHH-MM-SSZ ou sans param pour latest"
    }


@router.get("/stations/{doc}")
def network_stations_doc(doc: DocNameStations, request: Request, at: Optional[str] = None):
    """
    Proxy JSON pour les documents générés par le job Network Stations.

    Exemples:
      /monitoring/network/stations/kpis
      /monitoring/network/stations/centroids?at=2025-10-13T09-12-00Z
    """
    mon = _mon_prefix_or_500()   # ex: gs://velib-forecast-472820_cloudbuild/velib
    folder = _sanitize_at(at)    # 'latest' ou timestamp normalisé

    # si l'ENV finit déjà par /monitoring, on ne le redouble pas
    base = f"{mon}/monitoring" if not mon.endswith("/monitoring") else mon
    gs_uri = f"{base}/network/stations/{folder}/{doc}.json"

    ttl = _TTL_BY_DOC.get(doc, 120)
    return _proxy_json(gs_uri, request, ttl=ttl)


# ──────────────────────────────────────────────────────────────────────────────
# Legacy minimal (optionnel) : ancien JSON global si présent
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/stations")
def network_stations_legacy(request: Request):
    """
    Legacy endpoint — renvoie l'ancien JSON global si encore présent :
    gs://<...>/monitoring/network/stations.json
    """
    mon = _mon_prefix_or_500()
    base = f"{mon}/monitoring" if not mon.endswith("/monitoring") else mon
    gs_uri = f"{base}/network/stations.json"
    return _proxy_json(gs_uri, request, ttl=120)
