# apps/api/routes/monitoring/data_health.py
from __future__ import annotations
import os, json, math
from typing import Literal, Tuple, Any

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage requis côté API") from e


# ───────────────────────────── Router ─────────────────────────────
# Prefix "data" uniquement, puis sous-chemin "health/..." comme overview
router = APIRouter(prefix="/monitoring/data")


# ───────────────────────────── GCS Helpers ─────────────────────────────

def _split_gs(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Bad GCS URI: {uri}")
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def _gcs_read_json(gs_uri: str) -> Any:
    """Lit un blob JSON GCS (dict ou list)."""
    bkt, key = _split_gs(gs_uri)
    client = storage.Client()
    bucket = client.bucket(bkt)
    blob = bucket.blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"GCS blob not found: {gs_uri}")
    data = blob.download_as_bytes()
    return json.loads(data.decode("utf-8"))


def _mon_prefix_or_500() -> str:
    raw = os.environ.get("GCS_MONITORING_PREFIX", "")
    mon = raw.strip().strip("'\"").rstrip("/")
    if not mon.startswith("gs://"):
        print(f"[data.health.router] GCS_MONITORING_PREFIX invalide: '{raw}' → '{mon}'")
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX manquant ou invalide")
    return mon


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _proxy_json(gs_uri: str, request: Request, ttl: int = 120) -> JSONResponse:
    """Lit un JSON sur GCS et le proxifie avec cache; 204 si absent."""
    try:
        payload = _gcs_read_json(gs_uri)
    except FileNotFoundError:
        resp = JSONResponse({}, status_code=204)
        resp.headers["Cache-Control"] = f"public, max-age={ttl}"
        resp.headers.setdefault("Access-Control-Allow-Origin", "*")
        return resp
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur lecture GCS: {e}")

    safe = _sanitize(payload)
    resp = JSONResponse(safe)
    resp.headers["Cache-Control"] = f"public, max-age={ttl}"
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    return resp


# ───────────────────────────── ROUTES ─────────────────────────────

DocName = Literal[
    "kpis",
    "station_health",
    "coverage_by_hour",
    "anomalies",
    "alerts",
]

_TTL = {
    "kpis": 60,
    "station_health": 300,
    "coverage_by_hour": 300,
    "anomalies": 120,
    "alerts": 60,
}


@router.get("/health/available")
def available_docs():
    """Petit index des documents disponibles (latest only)."""
    return {
        "docs": list(_TTL.keys()),
        "time_travel": "non — ces artefacts ne sont pas versionnés (latest-only)",
    }


@router.get("/health/{doc}")
def get_doc(doc: DocName, request: Request):
    """
    Proxy JSON pour les documents générés par build_data_health.py

    Exemples:
      /monitoring/data/health/kpis
      /monitoring/data/health/station_health
    """
    mon = _mon_prefix_or_500()
    base = f"{mon}/monitoring" if not mon.endswith("/monitoring") else mon
    gs_uri = f"{base}/data/health/latest/{doc}.json"
    return _proxy_json(gs_uri, request, ttl=_TTL.get(doc, 120))
