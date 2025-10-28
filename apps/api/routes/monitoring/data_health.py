# apps/api/routes/monitoring/data_health.py
from __future__ import annotations
import os, json, math
from typing import Literal, Tuple, Any, Optional, List, Dict

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage requis côté API") from e


# ───────────────────────────── Router ─────────────────────────────
# Prefix "data" uniquement, puis sous-chemin "health/..."
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
    """Lit GCS_MONITORING_PREFIX et valide 'gs://...'; 500 sinon."""
    raw = os.environ.get("GCS_MONITORING_PREFIX", "")
    mon = raw.strip().strip("'\"").rstrip("/")
    if not mon.startswith("gs://"):
        print(f"[data.health.router] GCS_MONITORING_PREFIX invalide: '{raw}' → '{mon}'")
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX manquant ou invalide")
    return mon


def _mon_base_or_500() -> str:
    """Retourne le préfixe '.../monitoring' (ajouté si absent)."""
    mon = _mon_prefix_or_500()
    return f"{mon}/monitoring" if not mon.endswith("/monitoring") else mon


def _sanitize(obj):
    """Remplace les floats non finis par null (sécurité JSON)."""
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


def _proxy_file(rel: str, request: Request, ttl: int) -> JSONResponse:
    """Proxy utilitaire sur un chemin relatif depuis latest/."""
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
    "anomalies_manifest",   # nouveau: retourne manifest.json
    "anomalies_flat",       # nouveau: flat.json
    "anomalies_duplicates", # nouveau: duplicates.json
    "anomalies_missing",    # nouveau: missing.json
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
    """Petit index des documents disponibles (latest only)."""
    return {
        "docs": list(_TTL.keys()),
        "time_travel": "non — latest-only",
    }


@router.get("/health/anomalies_manifest")
def anomalies_manifest(request: Request):
    return _proxy_file("anomalies/manifest.json", request, _TTL["anomalies_manifest"])


@router.get("/health/anomalies_flat")
def anomalies_flat(request: Request):
    return _proxy_file("anomalies/flat.json", request, _TTL["anomalies_flat"])


@router.get("/health/anomalies_duplicates")
def anomalies_duplicates(request: Request):
    return _proxy_file("anomalies/duplicates.json", request, _TTL["anomalies_duplicates"])


@router.get("/health/anomalies_missing")
def anomalies_missing(request: Request):
    return _proxy_file("anomalies/missing.json", request, _TTL["anomalies_missing"])


@router.get("/health/{doc}")
def get_doc(doc: DocName, request: Request):
    """
    Proxy JSON pour les artefacts data health.

    Endpoints :
      /monitoring/data/health/kpis
      /monitoring/data/health/station_health
      /monitoring/data/health/coverage_by_hour
      /monitoring/data/health/alerts
      /monitoring/data/health/anomalies                → fusionne flat+duplicates+missing (compat)
      /monitoring/data/health/anomalies_manifest       → manifest.json
      /monitoring/data/health/anomalies_flat           → flat.json
      /monitoring/data/health/anomalies_duplicates     → duplicates.json
      /monitoring/data/health/anomalies_missing        → missing.json
    """

    # Compat ascendante : fusionner les shards si /anomalies est demandé
    if doc == "anomalies":
        base = _mon_base_or_500()
        parts: List[Any] = []
        for name in ("flat", "duplicates", "missing"):
            try:
                parts.append(_gcs_read_json(f"{base}/data/health/latest/anomalies/{name}.json"))
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
