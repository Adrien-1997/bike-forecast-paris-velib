# apps/api/routes/network_dynamics.py
from __future__ import annotations
import os
import json
import re
from typing import Optional, Literal, Tuple
import math

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage requis côté API") from e


router = APIRouter(prefix="/monitoring/network")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers GCS
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
        print(f"[network_dynamics] GCS_MONITORING_PREFIX invalide. Lu='{mon_raw}' nettoyé='{mon}'")
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX manquant ou invalide")
    print(f"[network_dynamics] GCS_MONITORING_PREFIX='{mon}'")
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
    "heatmaps_profiles": 300,   # plutôt stable
    "hourly_pen_sat":    120,   # plus “live”
    "episodes":          180,
    "by_zone":           300,
    "tension_by_station":180,
    "regularity_today":  120,
}

@router.get("/dynamics/available")
def network_dynamics_available():
    """Petit index des documents disponibles + rappel du time-travel."""
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
    """
    Proxy JSON pour les documents générés par le job 'build_network_dynamics.py'.

    Exemples:
      /monitoring/network/dynamics/heatmaps_profiles
      /monitoring/network/dynamics/episodes?at=2025-10-13T09-12-00Z
    """
    mon = _mon_prefix_or_500()  # ex: gs://velib-forecast-472820_cloudbuild/velib
    folder = _sanitize_at(at)   # 'latest' ou timestamp normalisé

    # si l'ENV finit déjà par /monitoring, on ne le redouble pas
    base = f"{mon}/monitoring" if not mon.endswith("/monitoring") else mon
    gs_uri = f"{base}/network/dynamics/{folder}/{doc}.json"

    ttl = _TTL_BY_DOC.get(doc, 120)
    return _proxy_json(gs_uri, request, ttl=ttl)
