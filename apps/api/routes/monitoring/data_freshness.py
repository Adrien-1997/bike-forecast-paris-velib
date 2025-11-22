# apps/api/routes/monitoring/data_freshness.py
from __future__ import annotations
import os, json, math
from typing import Tuple

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage requis côté API") from e


# Préfixe global de ce module
router = APIRouter(prefix="/monitoring/data")


# ──────────────────────────────────────────────────────────────
# Helpers GCS — identiques à ceux des autres routes monitoring
# ──────────────────────────────────────────────────────────────

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
        print(f"[data_freshness] GCS_MONITORING_PREFIX invalide. Lu='{mon_raw}' nettoyé='{mon}'")
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX manquant ou invalide")
    print(f"[data_freshness] GCS_MONITORING_PREFIX='{mon}'")
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
    """Lit un JSON sur GCS et le renvoie avec cache-control."""
    try:
        payload = _gcs_read_json(gs_uri)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document non trouvé")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur lecture GCS: {e}")

    safe = _json_sanitize(payload)
    resp = JSONResponse(safe)
    resp.headers["Cache-Control"] = f"public, max-age={max(0, int(ttl))}"
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    return resp


# ──────────────────────────────────────────────────────────────
# ROUTE UNIQUE /monitoring/data/freshness
# ──────────────────────────────────────────────────────────────

@router.get("/freshness")
def get_data_freshness_latest(request: Request):
    """Renvoie le freshness latest.json (P95, P50, météo, etc.)."""
    prefix = _mon_prefix_or_500()
    base = f"{prefix}/monitoring" if not prefix.endswith("/monitoring") else prefix
    uri = f"{base}/data/freshness/latest.json"
    print(f"[data_freshness] reading {uri}")
    return _proxy_json(uri, request, ttl=30)