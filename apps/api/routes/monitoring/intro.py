# apps/api/routes/monitoring/intro.py
from __future__ import annotations
import os, json, re, math
from typing import Optional, Tuple

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage requis côté API") from e


router = APIRouter(prefix="/monitoring")

# ───────────────────────── Helpers GCS ─────────────────────────

def _split_gs(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Bad GCS URI: {uri}")
    bucket, key = uri[5:].split("/", 1)
    return bucket, key

def _gcs_read_json(gs_uri: str) -> dict:
    bkt, key = _split_gs(gs_uri)
    client = storage.Client()
    blob = client.bucket(bkt).blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"GCS blob not found: {gs_uri}")
    data = blob.download_as_bytes()
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Invalid JSON in {gs_uri}: {e}") from e

def _mon_prefix_or_500() -> str:
    """
    Attend GCS_MONITORING_PREFIX. Accepte avec/sans suffixe /monitoring,
    on normalise vers .../monitoring.
    """
    raw = os.environ.get("GCS_MONITORING_PREFIX", "")
    mon = raw.strip().strip("'\"").rstrip("/")
    if not mon.startswith("gs://"):
        print(f"[monitoring.intro] GCS_MONITORING_PREFIX invalide. Lu='{raw}' nettoyé='{mon}'")
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX manquant ou invalide")
    if not mon.endswith("/monitoring"):
        mon = mon + "/monitoring"
    return mon

_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$")

def _sanitize_at(at: Optional[str]) -> str:
    if not at or at.strip().lower() in ("", "latest"):
        return "latest"
    s = at.strip()
    return s if _AT_RE.match(s) else "latest"

def _json_sanitize(obj):
    """Remplace NaN/±Inf par None pour un JSON valide."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj

def _proxy_json(gs_uri: str, request: Request, ttl: int = 60) -> JSONResponse:
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

# ───────────────────────── Endpoints ─────────────────────────

@router.get("/intro/available")
def intro_available():
    return {
        "docs": ["intro"],  # un seul document: intro.json
        "time_travel": "utiliser ?at=YYYY-MM-DDTHH-MM-SSZ ou sans param pour latest",
    }

@router.get("/intro")
def intro_doc(request: Request, at: Optional[str] = None):
    """
    Sert le document consolidé produit par build_monitoring_intro.py
    Chemin: gs://.../monitoring/intro/<latest|YYYY-MM-DDTHH-MM-SSZ>/intro.json
    """
    base = _mon_prefix_or_500()
    folder = _sanitize_at(at)
    gs_uri = f"{base}/intro/{folder}/intro.json"
    return _proxy_json(gs_uri, request, ttl=60)
