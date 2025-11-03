# apps/api/routes/forecast.py
from __future__ import annotations
import os, json, math
from typing import Tuple, List

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import JSONResponse

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage requis côté API") from e

router = APIRouter(prefix="/forecast", tags=["forecast"])

# ── Helpers GCS (mêmes patterns que model_performance.py) ─────────────────────
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

def _serving_prefix_or_500() -> str:
    raw = os.environ.get("SERVING_FORECAST_PREFIX", "")
    val = raw.strip().strip("'\"").rstrip("/")
    if not val.startswith("gs://"):
        print(f"[forecast] SERVING_FORECAST_PREFIX invalide. Lu='{raw}' → '{val}'")
        raise HTTPException(status_code=500, detail="SERVING_FORECAST_PREFIX manquant ou invalide")
    return val

def _json_sanitize(obj):
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

# ── Horizons supportés (comme perf utilise ?h=) ───────────────────────────────
def _supported_horizons() -> List[int]:
    envs = os.environ.get("FORECAST_SUPPORTED", "15,60")
    try:
        vals = sorted({int(x.strip()) for x in envs.split(",") if x.strip()})
        return vals or [15]
    except Exception:
        return [15]

def _validate_h(h: int) -> None:
    supported = set(_supported_horizons())
    if h not in supported:
        raise HTTPException(status_code=400, detail=f"h must be in {sorted(supported)}")

# ── Endpoints (style perf : 1 route + ?h=) ────────────────────────────────────
@router.get("/available")
def forecast_available():
    return {
        "prefix": os.environ.get("SERVING_FORECAST_PREFIX", ""),
        "horizons": _supported_horizons(),
        "time_travel": "latest uniquement",
        "docs": ["latest"],
        "examples": [
            "/forecast/latest?h=15",
            "/forecast/latest?h=60",
        ],
    }

@router.get("/latest")
def forecast_latest(
    request: Request,
    h: int = Query(15, description="Horizon en minutes (ex: 15, 60)"),
):
    """
    Proxy direct GCS, comme 'performance' :
    lit gs://…/serving/forecast/h{h}/latest.json
    """
    _validate_h(h)
    base = _serving_prefix_or_500()
    gs_uri = f"{base}/h{int(h)}/latest.json"
    return _proxy_json(gs_uri, request, ttl=60)
