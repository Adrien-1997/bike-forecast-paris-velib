# apps/api/routes/model_explainability.py
from __future__ import annotations
import os, json, re, math
from typing import Optional, Literal, Tuple

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import JSONResponse

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage requis côté API") from e

# ── Préfixe aligné sur les autres pages (network/*, model/*) ──────────────────
# /monitoring/model/explainability/*
router = APIRouter(prefix="/monitoring/model/explainability")

# ── Helpers GCS (repris du routeur performance pour cohérence) ───────────────
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
    mon_raw = os.environ.get("GCS_MONITORING_PREFIX", "")
    mon = mon_raw.strip().strip("'\"").rstrip("/")
    if not mon.startswith("gs://"):
        print(f"[model_explainability] GCS_MONITORING_PREFIX invalide. Lu='{mon_raw}' nettoyé='{mon}'")
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX manquant ou invalide")
    return mon

_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$")
def _sanitize_at(at: Optional[str]) -> str:
    # latest-only pour l’instant ; on garde 'at' future-proof
    if not at or at.strip().lower() in ("", "latest"):
        return "latest"
    s = at.strip()
    if not _AT_RE.match(s):
        return "latest"
    return s

def _json_sanitize(obj):
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
    safe = _json_sanitize(payload)
    resp = JSONResponse(safe)
    resp.headers["Cache-Control"] = f"public, max-age={max(0, int(ttl))}"
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    return resp

# ── Endpoints EXPLAINABILITY (latest only) ────────────────────────────────────

DocNameExplain = Literal["overview", "residuals", "calibration", "uncertainty"]

_TTL_BY_DOC = {
    "overview": 60,
    "residuals": 600,     # hist/qq/acf peu volatiles
    "calibration": 300,   # binning/heure
    "uncertainty": 300,   # couverture
}

@router.get("/available")
def model_explain_available():
    return {
        "docs": ["overview", "residuals", "calibration", "uncertainty"],
        "time_travel": "latest uniquement (param ?at=… ignoré si non valide)",
        "examples": [
            "/monitoring/model/explainability/overview",
            "/monitoring/model/explainability/residuals",
            "/monitoring/model/explainability/calibration",
            "/monitoring/model/explainability/uncertainty",
            "/monitoring/model/explainability/manifest",
        ],
    }

@router.get("/manifest")
def model_explain_manifest(request: Request, at: Optional[str] = None):
    mon = _mon_prefix_or_500()
    folder = _sanitize_at(at)  # 'latest'
    base = f"{mon}/monitoring" if not mon.endswith("/monitoring") else mon
    # pas de manifest dédié côté job explain -> on peut renvoyer overview.json comme "manifest"
    gs_uri = f"{base}/model/explainability/{folder}/overview.json"
    return _proxy_json(gs_uri, request, ttl=60)

@router.get("/{doc}")
def model_explain_doc(
    doc: DocNameExplain,
    request: Request,
    at: Optional[str] = None,
):
    mon = _mon_prefix_or_500()
    folder = _sanitize_at(at)  # 'latest'
    base = f"{mon}/monitoring" if not mon.endswith("/monitoring") else mon
    gs_uri = f"{base}/model/explainability/{folder}/{doc}.json"
    ttl = _TTL_BY_DOC.get(doc, 180)
    return _proxy_json(gs_uri, request, ttl=ttl)
