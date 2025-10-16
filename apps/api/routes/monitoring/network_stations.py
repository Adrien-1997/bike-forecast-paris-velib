# apps/api/routes/monitoring/network_stations.py
from __future__ import annotations
from fastapi import APIRouter, Request, Response, HTTPException
from typing import Tuple, Optional, Literal
from core.settings import settings
from core.gcs import read_blob_bytes_cached, head_blob

router = APIRouter(prefix="/monitoring/network")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_head(res) -> Tuple[str | None, str | None]:
    if isinstance(res, tuple) and len(res) >= 2:
        return res[0], res[1]
    return None, None


def _proxy_json(uri: str, request: Request, ttl: int = 60) -> Response:
    inm = request.headers.get("if-none-match")
    ims = request.headers.get("if-modified-since")

    res = head_blob(uri)
    etag, lastmod = _normalize_head(res)

    # Conditional GET support
    if inm and etag and inm.strip() == etag:
        return Response(status_code=304)
    if ims and lastmod and ims.strip() == lastmod:
        return Response(status_code=304)

    try:
        raw = read_blob_bytes_cached(uri, ttl)
        body = raw[0] if isinstance(raw, tuple) else raw
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Object not found: {uri}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Read error: {e}")

    headers = {"Cache-Control": f"public, max-age={ttl}"}
    if etag:
        headers["ETag"] = etag
    if lastmod:
        headers["Last-Modified"] = lastmod

    return Response(content=body, media_type="application/json", headers=headers)


def _mon_prefix_or_500() -> str:
    mon = getattr(settings, "GCS_MONITORING_PREFIX", None)
    if not mon or not mon.startswith("gs://"):
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX missing or invalid")
    return mon.rstrip("/")


def _sanitize_at(at: Optional[str]) -> str:
    """
    Accept only folder-like timestamps '2025-10-12T21-52-19Z'.
    If missing or invalid, defaults to 'latest'.
    """
    if not at:
        return "latest"
    safe = "".join(ch for ch in at.strip() if ch.isdigit() or ch in "TZ:-")
    return safe or "latest"


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints — per visual JSON (latest or time-travel via ?at=)
# ──────────────────────────────────────────────────────────────────────────────

DocName = Literal["kpis", "centroids", "pca_scatter", "pca_circle", "stats7"]

@router.get("/stations/{doc}")
def network_stations_doc(doc: DocName, request: Request, at: Optional[str] = None):
    """
    Exemples :
      /monitoring/network/stations/kpis
      /monitoring/network/stations/centroids?at=2025-10-13T09-12-00Z
    """
    mon = _mon_prefix_or_500()
    folder = _sanitize_at(at)

    # Structure identique au job build_network_stations.py
    # → {MON_PREFIX}/monitoring/network/stations/<folder>/{doc}.json
    uri = f"{mon}/monitoring/network/stations/{folder}/{doc}.json"

    ttl = 60 if doc == "kpis" else 120
    return _proxy_json(uri, request, ttl=ttl)


@router.get("/stations/available")
def network_stations_available():
    return {
        "docs": ["kpis", "centroids", "pca_scatter", "pca_circle", "stats7"],
        "time_travel": "use ?at=YYYY-MM-DDTHH-MM-SSZ or omit for latest"
    }


# ──────────────────────────────────────────────────────────────────────────────
# Legacy endpoint minimal (pour compatibilité API /stations JSON only)
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/stations")
def network_stations_legacy(request: Request):
    """Legacy endpoint — renvoie l'ancien JSON global si encore présent."""
    mon = _mon_prefix_or_500()
    uri = f"{mon}/monitoring/network/stations.json"
    return _proxy_json(uri, request, ttl=120)