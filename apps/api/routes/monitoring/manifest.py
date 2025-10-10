from __future__ import annotations
from fastapi import APIRouter, Request, Response, HTTPException
from typing import Tuple
from core.settings import settings
from core.gcs import read_blob_bytes_cached, head_blob

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

def _normalize_head(res) -> Tuple[str | None, str | None]:
    if isinstance(res, tuple) and len(res) >= 2:
        return res[0], res[1]
    return None, None

@router.get("/manifest")
def get_manifest(request: Request):
    mon = getattr(settings, "GCS_MONITORING_PREFIX", None)
    if not mon or not mon.startswith("gs://"):
        raise HTTPException(status_code=500, detail="Monitoring prefix not configured")

    uri = f"{mon.rstrip('/')}/manifest.json"

    inm = request.headers.get("if-none-match")
    ims = request.headers.get("if-modified-since")

    res = head_blob(uri)
    etag, lastmod = _normalize_head(res)

    if inm and etag and inm.strip() == etag:
        return Response(status_code=304)
    if ims and lastmod and ims.strip() == lastmod:
        return Response(status_code=304)

    try:
        raw = read_blob_bytes_cached(uri, 60)  # positional TTL
        body = raw[0] if isinstance(raw, tuple) else raw
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="manifest not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"manifest read error: {e}")

    headers = {"Cache-Control": "public, max-age=60"}
    if etag:
        headers["ETag"] = etag
    if lastmod:
        headers["Last-Modified"] = lastmod
    return Response(content=body, media_type="application/json", headers=headers)
