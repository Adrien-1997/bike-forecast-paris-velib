from __future__ import annotations
from fastapi import APIRouter, Request, Response, HTTPException, Query
from typing import Tuple, Optional
from core.settings import settings
from core.gcs import read_blob_bytes_cached, head_blob

router = APIRouter(prefix="/monitoring/model/perf", tags=["monitoring:perf"])

def _normalize_head(res) -> Tuple[str | None, str | None]:
    if isinstance(res, tuple) and len(res) >= 2:
        return res[0], res[1]
    return None, None

def _proxy(uri: str, request: Request, ttl: int = 120) -> Response:
    inm = request.headers.get("if-none-match")
    ims = request.headers.get("if-modified-since")

    res = head_blob(uri)
    etag, lastmod = _normalize_head(res)

    if inm and etag and inm.strip() == etag:
        return Response(status_code=304)
    if ims and lastmod and ims.strip() == lastmod:
        return Response(status_code=304)

    try:
        raw = read_blob_bytes_cached(uri, ttl)
        body = raw[0] if isinstance(raw, tuple) else raw
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="object not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"read error: {e}")

    headers = {"Cache-Control": f"public, max-age={ttl}"}
    if etag:
        headers["ETag"] = etag
    if lastmod:
        headers["Last-Modified"] = lastmod
    return Response(content=body, media_type="application/json", headers=headers)

@router.get("/daily")
def perf_daily(
    request: Request,
    h: int = Query(..., description="horizon in minutes, e.g. 15 or 60"),
    v: Optional[str] = Query(None, description="optional version tag like YYYYMMDD"),
):
    mon = getattr(settings, "GCS_MONITORING_PREFIX", None)
    if not mon or not mon.startswith("gs://"):
        raise HTTPException(status_code=500, detail="Monitoring prefix not configured")
    if v:
        uri = f"{mon.rstrip('/')}/model/perf/daily_h{h}_{v}.json"
    else:
        uri = f"{mon.rstrip('/')}/model/perf/daily_h{h}.json"
    return _proxy(uri, request, ttl=120)

@router.get("/segments")
def perf_segments(
    request: Request,
    h: int = Query(..., description="horizon in minutes, e.g. 15 or 60"),
    v: Optional[str] = Query(None, description="optional version tag like YYYYMMDD"),
):
    mon = getattr(settings, "GCS_MONITORING_PREFIX", None)
    if not mon or not mon.startswith("gs://"):
        raise HTTPException(status_code=500, detail="Monitoring prefix not configured")
    if v:
        uri = f"{mon.rstrip('/')}/model/perf/segments_h{h}_{v}.json"
    else:
        uri = f"{mon.rstrip('/')}/model/perf/segments_h{h}.json"
    return _proxy(uri, request, ttl=120)
