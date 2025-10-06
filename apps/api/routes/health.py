# apps/api/routes/health.py
from __future__ import annotations

import os
import datetime as dt
from typing import Dict, Any, Optional, List

from fastapi import APIRouter
from api.core.settings import settings

try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    storage = None  # type: ignore

router = APIRouter(prefix="/health", tags=["health"])


def _gcs_blob_meta(uri: str) -> Optional[Dict[str, Any]]:
    if not (uri and isinstance(uri, str) and uri.startswith("gs://") and storage is not None):
        return None
    try:
        bucket, key = uri[5:].split("/", 1)
        cli = storage.Client()
        blob = cli.bucket(bucket).get_blob(key)
        if not blob:
            return {"uri": uri, "etag": None, "size": None}
        blob.reload()
        return {
            "uri": uri,
            "etag": getattr(blob, "etag", None),
            "size": getattr(blob, "size", None),
            "updated": getattr(blob, "updated", None) and blob.updated.isoformat(),
        }
    except Exception:
        return {"uri": uri, "etag": None, "size": None}


def _forecast_bundle_uri() -> str:
    base = getattr(settings, "SERVING_FORECAST_PREFIX", None) or settings.GCS_SERVING_PREFIX
    base = base.rstrip("/")
    return f"{base}/latest_forecast.json"


def _supported_horizons() -> List[int]:
    raw = (getattr(settings, "FORECAST_SUPPORTED", "") or "15").strip()
    if not raw:
        return [15]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


@router.get("")
def health():
    out: Dict[str, Any] = {
        "status": "ok",
        "image_tag": os.getenv("IMAGE_TAG", "") or getattr(settings, "IMAGE_TAG", ""),
        "utc": dt.datetime.utcnow().isoformat() + "Z",
    }

    # Forecast bundle (single JSON)
    try:
        uri = _forecast_bundle_uri()
        out["forecast_bundle"] = _gcs_blob_meta(uri) or {"uri": uri, "etag": None, "size": None}
        out["supported_horizons"] = _supported_horizons()
    except Exception as e:
        out["forecast_error"] = str(e)

    # Model info (optional)
    try:
        model_uri = getattr(settings, "GCS_MODEL_URI", None) or getattr(settings, "models_prefix", None)
        if model_uri:
            out["model"] = _gcs_blob_meta(str(model_uri)) or {"uri": str(model_uri), "etag": None, "size": None}
    except Exception as e:
        out["model_error"] = str(e)

    return out
