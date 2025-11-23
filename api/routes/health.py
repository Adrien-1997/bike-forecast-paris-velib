# api/routes/health.py

"""API health endpoint for Vélib’ Forecast.

This module exposes a lightweight `/health` route that surfaces:
- basic process status and UTC time,
- Docker image tag (if configured),
- minimal GCS metadata for:
  - the main forecast bundle (`latest_forecast.json`),
  - the current model artefact (if any),
- the list of supported forecast horizons.

It is deliberately tolerant:
- if GCS is not available or misconfigured, the endpoint still returns `status="ok"`,
  with `forecast_error` / `model_error` fields explaining the issue.
"""

from __future__ import annotations

import os
import datetime as dt
from typing import Dict, Any, Optional, List

from fastapi import APIRouter
from core.settings import settings

try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    # In local/dev without GCS, we keep storage optional and degrade gracefully.
    storage = None  # type: ignore

router = APIRouter(prefix="/health", tags=["health"])


def _gcs_blob_meta(uri: str) -> Optional[Dict[str, Any]]:
    """Return lightweight metadata for a given GCS blob URI.

    Parameters
    ----------
    uri : str
        GCS URI in the form `gs://bucket/key`.

    Returns
    -------
    dict | None
        Typical structure:
        {
          "uri": "gs://bucket/path/to/blob",
          "etag": "...",
          "size": 123456,
          "updated": "2025-10-21T12:34:56.789012+00:00"
        }

        - `None` if:
          * the URI is not a valid `gs://` URI, or
          * `google-cloud-storage` is not available.
        - `{ "uri": uri, "etag": None, "size": None }` if the blob is missing
          or if an error occurs while querying GCS (tolerant behaviour).
    """
    if not (uri and isinstance(uri, str) and uri.startswith("gs://") and storage is not None):
        return None
    try:
        bucket, key = uri[5:].split("/", 1)
        cli = storage.Client()
        blob = cli.bucket(bucket).get_blob(key)
        if not blob:
            # Blob not found: we return a stub with only the URI.
            return {"uri": uri, "etag": None, "size": None}
        blob.reload()
        return {
            "uri": uri,
            "etag": getattr(blob, "etag", None),
            "size": getattr(blob, "size", None),
            "updated": getattr(blob, "updated", None) and blob.updated.isoformat(),
        }
    except Exception:
        # Any GCS error is swallowed: /health must never break the stack.
        return {"uri": uri, "etag": None, "size": None}


def _forecast_bundle_uri() -> str:
    """Build the GCS URI of the main forecast bundle (`latest_forecast.json`).

    Priority:
    - use `SERVING_FORECAST_PREFIX` if present on `settings`,
    - otherwise fall back to `settings.GCS_SERVING_PREFIX`.

    The `/latest_forecast.json` suffix is always appended.
    """
    base = getattr(settings, "SERVING_FORECAST_PREFIX", None) or settings.GCS_SERVING_PREFIX
    base = base.rstrip("/")
    return f"{base}/latest_forecast.json"


def _supported_horizons() -> List[int]:
    """Return the list of supported forecast horizons (in minutes).

    Reads `FORECAST_SUPPORTED` from `settings`, e.g. "15,60".
    Defaults to [15] if empty or misconfigured.
    """
    raw = (getattr(settings, "FORECAST_SUPPORTED", "") or "15").strip()
    if not raw:
        return [15]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


@router.get("")
def health():
    """Global API health endpoint.

    Response payload
    ----------------
    - `status`: always "ok" as long as the process is running;
    - `image_tag`: Docker image tag (from `IMAGE_TAG` env or `settings.IMAGE_TAG`);
    - `utc`: current server time in UTC, ISO format with "Z" suffix;
    - `forecast_bundle`: GCS metadata for `latest_forecast.json` (if resolvable);
    - `supported_horizons`: list of forecast horizons exposed by the API;
    - `model`: GCS metadata for the current model artefact (if configured);
    - `forecast_error` / `model_error`: plain-text error messages if GCS /
      configuration issues occur while resolving forecast bundle or model.

    Notes
    -----
    - This endpoint never raises HTTP errors due to GCS failures: exceptions
      are caught and surfaced as `*_error` fields, which is convenient for
      monitoring and for simple load balancer health checks.
    """
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
        # We stay tolerant: the API must still answer even if the bundle
        # hasn't been deployed yet or if GCS is temporarily unavailable.
        out["forecast_error"] = str(e)

    # Model info (optional)
    try:
        model_uri = getattr(settings, "GCS_MODEL_URI", None) or getattr(settings, "models_prefix", None)
        if model_uri:
            out["model"] = _gcs_blob_meta(str(model_uri)) or {"uri": str(model_uri), "etag": None, "size": None}
    except Exception as e:
        out["model_error"] = str(e)

    return out
