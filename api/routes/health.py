# api/routes/health.py

"""API health endpoint for Vélib’ Forecast.

This module exposes a lightweight `/health` route that surfaces:
- basic process status and UTC time,
- Docker image tag (if configured),
- minimal metadata for:
  - the main forecast bundle (`latest_forecast.json`),
  - the current model artefact (if any),
- the list of supported forecast horizons.

Storage backends
----------------
- If STORAGE_BACKEND = "local":
    * forecast/model artefacts are resolved on local disk under DATA_ROOT,
      by mapping a `gs://...` URI to a relative path.
- Otherwise:
    * metadata are fetched from GCS via `google-cloud-storage` (best-effort).

The endpoint is deliberately tolerant:
- GCS / local failures never break `/health`,
- errors are surfaced via `forecast_error` / `model_error` fields.
"""

from __future__ import annotations

import os
import datetime as dt
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import APIRouter
from core.settings import settings

try:  # pragma: no cover
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    storage = None  # type: ignore

router = APIRouter(prefix="/health", tags=["health"])

# ───────────────────────────── Backend (local vs GCS) ─────────────────────────────

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "gcs").lower()
USE_LOCAL = STORAGE_BACKEND == "local"

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("DATA_ROOT", REPO_ROOT / "data"))

# Mapping GCS → local disk when STORAGE_BACKEND=local
# Example:
#   GCS_LOCAL_ROOT_PREFIX = "gs://velib-forecast-472820_cloudbuild/velib/"
#   URI: gs://velib-forecast-472820_cloudbuild/velib/serving/forecast/...
#   → DATA_ROOT / "serving/forecast/..."
GCS_LOCAL_ROOT_PREFIX = (
    os.getenv(
        "GCS_LOCAL_ROOT_PREFIX",
        "gs://velib-forecast-472820_cloudbuild/velib/",
    ).rstrip("/")
    + "/"
)


def _local_blob_meta(uri: str) -> Optional[Dict[str, Any]]:
    """Return lightweight metadata for a "logical" GCS URI, using the local FS."""
    if not USE_LOCAL:
        return None
    if not uri.startswith(GCS_LOCAL_ROOT_PREFIX):
        return None

    rel = uri[len(GCS_LOCAL_ROOT_PREFIX):]
    path = DATA_ROOT / rel.lstrip("/")

    if not path.exists():
        return {"uri": uri, "etag": None, "size": None}

    # If it maps to a directory, it's a configuration/layout issue.
    # We stay tolerant but avoid returning misleading size=0.
    if path.is_dir():
        return {
            "uri": uri,
            "etag": None,
            "size": None,
            "updated": None,
            "warning": "URI maps to a directory on local FS",
        }

    try:
        stat = path.stat()
        updated = dt.datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z"
        return {
            "uri": uri,
            "etag": None,
            "size": stat.st_size if stat.st_size > 0 else None,
            "updated": updated,
        }
    except Exception:
        return {"uri": uri, "etag": None, "size": None}


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

        - In local mode, metadata are derived from the local FS.
        - `None` if:
          * the URI is not a valid `gs://` URI, or
          * `google-cloud-storage` is not available (non-local mode).
        - `{ "uri": uri, "etag": None, "size": None }` if the blob is missing
          or if an error occurs while querying GCS (tolerant behaviour).
    """
    # 1) Local mode: try mapping to DATA_ROOT
    if USE_LOCAL:
        return _local_blob_meta(uri)

    # 2) GCS mode
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
        # Any backend error is swallowed: /health must never break the stack.
        return {"uri": uri, "etag": None, "size": None}


def _forecast_bundle_uri() -> str:
    """Build the (logical) URI of the main forecast bundle (`latest_forecast.json`).

    Priority:
    - use `SERVING_FORECAST_PREFIX` if present on `settings`,
    - otherwise fall back to `settings.GCS_SERVING_PREFIX`.

    The `/latest_forecast.json` suffix is always appended.

    Notes
    -----
    - Even in local mode, the URI is expressed as `gs://...` and mapped to
      DATA_ROOT via `GCS_LOCAL_ROOT_PREFIX` by `_local_blob_meta`.
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
    try:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    except Exception:
        return [15]


@router.get("")
def health():
    """Global API health endpoint.

    Response payload
    ----------------
    - `status`: always "ok" as long as the process is running;
    - `image_tag`: Docker image tag (from `IMAGE_TAG` env or `settings.IMAGE_TAG`);
    - `utc`: current server time in UTC, ISO format with "Z" suffix;
    - `forecast_bundle`: metadata for `latest_forecast.json` (local or GCS);
    - `supported_horizons`: list of forecast horizons exposed by the API;
    - `model`: metadata for the current model artefact (local or GCS);
    - `forecast_error` / `model_error`: plain-text error messages if backend
      issues occur while resolving forecast bundle or model.

    Notes
    -----
    - This endpoint never raises HTTP errors due to backend failures: exceptions
      are caught and surfaced as `*_error` fields, which is convenient for
      monitoring and for simple load balancer health checks.
    """
    out: Dict[str, Any] = {
        "status": "ok",
        "image_tag": os.getenv("IMAGE_TAG", "") or getattr(settings, "IMAGE_TAG", ""),
        "utc": dt.datetime.utcnow().isoformat() + "Z",
        "backend": STORAGE_BACKEND,
        "data_root": str(DATA_ROOT) if USE_LOCAL else None,
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
            model_uri_str = str(model_uri)
            out["model"] = _gcs_blob_meta(model_uri_str) or {
                "uri": model_uri_str,
                "etag": None,
                "size": None,
            }
    except Exception as e:
        out["model_error"] = str(e)

    return out