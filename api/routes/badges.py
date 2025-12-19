# api/routes/badges.py

"""Badges endpoint: small metadata for the UI header.

This route exposes lightweight "badge" information consumed by the UI, such as:

- **weather**: live weather snapshot (via the API `/serving/weather` endpoint),
- **freshness**: recency of the latest forecast bundle on storage
  (`.../serving/forecast/latest/.../forecast.json → generated_at`),
- a small `meta` block with:
  - `updated_at`: best timestamp available (forecast_generated_at or ts_utc),
  - `freshness_min`: age of the forecast in minutes.

The implementation is deliberately defensive:
- backend (local vs GCS) is selected via STORAGE_BACKEND,
- failures (file missing, GCS error, etc.) result in empty freshness,
- live weather failures result in empty weather,
- the route always returns a 200 with `None` for missing parts.
"""

from __future__ import annotations

import os
import json
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, List

import httpx
from fastapi import APIRouter, Response, HTTPException

# google-cloud-storage is optional: only used when STORAGE_BACKEND != "local"
try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    storage = None  # type: ignore

router = APIRouter(prefix="/serving/badges", tags=["serving-badges"])

# ───────────────────────────── Backend selection (local vs GCS) ─────────────────────────────

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "gcs").lower()
USE_LOCAL = STORAGE_BACKEND == "local"

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("DATA_ROOT", REPO_ROOT / "data"))

GCS_LOCAL_ROOT_PREFIX = (
    os.getenv(
        "GCS_LOCAL_ROOT_PREFIX",
        "gs://velib-forecast-472820_cloudbuild/velib/",
    ).rstrip("/")
    + "/"
)

# Forecast serving prefix (logical gs://... even in local mode)
# Example logical tree (latest-only):
#   <SERVING_FORECAST_PREFIX>/latest/h60/forecast.json
#   <SERVING_FORECAST_PREFIX>/latest/manifest.json
def _forecast_serving_prefix_or_500() -> str:
    raw = (
        os.environ.get("GCS_FORECAST_PREFIX", "")
        or os.environ.get("SERVING_FORECAST_PREFIX", "")
    )
    val = raw.strip().strip("'\"").rstrip("/")
    if not val.startswith("gs://"):
        raise HTTPException(status_code=500, detail="SERVING_FORECAST_PREFIX manquant ou invalide")
    return val


def _split_gs(uri: str) -> Tuple[str, str]:
    """Split a `gs://bucket/key` URI into `(bucket, key)`."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Bad GCS URI: {uri}")
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def _local_path_from_gs(uri: str) -> Path:
    """Map a velib gs:// URI to local DATA_ROOT path (STORAGE_BACKEND=local)."""
    if not uri.startswith(GCS_LOCAL_ROOT_PREFIX):
        raise FileNotFoundError(f"Local mapping failed for URI: {uri}")
    rel = uri[len(GCS_LOCAL_ROOT_PREFIX) :]
    return DATA_ROOT / rel.lstrip("/")


def _read_json_any(uri: str) -> Optional[dict]:
    """Read a JSON document from either local disk or GCS, depending on STORAGE_BACKEND.

    Returns None on any error.
    """
    if not uri:
        return None

    if USE_LOCAL:
        try:
            path = _local_path_from_gs(uri)
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    if storage is None or not uri.startswith("gs://"):
        return None
    try:
        bkt, key = _split_gs(uri)
        cli = storage.Client()
        blob = cli.bucket(bkt).blob(key)
        if not blob.exists():
            return None
        data = blob.download_as_bytes()
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def _json_sanitize(obj):
    """Replace NaN/±Inf with None recursively."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


# ───────────────────────────── Helpers: weather ─────────────────────────────

def _api_self_base() -> str:
    """Base URL of the current API for internal calls (local dev default)."""
    return (os.environ.get("API_SELF_BASE") or "http://127.0.0.1:8081").rstrip("/")


def _weather_from_live_api() -> Dict[str, Any]:
    """Fetch weather from the internal live endpoint `/serving/weather`."""
    url = os.environ.get("LIVE_WEATHER_URL") or "/serving/weather"
    if url.startswith("/"):
        url = f"{_api_self_base()}{url}"

    try:
        with httpx.Client(timeout=2.5) as cli:
            r = cli.get(url)
            r.raise_for_status()
            payload = r.json()
    except Exception:
        return {}

    if isinstance(payload, dict):
        out = {
            "ts_utc": payload.get("ts_utc"),
            "temp_C": payload.get("temp_C"),
            "precip_mm": payload.get("precip_mm"),
            "wind_mps": payload.get("wind_mps"),
        }
        # keep only if at least one metric exists
        if any(out.get(k) is not None for k in ("temp_C", "precip_mm", "wind_mps")):
            return _json_sanitize(out)
    return {}


# ───────────────────────────── Helpers: forecast freshness ─────────────────────────────

def _supported_horizons() -> List[int]:
    envs = os.environ.get("FORECAST_SUPPORTED", "15,60")
    try:
        vals = sorted({int(x.strip()) for x in envs.split(",") if x.strip()})
        return vals or [15]
    except Exception:
        return [15]


def _freshness_from_forecast(h: int) -> Dict[str, Any]:
    """Compute freshness from `.../latest/h{h}/forecast.json` (expects `generated_at`)."""
    base = _forecast_serving_prefix_or_500()
    uri = f"{base}/latest/h{int(h)}/forecast.json"
    doc = _read_json_any(uri)
    if not isinstance(doc, dict):
        return {}

    ts = doc.get("generated_at")
    if not ts:
        return {}

    try:
        # expected ISO8601 with Z
        ts_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        age_min = (now - ts_dt).total_seconds() / 60.0
        return {
            "forecast_generated_at": ts_dt.isoformat().replace("+00:00", "Z"),
            "age_minutes": round(age_min, 1),
            "h": int(h),
        }
    except Exception:
        return {}


# ───────────────────────────── Route ─────────────────────────────

@router.get("")
def get_badges(response: Response):
    """Return small "badge" metadata for the UI (weather + forecast freshness)."""
    # Weather (live)
    weather = _weather_from_live_api()

    # Freshness (pick the first supported horizon, prefer smallest)
    horizons = _supported_horizons()
    h0 = horizons[0] if horizons else 15
    freshness = _freshness_from_forecast(h0)

    updated_at = freshness.get("forecast_generated_at") or weather.get("ts_utc")
    freshness_min = freshness.get("age_minutes")

    # Badges should always reflect the latest state → no caching.
    response.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"

    return {
        "weather": weather or None,
        "freshness": freshness or None,
        "meta": {
            "updated_at": updated_at,
            "freshness_min": freshness_min,
        },
    }