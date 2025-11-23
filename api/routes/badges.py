# api/routes/badges.py

"""Badges endpoint: small metadata for the UI header.

This route exposes lightweight "badge" information consumed by the UI, such as:

- **weather**: live weather snapshot (via the API `/snapshot` endpoint),
- **freshness**: recency of the latest forecast bundle on GCS
  (`latest_forecast.json → generated_at`),
- a small `meta` block with:
  - `updated_at`: best timestamp available (forecast_generated_at or ts_utc),
  - `freshness_min`: age of the forecast in minutes.

The implementation is deliberately defensive:
- GCS failures result in empty freshness,
- live snapshot failures result in empty weather,
- the route always returns a 200 with `None` for missing parts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
import pandas as pd
from fastapi import APIRouter, Response

from core.settings import settings

try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    # Optional dependency: in local/dev without GCS, storage may be missing.
    storage = None  # type: ignore

router = APIRouter(prefix="/badges", tags=["badges"])


# ───────────────────────────── Helpers: config / URIs ─────────────────────────────
def _forecast_serving_prefix() -> str:
    """Return the forecast serving prefix from settings (normalized).

    Looks up both camelCase / snake_case variants for compatibility:
    - SERVING_FORECAST_PREFIX
    - serving_forecast_prefix (legacy)

    Returns an empty string if not configured.
    """
    p = getattr(settings, "SERVING_FORECAST_PREFIX", None) or getattr(settings, "serving_forecast_prefix", None)
    return (p or "").rstrip("/")


def _bundle_uri() -> str:
    """Build the GCS URI for the forecast bundle `latest_forecast.json`."""
    base = _forecast_serving_prefix()
    return f"{base}/latest_forecast.json"


# ───────────────────────────── Helpers: GCS read ─────────────────────────────
def _read_json_from_gcs(uri: str) -> Optional[dict]:
    """Read a JSON document from GCS and return it as a dict.

    Parameters
    ----------
    uri : str
        GCS URI, e.g. "gs://bucket/path/to/latest_forecast.json".

    Returns
    -------
    dict | None
        Parsed JSON dict, or None if:
        - `google-cloud-storage` is not available,
        - the URI is not a `gs://` URI,
        - the blob does not exist,
        - or any error occurs during download / parsing.
    """
    if storage is None or not uri.startswith("gs://"):
        return None
    try:
        bkt, key = uri[5:].split("/", 1)
        cli = storage.Client()
        blob = cli.bucket(bkt).get_blob(key)
        if not blob:
            return None
        data = blob.download_as_bytes()
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def _self_base() -> str:
    """Base URL of the current API, used to call internal endpoints.

    Reads `API_SELF_BASE` from settings, defaulting to `http://127.0.0.1:8081`
    for local development.
    """
    return (getattr(settings, "API_SELF_BASE", None) or "http://127.0.0.1:8081").rstrip("/")


# ───────────────────────────── Helpers: weather source ─────────────────────────────
def _weather_from_live_api() -> Dict[str, Any]:
    """Fetch weather information from the live snapshot API.

    Strategy
    --------
    1. Determine the snapshot URL:
       - use `LIVE_SNAPSHOT_URL` from settings, default `/snapshot`,
       - if it starts with `/`, prefix it with `_self_base()`.

    2. Perform a GET request and decode JSON.

    3. Try to extract weather data from several possible shapes:
       - `{"weather": {...}}` (direct weather object),
       - `{"data": [...]}` or `{"data": {...}}`,
       - raw list of records (use latest timestamp and average numeric fields).

    Returns
    -------
    dict
        A dict with at least the keys:
        - "ts_utc"   (or "tbin_utc" converted to ISO),
        - "temp_C",
        - "precip_mm",
        - "wind_mps",
        when these are available. Otherwise, `{}`.
    """
    url = getattr(settings, "LIVE_SNAPSHOT_URL", None) or "/snapshot"
    if url.startswith("/"):
        url = f"{_self_base()}{url}"

    try:
        with httpx.Client(timeout=2.5) as cli:
            r = cli.get(url)
            r.raise_for_status()
            payload = r.json()
    except Exception:
        return {}

    # Normalize potential payload shapes
    if isinstance(payload, dict) and "weather" in payload:
        payload = payload["weather"]
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
        payload = payload["data"]

    # Case 1: single dict
    if isinstance(payload, dict):
        out = {
            "ts_utc": payload.get("ts_utc") or payload.get("tbin_utc"),
            "temp_C": payload.get("temp_C"),
            "precip_mm": payload.get("precip_mm"),
            "wind_mps": payload.get("wind_mps"),
        }
        if any(k in payload for k in ("temp_C", "precip_mm", "wind_mps")):
            return out

    # Case 2: list of records → use latest timestamp and average metrics
    if isinstance(payload, list) and payload:
        try:
            df = pd.DataFrame(payload)
            tcol = "tbin_utc" if "tbin_utc" in df.columns else ("ts_utc" if "ts_utc" in df.columns else None)
            if tcol:
                df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
                tmax = df[tcol].max()
                if pd.isna(tmax):
                    return {}
                cut = df[df[tcol] == tmax]

                def m(col: str) -> Optional[float]:
                    if col not in cut.columns:
                        return None
                    v = pd.to_numeric(cut[col], errors="coerce").dropna()
                    return float(v.mean()) if not v.empty else None

                return {
                    "ts_utc": tmax.isoformat().replace("+00:00", "Z"),
                    "temp_C": m("temp_C"),
                    "precip_mm": m("precip_mm"),
                    "wind_mps": m("wind_mps"),
                }
        except Exception:
            return {}
    return {}


# ───────────────────────────── Helpers: forecast freshness ─────────────────────────────
def _freshness_from_forecast_bundle() -> Dict[str, Any]:
    """Compute freshness metadata from `latest_forecast.json`.

    Expects a JSON structure like:
        {
          "generated_at": "...Z",
          "horizons": [15, 60],
          "data": { ... }
        }

    Returns
    -------
    dict
        Example:
        {
          "forecast_generated_at": "2025-10-21T12:30:00Z",
          "age_minutes": 4.2
        }
        or `{}` if the bundle is missing / invalid.
    """
    uri = _bundle_uri()
    meta = _read_json_from_gcs(uri)
    if not isinstance(meta, dict):
        return {}

    ts = meta.get("generated_at")
    if not ts:
        return {}

    try:
        ts_dt = pd.to_datetime(ts, utc=True, errors="coerce")
        if pd.isna(ts_dt):
            return {}
        now = datetime.now(timezone.utc)
        age_min = (now - ts_dt.to_pydatetime()).total_seconds() / 60.0
        return {
            "forecast_generated_at": ts_dt.isoformat().replace("+00:00", "Z"),
            "age_minutes": round(age_min, 1),
        }
    except Exception:
        return {}


# ───────────────────────────── Route ─────────────────────────────
@router.get("")
def get_badges(response: Response):
    """Return small "badge" metadata for the UI (weather + freshness).

    Response payload structure:
    ---------------------------
    {
      "weather": {
        "ts_utc": "...",
        "temp_C": ...,
        "precip_mm": ...,
        "wind_mps": ...
      } | null,
      "freshness": {
        "forecast_generated_at": "...",
        "age_minutes": ...
      } | null,
      "meta": {
        "updated_at": "...",      # best timestamp (forecast or weather)
        "freshness_min": ...      # age_minutes if available
      }
    }

    Notes
    -----
    - The route always returns 200, with `weather` and/or `freshness`
      possibly set to `null` in case of upstream failures.
    - `Cache-Control` is enforced to `no-store` to keep the header real-time.
    """
    # Weather from live snapshot API
    weather = _weather_from_live_api()

    # Freshness from forecast bundle on GCS
    freshness = _freshness_from_forecast_bundle()

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
