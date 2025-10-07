# apps/api/routes/badges.py
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
    storage = None  # type: ignore

router = APIRouter(prefix="/badges", tags=["badges"])


def _forecast_serving_prefix() -> str:
    p = getattr(settings, "SERVING_FORECAST_PREFIX", None) or getattr(settings, "serving_forecast_prefix", None)
    return (p or "").rstrip("/")


def _bundle_uri() -> str:
    base = _forecast_serving_prefix()
    return f"{base}/latest_forecast.json"


def _read_json_from_gcs(uri: str) -> Optional[dict]:
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
    return (getattr(settings, "API_SELF_BASE", None) or "http://127.0.0.1:8081").rstrip("/")


def _weather_from_live_api() -> Dict[str, Any]:
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

    if isinstance(payload, dict) and "weather" in payload:
        payload = payload["weather"]
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
        payload = payload["data"]

    if isinstance(payload, dict):
        out = {
            "ts_utc": payload.get("ts_utc") or payload.get("tbin_utc"),
            "temp_C": payload.get("temp_C"),
            "precip_mm": payload.get("precip_mm"),
            "wind_mps": payload.get("wind_mps"),
        }
        if any(k in payload for k in ("temp_C", "precip_mm", "wind_mps")):
            return out

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


def _freshness_from_forecast_bundle() -> Dict[str, Any]:
    """
    Freshness basée sur latest_forecast.json:
      { "generated_at": "...Z", "horizons": [...], "data": {...} }
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


@router.get("")
def get_badges(response: Response):
    """
    - weather: from LIVE snapshot API
    - freshness: from forecast bundle latest_forecast.json (generated_at)
    """
    weather = _weather_from_live_api()
    freshness = _freshness_from_forecast_bundle()

    updated_at = freshness.get("forecast_generated_at") or weather.get("ts_utc")
    freshness_min = freshness.get("age_minutes")

    response.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"

    return {
        "weather": weather or None,
        "freshness": freshness or None,
        "meta": {
            "updated_at": updated_at,
            "freshness_min": freshness_min,
        },
    }
