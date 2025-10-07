from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

# ---------- Config ----------
OPEN_METEO_URL = os.getenv("OPENMETEO_URL", "https://api.open-meteo.com/v1/forecast")
LAT = float(os.getenv("OPENMETEO_LAT", "48.8566"))
LON = float(os.getenv("OPENMETEO_LON", "2.3522"))
TIMEOUT_S = float(os.getenv("OPENMETEO_TIMEOUT", "5.0"))


# ---------- Core fetch ----------
def fetch_live_weather() -> Dict[str, Any]:
    """
    Fetch current weather from Open-Meteo (UTC timestamps).
    Returns a dict:
      {
        "ts_utc": "2025-10-05T16:00:00Z",
        "temp_C": 17.4,
        "precip_mm": 0.1,
        "wind_mps": 3.2
      }
    """
    params = {
        "latitude": LAT,
        "longitude": LON,
        "current": "temperature_2m,precipitation,wind_speed_10m",
        "windspeed_unit": "ms",
        "timezone": "UTC",
    }
    try:
        with httpx.Client(timeout=TIMEOUT_S) as cli:
            r = cli.get(OPEN_METEO_URL, params=params)
            r.raise_for_status()
            j = r.json()
        cur = j.get("current") or {}
        return {
            "ts_utc": cur.get("time"),
            "temp_C": cur.get("temperature_2m"),
            "precip_mm": cur.get("precipitation"),
            "wind_mps": cur.get("wind_speed_10m"),
        }
    except Exception as e:
        print(f"[weather_live] fetch failed: {e}")
        return {}
