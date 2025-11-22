"""Live weather fetcher for Vélib’ Forecast (Open-Meteo).

This module provides a tiny, self-contained helper to retrieve **current
weather conditions** from the Open-Meteo public API and expose them as a
simple Python `dict` with UTC timestamps:

    {
        "ts_utc": "2025-10-05T16:00:00Z",
        "temp_C": 17.4,
        "precip_mm": 0.1,
        "wind_mps": 3.2
    }

It is typically used by the API layer to enrich:
- live snapshots of the Vélib' network,
- UI endpoints that display "right-now" conditions in Paris.

Configuration is driven by environment variables:
- `OPENMETEO_URL`   (default: https://api.open-meteo.com/v1/forecast)
- `OPENMETEO_LAT`   (default: 48.8566)
- `OPENMETEO_LON`   (default: 2.3522)
- `OPENMETEO_TIMEOUT` (default: 5.0 seconds)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

# ---------- Config ----------
# Open-Meteo endpoint and coordinates for the "live" weather.
# All values can be overridden via environment variables for other cities.
OPEN_METEO_URL = os.getenv("OPENMETEO_URL", "https://api.open-meteo.com/v1/forecast")
LAT = float(os.getenv("OPENMETEO_LAT", "48.8566"))
LON = float(os.getenv("OPENMETEO_LON", "2.3522"))
TIMEOUT_S = float(os.getenv("OPENMETEO_TIMEOUT", "5.0"))


# ---------- Core fetch ----------
def fetch_live_weather() -> Dict[str, Any]:
    """
    Fetch current weather from Open-Meteo and return it as a flat dict.

    The request is made using the "current" API fields with:
    - UTC timezone,
    - temperature at 2m,
    - total precipitation,
    - wind speed at 10m in m/s.

    Returns
    -------
    dict
        A dictionary with the following keys (when the request succeeds):
        - "ts_utc": ISO8601 UTC timestamp string, e.g. "2025-10-05T16:00:00Z"
        - "temp_C": float | None, air temperature at 2m (°C)
        - "precip_mm": float | None, precipitation (mm)
        - "wind_mps": float | None, wind speed at 10m (m/s)

        On error, an empty dict `{}` is returned and a log is printed
        to stdout/stderr.

    Notes
    -----
    - Errors are deliberately swallowed (with a print) to avoid hard-failing
      API endpoints that consume this helper.
    - Callers must be prepared to handle missing keys / empty dict.
    """
    params = {
        "latitude": LAT,
        "longitude": LON,
        "current": "temperature_2m,precipitation,wind_speed_10m",
        "windspeed_unit": "ms",
        "timezone": "UTC",
    }
    try:
        # Using a short-lived httpx.Client for this one-off call. If this were
        # called many times per second, we could reuse a global client instead.
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
        # We keep this very defensive: upstream weather failures should not
        # break the whole application; callers will have to handle `{}`.
        print(f"[weather_live] fetch failed: {e}")
        return {}
