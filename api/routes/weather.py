# api/routes/stations.py

"""Weather endpoints for Vélib’ Forecast.

This router exposes lightweight weather-related endpoints.

Currently:
- `/weather/live`:
    - fetches current weather from Open-Meteo via `core.weather_live.fetch_live_weather`,
    - returns a small JSON payload with:
        * `ts_utc`    (ISO8601 UTC timestamp),
        * `temp_C`    (°C),
        * `precip_mm` (mm),
        * `wind_mps`  (m/s),
    - sets `Cache-Control: no-store` to avoid client-side caching and keep
      the value as real-time as possible.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from core.weather_live import fetch_live_weather

router = APIRouter(prefix="/weather", tags=["weather"])


@router.get("/live")
def get_weather_live(response: Response):
    """Return current weather from Open-Meteo (no caching).

    Response example
    ----------------
    {
      "ts_utc": "2025-10-05T16:00:00Z",
      "temp_C": 17.4,
      "precip_mm": 0.1,
      "wind_mps": 3.2
    }

    Notes
    -----
    - The underlying call is delegated to `fetch_live_weather()`.
    - The `Cache-Control` header is set to `no-store` so that browsers and
      intermediate proxies do not cache the response.
    """
    data = fetch_live_weather()
    response.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
    return data
