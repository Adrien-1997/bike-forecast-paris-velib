from __future__ import annotations

from fastapi import APIRouter, Response
from core.weather_live import fetch_live_weather

router = APIRouter(prefix="/weather", tags=["weather"])


@router.get("/live")
def get_weather_live(response: Response):
    """
    Return current weather from Open-Meteo.
    Example:
      {
        "ts_utc": "2025-10-05T16:00:00Z",
        "temp_C": 17.4,
        "precip_mm": 0.1,
        "wind_mps": 3.2
      }
    """
    data = fetch_live_weather()
    response.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
    return data
