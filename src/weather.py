# src/weather.py
from __future__ import annotations
import math
from functools import lru_cache
from typing import Dict, Any, List
import pandas as pd

LAT, LON = 48.8566, 2.3522  # Paris

def _make_session():
    import requests
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter

    s = requests.Session()
    retry = Retry(
        total=8, read=8, connect=8,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "velib-weather/1.1"})
    return s

def _get_json(url: str, timeout: int = 25) -> Dict[str, Any]:
    s = _make_session()
    r = s.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _floor_hour_naive(x):
    dt = pd.to_datetime(x, errors="coerce", utc=True)
    return dt.dt.floor("h").dt.tz_localize(None).astype("datetime64[ns]")

def _to_df(hourly: Dict[str, Any]) -> pd.DataFrame:
    if not hourly:
        return pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])
    rows: List[Dict[str, Any]] = []
    t  = hourly.get("time", []) or []
    tc = hourly.get("temperature_2m", []) or []
    pr = hourly.get("precipitation", []) or []
    ws = hourly.get("wind_speed_10m", []) or []
    for hh, a, b, c in zip(t, tc, pr, ws):
        rows.append({"hour_utc": hh, "temp_C": a, "precip_mm": b, "wind_mps": c})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["hour_utc"] = _floor_hour_naive(df["hour_utc"])
    for c in ["temp_C","precip_mm","wind_mps"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("hour_utc").sort_values("hour_utc").reset_index(drop=True)

@lru_cache(maxsize=8)
def _fetch_window_cached(start_floor_iso: str, end_floor_iso: str) -> pd.DataFrame:
    """Un seul appel Open-Meteo pour toute la fenêtre [start; end]. Cache par (start,end) arrondis à l'heure."""
    start = pd.Timestamp(start_floor_iso)
    end   = pd.Timestamp(end_floor_iso)

    # On élargit de ±1h pour les bords
    start_q = start - pd.Timedelta(hours=1)
    end_q   = end   + pd.Timedelta(hours=1)

    span_days = max(1, min(7, int(math.ceil((end_q - start_q).total_seconds()/86400)) + 1))
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m,precipitation,wind_speed_10m"
        "&windspeed_unit=ms&precipitation_unit=mm"
        f"&past_days={span_days}"
        "&timezone=UTC"
    )
    try:
        js = _get_json(url, timeout=25)
        df = _to_df(js.get("hourly") or {})
    except Exception as e:
        print(f"[weather] fetch_window error: {e}")
        return pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])

    # Filtre stricte -> [start; end]
    return df[(df["hour_utc"] >= start) & (df["hour_utc"] <= end)].reset_index(drop=True)

def fetch_history(start_ts, end_ts) -> pd.DataFrame:
    start = pd.to_datetime(start_ts, utc=True).floor("h").tz_convert(None)
    end   = pd.to_datetime(end_ts,   utc=True).floor("h").tz_convert(None)
    if pd.isna(start) or pd.isna(end) or start > end:
        return pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])
    return _fetch_window_cached(start.isoformat(), end.isoformat())

def fetch_forecast(start_ts, horizon_h: int = 36) -> pd.DataFrame:
    start = pd.to_datetime(start_ts, utc=True).floor("h").tz_convert(None)
    if pd.isna(start):
        return pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])
    end = (start + pd.Timedelta(hours=horizon_h)).floor("h")
    # NB: grâce au cache, si fetch_history vient d'appeler la même fenêtre, aucun nouvel HTTP
    return _fetch_window_cached(start.isoformat(), end.isoformat())
