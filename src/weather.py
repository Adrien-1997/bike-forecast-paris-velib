# src/weather.py
from __future__ import annotations
import math
from typing import Optional, Dict, Any, List
import pandas as pd

LAT, LON = 48.8566, 2.3522  # Paris

# ---------- HTTP avec retries ----------
def _make_session():
    import requests
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter

    s = requests.Session()
    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "velib-weather/1.0"})
    return s

def _get_json(url: str, timeout: int = 25) -> Dict[str, Any]:
    s = _make_session()
    r = s.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ---------- Normalisation ----------
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
    # conversions numériques
    for c in ["temp_C","precip_mm","wind_mps"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # dédupli / tri
    df = df.drop_duplicates("hour_utc").sort_values("hour_utc")
    return df

# ---------- Fetchers ----------
def fetch_history(start_ts, end_ts) -> pd.DataFrame:
    """
    Historique récent via Open-Meteo.
    Élargit la fenêtre [-1h ; +1h] pour éviter les bords manqués puis re-filtre.
    """
    start = pd.to_datetime(start_ts, utc=True)
    end   = pd.to_datetime(end_ts,   utc=True)
    if pd.isna(start) or pd.isna(end) or start > end:
        return pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])

    start_q = (start - pd.Timedelta(hours=1)).tz_convert(None)
    end_q   = (end   + pd.Timedelta(hours=1)).tz_convert(None)

    # past_days max 7 — calcule au plus large mais on re-filtrera ensuite
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
        print(f"[weather] fetch_history error: {e}")
        return pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])

    # filtre stricte à la vraie fenêtre
    df = df[(df["hour_utc"] >= start.tz_convert(None).floor("h")) &
            (df["hour_utc"] <= end.tz_convert(None).floor("h"))]
    return df.reset_index(drop=True)

def fetch_forecast(start_ts, horizon_h: int = 36) -> pd.DataFrame:
    """
    Prévision horaire à partir de start_ts pour horizon_h heures.
    Élargit la fenêtre [+1h] en aval pour capturer l'heure suivante.
    """
    start = pd.to_datetime(start_ts, utc=True)
    if pd.isna(start):
        return pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])

    end = start + pd.Timedelta(hours=horizon_h + 1)

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m,precipitation,wind_speed_10m"
        "&windspeed_unit=ms&precipitation_unit=mm"
        "&timezone=UTC"
    )
    try:
        js = _get_json(url, timeout=25)
        df = _to_df(js.get("hourly") or {})
    except Exception as e:
        print(f"[weather] fetch_forecast error: {e}")
        return pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])

    df = df[(df["hour_utc"] >  start.tz_convert(None).floor("h")) &
            (df["hour_utc"] <= end.tz_convert(None).floor("h"))]
    return df.reset_index(drop=True)
