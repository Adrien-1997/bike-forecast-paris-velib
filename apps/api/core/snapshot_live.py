# apps/api/core/snapshot_live.py
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Tuple

import pandas as pd
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


# ---------- Config (surchargable via env) ----------
URL_STATUS = os.getenv(
    "VELIB_STATUS_URL",
    "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_status.json",
)
URL_INFO = os.getenv(
    "VELIB_INFO_URL",
    "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json",
)

OPEN_METEO = os.getenv("OPENMETEO_URL", "https://api.open-meteo.com/v1/forecast")
LAT = float(os.environ.get("OPENMETEO_LAT", "48.8566"))
LON = float(os.environ.get("OPENMETEO_LON", "2.3522"))
METEO_DISABLE = os.environ.get("METEO_DISABLE", "0") == "1"

COLS_ORDER = [
    "ts_utc","tbin_utc","station_id","bikes","capacity","mechanical","ebike",
    "status","lat","lon","name","temp_C","precip_mm","wind_mps"
]

# ---------- HTTP session with retries ----------
_session: requests.Session | None = None
def _session_get() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=0.4,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
            raise_on_status=False,
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.headers.update({"User-Agent": "velib-api-snapshot-live/1.0"})
        _session = s
    return _session

def _http_get_json(url: str, timeout: int = 20) -> dict:
    r = _session_get().get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ---------- GBFS fetch ----------
def _fetch_gbfs_velib_df() -> pd.DataFrame:
    js_status = _http_get_json(URL_STATUS)
    js_info   = _http_get_json(URL_INFO)

    status_list = (js_status.get("data") or {}).get("stations") or []
    info_list   = (js_info.get("data") or {}).get("stations") or []
    if not status_list or not info_list:
        print("[gbfs] empty payloads — status:", len(status_list), "info:", len(info_list))
        return pd.DataFrame(columns=[
            "ts_utc","tbin_utc","station_id","bikes","capacity","mechanical","ebike",
            "status","lat","lon","name"
        ])

    st_rows = []
    for s in status_list:
        sid = s.get("station_id")
        if not sid:
            continue
        # types can be list or dict; normalize
        v = s.get("num_bikes_available_types")
        types = (v[0] if isinstance(v, list) and v else (v if isinstance(v, dict) else {})) or {}
        mech = types.get("mechanical", 0) or 0
        ebi  = types.get("ebike", 0) or 0
        ts   = pd.to_datetime(s.get("last_reported"), unit="s", utc=True, errors="coerce")
        st_rows.append({
            "station_id": sid,
            "ts_utc": (ts.tz_convert(None) if ts is not pd.NaT else pd.NaT),
            "bikes": s.get("num_bikes_available"),
            "mechanical": mech,
            "ebike": ebi,
            "status": s.get("station_status") or ("OK" if (s.get("is_renting",1) and s.get("is_returning",1)) else "CLOSED"),
        })
    df_st = pd.DataFrame(st_rows)

    in_rows = []
    for i in info_list:
        sid = i.get("station_id")
        if not sid:
            continue
        in_rows.append({
            "station_id": sid,
            "name": i.get("name"),
            "lat": i.get("lat"),
            "lon": i.get("lon"),
            "capacity": i.get("capacity"),
        })
    df_in = pd.DataFrame(in_rows)

    df = df_st.merge(df_in, on="station_id", how="inner")
    if df.empty:
        print("[gbfs] merge empty — no matching station_id")
        return df

    df["tbin_utc"] = (
        pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
        .dt.floor("5min")
        .dt.tz_convert(None)
    )
    return df

# ---------- Weather fetch ----------
def _weather_window(df_velib: pd.DataFrame) -> Tuple[datetime, datetime]:
    hours = (
        pd.to_datetime(df_velib["tbin_utc"], utc=True, errors="coerce")
        .dt.floor("h")
        .dt.tz_convert(None)
    )
    lo = hours.min()
    hi = hours.max()
    if pd.isna(lo) or pd.isna(hi):
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        return now - timedelta(hours=3), now + timedelta(hours=3)
    lo = (lo - timedelta(hours=3)).replace(tzinfo=None)
    hi = (hi + timedelta(hours=3)).replace(tzinfo=None)
    return lo, hi

def _fetch_weather_df(df_velib: pd.DataFrame) -> pd.DataFrame:
    if METEO_DISABLE:
        print("[weather] disabled by METEO_DISABLE=1")
        return pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])

    try:
        lo, hi = _weather_window(df_velib)
        params = {
            "latitude": LAT,
            "longitude": LON,
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "windspeed_unit": "ms",
            "past_days": 2,
            "forecast_days": 2,
            "timezone": "UTC",
        }
        resp = _session_get().get(OPEN_METEO, params=params, timeout=20)
        resp.raise_for_status()
        j = resp.json()

        hours = pd.to_datetime(j["hourly"]["time"], utc=True, errors="coerce").tz_convert(None)
        dfw = pd.DataFrame({
            "hour_utc":  hours,
            "temp_C":    j["hourly"]["temperature_2m"],
            "precip_mm": j["hourly"]["precipitation"],
            "wind_mps":  j["hourly"]["wind_speed_10m"],
        })
        dfw = dfw[(dfw["hour_utc"] >= lo) & (dfw["hour_utc"] <= hi)].copy()
        return dfw
    except Exception as e:
        print(f"[weather] error: {e}")
        return pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])

# ---------- Public API ----------
def fetch_live_snapshot() -> pd.DataFrame:
    """
    Retourne un DataFrame "snapshot live" + météo, schéma:
      ts_utc, tbin_utc, station_id, bikes, capacity, mechanical, ebike,
      status, lat, lon, name, temp_C, precip_mm, wind_mps
    """
    df_v = _fetch_gbfs_velib_df()
    if df_v.empty:
        return pd.DataFrame(columns=COLS_ORDER)

    # Join with weather by hour
    df_v["hour_utc"] = (
        pd.to_datetime(df_v["tbin_utc"], utc=True, errors="coerce")
        .dt.floor("h")
        .dt.tz_convert(None)
    )
    df_w = _fetch_weather_df(df_v)

    if not df_w.empty:
        df = df_v.merge(df_w, on="hour_utc", how="left")
    else:
        df = df_v.assign(temp_C=None, precip_mm=None, wind_mps=None)

    df = df.drop(columns=["hour_utc"])

    out = pd.DataFrame({
        "ts_utc":     pd.to_datetime(df["ts_utc"],   utc=True, errors="coerce").dt.tz_convert(None),
        "tbin_utc":   pd.to_datetime(df["tbin_utc"], utc=True, errors="coerce").dt.tz_convert(None),
        "station_id": df["station_id"].astype(str),
        "bikes":      pd.to_numeric(df["bikes"],      errors="coerce").fillna(0).astype(int),
        "capacity":   pd.to_numeric(df["capacity"],   errors="coerce").fillna(0).astype(int),
        "mechanical": pd.to_numeric(df["mechanical"], errors="coerce").fillna(0).astype(int),
        "ebike":      pd.to_numeric(df["ebike"],      errors="coerce").fillna(0).astype(int),
        "status":     df["status"].astype(str),
        "lat":        pd.to_numeric(df["lat"],        errors="coerce"),
        "lon":        pd.to_numeric(df["lon"],        errors="coerce"),
        "name":       df["name"].astype(str),
        "temp_C":     pd.to_numeric(df["temp_C"],     errors="coerce"),
        "precip_mm":  pd.to_numeric(df["precip_mm"],  errors="coerce"),
        "wind_mps":   pd.to_numeric(df["wind_mps"],   errors="coerce"),
    })[COLS_ORDER]

    return out
