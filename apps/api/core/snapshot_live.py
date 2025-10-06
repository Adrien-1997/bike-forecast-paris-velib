from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Any

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

COLS_ORDER = [
    "ts_utc", "tbin_utc", "station_id",
    "bikes", "capacity", "mechanical", "ebike",
    "status", "lat", "lon", "name"
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
        return pd.DataFrame(columns=COLS_ORDER)

    st_rows = []
    for s in status_list:
        sid = s.get("station_id")
        if not sid:
            continue
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

# ---------- Public API ----------
def fetch_live_snapshot() -> pd.DataFrame:
    """
    Retourne un DataFrame "snapshot live" (sans météo), schéma :
      ts_utc, tbin_utc, station_id, bikes, capacity, mechanical, ebike,
      status, lat, lon, name
    """
    df_v = _fetch_gbfs_velib_df()
    if df_v.empty:
        return pd.DataFrame(columns=COLS_ORDER)

    out = pd.DataFrame({
        "ts_utc":     pd.to_datetime(df_v["ts_utc"],   utc=True, errors="coerce").dt.tz_convert(None),
        "tbin_utc":   pd.to_datetime(df_v["tbin_utc"], utc=True, errors="coerce").dt.tz_convert(None),
        "station_id": df_v["station_id"].astype(str),
        "bikes":      pd.to_numeric(df_v["bikes"],      errors="coerce").fillna(0).astype(int),
        "capacity":   pd.to_numeric(df_v["capacity"],   errors="coerce").fillna(0).astype(int),
        "mechanical": pd.to_numeric(df_v["mechanical"], errors="coerce").fillna(0).astype(int),
        "ebike":      pd.to_numeric(df_v["ebike"],      errors="coerce").fillna(0).astype(int),
        "status":     df_v["status"].astype(str),
        "lat":        pd.to_numeric(df_v["lat"],        errors="coerce"),
        "lon":        pd.to_numeric(df_v["lon"],        errors="coerce"),
        "name":       df_v["name"].astype(str),
    })[COLS_ORDER]

    return out
