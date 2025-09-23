# src/ingest.py
import os
from typing import Optional, List, Dict, Any
import pandas as pd

# -----------------------------
# HTTP utils
# -----------------------------
def _make_session():
    import requests
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter

    s = requests.Session()
    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "velib-ingest/1.0 (+github.com/Adrien-1997)"})
    return s

def _http_get(url: str, timeout: int = 20):
    s = _make_session()
    verify_ssl = os.environ.get("NO_SSL_VERIFY", "0") != "1"
    r = s.get(url, timeout=timeout, verify=verify_ssl)
    r.raise_for_status()
    return r

# -----------------------------
# Fetch data
# -----------------------------
def _fetch_from_client() -> Optional[pd.DataFrame]:
    try:
        from src.velib_client import fetch_snapshot_df  # type: ignore
        df = fetch_snapshot_df()
        if df is not None and not df.empty:
            return _coerce_schema(df)
    except Exception as e:
        print(f"[ingest] src.velib_client fallback (reason: {e})")
    return None

def _fetch_from_gbfs() -> pd.DataFrame:
    URL_STATUS = "https://velib-metropole-opendata.smoove.pro/opendata/Velib_Metropole/station_status.json"
    URL_INFO   = "https://velib-metropole-opendata.smoove.pro/opendata/Velib_Metropole/station_information.json"

    rs = _http_get(URL_STATUS, timeout=20)
    ri = _http_get(URL_INFO,   timeout=20)

    js_status = rs.json()
    js_info   = ri.json()

    status_list = (js_status.get("data") or {}).get("stations") or []
    info_list   = (js_info.get("data")   or {}).get("stations") or []
    if not status_list or not info_list:
        raise RuntimeError("GBFS returned empty lists")

    st_rows: List[Dict[str, Any]] = []
    for s in status_list:
        sid = s.get("station_id")
        if not sid:
            continue
        types = s.get("num_bikes_available_types") or {}
        mech  = types.get("mechanical") or types.get("classic") or 0
        ebi   = types.get("ebike") or types.get("ebikes") or 0
        ts    = pd.to_datetime(s.get("last_reported"), unit="s", utc=True, errors="coerce")
        st_rows.append({
            "station_id": sid,
            "ts_utc": ts.tz_localize(None) if ts is not pd.NaT else pd.NaT,
            "numbikesavailable": s.get("num_bikes_available"),
            "numdocksavailable": s.get("num_docks_available"),
            "mechanical": mech,
            "ebike": ebi,
        })
    df_st = pd.DataFrame(st_rows)

    in_rows: List[Dict[str, Any]] = []
    for i in info_list:
        sid = i.get("station_id")
        if not sid:
            continue
        in_rows.append({
            "station_id": sid,
            "stationcode": i.get("station_code") or sid,
            "name": i.get("name"),
            "lat": i.get("lat"),
            "lon": i.get("lon"),
            "capacity": i.get("capacity"),
        })
    df_in = pd.DataFrame(in_rows)

    df = df_st.merge(df_in, on="station_id", how="inner")
    return _coerce_schema(df)

def _fetch_from_opendata_v1() -> pd.DataFrame:
    URL = (
        "https://opendata.paris.fr/api/records/1.0/search/"
        "?dataset=velib-disponibilite-en-temps-reel&rows=2000"
    )
    r = _http_get(URL, timeout=25)
    j = r.json()
    recs = j.get("records") or []
    rows: List[Dict[str, Any]] = []
    for rec in recs:
        f = rec.get("fields") or {}
        coord = f.get("coordonnees_geo") or f.get("coord")
        lat = lon = None
        if isinstance(coord, dict):
            lat, lon = coord.get("lat"), coord.get("lon")
        elif isinstance(coord, (list, tuple)) and len(coord) == 2:
            lat, lon = coord[0], coord[1]
        ts = pd.to_datetime(rec.get("record_timestamp"), utc=True, errors="coerce")
        rows.append({
            "ts_utc": ts.tz_localize(None) if ts is not pd.NaT else pd.NaT,
            "stationcode": f.get("stationcode") or f.get("station_id"),
            "name": f.get("name"),
            "lat": lat,
            "lon": lon,
            "numbikesavailable": f.get("numbikesavailable"),
            "numdocksavailable": f.get("numdocksavailable"),
            "capacity": f.get("capacity"),
            "mechanical": f.get("mechanical"),
            "ebike": f.get("ebike"),
        })
    return _coerce_schema(pd.DataFrame(rows))

def _synthetic_snapshot(n: int = 10) -> pd.DataFrame:
    import numpy as np
    now = pd.Timestamp.utcnow().replace(minute=0, second=0, microsecond=0)
    rows = []
    rng = np.random.default_rng(42)
    for k in range(n):
        cap = rng.integers(20, 45)
        bikes = rng.integers(0, cap)
        rows.append({
            "ts_utc": now,
            "stationcode": f"000{k:03d}",
            "name": f"Station {k:03d}",
            "lat": 48.85 + rng.normal(0, 0.01),
            "lon": 2.35 + rng.normal(0, 0.01),
            "numbikesavailable": int(bikes),
            "numdocksavailable": int(cap - bikes),
            "capacity": int(cap),
            "mechanical": int(rng.integers(0, bikes)),
            "ebike": int(max(0, bikes - rng.integers(0, bikes))),
        })
    return pd.DataFrame(rows)

# -----------------------------
# Normalisation
# -----------------------------
def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "timestamp": "ts_utc",
        "ts": "ts_utc",
        "bike_available": "numbikesavailable",
        "dock_available": "numdocksavailable",
        "station_id": "stationcode",
    }
    df = df.rename(columns=rename_map).copy()

    needed = [
        "ts_utc","stationcode","name","lat","lon",
        "numbikesavailable","numdocksavailable","capacity","mechanical","ebike",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = pd.NA

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True).dt.tz_localize(None)
    for c in ["numbikesavailable","numdocksavailable","capacity","mechanical","ebike"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["lat","lon"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["stationcode","ts_utc"])
    return df[needed].reset_index(drop=True)

# -----------------------------
# API principale
# -----------------------------
def fetch_snapshot() -> pd.DataFrame:
    df = _fetch_from_client()
    if df is not None and not df.empty:
        return df
    try:
        return _fetch_from_gbfs()
    except Exception as e:
        print(f"[ingest] GBFS failed: {e}")
    try:
        return _fetch_from_opendata_v1()
    except Exception as e:
        print(f"[ingest] Opendata v1 failed: {e}")
    print("[ingest] Network unavailable → using synthetic snapshot")
    return _synthetic_snapshot(30)

def ingest_once() -> pd.DataFrame:
    df = fetch_snapshot()
    if df is None or df.empty:
        print("[ingest] no data")
        return pd.DataFrame()
    return df

def main(return_df: bool = False) -> pd.DataFrame:
    df = ingest_once()
    print(f"[ingest] rows fetched: {len(df)}")

    # --- Handoff parquet pour aggregate (staging, ne touche pas à ta BDD principale) ---
    STAGING_PATH = "exports/staging_ingest.parquet"
    # Pas de mkdir: on suppose que 'exports/' existe dans le repo/Docker image
    try:
        df.to_parquet(STAGING_PATH, index=False)
        print(f"[ingest] saved -> {STAGING_PATH}")
    except FileNotFoundError:
        print("[ingest][FATAL] 'exports/' folder is missing; create it in the repo.")
        raise

    return df if return_df else df

if __name__ == "__main__":
    main()
