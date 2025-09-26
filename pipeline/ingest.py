# pipeline/ingest.py
# Ingest Vélib' + météo (Open-Meteo), écrit un snapshot Parquet
# - local (staging) si INGEST_SAVE_PARQUET=1
# - et/ou GCS si INGEST_TO_GCS=1 (GCS_RAW_PREFIX=gs://.../velib/bronze)
# Option: append dans DuckDB locale si APPEND_DB=1 et DB_LOCAL existe.
#
# Exécution: python -m pipeline.ingest

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple

import pandas as pd
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Optional dependency; DB append skipped if not present / DB file missing
try:
    import duckdb  # type: ignore
except Exception:  # pragma: no cover
    duckdb = None  # type: ignore

# Optional: GCS upload
try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    storage = None  # type: ignore

# --------------------------
# Config
# --------------------------
URL_STATUS = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_status.json"
URL_INFO   = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json"

BASE_METEO = "https://api.open-meteo.com/v1/forecast"
LAT, LON = 48.8566, 2.3522  # Paris centre

DATA_DIR = Path("data")
STAGING_DIR = DATA_DIR / "staging"

# --------------------------
# HTTP utils
# --------------------------
def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "velib-hf-pipeline/1.0"})
    return s

def _http_get(url: str, timeout: int = 20) -> dict:
    verify_ssl = os.environ.get("NO_SSL_VERIFY", "0") != "1"
    s = _make_session()
    r = s.get(url, timeout=timeout, verify=verify_ssl)
    r.raise_for_status()
    return r.json()

# --------------------------
# Helpers
# --------------------------
def _floor_to_5min(ts_utc: datetime) -> datetime:
    minute = (ts_utc.minute // 5) * 5
    return ts_utc.replace(minute=minute, second=0, microsecond=0)

def _upload_parquet_gcs(df: pd.DataFrame, gcs_url: str):
    if storage is None:
        raise RuntimeError("google-cloud-storage not installed")
    assert gcs_url.startswith("gs://"), "gcs_url must start with gs://"
    bkt, key = gcs_url[5:].split("/", 1)
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    storage.Client().bucket(bkt).blob(key).upload_from_file(
        buf, content_type="application/octet-stream"
    )

# --------------------------
# Fetch Vélib (status + info)
# --------------------------
def fetch_velib() -> pd.DataFrame:
    js_status = _http_get(URL_STATUS)
    js_info   = _http_get(URL_INFO)

    status_list = (js_status.get("data") or {}).get("stations") or []
    info_list   = (js_info.get("data") or {}).get("stations") or []

    if not status_list or not info_list:
        print("[ingest] GBFS empty → empty snapshot")
        return pd.DataFrame()

    # status
    st_rows = []
    for s in status_list:
        sid = s.get("station_id")
        if not sid:
            continue

        # dict or list-of-dict depending on payload
        types = {}
        v = s.get("num_bikes_available_types")
        if isinstance(v, list) and v:
            types = v[0] or {}
        elif isinstance(v, dict):
            types = v

        mech = types.get("mechanical", 0) or 0
        ebi  = types.get("ebike", 0) or 0
        ts   = pd.to_datetime(s.get("last_reported"), unit="s", utc=True, errors="coerce")

        st_rows.append({
            "station_id": sid,
            "ts_utc": ts.tz_localize(None) if ts is not pd.NaT else pd.NaT,
            "numbikesavailable": s.get("num_bikes_available"),
            "numdocksavailable": s.get("num_docks_available"),
            "mechanical": mech,
            "ebike": ebi,
            "status": s.get("station_status") or ("OK" if (s.get("is_renting", 1) and s.get("is_returning", 1)) else "CLOSED"),
        })

    df_st = pd.DataFrame(st_rows)

    # info
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

    # merge
    df = df_st.merge(df_in, on="station_id", how="inner")

    # normalize
    if not df.empty:
        df["tbin_utc"] = pd.to_datetime(df["ts_utc"]).dt.floor("5min")

    return df

# --------------------------
# Fetch météo (UTC hourly) → m/s
# --------------------------
def fetch_weather() -> pd.DataFrame:
    try:
        url = (
            f"{BASE_METEO}?latitude={LAT}&longitude={LON}"
            "&hourly=temperature_2m,precipitation,wind_speed_10m"
            "&windspeed_unit=ms&past_days=1&forecast_days=1&timezone=UTC"
        )
        j = _http_get(url, timeout=20)
    except Exception as e:
        print(f"[weather] fetch failed: {e}")
        return pd.DataFrame(columns=["hour_utc", "temp_C", "precip_mm", "wind_mps", "weather_src"])

    hours = pd.to_datetime(j["hourly"]["time"], utc=True, errors="coerce")
    df = pd.DataFrame({
        "hour_utc": hours,
        "temp_C": j["hourly"]["temperature_2m"],
        "precip_mm": j["hourly"]["precipitation"],
        "wind_mps": j["hourly"]["wind_speed_10m"],
        "weather_src": ["open-meteo"] * len(hours),
    })
    return df

# --------------------------
# Consolidation (Parquet)
# --------------------------
def consolidate_snapshot() -> pd.DataFrame:
    df_velib = fetch_velib()
    df_weather = fetch_weather()

    if df_velib.empty:
        return df_velib

    # join key = hour_utc (UTC naive)
    df_velib["hour_utc"] = (
        pd.to_datetime(df_velib["tbin_utc"], utc=True, errors="coerce")
        .dt.floor("h")
        .dt.tz_convert(None)
    )
    if not df_weather.empty:
        df_weather["hour_utc"] = (
            pd.to_datetime(df_weather["hour_utc"], utc=True, errors="coerce")
            .dt.tz_convert(None)
        )

    df = df_velib.merge(df_weather, on="hour_utc", how="left")
    df = df.drop(columns=["hour_utc"])
    return df

# --------------------------
# API principale
# --------------------------

def ingest_once(save: bool = True) -> pd.DataFrame:
    print(f"[ingest][cfg] TO_GCS={os.environ.get('INGEST_TO_GCS')} RAW_PREFIX={os.environ.get('GCS_RAW_PREFIX')}", flush=True)
    df = consolidate_snapshot()

    # bin courant (UTC)
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    tbin = (now.minute // 5) * 5
    tbin_dt = now.replace(minute=tbin)
    fname = tbin_dt.strftime("%Y-%m-%dT%H-%M.parquet")

    # 1) Parquet local (optionnel)
    if save:
        staging_dir = STAGING_DIR / tbin_dt.strftime("%Y-%m-%d")
        staging_dir.mkdir(parents=True, exist_ok=True)
        out_path = staging_dir / fname
        df.to_parquet(out_path, index=False)
        print(f"[ingest] saved {len(df)} rows → {out_path}")

    # 2) Parquet GCS (recommandé en prod)
    to_gcs = os.environ.get("INGEST_TO_GCS", "0") == "1"
    raw_prefix = os.environ.get("GCS_RAW_PREFIX")
    if to_gcs:
        if not raw_prefix:
            raise RuntimeError("INGEST_TO_GCS=1 but GCS_RAW_PREFIX is not set")
        day = tbin_dt.strftime("%Y-%m-%d")
        hour = tbin_dt.strftime("%H")
        gcs_path = f"{raw_prefix}/date={day}/hour={hour}/{fname}"
        _upload_parquet_gcs(df, gcs_path)
        print(f"[ingest] uploaded {len(df)} rows → {gcs_path}")

    # 3) Append DB locale (facultatif, pour runs locaux)
    try:
        if os.environ.get("APPEND_DB", "0") != "1":
            return df
        if duckdb is None:
            raise RuntimeError("duckdb not installed")
        db_local = os.environ.get("DB_LOCAL", "velib.duckdb")
        if not os.path.exists(db_local):
            print(f"[ingest][warn] DB not found at {db_local}, skipping DB append")
            return df

        df_db = df.copy()
        df_db.rename(columns={"numbikesavailable": "bikes"}, inplace=True)
        for c in ["bikes", "capacity", "mechanical", "ebike"]:
            if c not in df_db.columns: df_db[c] = None
        if "status" not in df_db.columns: df_db["status"] = "OK"

        df_db["tbin_utc"] = pd.to_datetime(df_db.get("tbin_utc"), utc=True, errors="coerce").dt.tz_convert(None)
        df_db["ts_utc"]   = pd.to_datetime(df_db.get("ts_utc"),   utc=True, errors="coerce").dt.tz_convert(None)
        df_db.loc[df_db["ts_utc"].isna(), "ts_utc"] = df_db["tbin_utc"]
        df_db["ts_paris"] = pd.to_datetime(df_db["tbin_utc"], utc=True, errors="coerce").dt.tz_convert("Europe/Paris").dt.tz_localize(None)

        df_db["lat"] = pd.to_numeric(df_db.get("lat"), errors="coerce")
        df_db["lon"] = pd.to_numeric(df_db.get("lon"), errors="coerce")

        now_utc = datetime.now(timezone.utc).replace(second=0, microsecond=0).replace(tzinfo=None)
        df_db["ingested_at"] = now_utc
        df_db["ingest_latency_s"] = (now_utc - pd.to_datetime(df_db["ts_utc"])).dt.total_seconds()
        df_db["source_etag"] = None

        keep = [
            "ts_utc","ts_paris","tbin_utc","station_id",
            "bikes","capacity","mechanical","ebike","status",
            "lat","lon","ingested_at","ingest_latency_s","source_etag"
        ]
        for c in keep:
            if c not in df_db.columns: df_db[c] = None
        df_db = df_db[keep]

        con = duckdb.connect(db_local)
        con.execute("PRAGMA threads=4;")
        con.register("snap_raw", df_db)
        con.execute("""
            INSERT INTO bronze.raw_snapshots_5min (
                ts_utc, ts_paris, tbin_utc, station_id,
                bikes, capacity, mechanical, ebike, status,
                lat, lon, ingested_at, ingest_latency_s, source_etag
            )
            SELECT
                ts_utc,
                ts_paris,
                tbin_utc,
                TRY_CAST(station_id AS BIGINT),
                CAST(bikes AS INTEGER),
                CAST(capacity AS INTEGER),
                CAST(mechanical AS INTEGER),
                CAST(ebike AS INTEGER),
                status,
                CAST(lat AS DOUBLE),
                CAST(lon AS DOUBLE),
                ingested_at,
                CAST(ingest_latency_s AS DOUBLE),
                source_etag
            FROM snap_raw
        """)
        # météo horaire
        w = fetch_weather()
        if not w.empty:
            w["tbin_utc"] = pd.to_datetime(w["hour_utc"], utc=True, errors="coerce").dt.tz_convert(None)
            w = w[["tbin_utc","temp_C","wind_mps","precip_mm","weather_src"]].dropna(subset=["tbin_utc"]).drop_duplicates("tbin_utc")
            con.register("wcur", w)
            con.execute("DELETE FROM bronze.weather_5min WHERE tbin_utc IN (SELECT tbin_utc FROM wcur)")
            con.execute("INSERT INTO bronze.weather_5min SELECT * FROM wcur")
        con.close()
        print("[ingest] DB append OK → bronze.raw_snapshots_5min (+ weather)")
    except Exception as e:
        print(f"[ingest][warn] DB append skipped: {e}")

    return df

if __name__ == "__main__":
    # En prod (Cloud Run): mettre INGEST_SAVE_PARQUET=0, INGEST_TO_GCS=1
    save_local = os.environ.get("INGEST_SAVE_PARQUET", "1") == "1"
    ingest_once(save=save_local)
