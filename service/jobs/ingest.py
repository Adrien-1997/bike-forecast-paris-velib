# service/jobs/ingest.py
# Ingestion 5 minutes : GBFS (status+info → name) + météo (Open-Meteo)
#
# Schéma strict écrit (timestamps internes en UTC naïf) :
# ts_utc, tbin_utc, station_id, bikes, capacity, mechanical, ebike, status,
# lat, lon, name, temp_C, precip_mm, wind_mps
#
# Chemin et nom de fichier en UTC :
#   data_local/raw/date=YYYY-MM-DD/hour=HH/YYYY-MM-DDT%H-%M.parquet
#   gs://.../bronze/date=YYYY-MM-DD/hour=HH/YYYY-MM-DDT%H-%M.parquet
#
# Env utiles :
#   INGEST_SAVE_PARQUET=1|0     (défaut 1)
#   LOCAL_RAW_DIR=...           (défaut data_local/raw)
#   INGEST_TO_GCS=1|0           (défaut 0)
#   GCS_RAW_PREFIX=gs://.../bronze
#   OPENMETEO_LAT, OPENMETEO_LON
#   METEO_DISABLE=1|0
#   DIAG=1 pour logs détaillés
#
# Exécution :
#   python -m jobs.ingest

from __future__ import annotations
import os
from io import BytesIO
from datetime import datetime, timezone, timedelta
from typing import Tuple

import pandas as pd
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

try:
    from google.cloud import storage  # type: ignore
except Exception:
    storage = None  # type: ignore

# ───────────────────────── Config ─────────────────────────

URL_STATUS = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_status.json"
URL_INFO   = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

LAT = float(os.environ.get("OPENMETEO_LAT", "48.8566"))
LON = float(os.environ.get("OPENMETEO_LON", "2.3522"))
METEO_DISABLE = os.environ.get("METEO_DISABLE", "0") == "1"
DIAG = os.environ.get("DIAG", "0") in ("1", "true", "True")

COLS_ORDER = [
    "ts_utc","tbin_utc","station_id","bikes","capacity","mechanical","ebike",
    "status","lat","lon","name","temp_C","precip_mm","wind_mps"
]

# ───────────────────── HTTP session (retries) ─────────────────────

_session: requests.Session | None = None
def _session_get() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(
            total=5, connect=5, read=5,
            backoff_factor=0.4,
            status_forcelist=[429,500,502,503,504],
            allowed_methods=["GET","HEAD"],
            raise_on_status=False,
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.headers.update({"User-Agent": "velib-pipeline/1.1"})
        _session = s
    return _session

def _http_get_json(url: str, timeout: int = 20) -> dict:
    r = _session_get().get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ───────────────────────── GBFS ─────────────────────────

def fetch_velib_df() -> pd.DataFrame:
    """Lit status+info et force l’horodatage au *moment UTC du job* (ts_utc=now_utc)."""
    js_status = _http_get_json(URL_STATUS)
    js_info   = _http_get_json(URL_INFO)

    status_list = (js_status.get("data") or {}).get("stations") or []
    info_list   = (js_info.get("data") or {}).get("stations") or []
    if not status_list or not info_list:
        print("[ingest][gbfs] empty payloads — status:", len(status_list), "info:", len(info_list))
        return pd.DataFrame(columns=COLS_ORDER)

    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    tbin_now = pd.Timestamp(now_utc).floor("5min").to_pydatetime()

    # STATUS
    st_rows = []
    for s in status_list:
        sid = s.get("station_id")
        if not sid:
            continue
        types = {}
        v = s.get("num_bikes_available_types")
        if isinstance(v, list) and v:
            for d in v:
                if isinstance(d, dict):
                    types.update(d)
        elif isinstance(v, dict):
            types = v

        st_rows.append({
            "station_id": sid,
            "ts_utc": now_utc,
            "bikes": s.get("num_bikes_available"),
            "mechanical": types.get("mechanical", 0),
            "ebike": types.get("ebike", 0),
            "status": s.get("station_status") or (
                "OK" if (s.get("is_renting",1) and s.get("is_returning",1)) else "CLOSED"
            ),
        })
    df_st = pd.DataFrame(st_rows)

    # INFO
    in_rows = []
    for i in info_list:
        sid = i.get("station_id")
        if not sid:
            continue
        in_rows.append({
            "station_id": sid,
            "name": i.get("name"),
            "lat":  i.get("lat"),
            "lon":  i.get("lon"),
            "capacity": i.get("capacity"),
        })
    df_in = pd.DataFrame(in_rows)

    df = df_st.merge(df_in, on="station_id", how="inner")
    df["ts_utc"]   = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce").dt.tz_localize(None)
    df["tbin_utc"] = pd.to_datetime(tbin_now)

    for c in ("bikes","mechanical","ebike","capacity","lat","lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"[ingest][gbfs] rows={len(df):,} tbin_utc={tbin_now} (UTC)")
    if DIAG:
        print(df[["station_id","ts_utc","tbin_utc","bikes","capacity"]].head(5).to_string(index=False))
    return df

# ───────────────────────── Météo ─────────────────────────

def _weather_window(df_velib: pd.DataFrame) -> Tuple[datetime, datetime]:
    """Fenêtre ±3h autour du bin courant (UTC naïf)."""
    hours = pd.to_datetime(df_velib["tbin_utc"], errors="coerce").dt.floor("h")
    lo, hi = hours.min(), hours.max()
    if pd.isna(lo) or pd.isna(hi):
        now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        return now - timedelta(hours=3), now + timedelta(hours=3)
    return lo - timedelta(hours=3), hi + timedelta(hours=3)

def fetch_weather_df(df_velib: pd.DataFrame) -> pd.DataFrame:
    if METEO_DISABLE:
        print("[ingest][weather] disabled by METEO_DISABLE=1")
        return pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])
    try:
        lo, hi = _weather_window(df_velib)
        params = {
            "latitude": LAT,
            "longitude": LON,
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "windspeed_unit": "ms",
            "past_days": 2, "forecast_days": 2,
            "timezone": "UTC",
        }
        j = _session_get().get(OPEN_METEO, params=params, timeout=20)
        j.raise_for_status()
        j = j.json()

        hours = pd.to_datetime(j["hourly"]["time"], utc=True, errors="coerce").tz_convert(None)
        dfw = pd.DataFrame({
            "hour_utc":  hours,
            "temp_C":    j["hourly"]["temperature_2m"],
            "precip_mm": j["hourly"]["precipitation"],
            "wind_mps":  j["hourly"]["wind_speed_10m"],
        })
        dfw = dfw[(dfw["hour_utc"] >= lo) & (dfw["hour_utc"] <= hi)].copy()
        print(f"[ingest][weather] got={len(j['hourly']['time']):,} kept={len(dfw):,} window={lo}..{hi} (UTC)")
        return dfw
    except Exception as e:
        print(f"[ingest][weather] error: {e}")
        return pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])

# ─────────────────────── GCS helpers ───────────────────────

def _split_gcs(gcs_url: str) -> tuple[str,str]:
    assert gcs_url.startswith("gs://")
    b, p = gcs_url[5:].split("/", 1)
    return b, p

def _upload_parquet_gcs(df: pd.DataFrame, gcs_url: str):
    if storage is None:
        raise RuntimeError("google-cloud-storage not installed")
    bkt, key = _split_gcs(gcs_url)
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    storage.Client().bucket(bkt).blob(key).upload_from_file(buf, content_type="application/octet-stream")

# ───────────────────────── Main ingest ─────────────────────────

def ingest_once(save: bool = True) -> tuple[int, str | None]:
    df_v = fetch_velib_df()
    if df_v.empty:
        print("[ingest] GBFS empty — no output")
        return 0, None

    df_w = fetch_weather_df(df_v)
    df_v["hour_utc"] = pd.to_datetime(df_v["tbin_utc"], errors="coerce").dt.floor("h")
    if not df_w.empty:
        df = df_v.merge(df_w, on="hour_utc", how="left")
        print(f"[ingest][merge] velib={len(df_v):,} weather={len(df_w):,} → merged={len(df):,}")
    else:
        df = df_v.assign(temp_C=None, precip_mm=None, wind_mps=None)
        print(f"[ingest][merge] weather empty → filled NaN (rows={len(df):,})")
    df = df.drop(columns=["hour_utc"])

    df_out = df[COLS_ORDER].copy()
    latest_bin_utc = pd.to_datetime(df_out["tbin_utc"].iloc[0])
    day  = latest_bin_utc.strftime("%Y-%m-%d")
    hour = latest_bin_utc.strftime("%H")
    fname = latest_bin_utc.strftime("%Y-%m-%dT%H-%M.parquet")

    if DIAG:
        print(f"[ingest][diag] rows={len(df_out)} stations={df_out['station_id'].nunique()} bin={latest_bin_utc} → {day}/{hour}/{fname}")

    wrote_gcs: str | None = None
    if save or os.environ.get("INGEST_SAVE_PARQUET","1") == "1":
        local_root = os.environ.get("LOCAL_RAW_DIR", "data_local/raw")
        local_path = os.path.join(local_root, f"date={day}", f"hour={hour}", fname)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        df_out.to_parquet(local_path, index=False)
        print(f"[ingest][snapshot] wrote {len(df_out):,} rows → {local_path}")

    if os.environ.get("INGEST_TO_GCS","0") == "1":
        gcs_prefix = os.environ.get("GCS_RAW_PREFIX")
        if not gcs_prefix or not gcs_prefix.startswith("gs://"):
            raise RuntimeError("INGEST_TO_GCS=1 mais GCS_RAW_PREFIX absent ou invalide")
        gcs_url = f"{gcs_prefix}/date={day}/hour={hour}/{fname}"
        _upload_parquet_gcs(df_out, gcs_url)
        print(f"[ingest][snapshot] uploaded {len(df_out):,} rows → {gcs_url}")
        wrote_gcs = gcs_url

    return len(df_out), wrote_gcs


def main():
    print("[ingest] start")
    n, out = ingest_once(save=os.environ.get("INGEST_SAVE_PARQUET","1") == "1")
    print(f"[ingest] done rows={n} out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
