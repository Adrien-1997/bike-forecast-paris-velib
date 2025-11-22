# service/jobs/ingest.py

"""
5-minute ingestion job for the Velib Forecast pipeline.

This job:
- Fetches station status and station information from the official Velib GBFS
  endpoints.
- Forces the station timestamps to the *job execution time* (ts_utc = now_utc)
  while optionally keeping the original GBFS `last_reported` as an in-memory
  `src_ts_utc`.
- Computes a 5-minute time bin (tbin_utc) aligned on the job execution time.
- Optionally fetches hourly weather from the Open-Meteo API around the current
  time window.
- Merges bike and weather data into a single snapshot DataFrame.
- Writes the snapshot to local parquet and optionally to GCS ("bronze" layer).
- Computes "freshness" metrics for stations and weather and publishes them as
  JSON locally and (optionally) on GCS for the monitoring stack.

Schema (strict, UTC naive timestamps)
-------------------------------------
ts_utc, tbin_utc, station_id, bikes, capacity, mechanical, ebike, status,
lat, lon, name, temp_C, precip_mm, wind_mps

File layout (UTC)
-----------------
Local snapshots:
  data_local/raw/date=YYYY-MM-DD/hour=HH/YYYY-MM-DDT%H-%M.parquet

GCS snapshots (when INGEST_TO_GCS=1):
  gs://.../bronze/date=YYYY-MM-DD/hour=HH/YYYY-MM-DDT%H-%M.parquet

Environment variables
---------------------
INGEST_SAVE_PARQUET : "1" | "0"  (default "1")
    Control local parquet write for the raw 5-minute snapshot.

LOCAL_RAW_DIR : str (default "data_local/raw")
    Local root directory for raw snapshots.

INGEST_TO_GCS : "1" | "0"  (default "0")
    When "1", upload the parquet snapshot to GCS under GCS_RAW_PREFIX.

GCS_RAW_PREFIX : str
    GCS prefix (gs://bucket/path) for bronze snapshots when INGEST_TO_GCS=1.

OPENMETEO_LAT, OPENMETEO_LON : float (as strings)
    Coordinates used for the Open-Meteo hourly weather query.

METEO_DISABLE : "1" | "0"  (default "0")
    When "1", completely skip the weather API call.

DIAG : "1" | "true" | "True" (default "0")
    Enable verbose diagnostic logging (sample rows, paths, etc.).

GCS_MONITORING_PREFIX : str (optional)
    GCS prefix (gs://bucket/path) used to upload freshness JSONs for the
    monitoring stack.

Execution
---------
Run once from the repository root:

    python -m jobs.ingest
"""

from __future__ import annotations
import os
from io import BytesIO
from datetime import datetime, timezone, timedelta
from typing import Tuple
import json
import math
import numpy as np
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
    """
    Return a process-wide HTTP session configured with retries.

    The session:
    - Retries on typical transient HTTP errors (5xx, 429).
    - Uses an exponential backoff.
    - Sets a custom User-Agent for easier tracing on the provider side.
    """
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
    """
    Perform an HTTP GET request with the shared session and decode JSON.

    Parameters
    ----------
    url : str
        Absolute URL to query.
    timeout : int, default 20
        Request timeout in seconds.

    Returns
    -------
    dict
        Parsed JSON payload.

    Raises
    ------
    requests.HTTPError
        If the response status code is not successful.
    """
    r = _session_get().get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ───────────────────────── GBFS ─────────────────────────


def fetch_velib_df() -> pd.DataFrame:
    """
    Fetch Velib station status and information, merge them, and normalize fields.

    This function:
    - Calls the official Velib GBFS `station_status.json` and `station_information.json`.
    - Builds one row per station.
    - Forces `ts_utc` to the ingestion job time (now UTC, naive).
    - Adds a 5-minute time bin (`tbin_utc`) aligned on the job time.
    - Optionally keeps the source timestamp `last_reported` as an in-memory
      `src_ts_utc` column (not persisted to parquet).

    Returns
    -------
    pandas.DataFrame
        DataFrame with (at least) the columns listed in COLS_ORDER plus an
        in-memory `src_ts_utc` used later for freshness computation.
        If payloads are empty, returns an empty DataFrame with COLS_ORDER
        as columns.
    """
    js_status = _http_get_json(URL_STATUS)
    js_info   = _http_get_json(URL_INFO)

    status_list = (js_status.get("data") or {}).get("stations") or []
    info_list   = (js_info.get("data") or {}).get("stations") or []
    if not status_list or not info_list:
        print("[ingest][gbfs] empty payloads — status:", len(status_list), "info:", len(info_list))
        return pd.DataFrame(columns=COLS_ORDER)

    # "Job time" in UTC (naive), used as ts_utc and for the 5-minute bin.
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    tbin_now = pd.Timestamp(now_utc).floor("5min").to_pydatetime()

    # STATUS
    st_rows = []
    for s in status_list:
        sid = s.get("station_id")
        if not sid:
            continue

        # Extract mechanical / ebike counts when exposed by the provider.
        types = {}
        v = s.get("num_bikes_available_types")
        if isinstance(v, list) and v:
            for d in v:
                if isinstance(d, dict):
                    types.update(d)
        elif isinstance(v, dict):
            types = v

        # Source timestamp from GBFS (last_reported), used only for freshness.
        last_reported = s.get("last_reported")
        try:
            src_ts = datetime.utcfromtimestamp(int(last_reported)) if last_reported else now_utc
        except Exception:
            src_ts = now_utc

        st_rows.append({
            "station_id": sid,
            "ts_utc": now_utc,  # ingestion job timestamp
            "bikes": s.get("num_bikes_available"),
            "mechanical": types.get("mechanical", 0),
            "ebike": types.get("ebike", 0),
            "status": s.get("station_status") or (
                "OK" if (s.get("is_renting",1) and s.get("is_returning",1)) else "CLOSED"
            ),
            # Source timestamp used only in memory for freshness; not written to parquet.
            "src_ts_utc": src_ts,
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

    # Merge status + info on station_id
    df = df_st.merge(df_in, on="station_id", how="inner")
    df["ts_utc"]   = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce").dt.tz_localize(None)
    df["tbin_utc"] = pd.to_datetime(tbin_now)

    # Normalize numeric fields.
    for c in ("bikes","mechanical","ebike","capacity","lat","lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"[ingest][gbfs] rows={len(df):,} tbin_utc={tbin_now} (UTC)")
    if DIAG:
        print(df[["station_id","ts_utc","tbin_utc","bikes","capacity"]].head(5).to_string(index=False))
    return df

# ───────────────────────── Weather ─────────────────────────


def _weather_window(df_velib: pd.DataFrame) -> Tuple[datetime, datetime]:
    """
    Build a ±3h UTC time window around the current bin.

    The window is derived from the min/max of `tbin_utc` in `df_velib`,
    floored to the nearest hour. If the timestamps cannot be parsed,
    fall back to a ±3h window around "now UTC".

    Parameters
    ----------
    df_velib : pandas.DataFrame
        Velib DataFrame as returned by `fetch_velib_df`.

    Returns
    -------
    (datetime, datetime)
        Lower and upper bounds of the time window in UTC (naive datetimes).
    """
    hours = pd.to_datetime(df_velib["tbin_utc"], errors="coerce").dt.floor("h")
    lo, hi = hours.min(), hours.max()
    if pd.isna(lo) or pd.isna(hi):
        now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        return now - timedelta(hours=3), now + timedelta(hours=3)
    return lo - timedelta(hours=3), hi + timedelta(hours=3)


def fetch_weather_df(df_velib: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch hourly weather around the Velib time window from Open-Meteo.

    The function:
    - Derives a ±3h window around the Velib `tbin_utc` values.
    - Queries the Open-Meteo API for the last 2 days and next 2 days in UTC.
    - Filters the hourly series to keep only the hours within the window.
    - Returns temperature, precipitation and wind speed at 10m.

    Parameters
    ----------
    df_velib : pandas.DataFrame
        Velib snapshot DataFrame used to compute the time window.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ["hour_utc", "temp_C", "precip_mm", "wind_mps"].
        If METEO_DISABLE=1 or any error occurs, returns an empty DataFrame with
        these columns.
    """
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
    """
    Split a GCS URL into bucket and object key.

    Parameters
    ----------
    gcs_url : str
        URL starting with "gs://".

    Returns
    -------
    (str, str)
        Tuple (bucket_name, object_key).

    Raises
    ------
    AssertionError
        If the URL does not start with "gs://".
    """
    assert gcs_url.startswith("gs://")
    b, p = gcs_url[5:].split("/", 1)
    return b, p


def _upload_parquet_gcs(df: pd.DataFrame, gcs_url: str):
    """
    Upload a DataFrame as parquet to a GCS location.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to serialize.
    gcs_url : str
        Destination GCS URL (gs://bucket/path/to/file.parquet).

    Raises
    ------
    RuntimeError
        If `google-cloud-storage` is not installed.
    """
    if storage is None:
        raise RuntimeError("google-cloud-storage not installed")
    bkt, key = _split_gcs(gcs_url)
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    storage.Client().bucket(bkt).blob(key).upload_from_file(buf, content_type="application/octet-stream")


def _upload_json_gcs(payload: dict, gcs_url: str):
    """
    Upload a JSON-serializable payload as a JSON file to GCS.

    Parameters
    ----------
    payload : dict
        Python dictionary to encode as JSON.
    gcs_url : str
        Destination GCS URL (gs://bucket/path/to/file.json).

    Raises
    ------
    RuntimeError
        If `google-cloud-storage` is not installed.
    """
    if storage is None:
        raise RuntimeError("google-cloud-storage not installed")
    bkt, key = _split_gcs(gcs_url)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    storage.Client().bucket(bkt).blob(key).upload_from_string(data, content_type="application/json")


def _write_json_local(payload: dict, path: str):
    """
    Write a JSON payload to a local path, creating parent directories if needed.

    Parameters
    ----------
    payload : dict
        Python dictionary to encode as JSON.
    path : str
        Local file path where the JSON will be written.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _compute_freshness_payload(df_velib: pd.DataFrame, df_weather: pd.DataFrame) -> dict:
    """
    Compute station and weather freshness metrics for monitoring.

    Station freshness
    -----------------
    - Base timestamp:
        - If `src_ts_utc` exists (GBFS `last_reported`), use it.
        - Otherwise, fall back to `ts_utc` (job timestamp).
    - Freshness values (in minutes) are computed as:
        `now_utc - base_timestamp`.

    Weather freshness
    -----------------
    - If `df_weather` is non-empty:
        - Take the max `hour_utc` and compute `now_utc - last_hour` in minutes.

    The function also computes a small diagnostic payload with the top-k
    "oldest" stations (largest freshness).

    Parameters
    ----------
    df_velib : pandas.DataFrame
        In-memory Velib DataFrame, including the optional `src_ts_utc` column.
    df_weather : pandas.DataFrame
        Weather DataFrame returned by `fetch_weather_df`.

    Returns
    -------
    dict
        JSON-ready payload with the following structure:

        {
          "now_utc": <ISO datetime>,
          "stations": {
            "count": <int>,
            "freshness": {
              "p50_min": <float or null>,
              "p95_min": <float or null>,
              "max_min": <float or null>
            },
            "top_oldest": [
              {"station_id": ..., "freshness_min": ...},
              ...
            ]
          },
          "weather": {
            "freshness_min": <float or null>
          },
          "meta": {
            "bin_t_utc": <ISO datetime or null>,
            "schema": "v1"
          }
        }
    """
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

    # 1) Station freshness.
    if "src_ts_utc" in df_velib.columns:
        ts_base = pd.to_datetime(df_velib["src_ts_utc"], errors="coerce")
    else:
        ts_base = pd.to_datetime(df_velib["ts_utc"], errors="coerce")

    freshness_min_st = (pd.to_datetime(now_utc) - pd.to_datetime(ts_base)).dt.total_seconds() / 60.0
    freshness_min_st = pd.to_numeric(freshness_min_st, errors="coerce")
    f50 = float(np.nanquantile(freshness_min_st, 0.50)) if len(freshness_min_st) else float("nan")
    f95 = float(np.nanquantile(freshness_min_st, 0.95)) if len(freshness_min_st) else float("nan")
    fmax = float(np.nanmax(freshness_min_st)) if len(freshness_min_st) else float("nan")

    # Top-k "oldest" stations (largest freshness) — optional but useful in diagnostics.
    top_k = 50
    top_idx = np.argsort(freshness_min_st.values)[-top_k:] if len(freshness_min_st) else []
    top_payload = []
    if len(top_idx):
        sub = df_velib.iloc[top_idx]
        for sid, val in zip(sub["station_id"], freshness_min_st.iloc[top_idx]):
            if pd.notna(val):
                try:
                    top_payload.append({"station_id": int(sid), "freshness_min": round(float(val), 2)})
                except Exception:
                    top_payload.append({"station_id": sid, "freshness_min": round(float(val), 2)})
        top_payload.sort(key=lambda d: d["freshness_min"], reverse=True)

    # 2) Weather freshness (if any).
    weather_fresh_min = None
    if not df_weather.empty and "hour_utc" in df_weather.columns:
        last_hour = pd.to_datetime(df_weather["hour_utc"], errors="coerce").max()
        if pd.notna(last_hour):
            weather_fresh_min = round(
                (pd.to_datetime(now_utc) - pd.to_datetime(last_hour)).total_seconds() / 60.0, 2
            )

    payload = {
        "now_utc": pd.to_datetime(now_utc).isoformat(),
        "stations": {
            "count": int(df_velib["station_id"].nunique()) if "station_id" in df_velib.columns else int(len(df_velib)),
            "freshness": {
                "p50_min": None if math.isnan(f50) else round(f50, 2),
                "p95_min": None if math.isnan(f95) else round(f95, 2),
                "max_min": None if math.isnan(fmax) else round(fmax, 2),
            },
            # Top 50 "oldest" stations, useful for internal monitoring/debug.
            "top_oldest": top_payload,
        },
        "weather": {
            # Can be None if weather is disabled or unavailable.
            "freshness_min": weather_fresh_min,
        },
        "meta": {
            "bin_t_utc": pd.to_datetime(df_velib["tbin_utc"].iloc[0]).isoformat() if len(df_velib) else None,
            "schema": "v1",
        },
    }
    return payload

# ───────────────────────── Main ingest ─────────────────────────


def ingest_once(save: bool = True) -> tuple[int, str | None]:
    """
    Run a single 5-minute ingestion cycle.

    Steps
    -----
    1. Fetch Velib status+info and build the station snapshot.
    2. Optionally fetch weather and merge on the hour.
    3. Reorder columns according to COLS_ORDER.
    4. Compute the UTC date/hour and parquet filename from `tbin_utc`.
    5. Optionally write the snapshot to local parquet (controlled by `save` or
       `INGEST_SAVE_PARQUET`).
    6. Optionally upload the snapshot to GCS when `INGEST_TO_GCS=1`.
    7. Compute freshness metrics and write them locally and on GCS (if configured).

    Parameters
    ----------
    save : bool, default True
        If True, write the parquet snapshot locally, regardless of the
        `INGEST_SAVE_PARQUET` environment variable. If False, rely solely on
        the environment variable.

    Returns
    -------
    (int, str or None)
        Tuple `(n_rows, gcs_url)` where:
        - `n_rows` is the number of rows in the output snapshot.
        - `gcs_url` is the GCS URL of the parquet file if uploaded, otherwise
          None.
    """
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

    # ───────────── Freshness JSON (monitoring/data/freshness) ─────────────
    freshness = _compute_freshness_payload(df, df_w)

    # Local "latest" and timestamped copies.
    local_monitor_root = os.environ.get("LOCAL_MONITOR_DIR", "data_local/monitoring")
    isots = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    local_latest = os.path.join(local_monitor_root, "data/freshness/latest.json")
    local_dated  = os.path.join(local_monitor_root, f"data/freshness/{isots}.json")
    _write_json_local(freshness, local_latest)
    _write_json_local(freshness, local_dated)
    print(f"[ingest][freshness] wrote {local_latest} & {local_dated}")

    # Optional GCS uploads when monitoring prefix is configured.
    gcs_mon = os.environ.get("GCS_MONITORING_PREFIX", "").rstrip("/")
    if gcs_mon.startswith("gs://"):
        g_latest = f"{gcs_mon}/monitoring/data/freshness/latest.json"
        g_dated  = f"{gcs_mon}/monitoring/data/freshness/{isots}.json"
        _upload_json_gcs(freshness, g_latest)
        _upload_json_gcs(freshness, g_dated)
        print(f"[ingest][freshness] uploaded {g_latest} & {g_dated}")

    return len(df_out), wrote_gcs


def main() -> int:
    """
    CLI entrypoint for the ingestion job.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    print("[ingest] start")
    n, out = ingest_once(save=os.environ.get("INGEST_SAVE_PARQUET","1") == "1")
    print(f"[ingest] done rows={n} out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
