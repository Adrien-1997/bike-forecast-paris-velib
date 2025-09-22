# src/weather.py
import os, pathlib, json, time
import pandas as pd, requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "data"
CACHE.mkdir(parents=True, exist_ok=True)
HIST_PATH = CACHE / "weather_hourly.parquet"

LAT = float(os.environ.get("WEATHER_LAT", 48.8566))
LON = float(os.environ.get("WEATHER_LON", 2.3522))
BASE_URL = os.environ.get("OPEN_METEO_URL", "https://api.open-meteo.com/v1/forecast")
VERIFY_SSL = os.environ.get("NO_SSL_VERIFY", "0") != "1"  # possibilité de tester sans SSL
TIMEOUT = int(os.environ.get("WEATHER_TIMEOUT", "20"))

def _make_session() -> requests.Session:
    s = requests.Session()
    r = Retry(
        total=5, read=5, connect=5,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"]
    )
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://",  HTTPAdapter(max_retries=r))
    s.headers.update({"User-Agent": "velib-weather/1.0 (+github.com/Adrien-1997)"})
    return s

def _floor_hour_naive(x):
    dt = pd.to_datetime(x, errors="coerce", utc=True)
    try:
        return dt.floor("h").tz_localize(None)
    except AttributeError:
        return dt.dt.floor("h").dt.tz_localize(None)

def _fetch_hourly(past_days: int | None = None) -> pd.DataFrame:
    """Appel Open-Meteo avec retries + logs; renvoie DF possiblement vide mais typé."""
    params = {
        "latitude": LAT, "longitude": LON,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": "UTC",
    }
    if past_days is not None:
        params["past_days"] = int(max(0, min(7, past_days)))
    s = _make_session()
    try:
        resp = s.get(BASE_URL, params=params, timeout=TIMEOUT, verify=VERIFY_SSL)
        resp.raise_for_status()
        js = resp.json()
    except Exception as e:
        print(f"[weather] Open-Meteo request failed (past_days={past_days}): {e}")
        return pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])

    hourly = js.get("hourly") or {}
    rows = [{
        "hour_utc": t, "temp_C": a, "precip_mm": b, "wind_mps": c
    } for t, a, b, c in zip(hourly.get("time", []),
                            hourly.get("temperature_2m", []),
                            hourly.get("precipitation", []),
                            hourly.get("wind_speed_10m", []))]
    df = pd.DataFrame(rows)
    if df.empty:
        print("[weather] WARNING: hourly payload empty")
        return pd.DataFrame(columns=["hour_utc","temp_C","wind_mps","precip_mm"])
    df["hour_utc"] = _floor_hour_naive(df["hour_utc"])
    for c in ["temp_C","precip_mm","wind_mps"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fetch_history(start_ts, end_ts):
    """Historique (J à J-6). Si l'appel réseau échoue, on tente le cache parquet existant."""
    start = pd.to_datetime(start_ts, utc=True).tz_convert(None).floor("h")
    end   = pd.to_datetime(end_ts,   utc=True).tz_convert(None).floor("h")
    days  = max(1, int((end - start).ceil("D").days) + 1)
    days  = min(days, 7)

    df = _fetch_hourly(past_days=days)
    if df.empty:
        # Fallback sur cache si présent
        if HIST_PATH.exists():
            try:
                cache = pd.read_parquet(HIST_PATH)
                cache["hour_utc"] = pd.to_datetime(cache["hour_utc"], utc=True).tz_localize(None).floor("h")
                cache = cache[(cache["hour_utc"] >= start) & (cache["hour_utc"] <= end)].copy()
                if not cache.empty:
                    print("[weather] Using cached history fallback")
                    return cache
            except Exception as e:
                print(f"[weather] cache read failed: {e}")
        return df  # vide

    # découpe fenêtre demandée
    df = df[(df["hour_utc"] >= start) & (df["hour_utc"] <= end)].copy()

    # Merge & update cache
    try:
        if HIST_PATH.exists():
            old = pd.read_parquet(HIST_PATH)
            all_ = (pd.concat([old, df], ignore_index=True)
                      .drop_duplicates(subset=["hour_utc"], keep="last")
                      .sort_values("hour_utc"))
            all_.to_parquet(HIST_PATH, index=False)
        else:
            df.to_parquet(HIST_PATH, index=False)
    except Exception as e:
        print(f"[weather] cache write failed: {e}")

    return df

def fetch_forecast(start_ts, horizon_h=24):
    """Prévision horaire > start_ts jusqu'à horizon_h (inclus)."""
    df = _fetch_hourly(past_days=None)
    if df.empty:
        return df
    start = pd.to_datetime(start_ts, utc=True).tz_convert(None).floor("h")
    end   = start + pd.Timedelta(hours=horizon_h)
    return df[(df["hour_utc"] > start) & (df["hour_utc"] <= end)].copy()
