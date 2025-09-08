# src/weather.py
import pandas as pd, requests, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "data"
CACHE.mkdir(parents=True, exist_ok=True)
HIST_PATH = CACHE / "weather_hourly.parquet"

LAT, LON = 48.8566, 2.3522

def _to_utc_naive(s):
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_localize(None)

def fetch_history(start_ts, end_ts):
    """Historique horaire temp/precip/vent (UTC naive) + cache parquet (merge/dedup)"""
    start = pd.to_datetime(start_ts).date()
    end   = pd.to_datetime(end_ts).date()
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start}&end_date={end}"
        "&hourly=temperature_2m,precipitation,wind_speed_10m"
        "&timezone=UTC"
    )
    js = requests.get(url, timeout=30).json()
    if "hourly" not in js:  # fallback vide
        return pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])

    df = pd.DataFrame({
        "hour_utc": _to_utc_naive(js["hourly"]["time"]),
        "temp_C":   pd.to_numeric(js["hourly"]["temperature_2m"], errors="coerce"),
        "precip_mm":pd.to_numeric(js["hourly"]["precipitation"], errors="coerce"),
        "wind_mps": pd.to_numeric(js["hourly"]["wind_speed_10m"], errors="coerce"),
    })

    # Cache (append + dedup sur hour_utc)
    try:
        if HIST_PATH.exists():
            old = pd.read_parquet(HIST_PATH)
            all_ = (pd.concat([old, df], ignore_index=True)
                    .drop_duplicates(subset=["hour_utc"], keep="last")
                    .sort_values("hour_utc"))
            all_.to_parquet(HIST_PATH, index=False)
        else:
            df.to_parquet(HIST_PATH, index=False)
    except Exception:
        pass
    return df

def fetch_forecast(start_ts, horizon_h=24):
    """Prévision horaire prochaine 24h (UTC naive)"""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m,precipitation,wind_speed_10m"
        "&timezone=UTC"
    )
    js = requests.get(url, timeout=30).json()
    if "hourly" not in js:
        return pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])

    df = pd.DataFrame({
        "hour_utc": _to_utc_naive(js["hourly"]["time"]),
        "temp_C":   pd.to_numeric(js["hourly"]["temperature_2m"], errors="coerce"),
        "precip_mm":pd.to_numeric(js["hourly"]["precipitation"], errors="coerce"),
        "wind_mps": pd.to_numeric(js["hourly"]["wind_speed_10m"], errors="coerce"),
    })
    start = pd.to_datetime(start_ts, utc=True).tz_convert(None)
    end   = start + pd.Timedelta(hours=horizon_h)
    return df[(df["hour_utc"]>start) & (df["hour_utc"]<=end)].copy()
