# src/weather.py
import pandas as pd, requests, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "data"
CACHE.mkdir(parents=True, exist_ok=True)
HIST_PATH = CACHE / "weather_hourly.parquet"

LAT, LON = 48.8566, 2.3522  # Paris (tu pourras rendre ça dynamique plus tard)

def _to_utc_naive(x):
    dt = pd.to_datetime(x, errors="coerce", utc=True)
    try:
        return dt.tz_localize(None)
    except AttributeError:
        return dt.dt.tz_localize(None)

def _floor_hour_naive(x):
    dt = pd.to_datetime(x, errors="coerce", utc=True)
    try:
        return dt.floor("h").tz_localize(None)
    except AttributeError:
        return dt.dt.floor("h").dt.tz_localize(None)

def fetch_history(start_ts, end_ts):
    """Historique récent (J-0..J-6) via open-meteo forecast + past_days."""
    start = pd.to_datetime(start_ts, utc=True).tz_convert(None)
    end   = pd.to_datetime(end_ts,   utc=True).tz_convert(None)
    days  = max(1, int((end - start).ceil("D").days) + 1)
    days  = min(days, 7)

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m,precipitation,wind_speed_10m"
        f"&past_days={days}"
        "&timezone=UTC"
    )
    js = requests.get(url, timeout=30).json()
    hourly = js.get("hourly") or {}
    rows = [{
        "hour_utc": hh, "temp_C": a, "precip_mm": b, "wind_mps": c
    } for hh, a, b, c in zip(hourly.get("time", []),
                             hourly.get("temperature_2m", []),
                             hourly.get("precipitation", []),
                             hourly.get("wind_speed_10m", []))]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])

    df["hour_utc"] = _floor_hour_naive(df["hour_utc"])
    for c in ["temp_C","precip_mm","wind_mps"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[(df["hour_utc"] >= start.floor("h")) & (df["hour_utc"] <= end.floor("h"))].copy()

    # Cache parquet
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
    """Prévision horaire à partir de start_ts (ex: +1h)."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m,precipitation,wind_speed_10m"
        "&timezone=UTC"
    )
    js = requests.get(url, timeout=30).json()
    hourly = js.get("hourly") or {}
    rows = [{
        "hour_utc": hh, "temp_C": a, "precip_mm": b, "wind_mps": c
    } for hh, a, b, c in zip(hourly.get("time", []),
                             hourly.get("temperature_2m", []),
                             hourly.get("precipitation", []),
                             hourly.get("wind_speed_10m", []))]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])

    df["hour_utc"] = _floor_hour_naive(df["hour_utc"])
    for c in ["temp_C","precip_mm","wind_mps"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    start = pd.to_datetime(start_ts, utc=True).tz_convert(None).floor("h")
    end   = start + pd.Timedelta(hours=horizon_h)
    return df[(df["hour_utc"] > start) & (df["hour_utc"] <= end)].copy()
