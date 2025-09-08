# src/cal_features.py
import pandas as pd
import holidays

FR_HOL = holidays.France()

def add_calendar_features(df: pd.DataFrame, tz: str = "Europe/Paris") -> pd.DataFrame:
    df = df.copy()
    ts_utc = pd.to_datetime(df["hour_utc"], errors="coerce", utc=True)
    ts_loc = ts_utc.dt.tz_convert(tz)

    df["hour"] = ts_loc.dt.hour
    df["dow"]  = ts_loc.dt.dayofweek
    df["is_weekend"] = ts_loc.dt.weekday.isin([5, 6]).astype("int8")
    df["is_rush_am"] = ts_loc.dt.hour.between(7, 9).astype("int8")
    df["is_rush_pm"] = ts_loc.dt.hour.between(17, 19).astype("int8")

    dates = ts_loc.dt.date
    df["is_holiday"] = [1 if d in FR_HOL else 0 for d in dates]
    return df

def feature_cols(df: pd.DataFrame):
    base = ["hour","dow","is_weekend","is_rush_am","is_rush_pm","is_holiday"]
    weather = [c for c in ["temp_C","precip_mm","wind_mps"] if c in df.columns and not df[c].isna().all()]
    return base + weather
