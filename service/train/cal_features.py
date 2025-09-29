# service/train/cal_features.py
from __future__ import annotations
import numpy as np
import pandas as pd

# Optional FR holidays
try:
    import holidays
    FR_HOL = holidays.country_holidays("FR")
except Exception:
    FR_HOL = set()

def add_calendar_features(df: pd.DataFrame, tz: str = "Europe/Paris") -> pd.DataFrame:
    """
    Adds (UTC→local):
      - hour, dow, is_weekend, is_rush_am, is_rush_pm, is_holiday
      - hour_sin, hour_cos (24h cycle)
      - minute, min5_idx (0..11), minute_sin, minute_cos (60-min cycle)
      - tod_sin, tod_cos (minutes 0..1439)
    Expects:
      - 'hour_utc' (UTC naive, floored to hour)
      - optional 'tbin_utc' (UTC naive @5-min for minute-level signals)
    """
    out = df.copy()

    ts_utc = pd.to_datetime(out["hour_utc"], errors="coerce", utc=True)
    ts_loc = ts_utc.dt.tz_convert(tz)

    out["hour"] = ts_loc.dt.hour.astype("int16")
    out["dow"] = ts_loc.dt.dayofweek.astype("int16")
    out["is_weekend"] = ts_loc.dt.weekday.isin([5, 6]).astype("int8")
    out["is_rush_am"] = ts_loc.dt.hour.between(7, 9).astype("int8")
    out["is_rush_pm"] = ts_loc.dt.hour.between(17, 19).astype("int8")

    dates = ts_loc.dt.date
    if isinstance(FR_HOL, set):
        out["is_holiday"] = np.fromiter((1 if d in FR_HOL else 0 for d in dates),
                                        count=len(out), dtype=np.int8)
    else:
        out["is_holiday"] = np.fromiter((1 if d in FR_HOL else 0 for d in dates),
                                        count=len(out), dtype=np.int8)

    angle_h = 2.0 * np.pi * (out["hour"].to_numpy(dtype=np.float32) / 24.0)
    out["hour_sin"] = np.sin(angle_h).astype("float32")
    out["hour_cos"] = np.cos(angle_h).astype("float32")

    if "tbin_utc" in out.columns:
        tbin_utc = pd.to_datetime(out["tbin_utc"], errors="coerce", utc=True)
        tbin_loc = tbin_utc.dt.tz_convert(tz)
        minute = tbin_loc.dt.minute.fillna(0).astype("int16")
    else:
        minute = pd.Series(0, index=out.index, dtype="int16")

    out["minute"] = minute
    out["min5_idx"] = (out["minute"] // 5).astype("int16")

    angle_m = 2.0 * np.pi * (out["minute"].to_numpy(dtype=np.float32) / 60.0)
    out["minute_sin"] = np.sin(angle_m).astype("float32")
    out["minute_cos"] = np.cos(angle_m).astype("float32")

    tod_minutes = (out["hour"].astype("int32") * 60 + out["minute"].astype("int32")).to_numpy()
    angle_tod = 2.0 * np.pi * (tod_minutes.astype(np.float32) / 1440.0)
    out["tod_sin"] = np.sin(angle_tod).astype("float32")
    out["tod_cos"] = np.cos(angle_tod).astype("float32")

    return out

def feature_cols(df: pd.DataFrame) -> list[str]:
    base = [
        "hour", "dow", "is_weekend", "is_rush_am", "is_rush_pm", "is_holiday",
        "hour_sin", "hour_cos",
        "minute", "min5_idx", "minute_sin", "minute_cos",
        "tod_sin", "tod_cos",
    ]
    weather = [c for c in ["temp_C", "precip_mm", "wind_mps"]
               if c in df.columns and not df[c].isna().all()]
    return base + weather
