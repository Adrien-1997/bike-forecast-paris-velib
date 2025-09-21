# src/cal_features.py
from __future__ import annotations
import numpy as np
import pandas as pd

# Jours fériés FR (optionnel)
try:
    import holidays
    FR_HOL = holidays.country_holidays("FR")
except Exception:
    FR_HOL = set()

def add_calendar_features(df: pd.DataFrame, tz: str = "Europe/Paris") -> pd.DataFrame:
    """
    Ajoute (UTC→locale):
      - hour, dow, is_weekend, is_rush_am, is_rush_pm, is_holiday
      - hour_sin, hour_cos  (cycle 24h)
      - minute, min5_idx (0..11), minute_sin, minute_cos (cycle 60 min)
      - tod_sin, tod_cos (cycle 24h en minutes, plus fin que hour_sin/cos)

    Hypothèses:
      - df contient 'hour_utc' (UTC naïf, floored à l'heure).
      - Si présent, 'tbin_utc' (UTC naïf au pas 5 min) sert à dériver la minute locale;
        sinon, minute=0 par construction (fallback sûr).
    """
    out = df.copy()

    # Base horaire à partir de hour_utc (aware→locale)
    ts_utc = pd.to_datetime(out["hour_utc"], errors="coerce", utc=True)
    ts_loc = ts_utc.dt.tz_convert(tz)

    out["hour"] = ts_loc.dt.hour.astype("int16")
    out["dow"] = ts_loc.dt.dayofweek.astype("int16")
    out["is_weekend"] = ts_loc.dt.weekday.isin([5, 6]).astype("int8")
    out["is_rush_am"] = ts_loc.dt.hour.between(7, 9).astype("int8")
    out["is_rush_pm"] = ts_loc.dt.hour.between(17, 19).astype("int8")

    # Jours fériés (par date locale)
    dates = ts_loc.dt.date
    if isinstance(FR_HOL, set):
        out["is_holiday"] = np.fromiter((1 if d in FR_HOL else 0 for d in dates),
                                        count=len(out), dtype=np.int8)
    else:
        out["is_holiday"] = np.fromiter((1 if d in FR_HOL else 0 for d in dates),
                                        count=len(out), dtype=np.int8)

    # Encodage circulaire horaire (identique à ton implémentation)
    angle_h = 2.0 * np.pi * (out["hour"].to_numpy(dtype=np.float32) / 24.0)
    out["hour_sin"] = np.sin(angle_h).astype("float32")
    out["hour_cos"] = np.cos(angle_h).astype("float32")

    # ------- Nouveaux features intra-heure (pas 5 min) -------
    # Si 'tbin_utc' existe, on calcule la minute locale exacte; sinon minute=0.
    if "tbin_utc" in out.columns:
        tbin_utc = pd.to_datetime(out["tbin_utc"], errors="coerce", utc=True)
        tbin_loc = tbin_utc.dt.tz_convert(tz)
        minute = tbin_loc.dt.minute.fillna(0).astype("int16")
    else:
        minute = pd.Series(0, index=out.index, dtype="int16")

    out["minute"] = minute
    # Index 5-min dans l’heure: 0..11 (robuste même si minute≠multiple de 5)
    out["min5_idx"] = (out["minute"] // 5).astype("int16")

    # Encodage circulaire minute (cycle 60) et "time-of-day" en minutes (cycle 1440)
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
        # Intra-heure (5 min)
        "minute", "min5_idx", "minute_sin", "minute_cos",
        "tod_sin", "tod_cos",
    ]
    weather = [c for c in ["temp_C", "precip_mm", "wind_mps"]
               if c in df.columns and not df[c].isna().all()]
    return base + weather
