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
    Ajoute: hour, dow, is_weekend, is_rush_am, is_rush_pm, is_holiday,
            hour_sin, hour_cos.
    Utilise la colonne 'hour_utc' (UTC naïf) déjà présente dans l'agrégat.
    """
    out = df.copy()

    # hour_utc -> tz locale
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

    # Encodage circulaire (vectorisé)
    angle = 2.0 * np.pi * (out["hour"].to_numpy(dtype=np.float32) / 24.0)
    out["hour_sin"] = np.sin(angle).astype("float32")
    out["hour_cos"] = np.cos(angle).astype("float32")

    return out

def feature_cols(df: pd.DataFrame) -> list[str]:
    base = ["hour", "dow", "is_weekend", "is_rush_am", "is_rush_pm", "is_holiday",
            "hour_sin", "hour_cos"]
    weather = [c for c in ["temp_C", "precip_mm", "wind_mps"]
               if c in df.columns and not df[c].isna().all()]
    return base + weather
