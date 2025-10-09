# cal_features.py
# ============================================================================
# Petites features calendaires (UTC + Paris) et encodages sin/cos.
# Entrée attendue: une colonne temporelle (par défaut 'tbin_utc') au format
# pandas.Timestamp naive UTC.
# ============================================================================

from __future__ import annotations
import pandas as pd

def _safe_to_datetime_utc(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, utc=True, errors="coerce")
    # on garde naïf UTC (pas de tz-info) pour rester cohérent avec les parquets
    return t.dt.tz_convert(None)

def add_time_features(
    df: pd.DataFrame,
    ts_col: str = "tbin_utc",
    add_paris_derived: bool = True,
) -> pd.DataFrame:
    """
    Ajoute des colonnes temporelles:
      - hour, minute, dow (0=Mon), month, is_weekend
      - sin/cos encodages: hod_sin/cos (24h), dow_sin/cos (7j)
      - (optionnel) paris_hour, paris_dow, paris_is_we
    Le tout en place. Retourne df.
    """
    if ts_col not in df.columns:
        return df

    t = _safe_to_datetime_utc(df[ts_col])
    df["hour"]   = t.dt.hour
    df["minute"] = t.dt.minute
    df["dow"]    = t.dt.dayofweek
    df["month"]  = t.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")

    # encodages sin/cos
    import numpy as np
    df["hod_sin"] = np.sin(2 * np.pi * (df["hour"] + df["minute"]/60.0) / 24.0)
    df["hod_cos"] = np.cos(2 * np.pi * (df["hour"] + df["minute"]/60.0) / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * (df["dow"]  ) / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * (df["dow"]  ) / 7.0)

    if add_paris_derived:
        t_paris = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dt.tz_convert("Europe/Paris")
        df["paris_hour"] = t_paris.dt.hour.astype("Int64")
        df["paris_dow"]  = t_paris.dt.dayofweek.astype("Int64")
        df["paris_is_we"] = (df["paris_dow"] >= 5).astype("Int64")

    return df
