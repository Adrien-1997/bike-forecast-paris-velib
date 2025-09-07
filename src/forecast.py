# src/forecast.py
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = pd.to_datetime(df["hour_utc"]).dt.hour
    df["dow"]  = pd.to_datetime(df["hour_utc"]).dt.dayofweek
    # garde temp_C si présent
    if "temp_C" in df.columns:
        df["temp_C"] = pd.to_numeric(df["temp_C"], errors="coerce")
    return df

def _feature_cols(df: pd.DataFrame) -> list[str]:
    cols = ["hour", "dow"]
    if "temp_C" in df.columns and not df["temp_C"].isna().all():
        cols.append("temp_C")
    return cols

def train_and_forecast(df_hour: pd.DataFrame, horizon_h: int = 24) -> pd.DataFrame:
    """
    Entrée: df_hour avec colonnes ['hour_utc','stationcode','occ_ratio_hour', ...]
    Sortie: DataFrame ['stationcode','hour_utc','pred_occ']
    """
    if df_hour is None or df_hour.empty:
        return pd.DataFrame(columns=["stationcode","hour_utc","pred_occ"])

    out = []
    for sc, g in df_hour.groupby("stationcode"):
        g = _make_features(g).dropna(subset=["occ_ratio_hour"])
        if len(g) < 48:   # ~2 jours mini
            continue

        X_cols = _feature_cols(g)
        X = g[X_cols].values
        y = g["occ_ratio_hour"].values

        mdl = LGBMRegressor(random_state=42)
        mdl.fit(X, y)

        last = pd.to_datetime(g["hour_utc"].max())
        fut = pd.date_range(last + pd.Timedelta(hours=1), periods=horizon_h, freq="H")
        F = pd.DataFrame({"hour_utc": fut})
        F["hour"] = F["hour_utc"].dt.hour
        F["dow"]  = F["hour_utc"].dt.dayofweek
        if "temp_C" in g.columns and not g["temp_C"].isna().all():
            # sans forecast météo, on propage la dernière valeur connue (naïf)
            F["temp_C"] = float(g["temp_C"].dropna().iloc[-1]) if g["temp_C"].notna().any() else np.nan

        F["stationcode"] = sc
        F["pred_occ"] = mdl.predict(F[X_cols].values).clip(0, 1)
        out.append(F[["stationcode","hour_utc","pred_occ"]])

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["stationcode","hour_utc","pred_occ"])
