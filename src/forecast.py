# src/forecast.py
import pandas as pd, numpy as np, requests
from lightgbm import LGBMRegressor
from src.weather import fetch_forecast

def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["hour_utc"], utc=True)
    df["hour"] = ts.dt.hour
    df["dow"]  = ts.dt.dayofweek
    # cas météo: garde tel quel (déjà numérique côté aggregate)
    return df

def _feature_cols(df: pd.DataFrame):
    cols = ["hour","dow"]
    for c in ["temp_C","precip_mm","wind_mps"]:
        if c in df.columns and not df[c].isna().all():
            cols.append(c)
    return cols

def train_and_forecast(df_hour: pd.DataFrame, horizon_h: int = 24) -> pd.DataFrame:
    if df_hour is None or df_hour.empty:
        return pd.DataFrame(columns=["stationcode","hour_utc","pred_occ"])

    out = []
    for sc, g in df_hour.groupby("stationcode"):
        g = _make_features(g).dropna(subset=["occ_ratio_hour"])
        if len(g) < 24:
            continue

        X = g[_feature_cols(g)].values
        y = g["occ_ratio_hour"].values
        mdl = LGBMRegressor(random_state=42)
        mdl.fit(X, y)

        last = pd.to_datetime(g["hour_utc"].max(), utc=True)
        fut  = pd.date_range(last + pd.Timedelta(hours=1), periods=horizon_h, freq="H", tz="UTC")
        F    = pd.DataFrame({"hour_utc": fut.tz_convert(None)})
        F["hour"] = fut.hour
        F["dow"]  = fut.dayofweek
        F["stationcode"] = sc

        # météo pour le futur (si dispo côté g)
        use_weather = any(c in g.columns and not g[c].isna().all()
                          for c in ["temp_C","precip_mm","wind_mps"])
        if use_weather:
            try:
                wf = fetch_forecast(last, horizon_h)
                F = F.merge(wf, on="hour_utc", how="left")
            except Exception:
                # fallback: dernière valeur observée
                for c in ["temp_C","precip_mm","wind_mps"]:
                    if c in g.columns and g[c].notna().any():
                        F[c] = float(g[c].dropna().iloc[-1])

        F["pred_occ"] = mdl.predict(F[_feature_cols(F)].values).clip(0, 1)
        out.append(F[["stationcode","hour_utc","pred_occ"]])

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["stationcode","hour_utc","pred_occ"])
