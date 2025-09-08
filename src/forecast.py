# src/forecast.py
import pandas as pd
from lightgbm import LGBMRegressor
from src.weather import fetch_forecast
from src.cal_features import add_calendar_features, feature_cols

def train_and_forecast(df_hour: pd.DataFrame, horizon_h: int = 24) -> pd.DataFrame:
    if df_hour is None or df_hour.empty:
        return pd.DataFrame(columns=["stationcode","hour_utc","pred_occ"])

    out = []
    for sc, g in df_hour.groupby("stationcode"):
        g = g.dropna(subset=["occ_ratio_hour"]).copy()
        if len(g) < 24:
            continue

        # Features calendrier + météo (côté historique)
        g = add_calendar_features(g)

        X = g[feature_cols(g)].values
        y = g["occ_ratio_hour"].values
        mdl = LGBMRegressor(random_state=42)
        mdl.fit(X, y)

        # Futur 24h (UTC naïf)
        last = pd.to_datetime(g["hour_utc"], utc=True).max()
        fut  = pd.date_range(last + pd.Timedelta(hours=1), periods=horizon_h, freq="H", tz="UTC").tz_convert(None)
        F    = pd.DataFrame({"hour_utc": fut, "stationcode": sc})

        # Features calendrier sur le futur
        F = add_calendar_features(F)

        # Météo prévisionnelle
        try:
            wf = fetch_forecast(last, horizon_h)
            if not wf.empty:
                F = F.merge(wf, on="hour_utc", how="left")
        except Exception:
            pass

        F["pred_occ"] = mdl.predict(F[feature_cols(F)].values).clip(0, 1)
        out.append(F[["stationcode","hour_utc","pred_occ"]])

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["stationcode","hour_utc","pred_occ"])
