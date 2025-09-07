import pandas as pd, numpy as np
from lightgbm import LGBMRegressor

def make_features(df):
    df = df.copy()
    df["hour"] = pd.to_datetime(df["hour_utc"]).dt.hour
    df["dow"] = pd.to_datetime(df["hour_utc"]).dt.dayofweek
    if "temp_C" in df:
        df["temp_C"] = df["temp_C"].astype(float)
    return df


def train_forecast(df_hour: pd.DataFrame, horizon_h=24) -> pd.DataFrame:
    out = []
    for sc, g in df_hour.groupby("stationcode"):
        g = make_features(g).dropna(subset=["occ_ratio_hour"])
        if len(g) < 48:  # pas assez d'historique
            continue
        X = g[["hour","dow"]].values
        y = g["occ_ratio_hour"].values
        mdl = LGBMRegressor(random_state=42)
        mdl.fit(X, y)
        # future 24h à partir de la dernière heure
        last = pd.to_datetime(g["hour_utc"].max())
        fut = pd.date_range(last + pd.Timedelta(hours=1), periods=horizon_h, freq="H")
        F = pd.DataFrame({"hour_utc": fut})
        F["hour"] = F["hour_utc"].dt.hour
        F["dow"]  = F["hour_utc"].dt.dayofweek
        F["pred_occ"] = mdl.predict(F[["hour","dow"]].values).clip(0,1)
        F["stationcode"] = sc
        out.append(F[["stationcode","hour_utc","pred_occ"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
