# src/eval.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support
from src.config import RUPTURE_LOW, RUPTURE_HIGH
from src.forecast import train_and_forecast

def _to_naive_ns(x):
    dt = pd.to_datetime(x, errors="coerce", utc=True)
    try:
        dt = dt.dt.tz_localize(None)
    except AttributeError:
        dt = dt.tz_localize(None)
    return dt.astype("datetime64[ns]")

def make_labels(df: pd.DataFrame):
    y = {}
    y["y_reg"] = df["occ_ratio_hour"].astype(float)
    y["y_bike"] = (df["occ_ratio_hour"] <  RUPTURE_LOW).astype(int)
    y["y_dock"] = (df["occ_ratio_hour"] >  RUPTURE_HIGH).astype(int)
    return y

def backtest_24h(hourly: pd.DataFrame, max_stations=60):
    """Réentraîne sur (t<=tmax-24h) et teste sur (t>tmax-24h)."""
    d = hourly.copy()
    d["hour_utc"] = _to_naive_ns(d["hour_utc"])
    tmax = d["hour_utc"].max()
    train = d[d["hour_utc"] <= tmax - pd.Timedelta("24h")].copy()
    test  = d[d["hour_utc"] >  tmax - pd.Timedelta("24h")].copy()

    # limiter aux stations denses
    sizes = train.groupby("stationcode")["hour_utc"].count().sort_values(ascending=False)
    keep = list(map(str, sizes.index[:max_stations]))
    train = train[train["stationcode"].astype(str).isin(keep)]
    test  = test[test["stationcode"].astype(str).isin(keep)]

    if train.empty or test.empty:
        return {}, pd.DataFrame()

    preds = train_and_forecast(train, horizon_h=24)
    if preds is None or preds.empty:
        return {}, pd.DataFrame()

    preds["hour_utc"] = _to_naive_ns(preds["hour_utc"])
    test["hour_utc"]  = _to_naive_ns(test["hour_utc"])

    m = preds.merge(test[["stationcode","hour_utc","occ_ratio_hour"]],
                    on=["stationcode","hour_utc"], how="inner")
    if m.empty:
        return {}, pd.DataFrame()

    # Régression
    mae = mean_absolute_error(m["occ_ratio_hour"], m["pred_occ"])
    rmse = mean_squared_error(m["occ_ratio_hour"], m["pred_occ"], squared=False)

    # Classification (rupture vélos / bornes)
    y = make_labels(m)
    yhat_bike = (m["pred_occ"] <  RUPTURE_LOW).astype(int)
    yhat_dock = (m["pred_occ"] >  RUPTURE_HIGH).astype(int)
    pr_b, rc_b, f1_b, _ = precision_recall_fscore_support(y["y_bike"], yhat_bike, average="binary", zero_division=0)
    pr_d, rc_d, f1_d, _ = precision_recall_fscore_support(y["y_dock"], yhat_dock, average="binary", zero_division=0)

    metrics = {
        "mae":  float(mae),
        "rmse": float(rmse),
        "bike_precision": float(pr_b), "bike_recall": float(rc_b), "bike_f1": float(f1_b),
        "dock_precision": float(pr_d), "dock_recall": float(rc_d), "dock_f1": float(f1_d),
        "n_pairs": int(len(m)),
        "n_stations": int(m["stationcode"].nunique())
    }
    return metrics, m
