# src/features.py
from __future__ import annotations
import pandas as pd
import numpy as np

TZ = "Europe/Paris"

def _to_local_hour(x):
    # Accepte ts naive/aware → renvoie tz-aware Europe/Paris
    dt = pd.to_datetime(x, errors="coerce", utc=True)
    try:
        dt = dt.dt.tz_convert(TZ)
    except Exception:
        dt = dt.tz_convert(TZ)
    return dt.dt.floor("h")

def build_training_frame(horizon_hours: int = 1, lookback_days: int = 7):
    """
    (déjà existant dans ton repo normalement)
    Renvoie df, feat_cols pour l'entraînement.
    Laisse tel quel si tu as déjà une version plus aboutie.
    """
    import duckdb, os
    con = duckdb.connect("warehouse.duckdb")
    q = f"""
    WITH base AS (
      SELECT
        ts_utc::TIMESTAMP AS ts_utc,
        stationcode,
        COALESCE(numbikesavailable,0) AS bikes,
        COALESCE(numdocksavailable,0) AS docks,
        NULLIF(capacity,0) AS capacity
      FROM velib_snapshots
      WHERE ts_utc >= now() - INTERVAL {lookback_days} DAY
    ),
    hourly AS (
      SELECT
        date_trunc('hour', ts_utc) AS hour_utc,
        stationcode,
        CAST(avg(bikes) AS DOUBLE) AS nb_velos_hour,
        CAST(avg(docks) AS DOUBLE) AS nb_bornes_hour,
        max(capacity)              AS capacity_hour
      FROM base
      GROUP BY 1,2
    ),
    enriched AS (
      SELECT h.*,
             CASE WHEN capacity_hour>0 THEN nb_velos_hour/capacity_hour ELSE NULL END AS occ_ratio_hour
      FROM hourly h
    )
    SELECT * FROM enriched
    """
    df = con.execute(q).fetchdf()
    if df.empty:
        return df, []

    # Y = nb vélos à T+h
    df = df.sort_values(["stationcode","hour_utc"])
    df["y_nb"] = df.groupby("stationcode")["nb_velos_hour"].shift(-horizon_hours)

    # time features locales
    df["hour_local"] = _to_local_hour(df["hour_utc"])
    dt = df["hour_local"]
    df["hour"] = dt.dt.hour.astype("int16")
    df["dow"] = dt.dt.dayofweek.astype("int16")
    df["is_weekend"] = df["dow"].isin([5,6]).astype("int8")
    df["month"] = dt.dt.month.astype("int16")

    # météos placeholders si absentes
    for c in ["temp_C","precip_mm","wind_mps"]:
        if c not in df.columns:
            df[c] = 0.0

    # features finales (numériques only)
    feat_cols = [
        "nb_velos_hour","nb_bornes_hour","capacity_hour","occ_ratio_hour",
        "temp_C","precip_mm","wind_mps",
        "hour","dow","is_weekend","month",
    ]
    df = df.dropna(subset=["y_nb"])
    return df, feat_cols

def prepare_live_features(df_live: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """
    Aligne / reconstruit les features live pour coller à feat_cols du modèle.
    df_live attendu (au minimum):
      stationcode, name, lat, lon, capacity, numbikesavailable, numdocksavailable, fetched_at_utc
    """
    df = df_live.copy()

    # Robustesse: types numériques de base
    for c in ["capacity","numbikesavailable","numdocksavailable","lat","lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Heure locale à partir de fetched_at_utc
    ts = df.get("fetched_at_utc")
    if ts is None:
        ts = pd.Timestamp.utcnow()
    df["_hour_local"] = _to_local_hour(ts)

    # Reconstructions "live-friendly"
    cap = pd.to_numeric(df.get("capacity"), errors="coerce").fillna(0.0)
    nb  = pd.to_numeric(df.get("numbikesavailable"), errors="coerce").fillna(0.0)
    docks = pd.to_numeric(df.get("numdocksavailable"), errors="coerce").fillna(0.0)

    # Proxies raisonnables pour features d’agrégat horaire
    proxy = {
        "nb_velos_hour": nb.astype(float),
        "nb_bornes_hour": docks.astype(float),
        "capacity_hour": cap.astype(float),
        "occ_ratio_hour": np.where(cap>0, (nb/cap).astype(float), 0.0),
        "bikes_avg": nb.astype(float),     # si le modèle en attend
        "docks_avg": docks.astype(float),  # si le modèle en attend
        "temp_C": 0.0,
        "precip_mm": 0.0,
        "wind_mps": 0.0,
    }

    # Time features
    dt = df["_hour_local"]
    df["hour"] = dt.dt.hour.astype("int16")
    df["dow"] = dt.dt.dayofweek.astype("int16")
    df["is_weekend"] = df["dow"].isin([5,6]).astype("int8")
    df["month"] = dt.dt.month.astype("int16")

    # Assemble X dans l’ordre des features attendues
    X = pd.DataFrame(index=df.index)
    for col in feat_cols:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors="coerce")
        elif col in proxy:
            # valeur scalaire → broadcast ; Series → alignement
            val = proxy[col]
            if isinstance(val, (int, float, np.floating, np.integer)):
                X[col] = float(val)
            else:
                X[col] = pd.to_numeric(val, errors="coerce")
        else:
            # inconnue du live → 0.0
            X[col] = 0.0

    # Remplissages finaux & dtypes
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for c in X.columns:
        # on force en float pour LGBM num-only (use_cats=False)
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0).astype("float32")

    return X