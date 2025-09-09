# src/features.py
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
import holidays

TZ = "Europe/Paris"

# après (robuste série/index)
def _to_local_hour(x):
    # Rend “tz-aware UTC”, convertit en Europe/Paris, arrondit à l’heure
    try:
        dt = pd.to_datetime(x, errors="coerce", utc=True).dt.tz_convert(TZ).dt.floor("h")
    except AttributeError:  # DatetimeIndex / scalar
        dt = pd.to_datetime(x, errors="coerce", utc=True).tz_convert(TZ).floor("h")
    return dt

def build_training_frame(db_path="warehouse.duckdb", hourly_path="exports/velib_hourly.parquet",
                         horizon_hours: int = 1, lookback_days: int = 60) -> pd.DataFrame:
    """
    Retourne un DataFrame features + target pour entraînement/pred.
    Cible: y_nb = nb_velos_hour(t + horizon).
    """
    con = duckdb.connect(db_path)
    if Path(hourly_path).exists():
        df = pd.read_parquet(hourly_path)
    else:
        # fallback rapide depuis la table
        q = """
        WITH base AS (
          SELECT ts_utc::TIMESTAMP AS ts_utc, stationcode,
                 COALESCE(numbikesavailable,0) AS bikes,
                 COALESCE(numdocksavailable,0) AS docks,
                 NULLIF(capacity,0) AS capacity
          FROM velib_snapshots
          WHERE ts_utc >= now() - INTERVAL 90 DAY
        )
        SELECT
          date_trunc('hour', ts_utc) AS hour_utc,
          stationcode,
          CAST(avg(bikes) AS INTEGER)       AS nb_velos_hour,
          CAST(avg(docks) AS INTEGER)       AS nb_bornes_hour,
          max(capacity)                     AS capacity_hour
        FROM base
        GROUP BY 1,2
        """
        df = con.execute(q).fetchdf()

    # garde fenêtre de lookback
    df["hour_utc"] = pd.to_datetime(df["hour_utc"])
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=lookback_days)
    df = df[df["hour_utc"] >= cutoff].copy()

    # ratio
    df["capacity_hour"] = pd.to_numeric(df["capacity_hour"], errors="coerce")
    df["occ_ratio_hour"] = (
        (pd.to_numeric(df["nb_velos_hour"], errors="coerce") / df["capacity_hour"])
        .where(df["capacity_hour"] > 0)
    ).clip(0, 1)

    # ===== Calendrier (Europe/Paris)
    df["hour_local"] = _to_local_hour(df["hour_utc"])
    df["hour"] = df["hour_local"].dt.hour
    df["dow"] = df["hour_local"].dt.dayofweek  # 0=Lundi
    df["week"] = df["hour_local"].dt.isocalendar().week.astype(int)
    df["month"] = df["hour_local"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    fr_holidays = holidays.country_holidays("FR")
    df["date_local"] = df["hour_local"].dt.date
    df["is_holiday_fr"] = df["date_local"].map(lambda d: int(d in fr_holidays))

    # Fourier heure/jour (saisonnalité douce)
    def fourier(col, period, K=2):
        col = col.astype(float)
        out = {}
        for k in range(1, K+1):
            out[f"{period}_sin{k}"] = np.sin(2*np.pi*k*col/period)
            out[f"{period}_cos{k}"] = np.cos(2*np.pi*k*col/period)
        return pd.DataFrame(out)

    df = pd.concat([df, fourier(df["hour"], 24, K=2), fourier(df["dow"], 7, K=2)], axis=1)

    # ===== Lags & rollings (par station)
    df = df.sort_values(["stationcode", "hour_utc"]).reset_index(drop=True)
    for col in ["nb_velos_hour", "occ_ratio_hour"]:
        for lag in [1, 2, 24]:
            df[f"{col}_lag{lag}"] = df.groupby("stationcode")[col].shift(lag)
        for w in [3, 6, 24]:
            df[f"{col}_roll{w}"] = (
                df.groupby("stationcode")[col].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
            )

    # ===== Target: y_nb (t + horizon)
    df["y_nb"] = df.groupby("stationcode")["nb_velos_hour"].shift(-horizon_hours)

    # Nettoyage
    feat_cols = [c for c in df.columns if c not in {
        "hour_local","date_local","y_nb"
    }]
    # drop lignes sans target ou features essentielles
    df = df.dropna(subset=["y_nb", "nb_velos_hour_lag1", "occ_ratio_hour_lag1"]).reset_index(drop=True)
    return df, feat_cols
