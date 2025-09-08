# src/aggregate.py
import os
import duckdb
import pandas as pd
from src.weather import fetch_history, fetch_forecast

CON = duckdb.connect("warehouse.duckdb")

def _to_utc_naive_hour(x):
    dt = pd.to_datetime(x, errors="coerce", utc=True)
    # Series vs DatetimeIndex
    try:
        return dt.dt.floor("h").dt.tz_localize(None)
    except AttributeError:
        return dt.floor("h").tz_localize(None)

def hourly_occupancy(with_weather: bool = True) -> pd.DataFrame:
    q = """
    WITH base AS (
      SELECT
        ts_utc::TIMESTAMP AS ts_utc,
        stationcode, name,
        COALESCE(numbikesavailable,0) AS bikes,
        COALESCE(numdocksavailable,0) AS docks,
        NULLIF(capacity,0) AS capacity,
        try_cast(lat as DOUBLE)  AS lat,
        try_cast(lon as DOUBLE)  AS lon
      FROM velib_snapshots
    ),
    enriched AS (
      SELECT *,
        CASE
          WHEN capacity IS NOT NULL THEN bikes / capacity
          WHEN bikes + docks > 0 THEN bikes / (bikes + docks)
          ELSE NULL
        END AS occ_ratio
      FROM base
    ),
    hourly AS (
      SELECT
        date_trunc('hour', ts_utc) AS hour_utc,
        stationcode,
        any_value(name)  AS name,
        avg(occ_ratio)   AS occ_ratio_hour,
        avg(bikes)       AS bikes_avg,
        avg(docks)       AS docks_avg,
        any_value(lat)   AS lat,
        any_value(lon)   AS lon
      FROM enriched
      GROUP BY 1,2
    )
    SELECT * FROM hourly
    ORDER BY hour_utc, stationcode;
    """
    df = CON.execute(q).fetchdf()
    if df.empty:
        return df

    # Normalise clé de jointure : UTC naïf, arrondi à l’heure
    df["hour_utc"] = _to_utc_naive_hour(df["hour_utc"])

    if with_weather:
        # 1) Historique météo sur la fenêtre utile
        w = pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])
        try:
            wf = fetch_history(df["hour_utc"].min(), df["hour_utc"].max())
            if wf is not None and not wf.empty:
                w = wf.copy()
        except Exception as e:
            print(f"[weather] fetch_history failed: {e}")

        if not w.empty:
            w["hour_utc"] = _to_utc_naive_hour(w["hour_utc"])
            df = df.merge(w, on="hour_utc", how="left")
        else:
            for c in ["temp_C","precip_mm","wind_mps"]:
                if c not in df.columns:
                    df[c] = pd.NA

        # 2) Backfill via prévision pour combler des trous récents
        try:
            if df[["temp_C","precip_mm","wind_mps"]].isna().any(axis=1).any():
                fx_start = pd.to_datetime(df["hour_utc"].max())  # déjà naïf
                wf = fetch_forecast(fx_start, 24)
                if not wf.empty:
                    wf["hour_utc"] = _to_utc_naive_hour(wf["hour_utc"])
                    df = df.merge(wf, on="hour_utc", how="left", suffixes=("", "_fx"))
                    for c in ["temp_C","precip_mm","wind_mps"]:
                        if c in df.columns and f"{c}_fx" in df.columns:
                            df[c] = df[c].fillna(df[f"{c}_fx"])
                    drop = [c for c in ["temp_C_fx","precip_mm_fx","wind_mps_fx"] if c in df.columns]
                    if drop:
                        df.drop(columns=drop, inplace=True)
        except Exception as e:
            print(f"[weather] forecast backfill skipped: {e}")

    return df

if __name__ == "__main__":
    os.makedirs("exports", exist_ok=True)
    out = hourly_occupancy(with_weather=True)
    if not out.empty:
        # Parquet (pyarrow/fastparquet sinon fallback DuckDB)
        try:
            out.to_parquet("exports/velib_hourly.parquet", index=False)
        except Exception:
            duckdb.register("out_tbl", out)
            duckdb.sql("COPY out_tbl TO 'exports/velib_hourly.parquet' (FORMAT PARQUET);")
        out.to_csv("exports/velib_hourly.csv", index=False)
    print("OK hourly -> exports/velib_hourly.parquet (et .csv)")
