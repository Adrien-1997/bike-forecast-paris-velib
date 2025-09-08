# src/aggregate.py
import os
import duckdb
import pandas as pd
from src.weather import fetch_history

CON = duckdb.connect("warehouse.duckdb")

def _to_utc_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_localize(None)

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
        any_value(name) AS name,
        avg(occ_ratio) AS occ_ratio_hour,
        avg(bikes) AS bikes_avg,
        avg(docks) AS docks_avg,
        any_value(lat) AS lat,
        any_value(lon) AS lon
      FROM enriched
      GROUP BY 1,2
    )
    SELECT * FROM hourly
    ORDER BY hour_utc, stationcode;
    """
    df = CON.execute(q).fetchdf()
    if df.empty:
        return df

    df["hour_utc"] = _to_utc_naive(df["hour_utc"])

    if with_weather:
        w = fetch_history(df["hour_utc"].min(), df["hour_utc"].max())
        if not w.empty:
            df = df.merge(w, on="hour_utc", how="left")
        else:
            for c in ["temp_C", "precip_mm", "wind_mps"]:
                df[c] = pd.NA
    return df

if __name__ == "__main__":
    os.makedirs("exports", exist_ok=True)
    out = hourly_occupancy(with_weather=True)
    if not out.empty:
        try:
            out.to_parquet("exports/velib_hourly.parquet", index=False)
        except Exception:
            duckdb.register("out", out)
            duckdb.sql("COPY out TO 'exports/velib_hourly.parquet' (FORMAT PARQUET);")
        out.to_csv("exports/velib_hourly.csv", index=False)
    print("OK hourly -> exports/velib_hourly.parquet (et .csv)")
