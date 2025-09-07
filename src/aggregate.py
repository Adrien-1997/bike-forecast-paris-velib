import duckdb, pandas as pd, numpy as np
import requests, pandas as pd
from datetime import datetime, timezone

CON = duckdb.connect("warehouse.duckdb")

def hourly_occupancy() -> pd.DataFrame:
    q = """
    WITH base AS (
      SELECT
        ts_utc::TIMESTAMP AS ts_utc,
        stationcode,
        name,
        COALESCE(numbikesavailable,0) AS bikes,
        COALESCE(numdocksavailable,0) AS docks,
        NULLIF(capacity,0) AS capacity
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
        stationcode, any_value(name) AS name,
        avg(occ_ratio) AS occ_ratio_hour,
        avg(bikes) AS bikes_avg, avg(docks) AS docks_avg
      FROM enriched
      GROUP BY 1,2
    )
    SELECT * FROM hourly
    ORDER BY hour_utc, stationcode;
    """
    return CON.execute(q).fetchdf()

if __name__ == "__main__":
    df = hourly_occupancy()
    df.to_parquet("exports/velib_hourly.parquet", index=False)
    print("OK hourly -> exports/velib_hourly.parquet")


# enrichir par température Paris (centre) à l'heure

def fetch_temp_series(start, end):
    url = ("https://archive-api.open-meteo.com/v1/archive"
           "?latitude=48.8566&longitude=2.3522"
           f"&start_date={start.date()}&end_date={end.date()}"
           "&hourly=temperature_2m&timezone=UTC")
    js = requests.get(url, timeout=30).json()
    if "hourly" not in js: return pd.DataFrame()
    t = pd.to_datetime(js["hourly"]["time"], utc=True)
    v = js["hourly"]["temperature_2m"]
    return pd.DataFrame({"hour_utc": t, "temp_C": v})

enr = fetch_temp_series(df["hour_utc"].min(), df["hour_utc"].max())
df = df.merge(enr, on="hour_utc", how="left")
