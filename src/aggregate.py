import duckdb, pandas as pd, numpy as np

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
