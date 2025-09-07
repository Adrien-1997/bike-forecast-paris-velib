import os, requests, duckdb, pandas as pd
CON = duckdb.connect("warehouse.duckdb")
def _to_utc_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce", utc=True); return s.dt.tz_localize(None)
def _fetch_temp_series(start, end) -> pd.DataFrame:
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude=48.8566&longitude=2.3522&start_date={pd.to_datetime(start).date()}"
           f"&end_date={pd.to_datetime(end).date()}&hourly=temperature_2m&timezone=UTC")
    js = requests.get(url, timeout=30).json()
    if "hourly" not in js: return pd.DataFrame()
    t = pd.to_datetime(js["hourly"]["time"], utc=True)
    v = pd.Series(js["hourly"]["temperature_2m"], dtype="float64")
    return pd.DataFrame({"hour_utc": t, "temp_C": v})
def hourly_occupancy(with_weather: bool = True) -> pd.DataFrame:
    q = """
    WITH base AS (
      SELECT ts_utc::TIMESTAMP AS ts_utc, stationcode, name,
             COALESCE(numbikesavailable,0) AS bikes,
             COALESCE(numdocksavailable,0) AS docks,
             NULLIF(capacity,0) AS capacity
      FROM velib_snapshots
    ),
    enriched AS (
      SELECT *, CASE WHEN capacity IS NOT NULL THEN bikes / capacity
                     WHEN bikes + docks > 0 THEN bikes / (bikes + docks)
                     ELSE NULL END AS occ_ratio
      FROM base
    ),
    hourly AS (
      SELECT date_trunc('hour', ts_utc) AS hour_utc,
             stationcode, any_value(name) AS name,
             avg(occ_ratio) AS occ_ratio_hour,
             avg(bikes) AS bikes_avg, avg(docks) AS docks_avg
      FROM enriched GROUP BY 1,2
    )
    SELECT * FROM hourly ORDER BY hour_utc, stationcode;
    """
    df = CON.execute(q).fetchdf()
    if df.empty: return df
    df["hour_utc"] = _to_utc_naive(df["hour_utc"])
    if with_weather:
        enr = _fetch_temp_series(df["hour_utc"].min(), df["hour_utc"].max())
        if not enr.empty:
            enr["hour_utc"] = _to_utc_naive(enr["hour_utc"])
            df = df.merge(enr, on="hour_utc", how="left")
        else:
            df["temp_C"] = pd.NA
    return df
if __name__ == "__main__":
    os.makedirs("exports", exist_ok=True)
    out = hourly_occupancy(True)
    if not out.empty:
        try: out.to_parquet("exports/velib_hourly.parquet", index=False)
        except Exception:
            duckdb.register("out", out)
            duckdb.sql("COPY out TO 'exports/velib_hourly.parquet' (FORMAT PARQUET);")
        out.to_csv("exports/velib_hourly.csv", index=False)
    print("OK hourly -> exports/velib_hourly.parquet (et .csv)")
