import os, duckdb
db = os.environ.get("DB_LOCAL","velib.duckdb")
con = duckdb.connect(db); con.execute("PRAGMA threads=4;")
# table rafraîchie (drop/create)
con.execute("DROP TABLE IF EXISTS gold.latest_rt;")
con.execute("""
CREATE TABLE gold.latest_rt AS
WITH last_bin AS (
  SELECT max(tbin_utc) AS tbin_utc FROM bronze.raw_snapshots_5min
),
snap AS (
  SELECT b.*
  FROM bronze.raw_snapshots_5min b, last_bin lb
  WHERE b.tbin_utc = lb.tbin_utc
),
wx AS (
  SELECT w.*
  FROM bronze.weather_5min w
  JOIN last_bin lb ON w.tbin_utc = date_trunc('hour', lb.tbin_utc)
)
SELECT s.station_id, s.tbin_utc, s.ts_utc, s.ts_paris,
       s.bikes, s.capacity, s.mechanical, s.ebike, s.status, s.lat, s.lon,
       wx.temp_C, wx.precip_mm, wx.wind_mps
FROM snap s
LEFT JOIN wx ON TRUE;
""")
con.close()
print("[latest] built gold.latest_rt")
