import os, duckdb
db = os.environ.get("DB_LOCAL","velib.duckdb")
con = duckdb.connect(db); con.execute("PRAGMA threads=4;")
con.execute("""
DELETE FROM bronze.raw_snapshots_5min
USING (
  SELECT station_id, tbin_utc, ts_utc, ingested_at,
         ROW_NUMBER() OVER (PARTITION BY station_id, tbin_utc ORDER BY ingested_at DESC, ts_utc DESC) AS rn
  FROM bronze.raw_snapshots_5min
) d
WHERE bronze.raw_snapshots_5min.station_id = d.station_id
  AND bronze.raw_snapshots_5min.tbin_utc   = d.tbin_utc
  AND bronze.raw_snapshots_5min.ts_utc     = d.ts_utc
  AND bronze.raw_snapshots_5min.ingested_at= d.ingested_at
  AND d.rn > 1;
""")
con.close()
print("[dedup] bronze.raw_snapshots_5min deduplicated")
