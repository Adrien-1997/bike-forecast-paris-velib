# src/ingest.py
import duckdb, os
from src.velib_client import fetch_snapshot

os.makedirs("data", exist_ok=True)
CON = duckdb.connect("warehouse.duckdb")

CON.execute("""
CREATE TABLE IF NOT EXISTS velib_snapshots (
  ts_utc TIMESTAMP,
  stationcode VARCHAR,
  name VARCHAR,
  numbikesavailable DOUBLE,
  numdocksavailable DOUBLE,
  mechanical DOUBLE,
  ebike DOUBLE,
  capacity DOUBLE,
  is_installed BOOLEAN,
  is_renting BOOLEAN,
  is_returning BOOLEAN,
  lat DOUBLE,
  lon DOUBLE
);
""")

def append_snapshot():
    df = fetch_snapshot(active_only=True)
    if "capacity" in df and df["capacity"].isna().all():
        df["capacity"] = (df["numbikesavailable"].fillna(0) + df["numdocksavailable"].fillna(0))
    CON.execute("INSERT INTO velib_snapshots SELECT * FROM df")

if __name__ == "__main__":
    append_snapshot()
    print("OK snapshot")
