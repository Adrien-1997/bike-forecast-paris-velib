# pipeline/compact_daily.py
import os, duckdb
from datetime import datetime, timedelta, timezone
from google.cloud import storage

RAW_PREFIX = os.environ["GCS_RAW_PREFIX"]          # gs://.../velib/bronze
OUT_DIR    = os.environ["GCS_DB_DAILY"]            # gs://.../velib/db/daily
DB_LOCAL   = os.environ.get("DB_LOCAL","/tmp/velib_daily.duckdb")
DAY        = os.environ.get("DAY")  # YYYY-MM-DD ; défaut = J-1 (UTC)

def _list_parquet_day(day: str):
    assert RAW_PREFIX.startswith("gs://")
    bkt, pfx = RAW_PREFIX[5:].split("/",1)
    pfx = f"{pfx}/date={day}/"
    cli = storage.Client()
    return [f"gs://{bkt}/{b.name}" for b in cli.list_blobs(bkt, prefix=pfx) if b.name.endswith(".parquet")]

def main():
    day = DAY or (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    files = _list_parquet_day(day)
    if not files:
        print(f"[compact] no raw files for {day} — exit 0"); return 0

    # (re)crée la DB de shard locale
    if os.path.exists(DB_LOCAL): os.remove(DB_LOCAL)
    con = duckdb.connect(DB_LOCAL)
    con.execute("PRAGMA threads=4;")
    con.execute("INSTALL gcs; LOAD gcs; SET gcs_use_default_credentials=true;")

    con.execute("CREATE SCHEMA IF NOT EXISTS bronze;")
    con.execute("""
        CREATE TABLE bronze.raw_snapshots_5min AS
        SELECT
          ts_utc, ts_paris, tbin_utc,
          CAST(station_id AS BIGINT) AS station_id,
          CAST(bikes AS INTEGER) AS bikes,
          CAST(capacity AS INTEGER) AS capacity,
          CAST(mechanical AS INTEGER) AS mechanical,
          CAST(ebike AS INTEGER) AS ebike,
          status,
          CAST(lat AS DOUBLE) AS lat,
          CAST(lon AS DOUBLE) AS lon,
          ingested_at,
          CAST(ingest_latency_s AS DOUBLE) AS ingest_latency_s,
          source_etag
        FROM read_parquet($files)
    """, {'files': files})
    con.close()

    # push shard vers GCS
    assert OUT_DIR.startswith("gs://")
    bkt, outpfx = OUT_DIR[5:].split("/",1)
    shard = f"{outpfx}/velib_{day.replace('-','')}.duckdb"
    storage.Client().bucket(bkt).blob(shard).upload_from_filename(DB_LOCAL)
    print(f"[compact] uploaded shard → gs://{bkt}/{shard}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
    