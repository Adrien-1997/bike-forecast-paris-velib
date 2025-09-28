# pipeline/compact_daily.py (robuste aux colonnes manquantes)
import os
import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from google.cloud import storage

RAW_PREFIX = os.environ["GCS_RAW_PREFIX"]      # gs://.../velib/bronze
OUT_DIR    = os.environ["GCS_DB_DAILY"]        # gs://.../velib/db/daily
DB_LOCAL   = os.environ.get("DB_LOCAL", "/tmp/velib_daily.duckdb")
DAY        = os.environ.get("DAY")             # YYYY-MM-DD ; défaut = J-1 (UTC)

def _list_parquet_day(day: str):
    assert RAW_PREFIX.startswith("gs://")
    bkt, pfx = RAW_PREFIX[5:].split("/", 1)
    pfx = f"{pfx}/date={day}/"
    cli = storage.Client()
    return [(bkt, b.name) for b in cli.list_blobs(bkt, prefix=pfx) if b.name.endswith(".parquet")]

def _download_all(blobs, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    cli = storage.Client()
    local_paths = []
    for bkt, key in blobs:
        fn = dest_dir / Path(key).name
        cli.bucket(bkt).blob(key).download_to_filename(str(fn))
        local_paths.append(str(fn))
    return local_paths

def _floor5(ts: pd.Series) -> pd.Series:
    # floor 5 min en UTC naïf
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    return t.dt.floor("5min").dt.tz_convert(None)

def main():
    day = DAY or (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    blobs = _list_parquet_day(day)
    if not blobs:
        print(f"[compact] no raw files for {day} — exit 0")
        return 0

    # 1) Télécharge localement
    tmp_dir = Path("/tmp/raw_day") / day
    files = _download_all(blobs, tmp_dir)
    print(f"[compact] downloaded {len(files)} files to {tmp_dir}")

    # 2) Charge toutes les shards en un DF (union_by_name)
    #    → plus tolérant aux schémas différents
    con_tmp = duckdb.connect(":memory:")
    con_tmp.execute("""
        SELECT * FROM read_parquet($files, union_by_name=true)
    """, {'files': files})
    df = con_tmp.fetch_df()
    con_tmp.close()

    # 3) Normalise les colonnes attendues (ajoute si manquantes)
    need_cols = [
        "ts_utc","ts_paris","tbin_utc","station_id","bikes","capacity",
        "mechanical","ebike","status","lat","lon","ingested_at",
        "ingest_latency_s","source_etag"
    ]
    for c in need_cols:
        if c not in df.columns:
            df[c] = None

    # tbin_utc si manquant
    if df["tbin_utc"].isna().all():
        df["tbin_utc"] = _floor5(df["ts_utc"])

    # ts_paris optionnel: si absent → derive depuis tbin_utc (affichage)
    if df["ts_paris"].isna().all():
        # on laisse NULL pour compaction; (optionnel) dériver ici si besoin
        pass

    # Casting doux côté pandas avant passage à DuckDB
    df["station_id"] = pd.to_numeric(df["station_id"], errors="coerce")
    for c in ["bikes","capacity","mechanical","ebike"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["lat","lon","ingest_latency_s"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4) (re)crée la DB locale + écrit la table typée
    if os.path.exists(DB_LOCAL):
        os.remove(DB_LOCAL)
    con = duckdb.connect(DB_LOCAL)
    con.execute("PRAGMA threads=4;")
    con.execute("CREATE SCHEMA IF NOT EXISTS bronze;")
    con.register("rawdf", df)

    con.execute("""
        CREATE TABLE bronze.raw_snapshots_5min AS
        SELECT
          CAST(ts_utc AS TIMESTAMP)                     AS ts_utc,
          CAST(ts_paris AS TIMESTAMP)                   AS ts_paris,
          CAST(tbin_utc AS TIMESTAMP)                   AS tbin_utc,
          CAST(station_id AS BIGINT)                    AS station_id,
          CAST(bikes AS INTEGER)                        AS bikes,
          CAST(capacity AS INTEGER)                     AS capacity,
          CAST(mechanical AS INTEGER)                   AS mechanical,
          CAST(ebike AS INTEGER)                        AS ebike,
          CAST(status AS VARCHAR)                       AS status,
          CAST(lat AS DOUBLE)                           AS lat,
          CAST(lon AS DOUBLE)                           AS lon,
          CAST(ingested_at AS TIMESTAMP)                AS ingested_at,
          CAST(ingest_latency_s AS DOUBLE)              AS ingest_latency_s,
          CAST(source_etag AS VARCHAR)                  AS source_etag
        FROM rawdf
    """)
    con.close()

    # 5) Upload du shard vers GCS
    assert OUT_DIR.startswith("gs://")
    bkt, outpfx = OUT_DIR[5:].split("/", 1)
    shard = f"{outpfx}/velib_{day.replace('-','')}.duckdb"
    storage.Client().bucket(bkt).blob(shard).upload_from_filename(DB_LOCAL)
    print(f"[compact] uploaded shard → gs://{bkt}/{shard}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
