# pipeline/build_daily.py
import os
import argparse
import duckdb
from google.cloud import storage
from datetime import datetime, timezone, timedelta

# Env:
# - DB_LOCAL      : /tmp/velib_reporting.duckdb
# - GCS_DB_URI    : gs://.../velib/db/reporting/velib_reporting.duckdb (téléchargée/uploadée par gcs_job.py)
# - GCS_DB_DAILY  : gs://.../velib/db/daily
# - SHARD_LOCAL   : /tmp/shard.duckdb
# - ON_MISSING    : "skip" (defaut) | "error"

def _download_shard(day: str, shard_local: str) -> bool:
    daily_root = os.environ["GCS_DB_DAILY"]  # gs://.../velib/db/daily
    assert daily_root.startswith("gs://")
    bkt, pfx = daily_root[5:].split("/", 1)
    fname = f"velib_{day.replace('-','')}.duckdb"
    blob = storage.Client().bucket(bkt).blob(f"{pfx}/{fname}")
    if not blob.exists(storage.Client()):
        return False
    os.makedirs(os.path.dirname(shard_local) or "/", exist_ok=True)
    blob.download_to_filename(shard_local)
    return True

def build_daily(day: str):
    db_local    = os.environ.get("DB_LOCAL", "/tmp/velib_reporting.duckdb")
    shard_local = os.environ.get("SHARD_LOCAL", "/tmp/shard.duckdb")
    on_missing  = os.environ.get("ON_MISSING", "skip").lower()

    # 1) Shard J (ou J-1 selon appelant)
    ok = _download_shard(day, shard_local)
    if not ok:
        msg = f"[build_daily] shard missing → gs://.../daily/velib_{day.replace('-','')}.duckdb"
        if on_missing == "error":
            raise FileNotFoundError(msg)
        print(msg + " — skip inserts (exit 0)")
        return 0

    # 2) DB reporting + ATTACH shard
    con = duckdb.connect(db_local)
    con.execute("PRAGMA threads=4;")
    con.execute("CREATE SCHEMA IF NOT EXISTS silver;")
    con.execute("CREATE SCHEMA IF NOT EXISTS gold;")
    con.execute(f"ATTACH '{shard_local}' AS shard (READ_ONLY);")

    # 2.a Crée les tables si besoin
    con.execute("""
    CREATE TABLE IF NOT EXISTS silver.daily_compact (
      date                 DATE,
      station_id           BIGINT,
      bins                 INTEGER,
      bins_present         INTEGER,
      completeness_pct     DOUBLE,
      bikes_median         DOUBLE,
      bikes_p90            DOUBLE,
      bikes_min            INTEGER,
      bikes_max            INTEGER,
      status_mode          VARCHAR,
      ingest_latency_p95_s DOUBLE
    );
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS silver.station_health_daily (
      date          DATE,
      station_id    BIGINT,
      gaps_count    INTEGER,
      max_gap_bins  INTEGER,
      outlier_bins  INTEGER,
      alerts_json   VARCHAR
    );
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS gold.data_health_daily (
      date                   DATE,
      ts_max                 TIMESTAMP,
      freshness_min          DOUBLE,
      completeness_pct_24h   DOUBLE,
      missing_bins_24h       INTEGER,
      ingest_latency_p95_s   DOUBLE,
      schema_ok              BOOLEAN
    );
    """)

    # 3) SILVER — daily_compact (ajout min/max/status_mode)
    con.execute(f"DELETE FROM silver.daily_compact WHERE date = DATE '{day}'")
    con.execute(f"""
    WITH base AS (
      SELECT *
      FROM shard.bronze.raw_snapshots_5min
      WHERE DATE(ts_utc) = DATE '{day}'
    ),
    status_cnt AS (
      SELECT station_id, status, COUNT(*) AS cnt
      FROM base
      GROUP BY 1,2
    ),
    status_mode AS (
      SELECT station_id, arg_max(status, cnt) AS status_mode
      FROM status_cnt
      GROUP BY 1
    )
    INSERT INTO silver.daily_compact
    SELECT
      DATE('{day}')                        AS date,
      b.station_id,
      288                                  AS bins,
      COUNT(*)                             AS bins_present,
      100.0 * COUNT(*) / 288.0             AS completeness_pct,
      median(b.bikes)                      AS bikes_median,
      quantile(b.bikes, 0.90)              AS bikes_p90,
      MIN(b.bikes)                         AS bikes_min,
      MAX(b.bikes)                         AS bikes_max,
      sm.status_mode                       AS status_mode,
      quantile(b.ingest_latency_s, 0.95)   AS ingest_latency_p95_s
    FROM base b
    LEFT JOIN status_mode sm USING (station_id)
    GROUP BY 1,2,10;
    """)

    # 4) SILVER — station_health_daily (gaps / max gap / outliers simples)
    # Gaps: on génère les 288 bins attendus et on détecte les séquences manquantes.
    con.execute(f"DELETE FROM silver.station_health_daily WHERE date = DATE '{day}'")
    con.execute(f"""
    WITH
    stations AS (SELECT DISTINCT station_id FROM shard.bronze.raw_snapshots_5min WHERE DATE(ts_utc)=DATE '{day}'),
    bins AS (
      SELECT generate_series(
        TIMESTAMP '{day} 00:00:00',
        TIMESTAMP '{day} 23:55:00',
        INTERVAL 5 MINUTE
      ) AS tbin_utc
    ),
    expected AS (
      SELECT s.station_id, b.tbin_utc
      FROM stations s CROSS JOIN bins b
    ),
    present AS (
      SELECT station_id, tbin_utc
      FROM shard.bronze.raw_snapshots_5min
      WHERE DATE(ts_utc)=DATE '{day}'
      GROUP BY 1,2
    ),
    missing AS (
      SELECT e.station_id, e.tbin_utc
      FROM expected e
      LEFT JOIN present p
      USING (station_id, tbin_utc)
      WHERE p.tbin_utc IS NULL
    ),
    runs AS (
      SELECT
        station_id,
        tbin_utc,
        -- id de run par gaps consécutifs: 5min = taille du pas
        (strftime(tbin_utc, '%s')::BIGINT/300) - row_number() OVER (PARTITION BY station_id ORDER BY tbin_utc) AS grp
      FROM missing
    ),
    gaps AS (
      SELECT station_id, COUNT(*) AS gap_len
      FROM runs
      GROUP BY station_id, grp
    ),
    outliers AS (
      -- outliers simples: bikes <0 ou bikes > capacity
      SELECT station_id, COUNT(*) AS outlier_bins
      FROM shard.bronze.raw_snapshots_5min
      WHERE DATE(ts_utc)=DATE '{day}'
        AND (bikes < 0 OR (capacity IS NOT NULL AND bikes > capacity))
      GROUP BY 1
    )
    INSERT INTO silver.station_health_daily
    SELECT
      DATE '{day}'                              AS date,
      s.station_id,
      COALESCE((SELECT COUNT(*) FROM gaps g WHERE g.station_id=s.station_id),0) AS gaps_count,
      COALESCE((SELECT MAX(gap_len) FROM gaps g WHERE g.station_id=s.station_id),0) AS max_gap_bins,
      COALESCE((SELECT outlier_bins FROM outliers o WHERE o.station_id=s.station_id),0) AS outlier_bins,
      NULL AS alerts_json
    FROM stations s;
    """)

    # 5) GOLD — data_health_daily (+ schema_ok)
    con.execute(f"DELETE FROM gold.data_health_daily WHERE date = DATE '{day}'")
    con.execute(f"""
    WITH last_ts AS (
      SELECT MAX(ts_utc) AS ts_max
      FROM shard.bronze.raw_snapshots_5min
      WHERE DATE(ts_utc)=DATE '{day}'
    ),
    agg AS (
      SELECT
        AVG(completeness_pct) AS completeness_pct_24h,
        SUM(288 - bins_present) AS missing_bins_24h,
        quantile(ingest_latency_p95_s, 0.95) AS ingest_latency_p95_s
      FROM silver.daily_compact
      WHERE date = DATE '{day}'
    ),
    checks AS (
      SELECT
        -- part des lignes "valides" sur la journée (bornes simples)
        AVG(CASE WHEN bikes IS NOT NULL AND capacity IS NOT NULL AND bikes BETWEEN 0 AND capacity THEN 1 ELSE 0 END)::DOUBLE AS share_valid
      FROM shard.bronze.raw_snapshots_5min
      WHERE DATE(ts_utc)=DATE '{day}'
    )
    INSERT INTO gold.data_health_daily
    SELECT
      DATE '{day}' AS date,
      ts_max,
      EXTRACT(EPOCH FROM (now() - ts_max))/60.0 AS freshness_min,
      agg.completeness_pct_24h,
      agg.missing_bins_24h,
      agg.ingest_latency_p95_s,
      (checks.share_valid >= 0.99) AS schema_ok
    FROM last_ts, agg, checks;
    """)

    con.close()
    print(f"[build_daily] OK → silver.daily_compact, silver.station_health_daily & gold.data_health_daily for {day}")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", help="YYYY-MM-DD (UTC day)")
    args = ap.parse_args()
    day = args.day or (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    raise SystemExit(build_daily(day))
