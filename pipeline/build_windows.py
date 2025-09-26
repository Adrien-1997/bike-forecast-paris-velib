# pipeline/build_windows.py
import os, argparse, duckdb
from datetime import datetime, timedelta, timezone

# Env:
# - DB_LOCAL   : /tmp/velib_reporting.duckdb  (déjà téléchargée par gcs_job.py)
# - WINDOW_DAYS: entier (defaut 7)
# - ANCHOR_DAY : YYYY-MM-DD (fin d’intervalle, defaut = J-1 UTC)

def build_windows(anchor_day: str, window_days: int = 7):
    db_local = os.environ.get("DB_LOCAL", "/tmp/velib_reporting.duckdb")
    con = duckdb.connect(db_local)
    con.execute("PRAGMA threads=4;")
    con.execute("CREATE SCHEMA IF NOT EXISTS gold;")

    # Tables (si absentes)
    con.execute("""
    CREATE TABLE IF NOT EXISTS gold.health_7d (
      date DATE,                         -- jour d'ancrage (fin de fenêtre)
      from_date DATE,                    -- début fenêtre (incl.)
      to_date DATE,                      -- fin fenêtre (incl.)
      completeness_pct_avg DOUBLE,
      missing_bins_sum INTEGER,
      ingest_latency_p95_s DOUBLE
    );
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS gold.station_health_7d (
      date DATE,                         -- jour d'ancrage
      station_id BIGINT,
      from_date DATE,
      to_date DATE,
      completeness_pct_avg DOUBLE,
      missing_bins_sum INTEGER,
      max_gap_bins_max INTEGER,
      gaps_count_sum INTEGER,
      outlier_bins_sum INTEGER
    );
    """)

    # Fenêtre
    day = anchor_day
    win = int(os.environ.get("WINDOW_DAYS", window_days))
    con.execute(f"DELETE FROM gold.health_7d WHERE date = DATE '{day}'")
    con.execute(f"DELETE FROM gold.station_health_7d WHERE date = DATE '{day}'")

    # Global (agrégé sur 7j depuis silver.daily_compact)
    con.execute(f"""
    WITH rng AS (
      SELECT DATE '{day}' AS to_date, DATE '{day}' - INTERVAL {win-1} DAY AS from_date
    ),
    agg AS (
      SELECT
        AVG(completeness_pct)               AS completeness_pct_avg,
        SUM(288 - bins_present)             AS missing_bins_sum,
        quantile(ingest_latency_p95_s,0.95) AS ingest_latency_p95_s
      FROM silver.daily_compact, rng
      WHERE date BETWEEN rng.from_date AND rng.to_date
    )
    INSERT INTO gold.health_7d
    SELECT DATE '{day}', (SELECT from_date FROM rng), (SELECT to_date FROM rng),
           agg.completeness_pct_avg, agg.missing_bins_sum, agg.ingest_latency_p95_s
    FROM agg;
    """)

    # Par station (joint avec station_health_daily)
    con.execute(f"""
    WITH rng AS (
      SELECT DATE '{day}' AS to_date, DATE '{day}' - INTERVAL {win-1} DAY AS from_date
    ),
    a AS (
      SELECT station_id,
             AVG(completeness_pct)     AS completeness_pct_avg,
             SUM(288 - bins_present)   AS missing_bins_sum
      FROM silver.daily_compact, rng
      WHERE date BETWEEN rng.from_date AND rng.to_date
      GROUP BY 1
    ),
    h AS (
      SELECT station_id,
             MAX(max_gap_bins)         AS max_gap_bins_max,
             SUM(gaps_count)           AS gaps_count_sum,
             SUM(outlier_bins)         AS outlier_bins_sum
      FROM silver.station_health_daily, rng
      WHERE date BETWEEN rng.from_date AND rng.to_date
      GROUP BY 1
    )
    INSERT INTO gold.station_health_7d
    SELECT
      DATE '{day}' AS date,
      COALESCE(a.station_id, h.station_id) AS station_id,
      (SELECT from_date FROM rng),
      (SELECT to_date   FROM rng),
      a.completeness_pct_avg,
      a.missing_bins_sum,
      h.max_gap_bins_max,
      h.gaps_count_sum,
      h.outlier_bins_sum
    FROM a
    FULL OUTER JOIN h USING (station_id);
    """)

    con.close()
    print(f"[build_windows] OK → gold.health_7d & gold.station_health_7d for {day}")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", help="YYYY-MM-DD (anchor, default J-1 UTC)")
    args = ap.parse_args()
    day = args.day or (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    raise SystemExit(build_windows(day))
