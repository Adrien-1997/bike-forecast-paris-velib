# tools/export_monthly.py
# Agrège depuis silver.daily_compact et exporte en CSV (pas besoin de table gold.monthly_metrics)
import os
from pathlib import Path
import argparse
import duckdb

SQL_ALL = """
WITH base AS (
  SELECT
    strftime(date, '%Y-%m') AS month,
    station_id,
    completeness_pct,
    ingest_latency_p95_s,
    bikes_median
  FROM silver.daily_compact
),
agg AS (
  SELECT
    month,
    station_id,
    COUNT(*)                                AS days,
    AVG(completeness_pct)                   AS avg_completeness,
    AVG(ingest_latency_p95_s)               AS avg_latency,
    median(bikes_median)                    AS bikes_median_p50,
    quantile(bikes_median, 0.90)            AS bikes_median_p90
  FROM base
  GROUP BY 1,2
)
SELECT * FROM agg
ORDER BY month, station_id
"""

SQL_ONE = """
WITH base AS (
  SELECT
    strftime(date, '%Y-%m') AS month,
    station_id,
    completeness_pct,
    ingest_latency_p95_s,
    bikes_median
  FROM silver.daily_compact
  WHERE strftime(date, '%Y-%m') = $month
),
agg AS (
  SELECT
    month,
    station_id,
    COUNT(*)                                AS days,
    AVG(completeness_pct)                   AS avg_completeness,
    AVG(ingest_latency_p95_s)               AS avg_latency,
    median(bikes_median)                    AS bikes_median_p50,
    quantile(bikes_median, 0.90)            AS bikes_median_p90
  FROM base
  GROUP BY 1,2
)
SELECT * FROM agg
ORDER BY station_id
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", help="YYYY-MM : n'exporter que ce mois")
    ap.add_argument("--out", default="exports/monthly_metrics.csv")
    args = ap.parse_args()

    # Utilise la DB reporting par défaut
    db_local = os.environ.get("DB_LOCAL", "velib_reporting.duckdb")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db_local)
    con.execute("PRAGMA threads=4;")
    if args.month:
        df = con.execute(SQL_ONE, {"$month": args.month}).fetchdf()
    else:
        df = con.execute(SQL_ALL).fetchdf()
    con.close()

    df.to_csv(out_path, index=False, float_format="%.3f")
    print(f"[export] wrote {len(df)} rows -> {out_path}")

if __name__ == "__main__":
    main()
