# tools/export_monthly.py
from __future__ import annotations
import os
from pathlib import Path
import argparse
import duckdb
import pandas as pd

SQL_ALL = """
WITH base AS (
  SELECT
    strftime(CAST(tbin_utc AS DATE), '%Y-%m') AS month,
    CAST(station_id AS BIGINT) AS station_id,
    CAST(bikes AS INTEGER)      AS bikes_sample,
    CAST(tbin_utc AS DATE)      AS d
  FROM read_parquet($files, union_by_name=true)
),
agg AS (
  SELECT
    month,
    station_id,
    COUNT(DISTINCT d)                      AS days,
    NULL::DOUBLE  AS avg_completeness,
    NULL::DOUBLE  AS avg_latency,
    median(bikes_sample)                   AS bikes_median_p50,
    quantile(bikes_sample,0.90)            AS bikes_median_p90
  FROM base
  GROUP BY 1,2
)
SELECT * FROM agg
ORDER BY month, station_id;
"""

SQL_DB = """
WITH base AS (
  SELECT
    strftime(date, '%Y-%m')      AS month,
    station_id,
    completeness_pct,
    ingest_latency_p95_s,
    bikes_median
  FROM silver.daily_compact
  {where}
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
ORDER BY {order};
"""

def _has_table(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> bool:
    try:
        return con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema=? AND table_name=?",
            [schema, table]
        ).fetchone() is not None
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser(description="Exporte des métriques mensuelles depuis DuckDB (si dispo) ou Parquet daily (fallback).")
    ap.add_argument("--month", help="YYYY-MM : n'exporter que ce mois")
    ap.add_argument("--out", default="exports/monthly_metrics.csv")
    ap.add_argument("--daily-dir", default=os.environ.get("LOCAL_DAILY_DIR","data_local/daily"))
    ap.add_argument("--db", default=os.environ.get("DB_LOCAL","velib_reporting.duckdb"))
    args = ap.parse_args()

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    daily_dir = Path(args.daily_dir)
    db_local  = Path(args.db)

    df = pd.DataFrame()

    # 1) Mode DB si disponible
    if db_local.exists():
        con = duckdb.connect(db_local.as_posix()); con.execute("PRAGMA threads=4;")
        if _has_table(con, "silver", "daily_compact"):
            where = f"WHERE strftime(date, '%Y-%m') = '{args.month}'" if args.month else ""
            order = "station_id" if args.month else "month, station_id"
            df = con.execute(SQL_DB.format(where=where, order=order)).fetchdf()
        con.close()

    # 2) Fallback Parquet si DF vide
    if df.empty and daily_dir.exists():
        files = sorted([p.as_posix() for p in daily_dir.glob("velib_*.parquet")])
        if args.month:
            prefix = f"velib_{args.month.replace('-','')}"
            files = [f for f in files if Path(f).name.startswith(prefix)]
        if files:
            con = duckdb.connect(":memory:"); con.execute("PRAGMA threads=4;")
            df = con.execute(SQL_ALL, {"files": files}).fetchdf()
            con.close()

    df.to_csv(out_path, index=False, float_format="%.3f")
    print(f"[export] wrote {len(df)} rows -> {out_path}")
    if df.empty:
        print("[export][warn] aucune donnée trouvée (ni DB silver.daily_compact, ni Parquet daily).")

if __name__ == "__main__":
    main()
