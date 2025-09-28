# pipeline/build_monthly.py
import os
import argparse
import duckdb

def build_monthly(month: str):
    """
    Consolide un mois (YYYY-MM) depuis silver.daily_compact
    → gold.monthly_metrics
    """
    db_local = os.environ.get("DB_LOCAL", "velib.duckdb")
    con = duckdb.connect(db_local)
    con.execute("PRAGMA threads=4;")

    # (Re)crée la table si besoin
    con.execute("""
    CREATE TABLE IF NOT EXISTS gold.monthly_metrics (
        month TEXT,
        station_id BIGINT,
        days INT,
        avg_completeness DOUBLE,
        avg_latency DOUBLE,
        bikes_median_p50 DOUBLE,
        bikes_median_p90 DOUBLE
    )
    """)

    # Nettoyer le mois cible avant réinsertion
    con.execute(f"DELETE FROM gold.monthly_metrics WHERE month = '{month}'")

    # Insérer les agrégats
    con.execute(f"""
        INSERT INTO gold.monthly_metrics
        SELECT
            '{month}' AS month,
            station_id,
            COUNT(*) AS days,
            avg(completeness_pct) AS avg_completeness,
            avg(ingest_latency_p95_s) AS avg_latency,
            median(bikes_median) AS bikes_median_p50,
            quantile(bikes_median, 0.90) AS bikes_median_p90
        FROM silver.daily_compact
        WHERE strftime(date, '%Y-%m') = '{month}'
        GROUP BY station_id
    """)

    con.close()
    print(f"[build_monthly] OK → gold.monthly_metrics for {month}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True, help="YYYY-MM")
    args = ap.parse_args()
    build_monthly(args.month)
