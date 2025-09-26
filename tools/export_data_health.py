# tools/export_data_health.py
import os
from pathlib import Path
import argparse
import duckdb
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="YYYY-MM-DD : n'exporter que >= cette date")
    ap.add_argument("--out", default="exports/data_health.csv", help="Chemin de sortie CSV")
    args = ap.parse_args()

    # DB reporting par défaut (override avec DB_LOCAL si besoin)
    db_local = os.environ.get("DB_LOCAL", "velib_reporting.duckdb")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db_local)
    con.execute("PRAGMA threads=4;")
    where = f"WHERE date >= DATE '{args.since}'" if args.since else ""
    df = con.execute(f"""
        SELECT date, ts_max, freshness_min, completeness_pct_24h,
               missing_bins_24h, ingest_latency_p95_s, schema_ok
        FROM gold.data_health_daily
        {where}
        ORDER BY date
    """).fetchdf()
    con.close()

    if not df.empty:
        df["ts_max"] = pd.to_datetime(df["ts_max"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    df.to_csv(out_path, index=False, float_format="%.3f")
    print(f"[export] wrote {len(df)} rows -> {out_path}")

if __name__ == "__main__":
    main()
