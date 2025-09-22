# tools/show_tail.py
import pandas as pd
from pathlib import Path

def main():
    outdir = Path("docs/exports")
    pq = outdir / "velib_local.parquet"
    csv = outdir / "velib_local.csv"

    if pq.exists():
        df = pd.read_parquet(pq)
        print(f"[show_tail] Loaded {pq} ({len(df)} rows)")
    elif csv.exists():
        df = pd.read_csv(csv, parse_dates=["tbin_utc","hour_utc"])
        print(f"[show_tail] Loaded {csv} ({len(df)} rows)")
    else:
        print("[show_tail] No local export found in docs/exports/")
        return

    pd.set_option("display.max_columns", None)
    print(df.tail(10))

if __name__ == "__main__":
    main()
