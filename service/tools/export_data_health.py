# export_data_health.py
from __future__ import annotations
import os, json
from pathlib import Path
import pandas as pd

def main():
    DAILY_DIR = os.environ.get("DAILY_DIR", "data_local/daily")
    OUT_DIR   = os.environ.get("HEALTH_OUT", "exports/health")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    files = sorted(Path(DAILY_DIR).glob("velib_*.parquet"))
    rows = []
    for f in files:
        day = f.stem.split("_")[-1]  # YYYYMMDD
        df = pd.read_parquet(f)
        if df.empty:
            continue
        df["tbin_utc"] = pd.to_datetime(df["tbin_utc"], utc=True, errors="coerce")

        stats = {
            "date": f"{day[:4]}-{day[4:6]}-{day[6:]}",
            "rows": int(len(df)),
            "stations": int(df["station_id"].nunique()),
            "bins": int(df["tbin_utc"].nunique()),
            "null_bikes": int(df["bikes"].isna().sum()),
            "null_temp": int(df["temp_C"].isna().sum()) if "temp_C" in df.columns else None,
        }
        rows.append(stats)

    out_csv = Path(OUT_DIR) / "data_health_daily.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    out_json = Path(OUT_DIR) / "data_health_daily.json"
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    print(f"[export_data_health] wrote {out_csv} & {out_json}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
