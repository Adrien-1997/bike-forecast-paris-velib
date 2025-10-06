# export_training_base.py
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

try:
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except Exception:
    ds = pq = None

REQ_COLS = [
    "ts_utc","tbin_utc","station_id",
    "bikes","capacity","mechanical","ebike","status",
    "lat","lon","name",
    "temp_C","precip_mm","wind_mps",
]

def _is_gcs(path: str) -> bool:
    return str(path).startswith("gs://")

def _list_daily_paths(root: str) -> list[str]:
    if _is_gcs(root):
        from google.cloud import storage
        bkt, pfx = root[5:].split("/", 1)
        cli = storage.Client()
        out = []
        for b in cli.list_blobs(bkt, prefix=pfx):
            if b.name.endswith(".parquet"):
                out.append(f"gs://{bkt}/{b.name}")
        return sorted(out)
    else:
        p = Path(root)
        return sorted(str(x) for x in p.glob("velib_*.parquet"))

def _read_parquets(paths: list[str]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(columns=REQ_COLS)
    if pq is not None:
        tbl = pq.read_table(paths)
        df = tbl.to_pandas()
    else:
        parts = [pd.read_parquet(p) for p in paths]
        df = pd.concat(parts, ignore_index=True)
    return df

def main():
    DAILY_DIR  = os.environ.get("DAILY_DIR", "data_local/daily")   # gs://... ou dossier
    OUT_PARQ   = os.environ.get("TRAIN_EXPORT", "exports/velib.parquet")
    Path(OUT_PARQ).parent.mkdir(parents=True, exist_ok=True)

    paths = _list_daily_paths(DAILY_DIR)
    if not paths:
        print("[export_training_base] no daily parquet found")
        return 0

    df = _read_parquets(paths)

    # Sécurité : ne garde que les colonnes utiles
    for c in REQ_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[REQ_COLS].copy()

    # ✅ plus aucune conversion en 'stationcode'
    df["tbin_utc"] = pd.to_datetime(df["tbin_utc"], utc=True, errors="coerce").dt.tz_convert(None)
    df["ts_utc"]   = pd.to_datetime(df["ts_utc"],   utc=True, errors="coerce").dt.tz_convert(None)

    df.to_parquet(OUT_PARQ, index=False)
    print(f"[export_training_base] wrote → {OUT_PARQ} (rows={len(df)})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
