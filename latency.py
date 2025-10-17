# test_latency_from_gcs.py
from __future__ import annotations
import argparse
import re
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# deps requises :
#   pip install google-cloud-storage pyarrow pandas numpy

def to_gs_uri(url: str) -> str:
    """
    Accepte soit un lien HTTPS 'storage.cloud.google.com/...'
    soit une URI 'gs://bucket/key' et renvoie une URI gs:// propre.
    """
    if url.startswith("gs://"):
        return url
    m = re.match(r"^https?://storage\.cloud\.google\.com/([^/]+)/(.+)$", url)
    if not m:
        raise ValueError(f"URL non reconnue: {url}")
    bucket, key = m.group(1), m.group(2)
    return f"gs://{bucket}/{key}"

def read_parquet_from_gcs(gs_uri: str) -> pd.DataFrame:
    from google.cloud import storage
    import pyarrow.parquet as pq
    from io import BytesIO

    assert gs_uri.startswith("gs://"), "gs_uri attendu"
    bucket, key = gs_uri[5:].split("/", 1)

    client = storage.Client()
    blob = client.bucket(bucket).blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"Blob introuvable: {gs_uri}")

    buf = BytesIO(blob.download_as_bytes())
    table = pq.read_table(buf)
    df = table.to_pandas()
    return df

def detect_cols(df: pd.DataFrame) -> Tuple[str, str, Optional[str]]:
    lower = {c.lower(): c for c in df.columns}
    def any_of(*cands):
        for c in cands:
            if c in lower:
                return lower[c]
        return None
    ts  = any_of("tbin_utc","ts","timestamp","datetime")
    sid = any_of("station_id","stationcode","station","id")
    ing = any_of("ingested_at","ingest_ts","ingest_time","received_at","etl_ts","load_ts","created_at")
    if not ts or not sid:
        raise KeyError(f"Colonnes minimales absentes (ts={ts}, station={sid})")
    return ts, sid, ing

def kpi_latency(df: pd.DataFrame, ts_col: str, ing_col: Optional[str]):
    if not ing_col or ing_col not in df.columns or df[ing_col].isna().all():
        return {"latency_p50_min": None, "latency_p95_min": None}, None
    lat = df.dropna(subset=[ts_col, ing_col]).copy()
    # normalisation type
    lat[ts_col] = pd.to_datetime(lat[ts_col], utc=True, errors="coerce")
    lat[ing_col] = pd.to_datetime(lat[ing_col], utc=True, errors="coerce")
    lat = lat.dropna(subset=[ts_col, ing_col])
    lat["latency_min"] = (lat[ing_col] - lat[ts_col]).dt.total_seconds() / 60.0
    if lat.empty:
        return {"latency_p50_min": None, "latency_p95_min": None}, None
    p50 = float(np.nanpercentile(lat["latency_min"], 50))
    p95 = float(np.nanpercentile(lat["latency_min"], 95))
    return {"latency_p50_min": round(p50, 2), "latency_p95_min": round(p95, 2)}, lat

def approx_global_latency_from_blob(gs_uri: str, ts_max: pd.Timestamp) -> Optional[float]:
    """Approximation grossière: (mtime blob - ts_max) en minutes (si positif)."""
    from google.cloud import storage
    assert gs_uri.startswith("gs://")
    bucket, key = gs_uri[5:].split("/", 1)
    client = storage.Client()
    blob = client.bucket(bucket).blob(key)
    if not blob.exists():
        return None
    # blob.updated est en UTC
    updated = pd.Timestamp(blob.updated, tz="UTC")
    if ts_max.tz is None:
        ts_max = ts_max.tz_localize("UTC")
    diff_min = (updated - ts_max).total_seconds() / 60.0
    return float(diff_min) if np.isfinite(diff_min) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="gs://bucket/key ou https://storage.cloud.google.com/bucket/key")
    ap.add_argument("--show", type=int, default=5, help="Afficher N lignes d'exemple de latence")
    ap.add_argument("--approx", action="store_true", help="Affiche une approximation globale si ingested_at absent")
    args = ap.parse_args()

    gs_uri = to_gs_uri(args.path)
    print(f"[i] lecture: {gs_uri}")

    df = read_parquet_from_gcs(gs_uri)
    print(f"[i] lignes={len(df):,}  colonnes={list(df.columns)}")

    ts_col, sid_col, ing_col = detect_cols(df)
    print(f"[i] colonnes détectées: ts={ts_col!r}  station={sid_col!r}  ingested_at={ing_col!r}")

    # types
    df = df.dropna(subset=[ts_col, sid_col]).copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    if ing_col and ing_col in df.columns:
        df[ing_col] = pd.to_datetime(df[ing_col], utc=True, errors="coerce")

    kpis, lat_df = kpi_latency(df, ts_col, ing_col)
    print("\n[KPI Latence]")
    print(kpis)

    if lat_df is not None and args.show > 0:
        cols = [c for c in ["station_id", sid_col] if c in lat_df.columns]
        cols = (cols[:1] or [sid_col]) + [ts_col]
        if ing_col and ing_col in lat_df.columns:
            cols += [ing_col]
        cols += ["latency_min"]
        print("\n[Exemples]")
        print(lat_df[cols].head(args.show).to_string(index=False))

    if kpis["latency_p50_min"] is None and args.approx:
        ts_max = pd.to_datetime(df[ts_col], utc=True, errors="coerce").max()
        approx = approx_global_latency_from_blob(gs_uri, ts_max)
        print("\n[Approx globale]")
        if approx is None:
            print("Impossible de calculer une approximation globale.")
        else:
            print(f"~{approx:.2f} min (blob.updated - ts_global_max)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERR] {e}")
        sys.exit(1)
