# pipeline/compact_monthly.py
# Compacte tous les dailies d'un mois UTC → 1 Parquet mensuel (schéma strict),
# puis (optionnel) supprime les dailies uniquement si le mois est CLOS.
#
# Schéma STRICT :
# ts_utc, tbin_utc, station_id, bikes, capacity, mechanical, ebike, status,
# lat, lon, name, temp_C, precip_mm, wind_mps
#
# Env requis :
#   GCS_DAILY_PREFIX    = gs://<bucket>/<root>/daily
#   GCS_MONTHLY_PREFIX  = gs://<bucket>/<root>/monthly
#   MONTH               = YYYY-MM  (mois UTC à compacter)
# Options :
#   DRY_RUN             = "1" → n'écrit/supprime rien, affiche ce qui serait fait
#
# Exécution :
#   python -m pipeline.compact_monthly

from __future__ import annotations

import os
from io import BytesIO
from typing import List, Tuple
from datetime import datetime, timezone
import re
import pandas as pd

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("pyarrow est requis pour compact_monthly.py") from e

try:
    from google.cloud import storage  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("google-cloud-storage est requis") from e


COLS = [
    "ts_utc","tbin_utc","station_id","bikes","capacity","mechanical","ebike",
    "status","lat","lon","name","temp_C","precip_mm","wind_mps"
]

def _split(gcs_url: str) -> Tuple[str, str]:
    assert gcs_url.startswith("gs://")
    b, p = gcs_url[5:].split("/", 1)
    return b, p.rstrip("/")

def _daily_key_to_date(key: str) -> datetime | None:
    fn = key.rsplit("/", 1)[-1]
    m1 = re.match(r"^compact_(\d{4})-(\d{2})-(\d{2})\.parquet$", fn)
    if m1:
        y, m, d = map(int, m1.groups())
        return datetime(y, m, d)
    m2 = re.match(r"^velib_(\d{4})(\d{2})(\d{2})\.parquet$", fn)
    if m2:
        y, m, d = map(int, m2.groups())
        return datetime(y, m, d)
    return None

def _list_month_dailies(daily_prefix: str, month: str) -> List[Tuple[str, str]]:
    bkt, pfx = _split(daily_prefix)
    cli = storage.Client()
    out: List[Tuple[str, str]] = []
    for b in cli.list_blobs(bkt, prefix=pfx):
        if not b.name.endswith(".parquet"):
            continue
        d = _daily_key_to_date(b.name)
        if d is None:
            continue
        if d.strftime("%Y-%m") == month:
            out.append((bkt, b.name))
    out.sort(key=lambda t: t[1])
    return out

def _download_parquet_to_df(bkt: str, key: str) -> pd.DataFrame:
    buf = BytesIO()
    storage.Client().bucket(bkt).blob(key).download_to_file(buf)
    buf.seek(0)
    table = pq.read_table(buf)
    return table.to_pandas()

def _upload_parquet_df(df: pd.DataFrame, gcs_url: str):
    bkt, key = _split(gcs_url)
    buf = BytesIO()
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)
    storage.Client().bucket(bkt).blob(key).upload_from_file(buf, content_type="application/octet-stream")

def _copy_blob(src_url: str, dst_url: str):
    src_bkt, src_key = _split(src_url)
    dst_bkt, dst_key = _split(dst_url)
    cli = storage.Client()
    dst_bucket = cli.bucket(dst_bkt)
    src_bucket = cli.bucket(src_bkt)
    src_blob = src_bucket.blob(src_key)
    dst_bucket.copy_blob(src_blob, dst_bucket, dst_key)

def _delete_blob(url: str):
    bkt, key = _split(url)
    cli = storage.Client()
    cli.bucket(bkt).blob(key).delete()

def _delete_keys(keys: List[Tuple[str, str]]):
    if not keys:
        return
    cli = storage.Client()
    with cli.batch():
        for bkt, key in keys:
            cli.bucket(bkt).blob(key).delete()

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    out = pd.DataFrame({
        "ts_utc":     pd.to_datetime(df["ts_utc"],   utc=True, errors="coerce").dt.tz_convert(None),
        "tbin_utc":   pd.to_datetime(df["tbin_utc"], utc=True, errors="coerce").dt.tz_convert(None),
        "station_id": pd.to_numeric(df["station_id"], errors="coerce"),
        "bikes":      pd.to_numeric(df["bikes"],      errors="coerce"),
        "capacity":   pd.to_numeric(df["capacity"],   errors="coerce"),
        "mechanical": pd.to_numeric(df["mechanical"], errors="coerce"),
        "ebike":      pd.to_numeric(df["ebike"],      errors="coerce"),
        "status":     df["status"].astype("string"),
        "lat":        pd.to_numeric(df["lat"],        errors="coerce"),
        "lon":        pd.to_numeric(df["lon"],        errors="coerce"),
        "name":       df["name"].astype("string"),
        "temp_C":     pd.to_numeric(df["temp_C"],     errors="coerce"),
        "precip_mm":  pd.to_numeric(df["precip_mm"],  errors="coerce"),
        "wind_mps":   pd.to_numeric(df["wind_mps"],   errors="coerce"),
    })
    for c in ["station_id","bikes","capacity","mechanical","ebike"]:
        out[c] = out[c].astype("Int64")
    return out[COLS]

def _filter_month_utc(df: pd.DataFrame, month: str) -> pd.DataFrame:
    start = pd.Timestamp(f"{month}-01 00:00:00")
    end_month = (start + pd.Timedelta(days=32)).replace(day=1)
    if "tbin_utc" not in df or df["tbin_utc"].isna().any():
        df["tbin_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce").dt.floor("5min")
    mask = (df["tbin_utc"] >= start) & (df["tbin_utc"] < end_month)
    kept = df.loc[mask].copy()
    dropped = len(df) - len(kept)
    if dropped:
        print(f"[monthly] filtered out {dropped:,} rows outside UTC month {month}")
    return kept

def _deduplicate_latest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station_id","tbin_utc","ts_utc"])
    df = df.groupby(["station_id","tbin_utc"], as_index=False).tail(1)
    return df

def _is_closed_month(month: str) -> bool:
    """True si MONTH < current UTC month (ex: '2025-09' < '2025-10')."""
    now = datetime.now(timezone.utc)
    current = (now.year, now.month)
    y, m = map(int, month.split("-", 1))
    return (y, m) < current

# ───────────────────────────── Main ─────────────────────────────

def main():
    DAILY_PREFIX   = os.environ.get("GCS_DAILY_PREFIX")
    MONTHLY_PREFIX = os.environ.get("GCS_MONTHLY_PREFIX")
    MONTH          = os.environ.get("MONTH")        # YYYY-MM
    DRY_RUN        = os.environ.get("DRY_RUN","0") == "1"

    if not (DAILY_PREFIX and DAILY_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_DAILY_PREFIX invalide ou manquant")
    if not (MONTHLY_PREFIX and MONTHLY_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_MONTHLY_PREFIX invalide ou manquant")
    if not MONTH:
        MONTH = datetime.now().strftime("%Y-%m")
        print(f"[monthly] MONTH not provided → default to current month: {MONTH}")

    print(f"[monthly] start MONTH={MONTH} daily={DAILY_PREFIX} out={MONTHLY_PREFIX} dry_run={int(DRY_RUN)}")

    daily_files = _list_month_dailies(DAILY_PREFIX, MONTH)
    if not daily_files:
        print(f"[monthly] aucun daily pour {MONTH} — rien à faire")
        return 0

    print(f"[monthly] {len(daily_files)} shards trouvés pour {MONTH}")
    for _, key in daily_files[:5]:
        print("  -", key)
    if len(daily_files) > 5:
        print(f"  … (+{len(daily_files)-5} autres)")

    dfs: List[pd.DataFrame] = []
    failed = 0
    for bkt, key in daily_files:
        try:
            dfs.append(_download_parquet_to_df(bkt, key))
        except Exception as e:
            failed += 1
            print(f"[monthly][warn] lecture échouée gs://{bkt}/{key}: {e}")

    if not dfs:
        print(f"[monthly] lecture impossible sur {MONTH}")
        return 0

    df_all = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"[monthly] concatenated rows={len(df_all):,} (failed_files={failed})")

    df_all = _ensure_schema(df_all)
    df_all = _filter_month_utc(df_all, MONTH)
    before_dedup = len(df_all)
    df_all = _deduplicate_latest(df_all)
    print(f"[monthly] filtered rows={before_dedup:,} | dedup kept={len(df_all):,}")

    df_all = df_all.sort_values(["tbin_utc","station_id"]).reset_index(drop=True)

    bins_present    = df_all["tbin_utc"].nunique()
    stations_unique = df_all["station_id"].nunique()
    print(f"[monthly] bins_present={bins_present} | stations_unique={stations_unique}")

    out_key_final = f"compact_{MONTH.replace('-','')}.parquet"  # compact_YYYYMM.parquet
    out_final = f"{MONTHLY_PREFIX.rstrip('/')}/{out_key_final}"
    out_tmp   = f"{MONTHLY_PREFIX.rstrip('/')}/_tmp_compact_{MONTH.replace('-','')}_{int(datetime.now(timezone.utc).timestamp())}.parquet"

    if DRY_RUN:
        print(f"[monthly][dry-run] écrirait {len(df_all):,} lignes → {out_final}")
    else:
        _upload_parquet_df(df_all, out_tmp)
        _copy_blob(out_tmp, out_final)
        _delete_blob(out_tmp)
        print(f"[monthly] wrote {len(df_all):,} rows → {out_final}")

        # Rolling logic : supprimer les dailies uniquement si le mois est clos
        if _is_closed_month(MONTH):
            _delete_keys(daily_files)
            print(f"[monthly] deleted {len(daily_files)} daily shards for {MONTH}")
        else:
            print(f"[monthly] keeping daily shards for {MONTH} (open month)")

    print("[monthly] done")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
