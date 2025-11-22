# service/jobs/compact_monthly.py

"""
Monthly compaction job for the Velib Forecast pipeline.

This job:
- Compacts all daily Parquet shards of a given UTC month into a single
  monthly Parquet file with a strict schema.
- Optionally deletes the daily shards once the month is considered *closed*
  (i.e. month < current UTC month).

Strict schema
-------------
ts_utc, tbin_utc, station_id, bikes, capacity, mechanical, ebike, status,
lat, lon, name, temp_C, precip_mm, wind_mps

Required environment variables
------------------------------
GCS_DAILY_PREFIX   = gs://<bucket>/<root>/daily
GCS_MONTHLY_PREFIX = gs://<bucket>/<root>/monthly
MONTH              = YYYY-MM  (suggested UTC month)

Options
-------
DRY_RUN = "1"
    When "1", the job does not write or delete anything and only logs what
    would be done.

Execution
---------
python -m jobs.compact_monthly
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import List, Tuple, Set
from datetime import datetime, timezone
import re
import pandas as pd

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception as e:
    raise RuntimeError("pyarrow est requis pour compact_monthly.py") from e

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage est requis") from e


COLS = [
    "ts_utc","tbin_utc","station_id","bikes","capacity","mechanical","ebike",
    "status","lat","lon","name","temp_C","precip_mm","wind_mps"
]

# ──────────────────────────── GCS Helpers ────────────────────────────

def _split(gcs_url: str) -> Tuple[str, str]:
    """
    Split a GCS URL of the form 'gs://bucket/path' into (bucket, key).

    Parameters
    ----------
    gcs_url : str
        GCS URL starting with 'gs://'.

    Returns
    -------
    (str, str)
        Tuple (bucket_name, object_key) with the key stripped of trailing '/'.
    """
    assert gcs_url.startswith("gs://")
    b, p = gcs_url[5:].split("/", 1)
    return b, p.rstrip("/")


def _daily_key_to_date(key: str) -> datetime | None:
    """
    Parse a daily Parquet object key and extract its UTC date.

    Accepted naming patterns:
    - compact_YYYY-MM-DD.parquet
    - velib_YYYYMMDD.parquet

    Parameters
    ----------
    key : str
        Full object key in the bucket.

    Returns
    -------
    datetime or None
        Parsed UTC date if the key matches a known pattern, otherwise None.
    """
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
    """
    List all daily Parquet files for a given UTC month.

    Parameters
    ----------
    daily_prefix : str
        GCS prefix for the daily layer (GCS_DAILY_PREFIX).
    month : str
        Target month in 'YYYY-MM' format (UTC).

    Returns
    -------
    list of (str, str)
        List of (bucket, key) tuples for all matching daily files.
    """
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


def _months_from_daily_keys(daily_files: List[Tuple[str, str]]) -> Set[str]:
    """
    Infer the set of 'YYYY-MM' months from a list of daily file keys.

    Parameters
    ----------
    daily_files : list of (str, str)
        List of (bucket, key) tuples for daily files.

    Returns
    -------
    set of str
        Unique months (YYYY-MM) derived from the filenames.
    """
    months: Set[str] = set()
    for _, key in daily_files:
        d = _daily_key_to_date(key)
        if d is not None:
            months.add(d.strftime("%Y-%m"))
    return months


def _download_parquet_to_df(bkt: str, key: str) -> pd.DataFrame:
    """
    Download a parquet file from GCS and load it as a pandas DataFrame.

    Parameters
    ----------
    bkt : str
        Bucket name.
    key : str
        Object key.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the parquet content.
    """
    buf = BytesIO()
    storage.Client().bucket(bkt).blob(key).download_to_file(buf)
    buf.seek(0)
    table = pq.read_table(buf)
    return table.to_pandas()


def _upload_parquet_df(df: pd.DataFrame, gcs_url: str):
    """
    Upload a pandas DataFrame as a parquet file to a GCS location.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to upload.
    gcs_url : str
        Destination GCS URL (gs://bucket/path/to/file.parquet).
    """
    bkt, key = _split(gcs_url)
    buf = BytesIO()
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)
    storage.Client().bucket(bkt).blob(key).upload_from_file(
        buf,
        content_type="application/octet-stream",
    )


def _copy_blob(src_url: str, dst_url: str):
    """
    Copy a blob from one GCS location to another.

    Parameters
    ----------
    src_url : str
        Source GCS URL.
    dst_url : str
        Destination GCS URL.
    """
    src_bkt, src_key = _split(src_url)
    dst_bkt, dst_key = _split(dst_url)
    cli = storage.Client()
    dst_bucket = cli.bucket(dst_bkt)
    src_bucket = cli.bucket(src_bkt)
    src_blob = src_bucket.blob(src_key)
    dst_bucket.copy_blob(src_blob, dst_bucket, dst_key)


def _delete_blob(url: str):
    """
    Delete a blob at the given GCS URL.

    Parameters
    ----------
    url : str
        GCS URL of the blob to delete.
    """
    bkt, key = _split(url)
    cli = storage.Client()
    cli.bucket(bkt).blob(key).delete()


def _delete_keys(keys: List[Tuple[str, str]]):
    """
    Delete multiple blobs in batch.

    Parameters
    ----------
    keys : list of (str, str)
        List of (bucket, key) tuples to delete.
    """
    if not keys:
        return
    cli = storage.Client()
    with cli.batch():
        for bkt, key in keys:
            cli.bucket(bkt).blob(key).delete()

# ──────────────────────────── Normalisation ────────────────────────────

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce the strict monthly schema and coerce dtypes.

    Missing columns are created and filled with None/NaN so that the resulting
    DataFrame has exactly COLS in the correct order.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw concatenated daily data.

    Returns
    -------
    pandas.DataFrame
        DataFrame with normalized columns and dtypes.
    """
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
    """
    Keep only rows whose tbin_utc falls within the given UTC month.

    If tbin_utc is missing or contains NaNs, it is rebuilt from ts_utc.floor('5min').

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame (typically already schema-normalized).
    month : str
        UTC month in 'YYYY-MM' format.

    Returns
    -------
    pandas.DataFrame
        Filtered view for the requested month.
    """
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
    """
    Keep only the latest snapshot per (station_id, tbin_utc) pair.

    Sorting order is by station_id, then tbin_utc, then ts_utc,
    and the last row in each group is retained.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing potential duplicates.

    Returns
    -------
    pandas.DataFrame
        Deduplicated DataFrame with one row per (station_id, tbin_utc).
    """
    df = df.sort_values(["station_id","tbin_utc","ts_utc"])
    df = df.groupby(["station_id","tbin_utc"], as_index=False).tail(1)
    return df


def _is_closed_month(month: str) -> bool:
    """
    Check whether a given UTC month is strictly before the current UTC month.

    Example:
    - '2025-09' < '2025-10' → True

    Parameters
    ----------
    month : str
        Month in 'YYYY-MM' format.

    Returns
    -------
    bool
        True if the month is closed (strictly < current UTC month).
    """
    now = datetime.now(timezone.utc)
    current = (now.year, now.month)
    y, m = map(int, month.split("-", 1))
    return (y, m) < current

# ─────────────────────────────── Main ───────────────────────────────

def main():
    """
    CLI entrypoint for the monthly compaction job.

    Steps
    -----
    1. List all daily Parquet shards for MONTH under GCS_DAILY_PREFIX.
    2. Normalize schema and filter rows to the target month.
    3. Deduplicate snapshots per (station_id, tbin_utc).
    4. Write a single compact monthly Parquet file under GCS_MONTHLY_PREFIX.
    5. If the month is closed and DRY_RUN != "1", delete all daily shards.
    """
    DAILY_PREFIX   = os.environ.get("GCS_DAILY_PREFIX")
    MONTHLY_PREFIX = os.environ.get("GCS_MONTHLY_PREFIX")
    MONTH          = os.environ.get("MONTH")
    DRY_RUN        = os.environ.get("DRY_RUN","0") == "1"

    if not (DAILY_PREFIX and DAILY_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_DAILY_PREFIX invalide ou manquant")
    if not (MONTHLY_PREFIX and MONTHLY_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_MONTHLY_PREFIX invalide ou manquant")
    if not MONTH:
        MONTH = datetime.now(timezone.utc).strftime("%Y-%m")
        print(f"[monthly] MONTH not provided → default to current UTC month: {MONTH}")

    print(f"[monthly] start MONTH={MONTH} daily={DAILY_PREFIX} out={MONTHLY_PREFIX} dry_run={int(DRY_RUN)}")

    # 1) Lister les dailies selon MONTH demandé
    daily_files = _list_month_dailies(DAILY_PREFIX, MONTH)
    if not daily_files:
        print(f"[monthly] aucun daily pour {MONTH} — rien à faire")
        return 0

    # 2) Verrouiller le mois depuis les chemins (source de vérité pour nommage & filtrage)
    months_from_paths = _months_from_daily_keys(daily_files)
    if not months_from_paths:
        print(f"[monthly][warn] impossible d'inférer YYYY-MM depuis les noms des dailies; fallback sur env MONTH={MONTH}")
        path_month = MONTH
    elif len(months_from_paths) > 1:
        raise RuntimeError(f"[monthly] plusieurs mois détectés dans les noms: {sorted(months_from_paths)}")
    else:
        path_month = next(iter(months_from_paths))
        if path_month != MONTH:
            print(f"[monthly][note] path_month={path_month} diffère de env MONTH={MONTH} — utilisation de path_month pour filtrage & nommage")

    print(f"[monthly] {len(daily_files)} shards trouvés pour {path_month}")
    for _, key in daily_files[:5]:
        print("  -", key)
    if len(daily_files) > 5:
        print(f"  … (+{len(daily_files)-5} autres)")

    # 3) Lecture + concat
    dfs: List[pd.DataFrame] = []
    failed = 0
    for bkt, key in daily_files:
        try:
            dfs.append(_download_parquet_to_df(bkt, key))
        except Exception as e:
            failed += 1
            print(f"[monthly][warn] lecture échouée gs://{bkt}/{key}: {e}")

    if not dfs:
        print(f"[monthly] lecture impossible sur {path_month}")
        return 0

    df_all = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"[monthly] concatenated rows={len(df_all):,} (failed_files={failed})")

    # 4) Schéma, filtre (sur path_month), dédup, tri
    df_all = _ensure_schema(df_all)
    df_all = _filter_month_utc(df_all, path_month)
    before_dedup = len(df_all)
    df_all = _deduplicate_latest(df_all)
    print(f"[monthly] filtered rows={before_dedup:,} | dedup kept={len(df_all):,}")

    df_all = df_all.sort_values(["tbin_utc","station_id"]).reset_index(drop=True)

    bins_present    = df_all["tbin_utc"].nunique()
    stations_unique = df_all["station_id"].nunique()
    print(f"[monthly] bins_present={bins_present} | stations_unique={stations_unique}")

    # 5) Nommage STRICT basé sur le mois des dailies (path_month)
    yyyymm = path_month.replace("-", "")
    out_key_final = f"compact_{yyyymm}.parquet"
    out_final = f"{MONTHLY_PREFIX.rstrip('/')}/{out_key_final}"
    out_tmp   = f"{MONTHLY_PREFIX.rstrip('/')}/_tmp_compact_{yyyymm}_{int(datetime.now(timezone.utc).timestamp())}.parquet"

    if DRY_RUN:
        print(f"[monthly][dry-run] écrirait {len(df_all):,} lignes → {out_final}")
    else:
        _upload_parquet_df(df_all, out_tmp)
        _copy_blob(out_tmp, out_final)
        _delete_blob(out_tmp)
        print(f"[monthly] wrote {len(df_all):,} rows → {out_final}")

        # 6) Purge alignée sur path_month
        if _is_closed_month(path_month):
            _delete_keys(daily_files)
            print(f"[monthly] deleted {len(daily_files)} daily shards for {path_month}")
        else:
            print(f"[monthly] keeping daily shards for {path_month} (open month)")

    print("[monthly] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
