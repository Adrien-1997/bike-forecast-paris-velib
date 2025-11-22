# service/jobs/compact_daily.py

"""
Daily compaction job for the Velib Forecast pipeline.

This job:
- Compacts all 5-minute raw snapshots of a given UTC day into a single Parquet
  file with a strict schema.
- Optionally deletes the raw 5-minute snapshots and their folders for that day
  once the day is considered "closed" (day < today_UTC).

Strict schema
-------------
ts_utc, tbin_utc, station_id, bikes, capacity, mechanical, ebike, status,
lat, lon, name, temp_C, precip_mm, wind_mps

Required environment variables
------------------------------
GCS_RAW_PREFIX       = gs://<bucket>/<root>/bronze
GCS_DAILY_PREFIX     = gs://<bucket>/<root>/daily
DAY                  = YYYY-MM-DD   (suggested UTC day)

Options
-------
DELETE_AFTER_COMPACT = 1|0  (default 1)
    When 1, delete raw snapshots for the day, but only if DAY < today_UTC.

Execution
---------
python -m service.jobs.compact_daily
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import List, Tuple
from datetime import datetime, timezone, date

import pandas as pd

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception as e:
    raise RuntimeError("pyarrow est requis pour la compaction") from e

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage est requis") from e


COLS = [
    "ts_utc","tbin_utc","station_id","bikes","capacity","mechanical","ebike",
    "status","lat","lon","name","temp_C","precip_mm","wind_mps"
]

# ───────────────────────── Helpers GCS ─────────────────────────

def _split(gcs_url: str) -> Tuple[str, str]:
    """
    Split a GCS URL of the form 'gs://bucket/path' into bucket name and object key.

    Parameters
    ----------
    gcs_url : str
        GCS URL starting with 'gs://'.

    Returns
    -------
    (str, str)
        Tuple (bucket_name, object_key) where object_key has no trailing slash.
    """
    assert gcs_url.startswith("gs://")
    b, p = gcs_url[5:].split("/", 1)
    return b, p.rstrip("/")


def _list_day_parquets(raw_prefix: str, day: str) -> List[storage.Blob]:
    """
    List all 5-minute parquet snapshot blobs for a given UTC day.

    Parameters
    ----------
    raw_prefix : str
        GCS prefix for the bronze layer (GCS_RAW_PREFIX).
    day : str
        UTC day in 'YYYY-MM-DD' format.

    Returns
    -------
    list of google.cloud.storage.Blob
        All blobs under '<raw_prefix>/date=DAY/' ending with '.parquet'.
    """
    bkt, pfx = _split(raw_prefix)
    pfx = f"{pfx}/date={day}/"
    cli = storage.Client()
    return [b for b in cli.list_blobs(bkt, prefix=pfx) if b.name.endswith(".parquet")]


def _day_from_blob_path(blob: storage.Blob) -> str | None:
    """
    Extract the 'YYYY-MM-DD' part from a blob path containing 'date=YYYY-MM-DD/'.

    Parameters
    ----------
    blob : google.cloud.storage.Blob
        Blob whose name is inspected.

    Returns
    -------
    str or None
        The day string if found, otherwise None.
    """
    name = blob.name
    i = name.find("date=")
    if i == -1:
        return None
    seg = name[i:].split("/", 1)[0]  # "date=YYYY-MM-DD"
    parts = seg.split("=", 1)
    if len(parts) == 2:
        return parts[1]
    return None


def _download_parquet_to_df(blob: storage.Blob) -> pd.DataFrame:
    """
    Download a parquet blob from GCS and load it into a pandas DataFrame.

    Parameters
    ----------
    blob : google.cloud.storage.Blob
        Source parquet file.

    Returns
    -------
    pandas.DataFrame
        Loaded parquet content.
    """
    buf = BytesIO()
    blob.download_to_file(buf)
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


def _purge_prefix_tree(raw_prefix: str, day: str):
    """
    Delete all objects and potential "marker" blobs under gs://.../date=DAY/.

    This is used after a successful compaction of a closed day when
    DELETE_AFTER_COMPACT=1, to free space in the bronze layer.
    """
    bkt, pfx = _split(raw_prefix)
    base = f"{pfx}/date={day}/"
    cli = storage.Client()
    bucket = cli.bucket(bkt)

    blobs = list(cli.list_blobs(bkt, prefix=base))
    if blobs:
        with cli.batch():
            for b in blobs:
                bucket.blob(b.name).delete()

    # Also try to remove potential "directory marker" blobs.
    candidates = {base, base.rstrip('/') + '/'}
    for hh in range(24):
        hp = f"{base}hour={hh:02d}/"
        candidates.add(hp)
        candidates.add(hp.rstrip('/') + '/')
    for key in candidates:
        try:
            bucket.blob(key).delete()
        except Exception:
            pass

    leftovers = list(cli.list_blobs(bkt, prefix=base, max_results=5))
    if leftovers:
        print(f"[daily][warn] leftovers still under gs://{bkt}/{base}:")
        for b in leftovers:
            print(" -", b.name)
    else:
        print(f"[daily] prefix fully empty: gs://{bkt}/{base}")

# ───────────────────── Normalisation & Qualité ─────────────────────

def _to_naive_utc(s: pd.Series) -> pd.Series:
    """
    Parse timestamps as UTC and return naive datetime64[ns] values.

    Rules
    -----
    - Naive datetimes are localized as UTC.
    - Aware datetimes are converted to UTC.
    - Output is timezone-naive but still represents UTC instants.
    """
    x = pd.to_datetime(s, utc=True, errors="coerce")
    return x.dt.tz_convert("UTC").dt.tz_localize(None)


def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce the strict daily schema, coercing columns to expected dtypes.

    Missing columns are created and filled with None/NaN so that the
    resulting DataFrame has exactly COLS in the correct order.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw concatenated snapshots.

    Returns
    -------
    pandas.DataFrame
        DataFrame with normalized columns and dtypes.
    """
    for c in COLS:
        if c not in df.columns:
            df[c] = None

    out = pd.DataFrame({
        "ts_utc":     _to_naive_utc(df["ts_utc"]),
        # IMPORTANT: respect original tbin_utc if present
        "tbin_utc":   _to_naive_utc(df["tbin_utc"]) if "tbin_utc" in df.columns else pd.NaT,
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

    # station_id and core counts are stored as nullable integers in the daily parquet.
    out["station_id"] = out["station_id"].astype("Int64")
    for c in ["bikes","capacity","mechanical","ebike"]:
        out[c] = out[c].astype("Int64")

    return out[COLS]


def _ensure_5min_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that tbin_utc is aligned on a 5-minute grid.

    - Fill missing tbin_utc values from ts_utc.floor('5min').
    - For existing tbin_utc values that are off-grid (minutes % 5 != 0 or seconds != 0),
      snap them down to floor('5min').

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with at least ts_utc and tbin_utc.

    Returns
    -------
    pandas.DataFrame
        Same DataFrame with tbin_utc corrected in-place.
    """
    tbin = df["tbin_utc"].copy()

    # Fill NaT values from ts_utc.
    mask_nat = tbin.isna()
    if mask_nat.any():
        tbin.loc[mask_nat] = pd.to_datetime(df.loc[mask_nat, "ts_utc"]).dt.floor("5min")

    # Fix values that are off the 5-minute grid.
    minutes = tbin.dt.minute
    seconds = tbin.dt.second
    offgrid = ((minutes % 5) != 0) | (seconds != 0)
    if offgrid.any():
        n = int(offgrid.sum())
        print(f"[daily] fixing {n} tbin_utc values to 5-min floor")
        tbin.loc[offgrid] = tbin.loc[offgrid].dt.floor("5min")

    df["tbin_utc"] = tbin.astype("datetime64[ns]")  # naive UTC
    return df


def _filter_day_utc(df: pd.DataFrame, day: str) -> pd.DataFrame:
    """
    Keep only rows whose tbin_utc falls within the given UTC day.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame (typically already schema-normalized).
    day : str
        UTC day in 'YYYY-MM-DD' format.

    Returns
    -------
    pandas.DataFrame
        Filtered view for the requested day.
    """
    start = pd.Timestamp(f"{day} 00:00:00")
    end   = start + pd.Timedelta(days=1)
    # IMPORTANT: only fill/correct the grid, do not recompute the whole column from scratch
    df = _ensure_5min_grid(df)
    mask = (df["tbin_utc"] >= start) & (df["tbin_utc"] < end)
    kept = df.loc[mask].copy()
    dropped = len(df) - len(kept)
    if dropped:
        print(f"[daily] filtered out {dropped:,} rows outside UTC day {day}")
    return kept


def _deduplicate_latest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the latest snapshot per (station_id, tbin_utc) pair.

    Sorting order is by station_id, then tbin_utc, then ts_utc,
    and we take the last row in each group.

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


def _is_closed_day(day: str) -> bool:
    """
    Decide whether a day is considered "closed" (strictly before today UTC).

    Parameters
    ----------
    day : str
        UTC day in 'YYYY-MM-DD' format.

    Returns
    -------
    bool
        True if the day is < today_UTC, False otherwise.
    """
    today = datetime.now(timezone.utc).date()
    d = datetime.strptime(day, "%Y-%m-%d").date()
    return d < today


def _infer_single_day_from_tbin(df: pd.DataFrame) -> date:
    """
    Infer the unique UTC day present in tbin_utc.

    Used as a sanity check after compaction. Fails if multiple
    different dates are present or if tbin_utc is empty.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a 'tbin_utc' column.

    Returns
    -------
    datetime.date
        The unique day detected in tbin_utc.

    Raises
    ------
    RuntimeError
        If no date is found or if several distinct days are present.
    """
    if "tbin_utc" not in df.columns or df["tbin_utc"].isna().all():
        raise RuntimeError("[daily] impossible d'inférer la date: tbin_utc absent ou vide")
    days = df["tbin_utc"].dt.normalize().dt.date.unique()
    if len(days) == 0:
        raise RuntimeError("[daily] aucune date détectée dans tbin_utc")
    if len(days) > 1:
        raise RuntimeError(f"[daily] plusieurs dates détectées: {sorted(map[str, str], map(str, days))}")
    return days[0]

# ───────────────────────────── Main ─────────────────────────────

def main():
    """
    CLI entrypoint for the daily compaction job.

    Reads all 5-minute snapshots for DAY from GCS_RAW_PREFIX, normalizes and
    deduplicates them, writes a single compact Parquet file under
    GCS_DAILY_PREFIX, and optionally purges raw snapshots.
    """
    RAW_PREFIX   = os.environ.get("GCS_RAW_PREFIX")
    DAILY_PREFIX = os.environ.get("GCS_DAILY_PREFIX")
    DAY          = os.environ.get("DAY")
    DELETE_FLAG  = os.environ.get("DELETE_AFTER_COMPACT","1") in ("1","true","True")

    if not (RAW_PREFIX and RAW_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_RAW_PREFIX invalide ou manquant")
    if not (DAILY_PREFIX and DAILY_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_DAILY_PREFIX invalide ou manquant")
    if not DAY:
        DAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        print(f"[daily] DAY not provided → default to UTC today: {DAY}")

    print(f"[daily] start DAY={DAY} raw={RAW_PREFIX} out={DAILY_PREFIX} delete_after={int(DELETE_FLAG)}")

    blobs = _list_day_parquets(RAW_PREFIX, DAY)
    if not blobs:
        print(f"[daily] no snapshots for {DAY} — nothing to do")
        return 0
    print(f"[daily] found {len(blobs)} snapshot files")

    # Day is enforced by the "date=YYYY-MM-DD" path structure
    days_from_paths = {_day_from_blob_path(b) for b in blobs}
    days_from_paths.discard(None)
    if not days_from_paths:
        print(f"[daily][warn] could not parse 'date=YYYY-MM-DD' from blob paths; fallback to env DAY={DAY}")
        path_day = DAY
    elif len(days_from_paths) > 1:
        raise RuntimeError(f"[daily] multiple 'date=YYYY-MM-DD' detected in paths: {sorted(days_from_paths)}")
    else:
        path_day = next(iter(days_from_paths))
        if path_day != DAY:
            print(f"[daily][note] blob-path day={path_day} ...rs from env DAY={DAY} — using path_day for filtering & naming")

    # Read all snapshots
    dfs: List[pd.DataFrame] = []
    failed = 0
    for b in blobs:
        try:
            dfs.append(_download_parquet_to_df(b))
        except Exception as e:
            failed += 1
            print(f"[daily][warn] read failed: gs://{b.bucket.name}/{b.name}: {e}")

    if not dfs:
        print(f"[daily] all snapshots failed to read for {path_day}")
        return 0

    df_all = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"[daily] concatenated rows={len(df_all):,} (failed_files={failed})")

    df_all = _enforce_schema(df_all)
    df_all = _filter_day_utc(df_all, path_day)
    before_dedup = len(df_all)
    df_all = _deduplicate_latest(df_all)
    print(f"[daily] filtered rows={before_dedup:,} | dedup kept={len(df_all):,}")

    df_all = df_all.sort_values(["tbin_utc","station_id"]).reset_index(drop=True)

    bins_present = df_all["tbin_utc"].nunique()
    stations_present = df_all["station_id"].nunique()
    print(f"[daily] bins_present={bins_present} | stations_present={stations_present}")

    # Cross-check inferred day from data
    try:
        day_actual = _infer_single_day_from_tbin(df_all)
        if str(day_actual) != path_day:
            print(f"[daily][warn] inferred day from data = {day_actual} but blob-path day = {path_day} — naming with path_day")
    except Exception as e:
        print(f"[daily][warn] could not infer day from data: {e}")

    # Write daily parquet (atomic pattern: tmp → final)
    out_key_final = f"compact_{path_day}.parquet"
    out_final = f"{DAILY_PREFIX.rstrip('/')}/{out_key_final}"
    out_tmp   = f"{DAILY_PREFIX.rstrip('/')}/_tmp_compact_{path_day}_{int(datetime.now(timezone.utc).timestamp())}.parquet"

    _upload_parquet_df(df_all, out_tmp)
    _copy_blob(out_tmp, out_final)
    _delete_blob(out_tmp)
    print(f"[daily] wrote {len(df_all):,} rows → {out_final}")

    # Purge raw snapshots if requested and the day is closed
    if DELETE_FLAG and _is_closed_day(path_day):
        _purge_prefix_tree(RAW_PREFIX, path_day)
        print(f"[daily] deleted snapshots & markers under {RAW_PREFIX}/date={path_day}/")
    else:
        print(f"[daily] keeping snapshots for {path_day} (open day or DELETE_AFTER_COMPACT=0)")

    print("[daily] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
