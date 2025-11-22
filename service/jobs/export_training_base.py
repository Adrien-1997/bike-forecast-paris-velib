# service/jobs/export_training_base.py

"""
Base training export job for the Velib Forecast pipeline.

This job:
- Scans a "daily" directory (local or GCS) containing daily Parquet files.
- Concatenates all matching daily Parquet shards into a single training base.
- Keeps only the required columns used for model training.
- Normalizes timestamps to naive UTC (ts_utc, tbin_utc).
- Writes the result either:
  - as a local Parquet file, or
  - directly to a GCS Parquet object (via a local temporary file).
- Optionally publishes an additional *dated* training export under a GCS prefix.

Input daily files (flexible naming)
-----------------------------------
The job accepts both historical formats:
- velib_YYYYMMDD.parquet
- compact_YYYY-MM-DD.parquet

and can work on:
- a local directory (recursively),
- or a GCS prefix (gs://bucket/path).

Required columns in output
--------------------------
ts_utc, tbin_utc, station_id,
bikes, capacity, mechanical, ebike, status,
lat, lon, name,
temp_C, precip_mm, wind_mps

Environment variables
---------------------
DAILY_DIR : str, default "data_local/daily"
    Root path for daily Parquets (local folder or GCS gs:// prefix).

TRAIN_EXPORT : str, default "exports/velib.parquet"
    Final training export path. Can be:
      - local path (e.g. "exports/velib.parquet")
      - GCS path (e.g. "gs://bucket/velib/training/latest.parquet")

TRAIN_EXPORT_GCS_PREFIX : str (optional)
    When set to a GCS prefix (gs://...), publish an additional dated version:
    "<prefix>/velib_training_YYYYMMDD.parquet".

TMP_OUT : str, default "/tmp/velib_training.parquet"
    Temporary local file used when uploading to GCS.

Execution
---------
Typical usage:

    python -m service.jobs.export_training_base
"""

from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import List
import pandas as pd

try:
    import pyarrow.dataset as ds  # noqa: F401
    import pyarrow.parquet as pq
except Exception:
    ds = pq = None

try:
    from google.cloud import storage  # optional, only if writing/reading GCS
except Exception:
    storage = None  # type: ignore

REQ_COLS = [
    "ts_utc","tbin_utc","station_id",
    "bikes","capacity","mechanical","ebike","status",
    "lat","lon","name",
    "temp_C","precip_mm","wind_mps",
]

# ───────────────────────────── helpers ─────────────────────────────

def _is_gcs(path: str) -> bool:
    """
    Return True if the given path is a GCS URL (starts with 'gs://').

    Parameters
    ----------
    path : str
        Path or URL to test.

    Returns
    -------
    bool
        True if the path starts with 'gs://', False otherwise.
    """
    return str(path).startswith("gs://")


def _split_gcs(url: str) -> tuple[str, str]:
    """
    Split a GCS URL of the form 'gs://bucket/path' into bucket and object key.

    Parameters
    ----------
    url : str
        GCS URL starting with 'gs://'.

    Returns
    -------
    (str, str)
        Tuple (bucket_name, object_key).
    """
    assert url.startswith("gs://")
    b, p = url[5:].split("/", 1)
    return b, p


def _list_daily_paths(root: str) -> list[str]:
    """
    List all daily Parquet files under a local directory or a GCS prefix.

    Accepted filename patterns:
    - velib_YYYYMMDD.parquet
    - compact_YYYY-MM-DD.parquet

    For local paths:
        - Recursively walks the directory and collects all *.parquet
          matching the patterns.

    For GCS prefixes:
        - Lists all blobs under the bucket/prefix and keeps only those
          whose names end with '.parquet' and match one of the patterns.

    Parameters
    ----------
    root : str
        Local root directory or GCS URL (gs://bucket/prefix).

    Returns
    -------
    list of str
        Sorted list of matching local or 'gs://'-style paths.
    """
    import re
    pat1 = re.compile(r".*/velib_\d{8}\.parquet$")
    pat2 = re.compile(r".*/compact_\d{4}-\d{2}-\d{2}\.parquet$")
    out: List[str] = []

    if _is_gcs(root):
        if storage is None:
            raise RuntimeError("google-cloud-storage required to read GCS paths")
        bkt, pfx = _split_gcs(root)
        cli = storage.Client()
        for b in cli.list_blobs(bkt, prefix=pfx):
            if b.name.endswith(".parquet") and (pat1.match(b.name) or pat2.match(b.name)):
                out.append(f"gs://{bkt}/{b.name}")
    else:
        p = Path(root)
        # Recursive search to tolerate nested layout
        for f in p.rglob("*.parquet"):
            s = str(f)
            if pat1.match(s) or pat2.match(s):
                out.append(s)

    return sorted(out)


def _read_parquets(paths: list[str]) -> pd.DataFrame:
    """
    Read and concatenate a list of Parquet files into a single DataFrame.

    Strategy
    --------
    - If pyarrow is available and all paths are local, attempt a single
      `pq.read_table(paths)` for efficiency.
    - Otherwise, fall back to reading each file individually (works for both
      local files and gs:// URLs, provided google-cloud-storage is available).

    Parameters
    ----------
    paths : list of str
        List of local paths or GCS URLs.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame. If no file can be read, an empty DataFrame
        with REQ_COLS as columns is returned.
    """
    if not paths:
        return pd.DataFrame(columns=REQ_COLS)

    if pq is not None:
        # Fast path: all-local list, read in one call
        try:
            if all(not _is_gcs(p) for p in paths):
                tbl = pq.read_table(paths)
                return tbl.to_pandas()
        except Exception:
            # If anything goes wrong, we will fall back to per-file reading
            pass

    # Fallback: read one by one (local or GCS)
    parts = []
    for pth in paths:
        try:
            if _is_gcs(pth):
                if storage is None:
                    raise RuntimeError("google-cloud-storage required to read GCS paths")
                from io import BytesIO
                bkt, key = _split_gcs(pth)
                buf = BytesIO()
                storage.Client().bucket(bkt).blob(key).download_to_file(buf)
                buf.seek(0)
                if pq is not None:
                    parts.append(pq.read_table(buf).to_pandas())
                else:
                    parts.append(pd.read_parquet(buf))
            else:
                parts.append(pd.read_parquet(pth))
        except Exception as e:
            print(f"[export_training_base][warn] read failed: {pth} — {e}")

    if not parts:
        return pd.DataFrame(columns=REQ_COLS)
    return pd.concat(parts, ignore_index=True, sort=False)


def _write_local_parquet(df: pd.DataFrame, out_path: str):
    """
    Write a DataFrame as a local Parquet file, creating parent dirs if needed.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to serialize.
    out_path : str
        Local file path where the Parquet file will be written.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[export_training_base] wrote local → {out_path}")


def _upload_file_to_gcs(local_path: str, gcs_url: str):
    """
    Upload an existing local file to a GCS location.

    Parameters
    ----------
    local_path : str
        Path to the file on the local filesystem.
    gcs_url : str
        GCS URL destination (gs://bucket/path/to/file.parquet).

    Raises
    ------
    RuntimeError
        If google-cloud-storage is not installed.
    """
    if storage is None:
        raise RuntimeError("google-cloud-storage required to write to GCS")
    bkt, key = _split_gcs(gcs_url)
    storage.Client().bucket(bkt).blob(key).upload_from_filename(
        local_path,
        content_type="application/octet-stream",
    )
    print(f"[export_training_base] uploaded → {gcs_url}")

# ───────────────────────────── main ─────────────────────────────

def main():
    """
    CLI entrypoint for exporting the base training dataset.

    Steps
    -----
    1. Discover daily Parquet files under DAILY_DIR (local or GCS).
    2. Read and concatenate all matching Parquet shards.
    3. Keep only REQ_COLS and coerce timestamps to naive UTC.
    4. Depending on TRAIN_EXPORT:
       - If local path → write directly to TRAIN_EXPORT
         - optionally also upload a dated copy under TRAIN_EXPORT_GCS_PREFIX.
       - If GCS path  → write to TMP_OUT then upload to TRAIN_EXPORT
         - optionally also upload a dated copy under TRAIN_EXPORT_GCS_PREFIX.

    Environment variables
    ---------------------
    DAILY_DIR : str, default "data_local/daily"
        Base directory or GCS prefix for daily Parquet files.

    TRAIN_EXPORT : str, default "exports/velib.parquet"
        Local or GCS path for the final training base.

    TRAIN_EXPORT_GCS_PREFIX : str (optional)
        If provided and starts with 'gs://', an extra dated copy
        'velib_training_YYYYMMDD.parquet' is uploaded under this prefix.

    TMP_OUT : str, default "/tmp/velib_training.parquet"
        Temporary local path used when uploading to GCS.
    """
    # Input root (local or GCS)
    DAILY_DIR  = os.environ.get("DAILY_DIR", "data_local/daily")   # gs://... or folder

    # Output configuration
    # 1) Final training export (local OR gs://…/xxx.parquet)
    TRAIN_EXPORT = os.environ.get("TRAIN_EXPORT", "exports/velib.parquet")
    # 2) Optional GCS prefix for an extra dated version
    TRAIN_EXPORT_GCS_PREFIX = os.environ.get("TRAIN_EXPORT_GCS_PREFIX")  # optional
    # 3) Temporary local file (used for uploading to GCS)
    TMP_OUT = os.environ.get("TMP_OUT", "/tmp/velib_training.parquet")

    paths = _list_daily_paths(DAILY_DIR)
    if not paths:
        print("[export_training_base] no daily parquet found")
        return 0

    df = _read_parquets(paths)

    # Keep only required columns (create missing ones as None)
    for c in REQ_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[REQ_COLS].copy()

    # Normalize timestamps to naive UTC
    df["tbin_utc"] = pd.to_datetime(df["tbin_utc"], utc=True, errors="coerce").dt.tz_convert(None)
    df["ts_utc"]   = pd.to_datetime(df["ts_utc"],   utc=True, errors="coerce").dt.tz_convert(None)

    wrote_any = False

    # Case A — TRAIN_EXPORT is a local file
    if not _is_gcs(TRAIN_EXPORT):
        _write_local_parquet(df, TRAIN_EXPORT)
        wrote_any = True

        # Optionally also push a dated version to GCS
        if TRAIN_EXPORT_GCS_PREFIX and TRAIN_EXPORT_GCS_PREFIX.startswith("gs://"):
            Path(TMP_OUT).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(TMP_OUT, index=False)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
            gcs_url = f"{TRAIN_EXPORT_GCS_PREFIX.rstrip('/')}/velib_training_{stamp}.parquet"
            _upload_file_to_gcs(TMP_OUT, gcs_url)

    # Case B — TRAIN_EXPORT is a direct GCS object
    else:
        # Write to local TMP_OUT first, then upload exactly to TRAIN_EXPORT
        Path(TMP_OUT).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(TMP_OUT, index=False)
        _upload_file_to_gcs(TMP_OUT, TRAIN_EXPORT)
        wrote_any = True

        # Optionally also push a dated version under TRAIN_EXPORT_GCS_PREFIX
        if TRAIN_EXPORT_GCS_PREFIX and TRAIN_EXPORT_GCS_PREFIX.startswith("gs://"):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
            gcs_url = f"{TRAIN_EXPORT_GCS_PREFIX.rstrip('/')}/velib_training_{stamp}.parquet"
            _upload_file_to_gcs(TMP_OUT, gcs_url)

    if not wrote_any:
        print("[export_training_base] nothing written (check TRAIN_EXPORT and/or TRAIN_EXPORT_GCS_PREFIX)")

    print(f"[export_training_base] rows={len(df):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
