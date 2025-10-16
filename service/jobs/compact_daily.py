# service/jobs/compact_daily.py
# Compacte tous les snapshots 5min d'un jour UTC → 1 Parquet daily (schéma strict),
# puis (optionnel) supprime les snapshots/dossiers du jour uniquement si le jour est CLOS.
#
# Schéma STRICT :
# ts_utc, tbin_utc, station_id, bikes, capacity, mechanical, ebike, status,
# lat, lon, name, temp_C, precip_mm, wind_mps
#
# Env requis :
#   GCS_RAW_PREFIX       = gs://<bucket>/<root>/bronze
#   GCS_DAILY_PREFIX     = gs://<bucket>/<root>/daily
#   DAY                  = YYYY-MM-DD   (jour UTC suggéré)
# Options :
#   DELETE_AFTER_COMPACT = 1|0  (défaut 1)  —> ET seulement si day < today_UTC
#
# Exécution :
#   python -m service.jobs.compact_daily

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
    assert gcs_url.startswith("gs://")
    b, p = gcs_url[5:].split("/", 1)
    return b, p.rstrip("/")

def _list_day_parquets(raw_prefix: str, day: str) -> List[storage.Blob]:
    bkt, pfx = _split(raw_prefix)
    pfx = f"{pfx}/date={day}/"
    cli = storage.Client()
    return [b for b in cli.list_blobs(bkt, prefix=pfx) if b.name.endswith(".parquet")]

def _day_from_blob_path(blob: storage.Blob) -> str | None:
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
    buf = BytesIO()
    blob.download_to_file(buf)
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

def _purge_prefix_tree(raw_prefix: str, day: str):
    """Supprime tout sous gs://.../date=DAY/ (objets + 'markers')."""
    bkt, pfx = _split(raw_prefix)
    base = f"{pfx}/date={day}/"
    cli = storage.Client()
    bucket = cli.bucket(bkt)

    blobs = list(cli.list_blobs(bkt, prefix=base))
    if blobs:
        with cli.batch():
            for b in blobs:
                bucket.blob(b.name).delete()

    # Supprimer marqueurs possibles
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
    Parse en UTC et rend naïf (horodatage UTC conservé).
    - Naive → localise en UTC.
    - Aware → convertit vers UTC.
    - Sortie dtype datetime64[ns] naïf (UTC).
    """
    x = pd.to_datetime(s, utc=True, errors="coerce")
    return x.dt.tz_convert("UTC").dt.tz_localize(None)

def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    for c in COLS:
        if c not in df.columns:
            df[c] = None

    out = pd.DataFrame({
        "ts_utc":     _to_naive_utc(df["ts_utc"]),
        # IMPORTANT: respecter tbin_utc source si présent
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

    # station_id numérique dans la daily
    out["station_id"] = out["station_id"].astype("Int64")
    for c in ["bikes","capacity","mechanical","ebike"]:
        out[c] = out[c].astype("Int64")

    return out[COLS]

def _ensure_5min_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantit que tbin_utc est calé sur une grille 5 min.
    - Remplit uniquement les tbin_utc manquants à partir de ts_utc.floor('5min').
    - Pour les tbin_utc présents mais off-grid, les rabat sur floor('5min') (peu fréquent).
    """
    tbin = df["tbin_utc"].copy()

    # Remplir les NaT à partir de ts_utc
    mask_nat = tbin.isna()
    if mask_nat.any():
        tbin.loc[mask_nat] = pd.to_datetime(df.loc[mask_nat, "ts_utc"]).dt.floor("5min")

    # Corriger les valeurs hors-grille (pas multiple de 5 minutes)
    # On détecte en regardant les minutes % 5 ou les secondes non nulles
    minutes = tbin.dt.minute
    seconds = tbin.dt.second
    offgrid = ((minutes % 5) != 0) | (seconds != 0)
    if offgrid.any():
        n = int(offgrid.sum())
        print(f"[daily] fixing {n} tbin_utc values to 5-min floor")
        tbin.loc[offgrid] = tbin.loc[offgrid].dt.floor("5min")

    df["tbin_utc"] = tbin.astype("datetime64[ns]")  # naïf UTC
    return df

def _filter_day_utc(df: pd.DataFrame, day: str) -> pd.DataFrame:
    start = pd.Timestamp(f"{day} 00:00:00")
    end   = start + pd.Timedelta(days=1)
    # IMPORTANT: ne plus recalculer toute la colonne — seulement remplir/corriger
    df = _ensure_5min_grid(df)
    mask = (df["tbin_utc"] >= start) & (df["tbin_utc"] < end)
    kept = df.loc[mask].copy()
    dropped = len(df) - len(kept)
    if dropped:
        print(f"[daily] filtered out {dropped:,} rows outside UTC day {day}")
    return kept

def _deduplicate_latest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station_id","tbin_utc","ts_utc"])
    df = df.groupby(["station_id","tbin_utc"], as_index=False).tail(1)
    return df

def _is_closed_day(day: str) -> bool:
    today = datetime.now(timezone.utc).date()
    d = datetime.strptime(day, "%Y-%m-%d").date()
    return d < today

def _infer_single_day_from_tbin(df: pd.DataFrame) -> date:
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

    # Jour verrouillé par la structure de chemin
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
            print(f"[daily][note] blob-path day={path_day} differs from env DAY={DAY} — using path_day for filtering & naming")

    # Lecture des snapshots
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

    # Contrôle croisé de la date inférée
    try:
        day_actual = _infer_single_day_from_tbin(df_all)
        if str(day_actual) != path_day:
            print(f"[daily][warn] inferred day from data = {day_actual} but blob-path day = {path_day} — naming with path_day")
    except Exception as e:
        print(f"[daily][warn] could not infer day from data: {e}")

    # Écriture daily
    out_key_final = f"compact_{path_day}.parquet"
    out_final = f"{DAILY_PREFIX.rstrip('/')}/{out_key_final}"
    out_tmp   = f"{DAILY_PREFIX.rstrip('/')}/_tmp_compact_{path_day}_{int(datetime.now(timezone.utc).timestamp())}.parquet"

    _upload_parquet_df(df_all, out_tmp)
    _copy_blob(out_tmp, out_final)
    _delete_blob(out_tmp)
    print(f"[daily] wrote {len(df_all):,} rows → {out_final}")

    # Purge
    if DELETE_FLAG and _is_closed_day(path_day):
        _purge_prefix_tree(RAW_PREFIX, path_day)
        print(f"[daily] deleted snapshots & markers under {RAW_PREFIX}/date={path_day}/")
    else:
        print(f"[daily] keeping snapshots for {path_day} (open day or DELETE_AFTER_COMPACT=0)")

    print("[daily] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
