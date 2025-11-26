# service/jobs/compact_daily.py

"""
Job de compaction quotidienne pour le pipeline Vélib’ Forecast.

Rôle
----
Ce job :
- compacte tous les snapshots bruts de 5 minutes d’une journée UTC donnée
  en un unique fichier Parquet avec un schéma strict ;
- supprime optionnellement les snapshots bruts 5 minutes (et leurs dossiers)
  pour cette journée, une fois qu’elle est considérée comme "fermée"
  (day < today_UTC).

Schéma strict
-------------
ts_utc, tbin_utc, station_id, bikes, capacity, mechanical, ebike, status,
lat, lon, name, temp_C, precip_mm, wind_mps

Variables d’environnement requises
----------------------------------
GCS_RAW_PREFIX       = gs://<bucket>/<root>/bronze
GCS_DAILY_PREFIX     = gs://<bucket>/<root>/daily
DAY                  = YYYY-MM-DD   (jour UTC suggéré)

Options
-------
DELETE_AFTER_COMPACT = 1|0  (par défaut 1)
    Quand vaut 1, supprime les snapshots bruts de la journée,
    mais uniquement si DAY < today_UTC.

Exécution
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
    Découper une URL GCS de la forme 'gs://bucket/path' en (bucket, clé).

    Paramètres
    ----------
    gcs_url : str
        URL GCS commençant par 'gs://'.

    Retour
    ------
    (str, str)
        Tuple (bucket_name, object_key), où object_key ne se termine pas
        par un slash.
    """
    assert gcs_url.startswith("gs://")
    b, p = gcs_url[5:].split("/", 1)
    return b, p.rstrip("/")


def _list_day_parquets(raw_prefix: str, day: str) -> List[storage.Blob]:
    """
    Lister tous les blobs parquet de snapshots 5 minutes pour un jour UTC donné.

    Paramètres
    ----------
    raw_prefix : str
        Préfixe GCS pour la couche bronze (GCS_RAW_PREFIX).
    day : str
        Jour UTC au format 'YYYY-MM-DD'.

    Retour
    ------
    list[google.cloud.storage.Blob]
        Tous les blobs sous '<raw_prefix>/date=DAY/' se terminant par '.parquet'.
    """
    bkt, pfx = _split(raw_prefix)
    pfx = f"{pfx}/date={day}/"
    cli = storage.Client()
    return [b for b in cli.list_blobs(bkt, prefix=pfx) if b.name.endswith(".parquet")]


def _day_from_blob_path(blob: storage.Blob) -> str | None:
    """
    Extraire la partie 'YYYY-MM-DD' d’un chemin de blob contenant 'date=YYYY-MM-DD/'.

    Paramètres
    ----------
    blob : google.cloud.storage.Blob
        Blob dont le nom est inspecté.

    Retour
    ------
    str ou None
        La chaîne du jour si trouvée, sinon None.
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
    Télécharger un blob parquet depuis GCS et le charger dans un DataFrame pandas.

    Paramètres
    ----------
    blob : google.cloud.storage.Blob
        Fichier parquet source.

    Retour
    ------
    pandas.DataFrame
        Contenu parquet chargé.
    """
    buf = BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    table = pq.read_table(buf)
    return table.to_pandas()


def _upload_parquet_df(df: pd.DataFrame, gcs_url: str):
    """
    Uploader un DataFrame pandas au format parquet vers une URL GCS.

    Paramètres
    ----------
    df : pandas.DataFrame
        Données à uploader.
    gcs_url : str
        URL GCS de destination (gs://bucket/path/to/file.parquet).
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
    Copier un blob d’une localisation GCS à une autre.

    Paramètres
    ----------
    src_url : str
        URL GCS source.
    dst_url : str
        URL GCS de destination.
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
    Supprimer un blob à l’URL GCS donnée.

    Paramètres
    ----------
    url : str
        URL GCS du blob à supprimer.
    """
    bkt, key = _split(url)
    cli = storage.Client()
    cli.bucket(bkt).blob(key).delete()


def _purge_prefix_tree(raw_prefix: str, day: str):
    """
    Supprimer tous les objets (et éventuels "marker" blobs) sous gs://.../date=DAY/.

    Utilisé après une compaction réussie d’une journée fermée lorsque
    DELETE_AFTER_COMPACT=1, pour libérer de l’espace dans la couche bronze.
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

    # Essayer aussi de supprimer d’éventuels blobs "marqueurs de dossier".
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
    Parser des timestamps en UTC et retourner des datetime64[ns] naïfs.

    Règles
    ------
    - Les datetimes naïfs sont localisés en UTC.
    - Les datetimes "aware" sont convertis en UTC.
    - La sortie est timezone-naïve mais représente toujours un instant UTC.
    """
    x = pd.to_datetime(s, utc=True, errors="coerce")
    return x.dt.tz_convert("UTC").dt.tz_localize(None)


def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appliquer le schéma quotidien strict, en forçant les dtypes attendus.

    Les colonnes manquantes sont créées et remplies par None/NaN, de sorte
    que le DataFrame résultant contienne exactement COLS dans le bon ordre.

    Paramètres
    ----------
    df : pandas.DataFrame
        Snapshots concaténés bruts.

    Retour
    ------
    pandas.DataFrame
        DataFrame avec colonnes et dtypes normalisés.
    """
    for c in COLS:
        if c not in df.columns:
            df[c] = None

    out = pd.DataFrame({
        "ts_utc":     _to_naive_utc(df["ts_utc"]),
        # IMPORTANT : respecter le tbin_utc original s’il est présent
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

    # station_id et les compteurs principaux sont stockés en entiers nullable
    # dans le parquet quotidien.
    out["station_id"] = out["station_id"].astype("Int64")
    for c in ["bikes","capacity","mechanical","ebike"]:
        out[c] = out[c].astype("Int64")

    return out[COLS]


def _ensure_5min_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    S’assurer que tbin_utc est bien aligné sur une grille de 5 minutes.

    - Remplit les valeurs tbin_utc manquantes à partir de ts_utc.floor('5min').
    - Pour les tbin_utc existants qui ne sont pas pile sur la grille
      (minutes % 5 != 0 ou secondes != 0), les aligne vers le bas avec floor('5min').

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame contenant au moins ts_utc et tbin_utc.

    Retour
    ------
    pandas.DataFrame
        Même DataFrame avec tbin_utc corrigé in-place.
    """
    tbin = df["tbin_utc"].copy()

    # Remplir les NaT à partir de ts_utc.
    mask_nat = tbin.isna()
    if mask_nat.any():
        tbin.loc[mask_nat] = pd.to_datetime(df.loc[mask_nat, "ts_utc"]).dt.floor("5min")

    # Corriger les valeurs hors grille 5 minutes.
    minutes = tbin.dt.minute
    seconds = tbin.dt.second
    offgrid = ((minutes % 5) != 0) | (seconds != 0)
    if offgrid.any():
        n = int(offgrid.sum())
        print(f"[daily] fixing {n} tbin_utc values to 5-min floor")
        tbin.loc[offgrid] = tbin.loc[offgrid].dt.floor("5min")

    df["tbin_utc"] = tbin.astype("datetime64[ns]")  # UTC naïf
    return df


def _filter_day_utc(df: pd.DataFrame, day: str) -> pd.DataFrame:
    """
    Ne conserver que les lignes dont tbin_utc tombe dans la journée UTC donnée.

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame d’entrée (généralement déjà normalisé au niveau du schéma).
    day : str
        Jour UTC au format 'YYYY-MM-DD'.

    Retour
    ------
    pandas.DataFrame
        Vue filtrée pour la journée demandée.
    """
    start = pd.Timestamp(f"{day} 00:00:00")
    end   = start + pd.Timedelta(days=1)
    # IMPORTANT : corriger la grille uniquement, ne pas recalculer toute la colonne
    df = _ensure_5min_grid(df)
    mask = (df["tbin_utc"] >= start) & (df["tbin_utc"] < end)
    kept = df.loc[mask].copy()
    dropped = len(df) - len(kept)
    if dropped:
        print(f"[daily] filtered out {dropped:,} rows outside UTC day {day}")
    return kept


def _deduplicate_latest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ne garder que le dernier snapshot par couple (station_id, tbin_utc).

    L’ordre de tri est : station_id, puis tbin_utc, puis ts_utc.
    On prend la dernière ligne dans chaque groupe.

    Paramètres
    ----------
    df : pandas.DataFrame
        Données pouvant contenir des doublons.

    Retour
    ------
    pandas.DataFrame
        DataFrame dédupliqué avec une ligne par (station_id, tbin_utc).
    """
    df = df.sort_values(["station_id","tbin_utc","ts_utc"])
    df = df.groupby(["station_id","tbin_utc"], as_index=False).tail(1)
    return df


def _is_closed_day(day: str) -> bool:
    """
    Indiquer si une journée est considérée comme "fermée" (strictement avant today UTC).

    Paramètres
    ----------
    day : str
        Jour UTC au format 'YYYY-MM-DD'.

    Retour
    ------
    bool
        True si day < today_UTC, False sinon.
    """
    today = datetime.now(timezone.utc).date()
    d = datetime.strptime(day, "%Y-%m-%d").date()
    return d < today


def _infer_single_day_from_tbin(df: pd.DataFrame) -> date:
    """
    Inférer l’unique jour UTC présent dans tbin_utc.

    Utilisé comme sanity check après compaction. Échoue s’il y a plusieurs
    dates différentes ou si tbin_utc est vide.

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame contenant une colonne 'tbin_utc'.

    Retour
    ------
    datetime.date
        La journée unique détectée dans tbin_utc.

    Lève
    ----
    RuntimeError
        Si aucune date n’est trouvée ou si plusieurs jours distincts sont présents.
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
    Point d’entrée CLI pour le job de compaction quotidienne.

    Lit tous les snapshots 5 minutes pour DAY depuis GCS_RAW_PREFIX,
    les normalise et les déduplique, écrit un unique fichier Parquet compact
    sous GCS_DAILY_PREFIX, et purge optionnellement les snapshots bruts.
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

    # Le jour est imposé par la structure de chemin "date=YYYY-MM-DD"
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

    # Lecture de tous les snapshots
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

    # Vérification croisée du jour inféré depuis les données
    try:
        day_actual = _infer_single_day_from_tbin(df_all)
        if str(day_actual) != path_day:
            print(f"[daily][warn] inferred day from data = {day_actual} but blob-path day = {path_day} — naming with path_day")
    except Exception as e:
        print(f"[daily][warn] could not infer day from data: {e}")

    # Écriture du parquet quotidien (pattern atomique : tmp → final)
    out_key_final = f"compact_{path_day}.parquet"
    out_final = f"{DAILY_PREFIX.rstrip('/')}/{out_key_final}"
    out_tmp   = f"{DAILY_PREFIX.rstrip('/')}/_tmp_compact_{path_day}_{int(datetime.now(timezone.utc).timestamp())}.parquet"

    _upload_parquet_df(df_all, out_tmp)
    _copy_blob(out_tmp, out_final)
    _delete_blob(out_tmp)
    print(f"[daily] wrote {len(df_all):,} rows → {out_final}")

    # Purge des snapshots bruts si demandé et si la journée est fermée
    if DELETE_FLAG and _is_closed_day(path_day):
        _purge_prefix_tree(RAW_PREFIX, path_day)
        print(f"[daily] deleted snapshots & markers under {RAW_PREFIX}/date={path_day}/")
    else:
        print(f"[daily] keeping snapshots for {path_day} (open day or DELETE_AFTER_COMPACT=0)")

    print("[daily] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
