# service/jobs/export_training_base.py

"""
Job d’export de base d’entraînement pour le pipeline Vélib’ Forecast.

Rôle
----
Ce job :
- parcourt un répertoire "daily" (local ou GCS) contenant des fichiers Parquet quotidiens ;
- concatène tous les shards quotidiens correspondants en une base d’entraînement unique ;
- ne conserve que les colonnes requises pour l’entraînement du modèle ;
- normalise les timestamps en UTC naïf (ts_utc, tbin_utc) ;
- écrit le résultat soit :
  - en Parquet local, soit
  - directement vers un objet Parquet sur GCS (via un fichier temporaire local) ;
- publie optionnellement une *version datée* supplémentaire sous un préfixe GCS.

Fichiers quotidiens en entrée (nommage flexible)
------------------------------------------------
Le job accepte les deux formats historiques :
- velib_YYYYMMDD.parquet
- compact_YYYY-MM-DD.parquet

et peut travailler sur :
- un répertoire local (récursivement),
- ou un préfixe GCS (gs://bucket/path).

Colonnes requises en sortie
---------------------------
ts_utc, tbin_utc, station_id,
bikes, capacity, mechanical, ebike, status,
lat, lon, name,
temp_C, precip_mm, wind_mps

Variables d’environnement
-------------------------
DAILY_DIR : str, défaut "data_local/daily"
    Chemin racine des Parquet quotidiens (dossier local ou préfixe GCS gs://).

TRAIN_EXPORT : str, défaut "exports/velib.parquet"
    Chemin de la base d’entraînement finale. Peut être :
      - un chemin local (ex. "exports/velib.parquet")
      - un chemin GCS (ex. "gs://bucket/velib/training/latest.parquet")

TRAIN_EXPORT_GCS_PREFIX : str (optionnel)
    Lorsqu’il est défini et pointe vers un préfixe GCS (gs://...),
    publie une version datée supplémentaire :
    "<prefix>/velib_training_YYYYMMDD.parquet".

TMP_OUT : str, défaut "/tmp/velib_training.parquet"
    Fichier temporaire local utilisé lors de l’upload vers GCS.

Exécution
---------
Usage typique :

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

# ───────────────────────────── Helpers ─────────────────────────────

def _is_gcs(path: str) -> bool:
    """
    Indique si le chemin fourni est une URL GCS (commence par 'gs://').

    Paramètres
    ----------
    path : str
        Chemin ou URL à tester.

    Retour
    ------
    bool
        True si le chemin commence par 'gs://', False sinon.
    """
    return str(path).startswith("gs://")


def _split_gcs(url: str) -> tuple[str, str]:
    """
    Découper une URL GCS de la forme 'gs://bucket/path' en (bucket, key).

    Paramètres
    ----------
    url : str
        URL GCS commençant par 'gs://'.

    Retour
    ------
    (str, str)
        Tuple (bucket_name, object_key).
    """
    assert url.startswith("gs://")
    b, p = url[5:].split("/", 1)
    return b, p


def _list_daily_paths(root: str) -> list[str]:
    """
    Lister tous les fichiers Parquet quotidiens sous un répertoire local
    ou un préfixe GCS.

    Modèles de nom de fichier acceptés :
    - velib_YYYYMMDD.parquet
    - compact_YYYY-MM-DD.parquet

    Cas chemin local :
        - Parcours récursif du répertoire et collecte de tous les *.parquet
          qui correspondent aux modèles ci-dessus.

    Cas préfixe GCS :
        - Liste tous les blobs sous bucket/prefix et ne garde que ceux
          dont le nom se termine par '.parquet' et matche l’un des modèles.

    Paramètres
    ----------
    root : str
        Répertoire racine local ou URL GCS (gs://bucket/prefix).

    Retour
    ------
    list[str]
        Liste triée des chemins locaux ou URLs 'gs://' correspondants.
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
        # Recherche récursive pour tolérer des layouts imbriqués
        for f in p.rglob("*.parquet"):
            s = str(f)
            if pat1.match(s) or pat2.match(s):
                out.append(s)

    return sorted(out)


def _read_parquets(paths: list[str]) -> pd.DataFrame:
    """
    Lire et concaténer une liste de fichiers Parquet dans un seul DataFrame.

    Stratégie
    ---------
    - Si pyarrow est disponible et que tous les chemins sont locaux, tenter
      un unique `pq.read_table(paths)` pour plus d’efficacité.
    - Sinon, basculer sur une lecture fichier par fichier (fonctionne à la fois
      pour les fichiers locaux et les URLs gs://, à condition que
      google-cloud-storage soit disponible).

    Paramètres
    ----------
    paths : list[str]
        Liste de chemins locaux ou URLs GCS.

    Retour
    ------
    pandas.DataFrame
        DataFrame concaténé. Si aucun fichier ne peut être lu, renvoie un
        DataFrame vide avec REQ_COLS comme colonnes.
    """
    if not paths:
        return pd.DataFrame(columns=REQ_COLS)

    if pq is not None:
        # Chemin rapide : tous les chemins sont locaux, lecture en un appel
        try:
            if all(not _is_gcs(p) for p in paths):
                tbl = pq.read_table(paths)
                return tbl.to_pandas()
        except Exception:
            # En cas de problème, on revient à la lecture fichier par fichier
            pass

    # Fallback : lecture unitaire (local ou GCS)
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
    Écrire un DataFrame en Parquet local, en créant les dossiers parents au besoin.

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame à sérialiser.
    out_path : str
        Chemin local où le fichier Parquet sera écrit.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[export_training_base] wrote local → {out_path}")


def _upload_file_to_gcs(local_path: str, gcs_url: str):
    """
    Uploader un fichier local existant vers une localisation GCS.

    Paramètres
    ----------
    local_path : str
        Chemin du fichier sur le système local.
    gcs_url : str
        URL GCS de destination (gs://bucket/path/to/file.parquet).

    Lève
    ----
    RuntimeError
        Si google-cloud-storage n’est pas installé.
    """
    if storage is None:
        raise RuntimeError("google-cloud-storage required to write to GCS")
    bkt, key = _split_gcs(gcs_url)
    storage.Client().bucket(bkt).blob(key).upload_from_filename(
        local_path,
        content_type="application/octet-stream",
    )
    print(f"[export_training_base] uploaded → {gcs_url}")

# ───────────────────────────── Main ─────────────────────────────

def main():
    """
    Point d’entrée CLI pour l’export de la base d’entraînement.

    Étapes
    ------
    1. Découvrir les fichiers Parquet quotidiens sous DAILY_DIR (local ou GCS).
    2. Lire et concaténer tous les shards quotidiens correspondants.
    3. Ne garder que REQ_COLS et forcer les timestamps en UTC naïf.
    4. Selon TRAIN_EXPORT :
       - Si chemin local → écrire directement dans TRAIN_EXPORT
         - et, optionnellement, uploader une copie datée sous TRAIN_EXPORT_GCS_PREFIX.
       - Si chemin GCS   → écrire d’abord dans TMP_OUT puis uploader vers TRAIN_EXPORT
         - et, optionnellement, uploader une copie datée sous TRAIN_EXPORT_GCS_PREFIX.

    Variables d’environnement
    -------------------------
    DAILY_DIR : str, défaut "data_local/daily"
        Dossier de base ou préfixe GCS pour les fichiers Parquet quotidiens.

    TRAIN_EXPORT : str, défaut "exports/velib.parquet"
        Chemin local ou GCS pour la base d’entraînement finale.

    TRAIN_EXPORT_GCS_PREFIX : str (optionnel)
        Si fourni et commence par 'gs://', une copie datée supplémentaire
        'velib_training_YYYYMMDD.parquet' est uploadée sous ce préfixe.

    TMP_OUT : str, défaut "/tmp/velib_training.parquet"
        Chemin local temporaire utilisé pour l’upload vers GCS.
    """
    # Racine des données quotidiennes (local ou GCS)
    DAILY_DIR  = os.environ.get("DAILY_DIR", "data_local/daily")   # gs://... ou dossier

    # Configuration de sortie
    # 1) Export d’entraînement final (local OU gs://…/xxx.parquet)
    TRAIN_EXPORT = os.environ.get("TRAIN_EXPORT", "exports/velib.parquet")
    # 2) Préfixe GCS optionnel pour une version datée supplémentaire
    TRAIN_EXPORT_GCS_PREFIX = os.environ.get("TRAIN_EXPORT_GCS_PREFIX")  # optionnel
    # 3) Fichier local temporaire (utilisé pour l’upload vers GCS)
    TMP_OUT = os.environ.get("TMP_OUT", "/tmp/velib_training.parquet")

    paths = _list_daily_paths(DAILY_DIR)
    if not paths:
        print("[export_training_base] no daily parquet found")
        return 0

    df = _read_parquets(paths)

    # Ne conserver que les colonnes requises (création de celles manquantes à None)
    for c in REQ_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[REQ_COLS].copy()

    # Normaliser les timestamps en UTC naïf
    df["tbin_utc"] = pd.to_datetime(df["tbin_utc"], utc=True, errors="coerce").dt.tz_convert(None)
    df["ts_utc"]   = pd.to_datetime(df["ts_utc"],   utc=True, errors="coerce").dt.tz_convert(None)

    wrote_any = False

    # Cas A — TRAIN_EXPORT est un fichier local
    if not _is_gcs(TRAIN_EXPORT):
        _write_local_parquet(df, TRAIN_EXPORT)
        wrote_any = True

        # Option : pousser également une version datée sur GCS
        if TRAIN_EXPORT_GCS_PREFIX and TRAIN_EXPORT_GCS_PREFIX.startswith("gs://"):
            Path(TMP_OUT).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(TMP_OUT, index=False)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
            gcs_url = f"{TRAIN_EXPORT_GCS_PREFIX.rstrip('/')}/velib_training_{stamp}.parquet"
            _upload_file_to_gcs(TMP_OUT, gcs_url)

    # Cas B — TRAIN_EXPORT est un objet GCS
    else:
        # Écrire d’abord dans TMP_OUT en local, puis uploader vers TRAIN_EXPORT
        Path(TMP_OUT).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(TMP_OUT, index=False)
        _upload_file_to_gcs(TMP_OUT, TRAIN_EXPORT)
        wrote_any = True

        # Option : pousser également une version datée sous TRAIN_EXPORT_GCS_PREFIX
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