# service/jobs/export_training_base.py
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
    return str(path).startswith("gs://")

def _split_gcs(url: str) -> tuple[str, str]:
    assert url.startswith("gs://")
    b, p = url[5:].split("/", 1)
    return b, p

def _list_daily_paths(root: str) -> list[str]:
    """Liste tous les Parquet daily, local ou GCS (formats velib_YYYYMMDD / compact_YYYY-MM-DD)."""
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
        # récursif pour tolérer des sous-dossiers
        for f in p.rglob("*.parquet"):
            s = str(f)
            if pat1.match(s) or pat2.match(s):
                out.append(s)
    return sorted(out)

def _read_parquets(paths: list[str]) -> pd.DataFrame:
    """Concatène tous les fichiers Parquet trouvés (pyarrow si dispo, sinon pandas)."""
    if not paths:
        return pd.DataFrame(columns=REQ_COLS)
    if pq is not None:
        # pyarrow accepte une liste de chemins locaux et de gs:// si gcsfs est configuré,
        # mais pour la robustesse, on lit en pandas quand gs:// (via download buffer) n'est pas garanti.
        try:
            # Si 100% locaux, on peut lire d'un coup via pq
            if all(not _is_gcs(p) for p in paths):
                tbl = pq.read_table(paths)
                return tbl.to_pandas()
        except Exception:
            pass
    # fallback: lire un par un (local ou GCS)
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
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[export_training_base] wrote local → {out_path}")

def _upload_file_to_gcs(local_path: str, gcs_url: str):
    if storage is None:
        raise RuntimeError("google-cloud-storage required to write to GCS")
    bkt, key = _split_gcs(gcs_url)
    storage.Client().bucket(bkt).blob(key).upload_from_filename(local_path, content_type="application/octet-stream")
    print(f"[export_training_base] uploaded → {gcs_url}")

# ───────────────────────────── main ─────────────────────────────

def main():
    # Entrée (lecture)
    DAILY_DIR  = os.environ.get("DAILY_DIR", "data_local/daily")   # gs://... ou dossier

    # Sorties (choisis librement l’un, l’autre, ou les deux)
    # 1) Chemin fichier final (local OU gs://…/xxx.parquet)
    TRAIN_EXPORT = os.environ.get("TRAIN_EXPORT", "exports/velib.parquet")
    # 2) Préfixe GCS pour publier aussi une version datée (ex: gs://bucket/velib/training)
    TRAIN_EXPORT_GCS_PREFIX = os.environ.get("TRAIN_EXPORT_GCS_PREFIX")  # optionnel
    # 3) Fichier temporaire local (utilisé pour uploader vers GCS)
    TMP_OUT = os.environ.get("TMP_OUT", "/tmp/velib_training.parquet")

    paths = _list_daily_paths(DAILY_DIR)
    if not paths:
        print("[export_training_base] no daily parquet found")
        return 0

    df = _read_parquets(paths)

    # ne garder que les colonnes utiles
    for c in REQ_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[REQ_COLS].copy()

    # harmoniser timestamps (UTC naïf)
    df["tbin_utc"] = pd.to_datetime(df["tbin_utc"], utc=True, errors="coerce").dt.tz_convert(None)
    df["ts_utc"]   = pd.to_datetime(df["ts_utc"],   utc=True, errors="coerce").dt.tz_convert(None)

    wrote_any = False

    # Cas A — TRAIN_EXPORT est un fichier local
    if not _is_gcs(TRAIN_EXPORT):
        _write_local_parquet(df, TRAIN_EXPORT)
        wrote_any = True
        # Si on veut aussi pousser une version datée sur GCS (en plus du local)
        if TRAIN_EXPORT_GCS_PREFIX and TRAIN_EXPORT_GCS_PREFIX.startswith("gs://"):
            Path(TMP_OUT).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(TMP_OUT, index=False)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
            gcs_url = f"{TRAIN_EXPORT_GCS_PREFIX.rstrip('/')}/velib_training_{stamp}.parquet"
            _upload_file_to_gcs(TMP_OUT, gcs_url)

    # Cas B — TRAIN_EXPORT est un objet GCS direct (gs://…/xxx.parquet)
    else:
        # on écrit d’abord en local puis on upload exactement à TRAIN_EXPORT
        Path(TMP_OUT).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(TMP_OUT, index=False)
        _upload_file_to_gcs(TMP_OUT, TRAIN_EXPORT)
        wrote_any = True
        # Optionnel: publier AUSSI une version datée si un préfixe est fourni
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
