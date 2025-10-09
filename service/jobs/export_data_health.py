# service/jobs/export_data_health.py
# Exporte un résumé "santé des données" sur les fichiers daily (local ou GCS)
# Produit :
#   - data_health_daily.json (toujours)
#   - data_health_daily.csv  (si CSV_ENABLE=1)
#
# Env:
#   (local)  DAILY_DIR              = data_local/daily (par défaut)
#            HEALTH_OUT             = exports/health   (par défaut)
#   (GCS)    GCS_DAILY_PREFIX       = gs://<bucket>/<root>/daily   (optionnel)
#            GCS_MONITORING_PREFIX  = gs://<bucket>/<root>/monitoring (optionnel upload)
#            CSV_ENABLE             = 1|0 (défaut 1)
#
# Exécution :
#   python -m service.jobs.export_data_health

from __future__ import annotations
import os, json, re
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pq = None  # fallback to pandas.read_parquet

try:
    from google.cloud import storage  # type: ignore
except Exception:
    storage = None  # OK si on reste local

# -------------------- patterns de fichiers acceptés --------------------

def _is_daily_name(name: str) -> bool:
    # velib_YYYYMMDD.parquet  OU  compact_YYYY-MM-DD.parquet
    return bool(
        re.match(r"^velib_\d{8}\.parquet$", name) or
        re.match(r"^compact_\d{4}-\d{2}-\d{2}\.parquet$", name)
    )

def _day_from_name(name: str) -> Optional[str]:
    m1 = re.match(r"^velib_(\d{8})\.parquet$", name)
    if m1:
        ymd = m1.group(1)
        return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
    m2 = re.match(r"^compact_(\d{4}-\d{2}-\d{2})\.parquet$", name)
    if m2:
        return m2.group(1)
    return None

# -------------------- Local --------------------

def _iter_local_daily_files(root: str) -> List[Path]:
    p = Path(root)
    if not p.exists():
        return []
    out: List[Path] = []
    for f in p.rglob("*.parquet"):
        if _is_daily_name(f.name):
            out.append(f)
    return sorted(set(out))

def _read_parquet_local(path: Path) -> pd.DataFrame:
    try:
        if pq is not None:
            return pq.read_table(path).to_pandas()
        return pd.read_parquet(path)
    except Exception as e:
        print(f"[data_health][warn] read local failed {path}: {e}")
        return pd.DataFrame()

# -------------------- GCS --------------------

def _is_gcs(url: Optional[str]) -> bool:
    return isinstance(url, str) and url.startswith("gs://")

def _split_gs(url: str) -> Tuple[str, str]:
    b, k = url[5:].split("/", 1)
    return b, k.rstrip("/")

def _iter_gcs_daily_files(prefix: str) -> List[Tuple[str, str]]:
    if storage is None:
        raise RuntimeError("google-cloud-storage not installed but GCS_DAILY_PREFIX is set")
    bkt, pfx = _split_gs(prefix)
    cli = storage.Client()
    out: List[Tuple[str, str]] = []
    for b in cli.list_blobs(bkt, prefix=pfx):
        if b.name.endswith(".parquet") and _is_daily_name(b.name.rsplit("/", 1)[-1]):
            out.append((bkt, b.name))
    out.sort(key=lambda t: t[1])
    return out

def _read_parquet_gcs(bkt: str, key: str) -> pd.DataFrame:
    if storage is None:
        raise RuntimeError("google-cloud-storage not installed")
    try:
        buf = BytesIO()
        storage.Client().bucket(bkt).blob(key).download_to_file(buf)
        buf.seek(0)
        if pq is not None:
            return pq.read_table(buf).to_pandas()
        return pd.read_parquet(buf)
    except Exception as e:
        print(f"[data_health][warn] read gcs failed gs://{bkt}/{key}: {e}")
        return pd.DataFrame()

def _upload_text_gcs(text: str, gcs_url: str, content_type: str = "application/json"):
    if storage is None:
        raise RuntimeError("google-cloud-storage not installed")
    bkt, key = _split_gs(gcs_url)
    storage.Client().bucket(bkt).blob(key).upload_from_string(text, content_type=content_type)
    print(f"[data_health] uploaded → {gcs_url}")

def _upload_bytes_gcs(data: bytes, gcs_url: str, content_type: str = "text/csv"):
    if storage is None:
        raise RuntimeError("google-cloud-storage not installed")
    bkt, key = _split_gs(gcs_url)
    storage.Client().bucket(bkt).blob(key).upload_from_string(data, content_type=content_type)
    print(f"[data_health] uploaded → {gcs_url}")

# -------------------- Stats --------------------

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    need = ["ts_utc","tbin_utc","station_id","bikes","temp_C"]
    for c in need:
        if c not in df.columns:
            df[c] = None
    out = df[need].copy()
    out["ts_utc"]   = pd.to_datetime(out["ts_utc"],   utc=True, errors="coerce").dt.tz_convert(None)
    out["tbin_utc"] = pd.to_datetime(out["tbin_utc"], utc=True, errors="coerce").dt.tz_convert(None)
    return out

def _stats_for(day_iso: str, df: pd.DataFrame) -> dict:
    if df.empty:
        return {"date": day_iso, "rows": 0, "stations": 0, "bins": 0, "null_bikes": 0, "null_temp": 0}
    d = _normalize(df)
    return {
        "date": day_iso,
        "rows": int(len(d)),
        "stations": int(d["station_id"].nunique()),
        "bins": int(d["tbin_utc"].nunique()),
        "null_bikes": int(d["bikes"].isna().sum()),
        "null_temp": int(d["temp_C"].isna().sum()),
    }

# -------------------- Main --------------------

def main():
    DAILY_DIR   = os.environ.get("DAILY_DIR", "data_local/daily")
    OUT_DIR     = os.environ.get("HEALTH_OUT", "exports/health")
    GCS_DAILY   = os.environ.get("GCS_DAILY_PREFIX")           # optionnel (lecture GCS)
    GCS_MON     = os.environ.get("GCS_MONITORING_PREFIX")      # optionnel (upload GCS)
    CSV_ENABLE  = os.environ.get("CSV_ENABLE", "1") in ("1","true","True")

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []

    if _is_gcs(GCS_DAILY):
        # --- lecture GCS ---
        files = _iter_gcs_daily_files(GCS_DAILY)  # (bkt, key)
        if not files:
            print(f"[data_health] no daily parquet under {GCS_DAILY}")
        for bkt, key in files:
            name = key.rsplit("/", 1)[-1]
            day_iso = _day_from_name(name) or "unknown"
            df = _read_parquet_gcs(bkt, key)
            rows.append(_stats_for(day_iso, df))
    else:
        # --- lecture locale ---
        paths = _iter_local_daily_files(DAILY_DIR)
        if not paths:
            print(f"[data_health] aucun fichier trouvé dans {DAILY_DIR} (attendus: velib_YYYYMMDD.parquet ou compact_YYYY-MM-DD.parquet)")
        for p in paths:
            day_iso = _day_from_name(p.name) or "unknown"
            df = _read_parquet_local(p)
            rows.append(_stats_for(day_iso, df))

    # tri par date si possible
    rows = sorted(rows, key=lambda r: r.get("date") or "")

    if not rows:
        print("[data_health] aucun jour valide trouvé — sortie vide")
        return 0

    # JSON (toujours)
    json_text = json.dumps(rows, indent=2, ensure_ascii=False)
    out_json = Path(OUT_DIR) / "data_health_daily.json"
    out_json.write_text(json_text, encoding="utf-8")
    print(f"[data_health] wrote {out_json}")

    # CSV (optionnel)
    if CSV_ENABLE:
        out_csv = Path(OUT_DIR) / "data_health_daily.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[data_health] wrote {out_csv}")

    # Upload GCS (optionnel)
    if _is_gcs(GCS_MON):
        _upload_text_gcs(json_text, f"{GCS_MON.rstrip('/')}/data_health_daily.json", content_type="application/json")
        if CSV_ENABLE:
            _upload_bytes_gcs(pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
                              f"{GCS_MON.rstrip('/')}/data_health_daily.csv",
                              content_type="text/csv")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
