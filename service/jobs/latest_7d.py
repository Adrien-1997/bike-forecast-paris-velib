# service/jobs/latest_7d.py
# Monitoring 7 jours basé sur les shards daily (Parquet) stockés sur GCS.
#
# Produit uniquement 2 fichiers JSON dans gs://…/monitoring :
#   - health_7d_<anchor:YYYYMMDD>.json         (agrégats globaux)
#   - station_health_7d_<anchor:YYYYMMDD>.json (par station)
#
# Env requis :
#   GCS_DAILY_PREFIX       = gs://<bucket>/<root>/daily
#   GCS_MONITORING_PREFIX  = gs://<bucket>/<root>/monitoring
#   WINDOW_DAYS            = 7 (défaut)
#   ANCHOR_DAY             = YYYY-MM-DD (optionnel)
#
# Exécution :
#   python -m service.jobs.latest_7d

from __future__ import annotations

import os
from io import BytesIO
from datetime import datetime, date, timezone
from typing import List, Tuple, Optional, Set
import pandas as pd

try:
    import pyarrow.parquet as pq  # lecture des daily
except Exception as e:
    raise RuntimeError("pyarrow est requis pour latest_7d.py") from e

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage est requis") from e


COLS = [
    "ts_utc","tbin_utc","station_id","bikes","capacity","mechanical","ebike",
    "status","lat","lon","name","temp_C","precip_mm","wind_mps"
]

# ─────────────────────────────── GCS Helpers ───────────────────────────────

def _split(gcs_url: str) -> Tuple[str, str]:
    assert gcs_url.startswith("gs://")
    b, p = gcs_url[5:].split("/", 1)
    return b, p.rstrip("/")

def _list_daily_files(daily_prefix: str) -> List[Tuple[str, str]]:
    """Retourne [(bucket, key)] des Parquet daily (compact_YYYY-MM-DD|velib_YYYYMMDD)."""
    bkt, pfx = _split(daily_prefix)
    cli = storage.Client()
    out: List[Tuple[str, str]] = []
    for b in cli.list_blobs(bkt, prefix=pfx):
        if not b.name.endswith(".parquet"):
            continue
        fn = b.name.rsplit("/", 1)[-1]
        if fn.startswith(("velib_", "compact_")):
            out.append((bkt, b.name))
    return out

def _parse_ymd_from_key(key: str) -> Optional[date]:
    fn = key.rsplit("/", 1)[-1]
    try:
        if fn.startswith("velib_"):
            return datetime.strptime(fn[6:14], "%Y%m%d").date()
        if fn.startswith("compact_"):
            return datetime.strptime(fn[8:18], "%Y-%m-%d").date()
    except Exception:
        return None
    return None

def _download_parquet_to_df(bkt: str, key: str) -> pd.DataFrame:
    buf = BytesIO()
    storage.Client().bucket(bkt).blob(key).download_to_file(buf)
    buf.seek(0)
    table = pq.read_table(buf)
    return table.to_pandas()

def _upload_json_df(df: pd.DataFrame, gcs_url: str):
    bkt, key = _split(gcs_url)
    storage.Client().bucket(bkt).blob(key).upload_from_string(
        df.to_json(orient="records", force_ascii=False),
        content_type="application/json"
    )

# ─────────────────────────────── Normalisation ───────────────────────────────

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

# ─────────────────────────────── Calculs ───────────────────────────────

def _compute_station_metrics(df7: pd.DataFrame, anchor: date) -> pd.DataFrame:
    if df7.empty:
        return pd.DataFrame(columns=[
            "anchor_date","station_id","bins_present","days_present",
            "bins_expected","completeness_pct","bikes_min","bikes_max","bikes_median"
        ])
    df7 = df7.copy()
    df7["day_utc"] = pd.to_datetime(df7["tbin_utc"]).dt.date
    grp = df7.groupby("station_id", dropna=True)
    met = pd.concat([
        grp.size().rename("bins_present"),
        grp["day_utc"].nunique().rename("days_present"),
        grp["bikes"].min().rename("bikes_min"),
        grp["bikes"].max().rename("bikes_max"),
        grp["bikes"].median().rename("bikes_median")
    ], axis=1).reset_index()
    met["bins_expected"] = (met["days_present"].fillna(0).astype(int) * 288).astype("Int64")
    denom = met["bins_expected"].replace({0: pd.NA})
    met["completeness_pct"] = (100.0 * met["bins_present"] / denom).astype("Float64")
    met.insert(0, "anchor_date", anchor.isoformat())
    return met

def _compute_global_metrics(df_station: pd.DataFrame, start: date, anchor: date) -> pd.DataFrame:
    if df_station.empty:
        return pd.DataFrame([{
            "anchor_date": anchor.isoformat(),
            "from_date": start.isoformat(),
            "to_date": anchor.isoformat(),
            "stations": 0,
            "bins_present_sum": 0,
            "bins_expected_sum": 0,
            "completeness_pct_avg": 0.0
        }])
    return pd.DataFrame([{
        "anchor_date": anchor.isoformat(),
        "from_date": start.isoformat(),
        "to_date": anchor.isoformat(),
        "stations": int(len(df_station)),
        "bins_present_sum": int(df_station["bins_present"].fillna(0).sum()),
        "bins_expected_sum": int(df_station["bins_expected"].fillna(0).sum()),
        "completeness_pct_avg": float(df_station["completeness_pct"].dropna().mean()) if df_station["completeness_pct"].notna().any() else 0.0
    }])

# ─────────────────────────────── Main ───────────────────────────────

def main():
    DAILY_PREFIX      = os.environ.get("GCS_DAILY_PREFIX")
    MONITORING_PREFIX = os.environ.get("GCS_MONITORING_PREFIX")
    WINDOW_DAYS       = int(os.environ.get("WINDOW_DAYS", "7"))
    ANCHOR_DAY        = os.environ.get("ANCHOR_DAY")

    if not (DAILY_PREFIX and DAILY_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_DAILY_PREFIX invalide ou manquant")
    if not (MONITORING_PREFIX and MONITORING_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX invalide ou manquant")

    files = _list_daily_files(DAILY_PREFIX)
    if not files:
        print("[7d] aucun daily trouvé — sortie vide")
        return 0

    today = datetime.now(timezone.utc).date()
    all_dates: List[date] = sorted({
        d for (_, key) in files
        if (d := _parse_ymd_from_key(key)) is not None and d <= today
    })
    if not all_dates:
        print("[7d] aucune date exploitable — sortie vide")
        return 0

    if ANCHOR_DAY:
        anchor = datetime.strptime(ANCHOR_DAY, "%Y-%m-%d").date()
        print(f"[7d] anchor explicite → {anchor}")
    else:
        anchor = all_dates[-1]
        print(f"[7d] anchor auto → {anchor}")

    last_days: List[date] = [d for d in all_dates if d <= anchor][-WINDOW_DAYS:]
    if not last_days:
        print(f"[7d] aucun shard dans la fenêtre (derniers {WINDOW_DAYS} jours ≤ {anchor})")
        return 0

    start = last_days[0]
    print(f"[7d] fenêtre = {start} → {anchor} ({len(last_days)} jours)")

    target_days: Set[date] = set(last_days)
    selected: List[Tuple[str, str]] = [
        (bkt, key) for (bkt, key) in files
        if (d := _parse_ymd_from_key(key)) in target_days
    ]
    if not selected:
        print(f"[7d] aucun fichier sélectionné pour {len(last_days)} jours")
        return 0

    dfs: List[pd.DataFrame] = []
    failed = 0
    for bkt, key in selected:
        try:
            dfs.append(_download_parquet_to_df(bkt, key))
        except Exception as e:
            failed += 1
            print(f"[7d][warn] lecture échouée gs://{bkt}/{key}: {e}")

    if not dfs:
        print("[7d] lecture impossible sur tous les shards sélectionnés — sortie vide")
        return 0

    df_all = pd.concat(dfs, ignore_index=True, sort=False)
    df_all = _ensure_schema(df_all)

    df_station = _compute_station_metrics(df_all, anchor)
    df_global  = _compute_global_metrics(df_station, start, anchor)

    suffix = anchor.strftime("%Y%m%d")
    out_global_json   = f"{MONITORING_PREFIX.rstrip('/')}/health_7d_{suffix}.json"
    out_station_json  = f"{MONITORING_PREFIX.rstrip('/')}/station_health_7d_{suffix}.json"

    _upload_json_df(df_global,  out_global_json)
    _upload_json_df(df_station, out_station_json)

    # 🔊 Log final, verbeux, mais bien en JSON (pas de Parquet dans le message)
    print(f"[7d] OK → {out_global_json} & {out_station_json} | stations={len(df_station)} | shards={len(selected)} | failed_reads={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
