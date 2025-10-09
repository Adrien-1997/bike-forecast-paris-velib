# service/jobs/build_datasets.py
# Construit 2 exports canoniques :
#  - gs://.../exports/events.parquet   (réseau au pas 5 min)
#  - gs://.../exports/perf.parquet     (paires (t, t+h) pour l'évaluation/serving)
#
# Entrée : dailies compacts GCS (compact_YYYY-MM-DD.parquet)
#
# ENV requis :
#   GCS_DAILY_PREFIX       = gs://<bucket>/velib/daily
#   GCS_EXPORTS_PREFIX     = gs://<bucket>/velib/exports
#
# Optionnel (transition) :
#   GCS_MONITORING_PREFIX  = gs://<bucket>/velib/monitoring  # si défini, on duplique dans monitoring/exports/
#
# Options :
#   EVENTS_WINDOW_DAYS     = 30 (défaut)
#   PERF_WINDOW_DAYS       = 30 (défaut)
#   FORECAST_HORIZONS      = "15,60" (en minutes)
#   PENURY_THRESH          = 2
#   SATURATION_THRESH      = 2
#   ANCHOR_DAY             = "YYYY-MM-DD" (par défaut today UTC)
#
# Exécution :
#   python -m service.jobs.build_datasets

from __future__ import annotations
import os, re, sys
from io import BytesIO
from typing import List, Tuple
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    raise RuntimeError("pyarrow est requis pour build_datasets.py") from e

try:
    from google.cloud import storage
except Exception as e:
    raise RuntimeError("google-cloud-storage est requis (GCS I/O)") from e

BIN_MIN = 5
COLS = [
    "ts_utc","tbin_utc","station_id","bikes","capacity","mechanical","ebike",
    "status","lat","lon","name","temp_C","precip_mm","wind_mps"
]

# ───────────────────────────── GCS helpers ─────────────────────────────

def _split(gs: str) -> Tuple[str, str]:
    assert gs.startswith("gs://"), f"bad GCS uri: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k.rstrip("/")

def _iter_dailies_for_window(daily_prefix: str, start_day: str, end_day: str) -> List[Tuple[str,str]]:
    bkt, pfx = _split(daily_prefix)
    cli = storage.Client()
    p = re.compile(r".*/compact_(\d{4}-\d{2}-\d{2})\.parquet$")
    out: List[Tuple[str,str]] = []
    for b in cli.list_blobs(bkt, prefix=pfx):
        m = p.match(b.name)
        if not m:
            continue
        day = m.group(1)
        if start_day <= day <= end_day:
            out.append((bkt, b.name))
    out.sort(key=lambda t: t[1])
    return out

def _read_gcs_parquet(bkt: str, key: str) -> pd.DataFrame:
    buf = BytesIO()
    storage.Client().bucket(bkt).blob(key).download_to_file(buf)
    buf.seek(0)
    return pq.read_table(buf).to_pandas()

def _write_gcs_parquet(df: pd.DataFrame, dest_gs: str):
    bkt, key = _split(dest_gs)
    buf = BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf, compression="snappy")
    buf.seek(0)
    storage.Client().bucket(bkt).blob(key).upload_from_file(buf, content_type="application/octet-stream")
    print(f"[build_datasets] wrote → {dest_gs} (rows={len(df):,})")

def _copy_gcs(src_gs: str, dst_gs: str):
    src_bkt, src_key = _split(src_gs)
    dst_bkt, dst_key = _split(dst_gs)
    cli = storage.Client()
    src_blob = cli.bucket(src_bkt).blob(src_key)
    cli.bucket(dst_bkt).copy_blob(src_blob, cli.bucket(dst_bkt), dst_key)
    print(f"[build_datasets] duplicated → {dst_gs}")

# ───────────────────────────── Dates & fenêtres ─────────────────────────────

def _anchor_day_utc() -> datetime:
    ad = os.environ.get("ANCHOR_DAY")
    if ad:
        return datetime.fromisoformat(ad).replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)

def _window_days(anchor: datetime, days: int) -> Tuple[str,str]:
    end_day = anchor.date().isoformat()
    start_day = (anchor - timedelta(days=days-1)).date().isoformat()
    return start_day, end_day

# ───────────────────────────── Normalisation ─────────────────────────────

def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    out = pd.DataFrame({
        "ts_utc":     pd.to_datetime(df["ts_utc"],   utc=True, errors="coerce").dt.tz_convert(None),
        "tbin_utc":   pd.to_datetime(df["tbin_utc"], utc=True, errors="coerce").dt.tz_convert(None),
        "station_id": df["station_id"].astype("string"),
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
    for c in ["bikes","capacity","mechanical","ebike"]:
        out[c] = out[c].astype("Int64")
    return out

def _dedup_latest(df: pd.DataFrame) -> pd.DataFrame:
    if not {"station_id","tbin_utc","ts_utc"}.issubset(df.columns):
        return df
    df = df.sort_values(["station_id","tbin_utc","ts_utc"])
    return df.groupby(["station_id","tbin_utc"], as_index=False).tail(1).reset_index(drop=True)

# ───────────────────────────── Events build ─────────────────────────────

def _build_events(dfs: List[pd.DataFrame], penury: int, saturation: int) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame(columns=[
            "tbin_utc","station_id","bikes","capacity","mechanical","ebike",
            "status","lat","lon","name","temp_C","precip_mm","wind_mps",
            "occ_ratio","is_penury","is_saturation","status_code"
        ])
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df = _enforce_schema(df)
    df = _dedup_latest(df)

    cap = pd.to_numeric(df["capacity"], errors="coerce")
    bikes = pd.to_numeric(df["bikes"], errors="coerce")
    df["occ_ratio"] = np.where((cap > 0) & bikes.notna(), bikes / cap, np.nan)
    df["is_penury"] = ((bikes <= penury) & bikes.notna()).astype("Int8")
    df["is_saturation"] = ((cap - bikes <= saturation) & cap.notna() & bikes.notna()).astype("Int8")

    if "status" in df.columns:
        cats = sorted([s for s in df["status"].dropna().unique()])
        s_map = {s:i for i,s in enumerate(cats)}
        df["status_code"] = df["status"].map(s_map).astype("Int64")
    else:
        df["status_code"] = pd.NA

    keep = [
        "tbin_utc","station_id","bikes","capacity","mechanical","ebike",
        "status","status_code","lat","lon","name","temp_C","precip_mm","wind_mps",
        "occ_ratio","is_penury","is_saturation"
    ]
    df = df[keep].sort_values(["tbin_utc","station_id"]).reset_index(drop=True)
    return df

# ───────────────────────────── Perf build ─────────────────────────────

def _build_perf(events: pd.DataFrame, horizons_min: List[int]) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=[
            "tbin_utc","station_id","horizon_bins","y_true","y_baseline_persist",
            "bikes","capacity","occ_ratio"
        ])
    out_frames: List[pd.DataFrame] = []
    events = events.sort_values(["station_id","tbin_utc"]).reset_index(drop=True)

    for hmin in horizons_min:
        hb = max(1, int(round(hmin / BIN_MIN)))
        g = events.groupby("station_id", group_keys=False, dropna=True)

        def _shift_merge(st: pd.DataFrame) -> pd.DataFrame:
            st = st[["tbin_utc","station_id","bikes","capacity","occ_ratio"]].copy()
            st["tbin_target"] = st["tbin_utc"] + timedelta(minutes=BIN_MIN*hb)
            tgt = st[["tbin_utc","bikes"]].rename(columns={"tbin_utc":"tbin_target","bikes":"y_true"})
            merged = st.merge(tgt, on=["tbin_target"], how="left")
            merged["y_baseline_persist"] = merged["bikes"]
            merged["horizon_bins"] = hb
            return merged

        part = g.apply(_shift_merge).reset_index(drop=True)
        out_frames.append(part)

    perf = pd.concat(out_frames, ignore_index=True, sort=False)
    keep = ["tbin_utc","station_id","horizon_bins","y_true","y_baseline_persist","bikes","capacity","occ_ratio"]
    perf = perf[keep].sort_values(["tbin_utc","station_id","horizon_bins"]).reset_index(drop=True)
    return perf

# ───────────────────────────── Main ─────────────────────────────

def main() -> int:
    DAILY_PREFIX  = os.environ.get("GCS_DAILY_PREFIX")
    EXPORTS_PREFIX = os.environ.get("GCS_EXPORTS_PREFIX")  # nouveau
    MON_PREFIX    = os.environ.get("GCS_MONITORING_PREFIX")  # transition (optionnel)

    if not (DAILY_PREFIX and DAILY_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_DAILY_PREFIX manquant ou invalide")
    if not (EXPORTS_PREFIX and EXPORTS_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_EXPORTS_PREFIX manquant ou invalide")

    EVENTS_WIN = int(os.environ.get("EVENTS_WINDOW_DAYS","30"))
    PERF_WIN   = int(os.environ.get("PERF_WINDOW_DAYS","30"))
    HORIZONS   = [int(x.strip()) for x in os.environ.get("FORECAST_HORIZONS","15,60").split(",") if x.strip()]
    PENURY_T   = int(os.environ.get("PENURY_THRESH","2"))
    SAT_T      = int(os.environ.get("SATURATION_THRESH","2"))

    anchor = _anchor_day_utc()
    ev_start, ev_end = _window_days(anchor, EVENTS_WIN)
    pf_start, pf_end = _window_days(anchor, PERF_WIN)

    print(f"[build_datasets] events window: {ev_start}..{ev_end} (days={EVENTS_WIN})")
    print(f"[build_datasets] perf   window: {pf_start}..{pf_end} (days={PERF_WIN})")
    print(f"[build_datasets] horizons (min): {HORIZONS}")

    # Events
    ev_files = _iter_dailies_for_window(DAILY_PREFIX, ev_start, ev_end)
    if not ev_files:
        print("[build_datasets] no daily files for events window — exit 0")
        return 0

    ev_dfs: List[pd.DataFrame] = []
    for bkt, key in ev_files:
        try:
            ev_dfs.append(_read_gcs_parquet(bkt, key))
        except Exception as e:
            print(f"[build_datasets][warn] read fail gs://{bkt}/{key}: {e}")

    events = _build_events(ev_dfs, penury=PENURY_T, saturation=SAT_T)
    if events.empty:
        print("[build_datasets] events empty — exit 0")
        return 0

    events_uri_main = f"{EXPORTS_PREFIX.rstrip('/')}/events.parquet"
    _write_gcs_parquet(events, events_uri_main)

    # Dupliquer dans monitoring/exports (transition)
    if MON_PREFIX and MON_PREFIX.startswith("gs://"):
        events_uri_mon = f"{MON_PREFIX.rstrip('/')}/exports/events.parquet"
        _copy_gcs(events_uri_main, events_uri_mon)

    # Perf
    pf_files = _iter_dailies_for_window(DAILY_PREFIX, pf_start, pf_end)
    if not pf_files:
        print("[build_datasets] no daily files for perf window — skip perf")
        return 0

    pf_dfs: List[pd.DataFrame] = []
    for bkt, key in pf_files:
        try:
            pf_dfs.append(_read_gcs_parquet(bkt, key))
        except Exception as e:
            print(f"[build_datasets][warn] read fail (perf) gs://{bkt}/{key}: {e}")

    ev_for_perf = _build_events(pf_dfs, penury=PENURY_T, saturation=SAT_T)
    perf = _build_perf(ev_for_perf, horizons_min=HORIZONS)
    if perf.empty:
        print("[build_datasets] perf empty — nothing written")
        return 0

    perf_uri_main = f"{EXPORTS_PREFIX.rstrip('/')}/perf.parquet"
    _write_gcs_parquet(perf, perf_uri_main)

    if MON_PREFIX and MON_PREFIX.startswith("gs://"):
        perf_uri_mon = f"{MON_PREFIX.rstrip('/')}/exports/perf.parquet"
        _copy_gcs(perf_uri_main, perf_uri_mon)

    print("[build_datasets] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
