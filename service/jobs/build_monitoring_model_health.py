# service/jobs/build_monitoring_model_health.py
# Produit des JSON de performance modèle pour l'UI, à partir de exports/perf.parquet
#
# Entrée:
#   - gs://.../velib/exports/perf.parquet   (colonnes: tbin_utc, station_id, horizon_bins, y_true,
#                                             y_baseline_persist, (optionnel) y_pred, bikes, capacity, occ_ratio)
#
# Sorties (par horizon):
#   - gs://.../velib/monitoring/model/perf/daily_h{H}.json
#   - gs://.../velib/monitoring/model/perf/daily_h{H}_YYYYMMDD.json
#   - gs://.../velib/monitoring/model/perf/segments_h{H}.json
#   - gs://.../velib/monitoring/model/perf/segments_h{H}_YYYYMMDD.json
#
# ENV requis:
#   GCS_EXPORTS_PREFIX     = gs://<bucket>/velib/exports
#   GCS_MONITORING_PREFIX  = gs://<bucket>/velib/monitoring
#
# Options:
#   HORIZONS               = "15,60"   (minutes)
#   TOP_STATIONS           = 200       (taille du "by_station_top")
#   MIN_SAMPLES_PER_GROUP  = 50        (filtre de robustesse pour segments)
#   DATE_RANGE_DAYS        = 60        (fenêtre max récente pour daily; None => tout)
#   ANCHOR_DAY             = "YYYY-MM-DD" (par défaut today UTC; sert au suffixe versionné)
#
# Exécution:
#   python -m service.jobs.build_monitoring_model_health

from __future__ import annotations
import os, json, sys
from io import BytesIO
from typing import List, Dict
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq  # type: ignore
    import pyarrow as pa          # type: ignore
except Exception as e:
    raise RuntimeError("pyarrow requis") from e

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage requis") from e


SCHEMA_VERSION = "1.0"


# ─────────────────────────── GCS helpers ───────────────────────────

def _split(gs: str):
    assert gs.startswith("gs://"), f"bad GCS uri: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k.rstrip("/")

def _read_parquet_gs(gs_uri: str) -> pd.DataFrame:
    bkt, key = _split(gs_uri)
    buf = BytesIO()
    storage.Client().bucket(bkt).blob(key).download_to_file(buf)
    buf.seek(0)
    tbl = pq.read_table(buf)
    return tbl.to_pandas()

def _upload_json_gs(obj: dict, gs_uri: str):
    bkt, key = _split(gs_uri)
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    storage.Client().bucket(bkt).blob(key).upload_from_string(
        data, content_type="application/json"
    )
    print(f"[model_health] wrote → {gs_uri} ({len(data):,} bytes)")

# ─────────────────────────── Time / window ───────────────────────────

def _anchor_day_utc() -> datetime:
    ad = os.environ.get("ANCHOR_DAY")
    if ad:
        return datetime.fromisoformat(ad).replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)

def _recent_cutoff(anchor: datetime, days: int | None) -> pd.Timestamp | None:
    if not days or days <= 0:
        return None
    start = (anchor - timedelta(days=days-1)).date()
    return pd.Timestamp(str(start))

# ─────────────────────────── Metrics helpers ───────────────────────────

def _safe_mae(y_true: pd.Series, y_pred: pd.Series) -> float | None:
    m = y_true.notna() & y_pred.notna()
    if m.sum() == 0:
        return None
    return float(np.abs(y_true[m].astype(float) - y_pred[m].astype(float)).mean())

def _group_metrics(df: pd.DataFrame) -> Dict[str, float | int | None]:
    out: Dict[str, float | int | None] = {}
    out["n"] = int(len(df))
    out["mae_baseline"] = _safe_mae(df["y_true"], df["y_baseline_persist"])
    if "y_pred" in df.columns:
        out["mae_model"] = _safe_mae(df["y_true"], df["y_pred"])
    else:
        out["mae_model"] = None
    # lift vs baseline (positif = amélioration)
    if out["mae_model"] is not None and out["mae_baseline"] not in (None, 0):
        out["lift_vs_baseline"] = max(0.0, 1.0 - (out["mae_model"] / out["mae_baseline"]))
    else:
        out["lift_vs_baseline"] = None
    # RMSE model (si dispo)
    if "y_pred" in df.columns:
        m = df["y_true"].notna() & df["y_pred"].notna()
        if m.sum() > 0:
            rmse = np.sqrt(((df.loc[m, "y_true"] - df.loc[m, "y_pred"]) ** 2).mean())
            out["rmse_model"] = float(rmse)
        else:
            out["rmse_model"] = None
        out["coverage_pred_pct"] = float(100.0 * m.mean())
    else:
        out["rmse_model"] = None
        out["coverage_pred_pct"] = 0.0
    return out

# ─────────────────────────── Build JSONs ───────────────────────────

def _build_daily_json(df: pd.DataFrame, horizon_bins: int, cutoff: pd.Timestamp | None) -> dict:
    # restreindre horizon + fenêtre récente (si définie)
    d = df[(df["horizon_bins"] == horizon_bins)].copy()
    d["date"] = pd.to_datetime(d["tbin_utc"], errors="coerce").dt.date.astype("string")
    if cutoff is not None:
        d = d[pd.to_datetime(d["date"]) >= cutoff]
    # regroupe par date
    rows = []
    for dt, grp in d.groupby("date", dropna=True):
        m = _group_metrics(grp)
        rows.append({
            "date": str(dt),
            "mae_model": m["mae_model"],
            "mae_baseline": m["mae_baseline"],
            "lift_vs_baseline": m["lift_vs_baseline"],
            "rmse_model": m["rmse_model"],
            "coverage_pred_pct": m["coverage_pred_pct"],
            "n": m["n"],
        })
    rows = sorted(rows, key=lambda r: r["date"])
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "horizon_min": int(horizon_bins * 5),
        "metrics": rows,
    }

def _build_segments_json(df: pd.DataFrame, horizon_bins: int, top_stations: int, min_samples: int) -> dict:
    d = df[(df["horizon_bins"] == horizon_bins)].copy()
    d["hour"] = pd.to_datetime(d["tbin_utc"], errors="coerce").dt.hour
    d["dow"]  = pd.to_datetime(d["tbin_utc"], errors="coerce").dt.dayofweek  # 0=Mon

    # by_station_top
    g_station = d.groupby("station_id", dropna=True)
    recs_station = []
    for sid, grp in g_station:
        if len(grp) < min_samples:
            continue
        m = _group_metrics(grp)
        recs_station.append({
            "station_id": str(sid),
            "mae_model": m["mae_model"],
            "mae_baseline": m["mae_baseline"],
            "lift_vs_baseline": m["lift_vs_baseline"],
            "n": m["n"],
        })
    # trier par n desc puis par mae_model asc si dispo
    recs_station.sort(key=lambda r: (-int(r["n"]), (r["mae_model"] if r["mae_model"] is not None else 1e9)))
    if len(recs_station) > top_stations:
        recs_station = recs_station[:top_stations]

    # by_hour
    recs_hour = []
    for hr, grp in d.groupby("hour"):
        if len(grp) < min_samples:
            continue
        m = _group_metrics(grp)
        recs_hour.append({
            "hour": int(hr),
            "mae_model": m["mae_model"],
            "mae_baseline": m["mae_baseline"],
            "lift_vs_baseline": m["lift_vs_baseline"],
            "coverage_pred_pct": m["coverage_pred_pct"],
            "n": m["n"],
        })
    recs_hour.sort(key=lambda r: r["hour"])

    # by_dow
    recs_dow = []
    for dw, grp in d.groupby("dow"):
        if len(grp) < min_samples:
            continue
        m = _group_metrics(grp)
        recs_dow.append({
            "dow": int(dw),
            "mae_model": m["mae_model"],
            "mae_baseline": m["mae_baseline"],
            "lift_vs_baseline": m["lift_vs_baseline"],
            "coverage_pred_pct": m["coverage_pred_pct"],
            "n": m["n"],
        })
    recs_dow.sort(key=lambda r: r["dow"])

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "horizon_min": int(horizon_bins * 5),
        "by_station_top": recs_station,
        "by_hour": recs_hour,
        "by_dow": recs_dow,
    }

# ─────────────────────────── Main ───────────────────────────

def main() -> int:
    EXPORTS_PREFIX = os.environ.get("GCS_EXPORTS_PREFIX")
    MON_PREFIX     = os.environ.get("GCS_MONITORING_PREFIX")
    if not (EXPORTS_PREFIX and EXPORTS_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_EXPORTS_PREFIX manquant ou invalide")
    if not (MON_PREFIX and MON_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX manquant ou invalide")

    HORIZONS_MIN = [int(x.strip()) for x in os.environ.get("HORIZONS","15,60").split(",") if x.strip()]
    TOP_ST = int(os.environ.get("TOP_STATIONS", "200"))
    MIN_S  = int(os.environ.get("MIN_SAMPLES_PER_GROUP", "50"))
    DAYS   = os.environ.get("DATE_RANGE_DAYS")
    DAYS   = int(DAYS) if (DAYS and DAYS.strip().isdigit()) else 60

    anchor = _anchor_day_utc()
    anchor_tag = anchor.strftime("%Y%m%d")
    cutoff = _recent_cutoff(anchor, DAYS)

    perf_uri = f"{EXPORTS_PREFIX.rstrip('/')}/perf.parquet"
    print(f"[model_health] read: {perf_uri}")
    df = _read_parquet_gs(perf_uri)

    # types & nettoyages minimaux
    df["tbin_utc"] = pd.to_datetime(df["tbin_utc"], errors="coerce")
    # facultatif: y_pred peut ne pas exister
    if "y_pred" not in df.columns:
        print("[model_health][warn] y_pred absent → métriques 'model' et lift indisponibles")

    for hmin in HORIZONS_MIN:
        hb = max(1, int(round(hmin / 5)))
        # daily
        daily = _build_daily_json(df, horizon_bins=hb, cutoff=cutoff)
        out_alias = f"{MON_PREFIX.rstrip('/')}/model/perf/daily_h{hmin}.json"
        out_ver   = f"{MON_PREFIX.rstrip('/')}/model/perf/daily_h{hmin}_{anchor_tag}.json"
        _upload_json_gs(daily, out_alias)
        _upload_json_gs(daily, out_ver)

        # segments
        segs = _build_segments_json(df, horizon_bins=hb, top_stations=TOP_ST, min_samples=MIN_S)
        out_alias = f"{MON_PREFIX.rstrip('/')}/model/perf/segments_h{hmin}.json"
        out_ver   = f"{MON_PREFIX.rstrip('/')}/model/perf/segments_h{hmin}_{anchor_tag}.json"
        _upload_json_gs(segs, out_alias)
        _upload_json_gs(segs, out_ver)

    print("[model_health] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
