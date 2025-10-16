# service/jobs/build_model_performance.py
# -----------------------------------------------------------------------------
# Build "Model Performance" monitoring artifacts from perf_YYYY-MM-DD.parquet
# ➜ VERSION "LATEST ONLY" (aucun dossier versionné horodaté)
#
# Inputs  (GCS):
#   GCS_EXPORTS_PREFIX/perf_YYYY-MM-DD.parquet
#
# Outputs (GCS JSON — LATEST ONLY):
#   <MONITORING_BASE>/model/performance/latest/manifest.json
#   <MONITORING_BASE>/model/performance/latest/h{H}/kpis.json
#   <MONITORING_BASE>/model/performance/latest/h{H}/daily_metrics.json
#   <MONITORING_BASE>/model/performance/latest/h{H}/by_hour.json
#   <MONITORING_BASE>/model/performance/latest/h{H}/by_dow.json
#   <MONITORING_BASE>/model/performance/latest/h{H}/by_station.json
#   <MONITORING_BASE>/model/performance/latest/h{H}/by_cluster.json  (optional)
#   <MONITORING_BASE>/model/performance/latest/h{H}/lift_curve.json
#   <MONITORING_BASE>/model/performance/latest/h{H}/hist_residuals.json
#
# ENV (required):
#   GCS_EXPORTS_PREFIX     = gs://<bucket>/velib/exports
#   GCS_MONITORING_PREFIX  = gs://<bucket>/velib   (or .../velib/monitoring)
#
# ENV (optional):
#   PERF_TZ              = Europe/Paris
#   PERF_LAST_DAYS       = 14
#   PERF_HORIZONS        = "15"          (CSV like "15,60")
#   PERF_RESID_BINS      = 40
#   PERF_TOP_STATIONS    = 10
#   PERF_CLUSTERS_CSV    = gs://.../station_clusters.csv  (mapping station_id,cluster)
#
# Run:
#   python -m service.jobs.build_model_performance
# -----------------------------------------------------------------------------

from __future__ import annotations
import os, re, json, sys
from io import BytesIO
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception as e:
    raise RuntimeError("pyarrow required") from e

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage required") from e


SCHEMA_VERSION = "1.0"  # initial schema for model/performance JSONs
BIN_MIN = 5             # 5-minute cadence (pipeline invariant)


# ──────────────────────────────────────────────────────────────────────────────
# GCS helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split(gs: str) -> Tuple[str, str]:
    assert gs.startswith("gs://"), f"bad GCS uri: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k.rstrip("/")

def _read_parquet_blob_to_df(blob: "storage.Blob") -> pd.DataFrame:
    buf = BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    tbl = pq.read_table(buf)
    return tbl.to_pandas()

def _list_perf_blobs(exports_prefix: str, start_date: datetime, end_date: datetime) -> List["storage.Blob"]:
    """List perf_YYYY-MM-DD.parquet blobs between start_date and end_date (UTC day)."""
    bkt, key_prefix = _split(exports_prefix)
    client = storage.Client()
    bucket = client.bucket(bkt)
    blobs = list(client.list_blobs(bucket, prefix=key_prefix.strip("/") + "/"))
    pat = re.compile(r"perf_(\d{4}-\d{2}-\d{2})\.parquet$")
    out = []
    for bl in blobs:
        m = pat.search(bl.name)
        if not m:
            continue
        d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        if start_date.date() <= d <= end_date.date():
            out.append(bl)
    out.sort(key=lambda b: b.name)
    return out

def _upload_json_gs(obj: dict, gs_uri: str):
    """Upload JSON with NaN/Inf sanitized to null."""
    def _san(o):
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_san(v) for v in o]
        if isinstance(o, float):
            return float(o) if np.isfinite(o) else None
        return o

    safe = _san(obj)
    bkt, key = _split(gs_uri)
    data = json.dumps(safe, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    storage.Client().bucket(bkt).blob(key).upload_from_string(data, content_type="application/json")
    print(f"[model.performance] wrote → {gs_uri} ({len(data):,} bytes)")


# ──────────────────────────────────────────────────────────────────────────────
# Time & base path helpers
# ──────────────────────────────────────────────────────────────────────────────

def _compute_window(now_utc: datetime, last_days: int) -> Tuple[datetime, datetime]:
    """Inclusive window: from 00:00 UTC of (now - last_days + 1) to now."""
    start = (now_utc - timedelta(days=last_days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return start, now_utc

def _ensure_mon_base(mon_prefix: str) -> str:
    """Ensure path ends with '/monitoring' once."""
    base = mon_prefix.rstrip("/")
    if not base.endswith("/monitoring"):
        base = base + "/monitoring"
    return base

def _to_local(utc_series: pd.Series, tzname: str) -> pd.Series:
    s = pd.to_datetime(utc_series, utc=True, errors="coerce")
    return s.dt.tz_convert(tzname)


# ──────────────────────────────────────────────────────────────────────────────
# Schema detection & normalization
# ──────────────────────────────────────────────────────────────────────────────

def _detect_perf_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Auto-detect key columns with robust fallbacks."""
    lower = {c.lower(): c for c in df.columns}

    def any_of(*cands):
        for c in cands:
            if c.lower() in lower:
                return lower[c.lower()]
        return None

    tbin = any_of("tbin_utc", "ts", "timestamp", "datetime")
    sid = any_of("station_id", "stationcode", "id", "station")
    ytrue = any_of("y_true", "y", "target", "bikes_true", "nb_velos_bin")
    ypred_int = any_of("y_pred_int", "bikes_pred_int")
    # float fallback if int missing
    ypred_float = any_of("y_pred", "bikes_pred", "yhat", "prediction", "pred")
    ybase = any_of("y_baseline_persist", "y_pred_baseline", "baseline", "persist")
    cap = any_of("capacity", "cap", "num_docks_total", "dock_count")
    hb = any_of("horizon_bins",)
    hmin = any_of("horizon_min",)

    if not tbin or not sid or not ytrue or not ybase:
        raise KeyError(f"[model.performance] Missing minimal columns (tbin={tbin}, station={sid}, y_true={ytrue}, baseline={ybase})")

    return dict(
        tbin=tbin, station=sid, y_true=ytrue, y_pred_int=ypred_int,
        y_pred_float=ypred_float, y_baseline=ybase, capacity=cap,
        horizon_bins=hb, horizon_min=hmin
    )

def _normalize_types(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    out = df.copy()
    out[cols["tbin"]] = pd.to_datetime(out[cols["tbin"]], utc=True, errors="coerce")
    out[cols["station"]] = out[cols["station"]].astype("string")
    out[cols["y_true"]] = pd.to_numeric(out[cols["y_true"]], errors="coerce")
    out[cols["y_baseline"]] = pd.to_numeric(out[cols["y_baseline"]], errors="coerce")
    if cols["y_pred_int"] and cols["y_pred_int"] in out.columns:
        out[cols["y_pred_int"]] = pd.to_numeric(out[cols["y_pred_int"]], errors="coerce")
    if cols["y_pred_float"] and cols["y_pred_float"] in out.columns:
        out[cols["y_pred_float"]] = pd.to_numeric(out[cols["y_pred_float"]], errors="coerce")
    if cols["capacity"] and cols["capacity"] in out.columns:
        out[cols["capacity"]] = pd.to_numeric(out[cols["capacity"]], errors="coerce")
    # horizons
    if cols["horizon_bins"] and cols["horizon_bins"] in out.columns:
        out[cols["horizon_bins"]] = pd.to_numeric(out[cols["horizon_bins"]], errors="coerce")
    if cols["horizon_min"] and cols["horizon_min"] in out.columns:
        out[cols["horizon_min"]] = pd.to_numeric(out[cols["horizon_min"]], errors="coerce")
    return out

def _select_pred_series(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.Series:
    """Return the best available prediction series (Int preferred, else float)."""
    if cols["y_pred_int"] and cols["y_pred_int"] in df.columns:
        return pd.to_numeric(df[cols["y_pred_int"]], errors="coerce")
    if cols["y_pred_float"] and cols["y_pred_float"] in df.columns:
        return pd.to_numeric(df[cols["y_pred_float"]], errors="coerce")
    # no predictions → NaNs
    return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")

def _normalize_horizon_bin(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.Series:
    """Return a Series 'hbin' of integer horizon bins."""
    if cols["horizon_bins"] and cols["horizon_bins"] in df.columns:
        s = pd.to_numeric(df[cols["horizon_bins"]], errors="coerce")
        s = s.round().astype("Int64")
        return s
    if cols["horizon_min"] and cols["horizon_min"] in df.columns:
        m = pd.to_numeric(df[cols["horizon_min"]], errors="coerce")
        s = (m / BIN_MIN).round().astype("Int64")
        return s
    # fallback unknown
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def _metrics(y_true: pd.Series, y_hat: pd.Series) -> Dict[str, float]:
    err = y_true.astype("float64") - y_hat.astype("float64")
    mae = float(np.nanmean(np.abs(err))) if len(err) else float("nan")
    rmse = float(np.sqrt(np.nanmean(np.square(err)))) if len(err) else float("nan")
    me = float(np.nanmean(err)) if len(err) else float("nan")
    return {"mae": mae, "rmse": rmse, "me": me}

def _lift(mae_base: float, mae_model: float) -> float:
    if mae_base is None or np.isnan(mae_base) or mae_base == 0 or mae_model is None or np.isnan(mae_model):
        return float("nan")
    return float((mae_base - mae_model) / mae_base)

def _coverage_pct(pred: pd.Series) -> float:
    return float((pred.notna().mean() * 100.0).round(2)) if len(pred) else float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Computations (global/daily/segments/hist)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_global(df: pd.DataFrame, cols: Dict[str, Optional[str]], tzname: str, window_days: int, horizon_min: Optional[int]) -> dict:
    y_true = pd.to_numeric(df[cols["y_true"]], errors="coerce")
    y_base = pd.to_numeric(df[cols["y_baseline"]], errors="coerce")
    y_pred = _select_pred_series(df, cols)

    m_model = _metrics(y_true, y_pred)
    m_base = _metrics(y_true, y_base)
    cov = _coverage_pct(y_pred)

    g = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "window_days": int(window_days),
        "horizon_min": int(horizon_min) if horizon_min is not None else None,
        "coverage_pred_pct": cov,
        "mae_model": m_model["mae"], "rmse_model": m_model["rmse"], "me_model": m_model["me"],
        "mae_baseline": m_base["mae"], "rmse_baseline": m_base["rmse"], "me_baseline": m_base["me"],
        "lift_vs_baseline": _lift(m_base["mae"], m_model["mae"]),
        "n_rows": int(len(df)),
        "n_stations": int(df[cols["station"]].nunique()),
        "ts_min_utc": pd.to_datetime(df[cols["tbin"]], utc=True, errors="coerce").min().isoformat().replace("+00:00","Z") if len(df) else None,
        "ts_max_utc": pd.to_datetime(df[cols["tbin"]], utc=True, errors="coerce").max().isoformat().replace("+00:00","Z") if len(df) else None,
    }
    return g

def _compute_daily(df: pd.DataFrame, cols: Dict[str, Optional[str]], tzname: str, last_days: int) -> List[dict]:
    tloc = _to_local(df[cols["tbin"]], tzname)
    y_true = pd.to_numeric(df[cols["y_true"]], errors="coerce")
    y_base = pd.to_numeric(df[cols["y_baseline"]], errors="coerce")
    y_pred = _select_pred_series(df, cols)

    tmp = pd.DataFrame({
        "_date": tloc.dt.date,
        "_y": y_true,
        "_b": y_base,
        "_p": y_pred
    })
    grp = tmp.groupby("_date", dropna=False)
    rows = []
    for d, g in grp:
        mm = _metrics(g["_y"], g["_p"])
        mb = _metrics(g["_y"], g["_b"])
        cov = _coverage_pct(g["_p"])
        rows.append({
            "date": str(d),
            "mae_model": mm["mae"],
            "mae_baseline": mb["mae"],
            "rmse_model": mm["rmse"],
            "rmse_baseline": mb["rmse"],
            "coverage_pred_pct": cov,
            "lift_vs_baseline": _lift(mb["mae"], mm["mae"]),
            "n": int(len(g)),
        })
    rows = sorted(rows, key=lambda r: r["date"])
    if last_days and last_days > 0:
        rows = rows[-last_days:]
    return rows

def _compute_by_hour(df: pd.DataFrame, cols: Dict[str, Optional[str]], tzname: str) -> List[dict]:
    tloc = _to_local(df[cols["tbin"]], tzname)
    y_true = pd.to_numeric(df[cols["y_true"]], errors="coerce")
    y_base = pd.to_numeric(df[cols["y_baseline"]], errors="coerce")
    y_pred = _select_pred_series(df, cols)

    tmp = pd.DataFrame({
        "_hour": tloc.dt.hour,
        "_y": y_true,
        "_b": y_base,
        "_p": y_pred
    })
    grp = tmp.groupby("_hour", dropna=False)
    rows = []
    for h, g in grp:
        mm = _metrics(g["_y"], g["_p"])
        mb = _metrics(g["_y"], g["_b"])
        cov = _coverage_pct(g["_p"])
        rows.append({
            "hour": int(h),
            "mae_model": mm["mae"],
            "mae_baseline": mb["mae"],
            "coverage_pred_pct": cov,
            "n": int(len(g)),
        })
    rows = sorted(rows, key=lambda r: r["hour"])
    return rows

def _compute_by_dow(df: pd.DataFrame, cols: Dict[str, Optional[str]], tzname: str) -> List[dict]:
    tloc = _to_local(df[cols["tbin"]], tzname)
    y_true = pd.to_numeric(df[cols["y_true"]], errors="coerce")
    y_base = pd.to_numeric(df[cols["y_baseline"]], errors="coerce")
    y_pred = _select_pred_series(df, cols)

    tmp = pd.DataFrame({
        "_dow": tloc.dt.dayofweek,
        "_y": y_true,
        "_b": y_base,
        "_p": y_pred
    })
    grp = tmp.groupby("_dow", dropna=False)
    rows = []
    for d, g in grp:
        mm = _metrics(g["_y"], g["_p"])
        mb = _metrics(g["_y"], g["_b"])
        cov = _coverage_pct(g["_p"])
        rows.append({
            "dow": int(d),
            "mae_model": mm["mae"],
            "mae_baseline": mb["mae"],
            "coverage_pred_pct": cov,
            "n": int(len(g)),
        })
    rows = sorted(rows, key=lambda r: r["dow"])
    return rows

def _compute_by_station(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> List[dict]:
    y_true = pd.to_numeric(df[cols["y_true"]], errors="coerce")
    y_base = pd.to_numeric(df[cols["y_baseline"]], errors="coerce")
    y_pred = _select_pred_series(df, cols)

    tmp = pd.DataFrame({
        "_sid": df[cols["station"]].astype("string"),
        "_y": y_true,
        "_b": y_base,
        "_p": y_pred
    })
    grp = tmp.groupby("_sid", dropna=False)
    rows = []
    for sid, g in grp:
        mm = _metrics(g["_y"], g["_p"])
        mb = _metrics(g["_y"], g["_b"])
        cov = _coverage_pct(g["_p"])
        rows.append({
            "station_id": str(sid),
            "mae_model": mm["mae"],
            "mae_baseline": mb["mae"],
            "coverage_pred_pct": cov,
            "n": int(len(g)),
            "lift_vs_baseline": _lift(mb["mae"], mm["mae"]),
        })
    rows = sorted(rows, key=lambda r: r["station_id"])
    return rows

def _compute_by_cluster(df: pd.DataFrame, cols: Dict[str, Optional[str]], clusters_df: Optional[pd.DataFrame]) -> Optional[List[dict]]:
    if clusters_df is None or clusters_df.empty:
        return None
    m = df.copy()
    m[cols["station"]] = m[cols["station"]].astype("string")
    clusters_df = clusters_df.copy()
    if "station_id" not in clusters_df.columns or "cluster" not in clusters_df.columns:
        return None
    clusters_df["station_id"] = clusters_df["station_id"].astype("string")
    m = m.merge(clusters_df[["station_id", "cluster"]], left_on=cols["station"], right_on="station_id", how="left")
    if m["cluster"].notna().sum() == 0:
        return None

    y_true = pd.to_numeric(m[cols["y_true"]], errors="coerce")
    y_base = pd.to_numeric(m[cols["y_baseline"]], errors="coerce")
    y_pred = _select_pred_series(m, cols)

    tmp = pd.DataFrame({
        "_cluster": m["cluster"],
        "_y": y_true, "_b": y_base, "_p": y_pred
    }).dropna(subset=["_cluster"])
    grp = tmp.groupby("_cluster", dropna=False)
    rows = []
    for cl, g in grp:
        mm = _metrics(g["_y"], g["_p"])
        mb = _metrics(g["_y"], g["_b"])
        cov = _coverage_pct(g["_p"])
        rows.append({
            "cluster": cl,
            "mae_model": mm["mae"],
            "mae_baseline": mb["mae"],
            "coverage_pred_pct": cov,
            "n": int(len(g)),
            "lift_vs_baseline": _lift(mb["mae"], mm["mae"]),
        })
    rows = sorted(rows, key=lambda r: (str(r["cluster"])))
    return rows

def _build_lift_curve(daily_rows: List[dict]) -> dict:
    pts = [{"date": r["date"], "lift_vs_baseline": r.get("lift_vs_baseline", None)} for r in daily_rows]
    return {"schema_version": SCHEMA_VERSION, "points": pts}

def _build_residual_hist(df: pd.DataFrame, cols: Dict[str, Optional[str]], bins: int) -> dict:
    y_true = pd.to_numeric(df[cols["y_true"]], errors="coerce")
    y_pred = _select_pred_series(df, cols)
    mask = y_pred.notna() & y_true.notna()
    if not mask.any():
        return {"schema_version": SCHEMA_VERSION, "bins": [], "counts": [], "n": 0}
    err = (y_true[mask] - y_pred[mask]).astype("float64").to_numpy()

    # symmetric range around 0 using 99th percentile of |err| to avoid heavy tails
    max_abs = float(np.nanpercentile(np.abs(err), 99)) if err.size else 1.0
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0
    hist_counts, bin_edges = np.histogram(err, bins=bins, range=(-max_abs, max_abs))
    return {
        "schema_version": SCHEMA_VERSION,
        "bins": [float(x) for x in bin_edges.tolist()],
        "counts": [int(c) for c in hist_counts.tolist()],
        "n": int(len(err))
    }


# ──────────────────────────────────────────────────────────────────────────────
# Clusters CSV (optional)
# ──────────────────────────────────────────────────────────────────────────────

def _read_clusters_csv(gs_uri: Optional[str]) -> Optional[pd.DataFrame]:
    if not gs_uri or not gs_uri.startswith("gs://"):
        return None
    bkt, key = _split(gs_uri)
    cli = storage.Client()
    bl = cli.bucket(bkt).blob(key)
    if not bl.exists():
        print(f"[model.performance] clusters csv not found: {gs_uri}")
        return None
    data = bl.download_as_bytes()
    try:
        df = pd.read_csv(BytesIO(data), dtype={"station_id": "string"})
        if "station_id" in df.columns and "cluster" in df.columns:
            return df[["station_id", "cluster"]].drop_duplicates()
        print(f"[model.performance][warn] clusters csv missing required columns: {list(df.columns)}")
        return None
    except Exception as e:
        print(f"[model.performance][warn] failed to read clusters csv: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Publishing (LATEST ONLY)
# ──────────────────────────────────────────────────────────────────────────────

def _publish_latest(base_alias: str, horizon_min: int, payloads: Dict[str, dict]) -> None:
    """Publish artifacts for a given horizon under latest/h{H} only."""
    hdir = f"h{int(horizon_min)}"
    for name, obj in payloads.items():
        _upload_json_gs(obj, f"{base_alias}/{hdir}/{name}.json")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    EXPORTS_PREFIX = os.environ.get("GCS_EXPORTS_PREFIX")     # gs://.../velib/exports
    MON_PREFIX     = os.environ.get("GCS_MONITORING_PREFIX")  # gs://.../velib (or .../monitoring)
    if not (EXPORTS_PREFIX and EXPORTS_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_EXPORTS_PREFIX missing or invalid")
    if not (MON_PREFIX and MON_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX missing or invalid")

    TZNAME         = os.environ.get("PERF_TZ", "Europe/Paris")
    LAST_DAYS      = int(os.environ.get("PERF_LAST_DAYS", "14"))
    RESID_BINS     = int(os.environ.get("PERF_RESID_BINS", "40"))
    HORIZONS       = [int(x.strip()) for x in os.environ.get("PERF_HORIZONS", "15").split(",") if x.strip()]
    CLUSTERS_URI   = os.environ.get("PERF_CLUSTERS_CSV", None)

    now = datetime.now(timezone.utc)
    start, end = _compute_window(now, LAST_DAYS)
    print(f"[model.performance] window UTC: {start.date()} → {end.date()} (days={LAST_DAYS}) | horizons={HORIZONS}")

    # Read perf blobs in window
    blobs = _list_perf_blobs(EXPORTS_PREFIX, start, end)
    if not blobs:
        print("[model.performance] no perf_* blobs in window — nothing to do")
        return 0

    frames: List[pd.DataFrame] = []
    for bl in blobs:
        print(f"[read] {bl.name}")
        try:
            df = _read_parquet_blob_to_df(bl)
            frames.append(df)
        except Exception as e:
            print(f"[warn] failed to read {bl.name}: {e}")

    if not frames:
        print("[model.performance] no readable data — nothing to do")
        return 0

    perf = pd.concat(frames, ignore_index=True)
    if perf.empty:
        print("[model.performance] perf empty — nothing to do")
        return 0

    # Detect & normalize columns / types
    cols = _detect_perf_columns(perf)
    perf = _normalize_types(perf, cols)
    # Explicit working columns
    tbin_col = cols["tbin"]; sid_col = cols["station"]
    y_col = cols["y_true"]; b_col = cols["y_baseline"]
    pred_series = _select_pred_series(perf, cols)
    hbin_series = _normalize_horizon_bin(perf, cols)

    # Add normalized columns for processing
    perf = perf.assign(
        __tbin=perf[tbin_col],
        __sid=perf[sid_col].astype("string"),
        __y=pd.to_numeric(perf[y_col], errors="coerce"),
        __b=pd.to_numeric(perf[b_col], errors="coerce"),
        __p=pred_series,
        __hbin=hbin_series
    ).dropna(subset=["__tbin", "__sid"])

    # Optional clusters
    clusters_df = _read_clusters_csv(CLUSTERS_URI)

    # Output base (LATEST ONLY)
    mon_base = _ensure_mon_base(MON_PREFIX)
    base_alias = f"{mon_base}/model/performance/latest"

    # Determine available horizons in data
    available_hbins = sorted([int(x) for x in pd.Series(perf["__hbin"].dropna().unique()).tolist() if pd.notna(x)])
    requested_hbins = sorted(list(set([max(1, int(round(h / BIN_MIN))) for h in HORIZONS])))
    target_hbins = [hb for hb in requested_hbins if hb in available_hbins]
    if not target_hbins:
        print(f"[model.performance] no matching horizons in data. available_hbins={available_hbins}, requested={requested_hbins}")
        # on publie tout de même un manifest minimal pour latest
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "window_days": int(LAST_DAYS),
            "latest_prefix": base_alias,
            "horizons": [],
            "horizons_info": []
        }
        _upload_json_gs(manifest, f"{base_alias}/manifest.json")
        return 0

    manifests_items = []
    for hb in target_hbins:
        hmin = hb * BIN_MIN
        sub = perf.loc[perf["__hbin"] == hb].copy()
        if sub.empty:
            print(f"[model.performance] skip h={hmin} (empty after filter)")
            continue

        # Compute artifacts
        kpis = _compute_global(sub, cols={
            "tbin": "__tbin", "station": "__sid",
            "y_true": "__y", "y_baseline": "__b",
            "y_pred_int": "__p", "y_pred_float": "__p",
            "capacity": cols.get("capacity")
        }, tzname=TZNAME, window_days=LAST_DAYS, horizon_min=hmin)

        daily_rows = _compute_daily(sub, cols={
            "tbin": "__tbin", "station": "__sid",
            "y_true": "__y", "y_baseline": "__b",
            "y_pred_int": "__p", "y_pred_float": "__p",
        }, tzname=TZNAME, last_days=LAST_DAYS)

        by_hour_rows = _compute_by_hour(sub, cols={
            "tbin": "__tbin", "station": "__sid",
            "y_true": "__y", "y_baseline": "__b",
            "y_pred_int": "__p", "y_pred_float": "__p",
        }, tzname=TZNAME)

        by_dow_rows = _compute_by_dow(sub, cols={
            "tbin": "__tbin", "station": "__sid",
            "y_true": "__y", "y_baseline": "__b",
            "y_pred_int": "__p", "y_pred_float": "__p",
        }, tzname=TZNAME)

        by_station_rows = _compute_by_station(sub, cols={
            "tbin": "__tbin", "station": "__sid",
            "y_true": "__y", "y_baseline": "__b",
            "y_pred_int": "__p", "y_pred_float": "__p",
        })

        by_cluster_rows = _compute_by_cluster(sub, cols={
            "tbin": "__tbin", "station": "__sid",
            "y_true": "__y", "y_baseline": "__b",
            "y_pred_int": "__p", "y_pred_float": "__p",
        }, clusters_df=clusters_df)

        lift_curve = _build_lift_curve(daily_rows)
        hist_res = _build_residual_hist(sub, cols={
            "tbin": "__tbin", "station": "__sid",
            "y_true": "__y", "y_baseline": "__b",
            "y_pred_int": "__p", "y_pred_float": "__p",
        }, bins=RESID_BINS)

        # Assemble payloads per horizon
        payloads: Dict[str, dict] = {
            "kpis": kpis,
            "daily_metrics": {"schema_version": SCHEMA_VERSION, "horizon_min": int(hmin), "rows": daily_rows},
            "by_hour": {"schema_version": SCHEMA_VERSION, "horizon_min": int(hmin), "rows": by_hour_rows},
            "by_dow": {"schema_version": SCHEMA_VERSION, "horizon_min": int(hmin), "rows": by_dow_rows},
            "by_station": {"schema_version": SCHEMA_VERSION, "horizon_min": int(hmin), "rows": by_station_rows},
            "lift_curve": dict(lift_curve, **{"horizon_min": int(hmin)}),
            "hist_residuals": dict(hist_res, **{"horizon_min": int(hmin)}),
        }
        if by_cluster_rows is not None:
            payloads["by_cluster"] = {"schema_version": SCHEMA_VERSION, "horizon_min": int(hmin), "rows": by_cluster_rows}

        # Publish LATEST ONLY under h{H}
        _publish_latest(base_alias, horizon_min=hmin, payloads=payloads)

        manifests_items.append({
            "horizon_min": int(hmin),
            "prefix_latest": f"{base_alias}/h{int(hmin)}",
            "artifacts": list(payloads.keys())
        })

    if not manifests_items:
        print("[model.performance] nothing published — all horizons empty")
        # manifest minimal
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "window_days": int(LAST_DAYS),
            "latest_prefix": base_alias,
            "horizons": [],
            "horizons_info": []
        }
        _upload_json_gs(manifest, f"{base_alias}/manifest.json")
        return 0

    # Top-level manifest (LATEST ONLY)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "window_days": int(LAST_DAYS),
        "latest_prefix": base_alias,
        "horizons": [int(it["horizon_min"]) for it in manifests_items],
        "horizons_info": manifests_items
    }
    _upload_json_gs(manifest, f"{base_alias}/manifest.json")

    print("[model.performance] done (latest only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
