# service/jobs/build_data_health.py
from __future__ import annotations
import os, re, json, sys, math
from io import BytesIO
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception as e:
    raise RuntimeError("pyarrow requis") from e

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage requis") from e

SCHEMA_VERSION = "1.1"  # 1.0 + allègement/sharding anomalies + en-tête params

# ─────────────────────────── Helpers GCS ───────────────────────────

def _split(gs: str) -> Tuple[str, str]:
    assert gs.startswith("gs://"), f"bad GCS uri: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k.rstrip("/")

def _list_event_blobs(exports_prefix: str, start_date: datetime, end_date: datetime) -> List["storage.Blob"]:
    bkt, key_prefix = _split(exports_prefix)
    client = storage.Client()
    bucket = client.bucket(bkt)
    blobs = list(client.list_blobs(bucket, prefix=key_prefix.strip("/") + "/"))
    pat = re.compile(r"events_(\d{4}-\d{2}-\d{2})\.parquet$")
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

def _read_parquet_blob_to_df(blob: "storage.Blob") -> pd.DataFrame:
    buf = BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    tbl = pq.read_table(buf)
    return tbl.to_pandas()

def _upload_json_gs(obj: object, gs_uri: str) -> None:
    # JSON safe: remplace NaN/±Inf par null, cast numpy types
    def san(o):
        if isinstance(o, dict):  return {k: san(v) for k, v in o.items()}
        if isinstance(o, list):  return [san(v) for v in o]
        if isinstance(o, (np.floating,)): return float(o) if np.isfinite(o) else None
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.bool_,)):    return bool(o)
        if o is None: return None
        try:
            if pd.isna(o): return None
        except Exception:
            pass
        return o
    safe = san(obj)
    bkt, key = _split(gs_uri)
    data = json.dumps(safe, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    storage.Client().bucket(bkt).blob(key).upload_from_string(data, content_type="application/json")
    print(f"[data.health] wrote → {gs_uri} ({len(data):,} bytes)")

# ───────────────────── Allègement helpers ─────────────────────

def _r(x: float, nd: int) -> Optional[float]:
    try:
        if not np.isfinite(x): return None
        return float(np.round(float(x), nd))
    except Exception:
        return None

def _rl(vals, nd: int):
    out = []
    for v in vals:
        if isinstance(v, (int, float, np.floating)):
            out.append(_r(float(v), nd))
        else:
            out.append(v)
    return out

# ───────────────────── Détection colonnes & utils ─────────────────────

def _detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    lower = {c.lower(): c for c in df.columns}
    def any_of(*cands):
        for c in cands:
            if c in lower:
                return lower[c]
        return None
    ts  = any_of("ts","tbin_utc","timestamp","datetime")
    sid = any_of("station_id","stationcode","station","id")
    bikes = any_of("bikes","nb_velos_bin","num_bikes_available","velos","velos_disponibles")
    docks = any_of("docks_avail","nb_docks_bin","num_docks_available","free_docks","places_disponibles")
    cap   = any_of("capacity","num_docks_total","dock_count","cap")
    ing   = any_of("ingested_at","ingest_ts","ingest_time","received_at","etl_ts","load_ts","created_at")
    lat   = any_of("lat","latitude"); lon = any_of("lon","lng","longitude"); name = any_of("name","station_name","nom")
    if not ts or not sid or not bikes:
        raise KeyError(f"[data.health] Colonnes minimales absentes (ts={ts}, station={sid}, bikes={bikes})")
    return dict(ts=ts, station=sid, bikes=bikes, docks=docks, capacity=cap, ingested_at=ing, lat=lat, lon=lon, name=name)

def _now_floor_utc(bin_min: int) -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor(f"{int(bin_min)}min").tz_convert(None)

def _expected_bins(tmin: pd.Timestamp, tmax: pd.Timestamp, bin_min: int) -> int:
    return max(1, math.ceil((tmax - tmin).total_seconds() / 60.0 / bin_min) + 1)

# ─────────────────────────── Calculs KPI ───────────────────────────

def kpi_freshness(df: pd.DataFrame, sid_col: str, ts_col: str, slo_min: int, bin_min: int) -> dict:
    now = _now_floor_utc(bin_min)
    last = df.groupby(sid_col)[ts_col].max().reset_index(name="last_ts")
    last["age_min"] = (now - last["last_ts"]).dt.total_seconds() / 60.0
    p50 = float(np.nanpercentile(last["age_min"], 50)) if len(last) else np.nan
    p95 = float(np.nanpercentile(last["age_min"], 95)) if len(last) else np.nan
    return {
        "now_utc": now.isoformat(),
        "ts_global_max": (df[ts_col].max().isoformat() if len(df) else None),
        "freshness_age_p50_min": _r(p50, 2),
        "freshness_age_p95_min": _r(p95, 2),
        "freshness_slo_min": float(slo_min),
        "bin_min": int(bin_min),
        "freshness_p95_ok": (p95 <= slo_min) if np.isfinite(p95) else None,
    }

def kpi_completeness(df: pd.DataFrame, sid_col: str, ts_col: str, current_days: int, bin_min: int, tz: Optional[str]) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    if current_days <= 0 or df.empty:
        return {"coverage_global_pct": np.nan}, pd.DataFrame(), pd.DataFrame()
    tmax = df[ts_col].max()
    tmin = tmax - pd.Timedelta(days=current_days)
    win = df[(df[ts_col] > tmin) & (df[ts_col] <= tmax)].copy()
    if win.empty:
        return {"coverage_global_pct": np.nan}, pd.DataFrame(), pd.DataFrame()

    exp = _expected_bins(tmin, tmax, bin_min)

    # par station (>=1 point dans la fenêtre)
    per_station = (win.groupby(sid_col)[ts_col].nunique().clip(upper=exp)
                     .rename_axis(sid_col).reset_index(name="obs"))
    per_station["expected"] = int(exp)
    per_station["coverage_pct"] = per_station["obs"] / per_station["expected"] * 100.0
    coverage_global = float(per_station["coverage_pct"].mean()) if len(per_station) else np.nan

    # coverage by hour (pour la page)
    if tz:
        win["_hour"] = pd.to_datetime(win[ts_col]).dt.tz_localize("UTC").dt.tz_convert(tz).dt.hour
    else:
        win["_hour"] = pd.to_datetime(win[ts_col]).dt.hour
    per_hour = (win.groupby(["_hour", sid_col])[ts_col].nunique().rename("obs").reset_index())
    exp_hour = current_days * (60 // max(1, int(bin_min)))
    per_hour["coverage_pct"] = per_hour["obs"].clip(upper=exp_hour) / float(exp_hour) * 100.0
    cov_by_hour = (per_hour.groupby("_hour")["coverage_pct"].mean()
                           .rename_axis("hour").reset_index())

    return {"coverage_global_pct": _r(coverage_global, 2)}, per_station, cov_by_hour

def kpi_latency(df: pd.DataFrame, ts_col: str, ing_col: Optional[str], cap_min: Optional[int] = None) -> Tuple[dict, Optional[pd.DataFrame]]:
    """
    Calcule la latence = (ingested_at - ts) en minutes.
    - Ignore valeurs négatives et > cap_min si fourni (outliers).
    - Si ing_col absent → retourne des métriques NaN.
    """
    if not ing_col or ing_col not in df.columns or df[ing_col].isna().all():
        return {"latency_p50_min": np.nan, "latency_p95_min": np.nan}, None

    lat = df.dropna(subset=[ts_col, ing_col]).copy()
    lat["latency_min"] = (lat[ing_col] - lat[ts_col]).dt.total_seconds() / 60.0

    # filtres de robustesse
    lat = lat[np.isfinite(lat["latency_min"])]
    lat = lat[lat["latency_min"] >= 0]
    if cap_min is not None and np.isfinite(cap_min):
        lat = lat[lat["latency_min"] <= float(cap_min)]

    if lat.empty:
        return {"latency_p50_min": np.nan, "latency_p95_min": np.nan}, None

    p50 = float(np.nanpercentile(lat["latency_min"], 50))
    p95 = float(np.nanpercentile(lat["latency_min"], 95))
    return {"latency_p50_min": _r(p50, 2), "latency_p95_min": _r(p95, 2)}, lat[["station_id","latency_min"]] if "station_id" in lat.columns else lat[["latency_min"]]

def duplication_stats(df: pd.DataFrame, sid_col: str, ts_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[sid_col, "dups"])
    dups = df.duplicated(subset=[ts_col, sid_col]).groupby(df[sid_col]).sum().astype(int)
    out = dups.rename_axis(sid_col).reset_index(name="dups").sort_values("dups", ascending=False)
    # assure types stables
    out[sid_col] = out[sid_col].astype("string")
    out["dups"] = pd.to_numeric(out["dups"], errors="coerce").fillna(0).astype(int)
    return out

def flat_sequences(df: pd.DataFrame, sid_col: str, ts_col: str, bikes_col: str, min_steps: int, current_days: int, bin_min: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[sid_col,"steps","start","end","duration_min"])
    tmax = df[ts_col].max(); tmin = tmax - pd.Timedelta(days=current_days)
    win = df[(df[ts_col] > tmin) & (df[ts_col] <= tmax)].copy()
    if win.empty:
        return pd.DataFrame(columns=[sid_col,"steps","start","end","duration_min"])
    win = win.sort_values([sid_col, ts_col])
    win["delta"] = win.groupby(sid_col)[bikes_col].diff().fillna(0.0).abs()
    win["is_flat"] = (win["delta"] == 0.0)
    win["grp"] = (~win["is_flat"]).groupby(win[sid_col]).cumsum()
    agg = (win[win["is_flat"]]
           .groupby([sid_col,"grp"])
           .agg(steps=("is_flat","size"), start=(ts_col,"min"), end=(ts_col,"max"))
           .reset_index())
    out = agg[agg["steps"] >= min_steps][[sid_col,"steps","start","end"]]
    out["duration_min"] = (out["steps"] * bin_min).astype(int)
    out = out.sort_values("steps", ascending=False).reset_index(drop=True)
    # types
    out[sid_col] = out[sid_col].astype("string")
    return out

# ─────────────────────────── Main ───────────────────────────

def main() -> int:
    EXPORTS_PREFIX = os.environ.get("GCS_EXPORTS_PREFIX")    # gs://bucket/velib/exports
    MON_PREFIX     = os.environ.get("GCS_MONITORING_PREFIX") # gs://bucket/velib (ou …/monitoring)
    if not (EXPORTS_PREFIX and EXPORTS_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_EXPORTS_PREFIX manquant ou invalide")
    if not (MON_PREFIX and MON_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX manquant ou invalide")

    # Paramètres (fenêtre & seuils)
    TZNAME            = os.environ.get("DATA_HEALTH_TZ", "Europe/Paris")
    CURRENT_DAYS      = int(os.environ.get("DATA_HEALTH_CURRENT_DAYS", "7"))
    BIN_MIN           = int(os.environ.get("DATA_HEALTH_BIN_MIN", "5"))
    FRESH_SLO_MIN     = int(os.environ.get("DATA_HEALTH_FRESH_SLO_MIN", "5"))
    DUP_ALERT_PCT     = float(os.environ.get("DATA_HEALTH_DUP_ALERT_PCT", "1.0"))
    FLAT_STEPS        = int(os.environ.get("DATA_HEALTH_FLAT_STEPS", "6"))
    COMPL_ALERT_PCT   = float(os.environ.get("DATA_HEALTH_COMPL_ALERT_PCT", "98.0"))
    LAT_MAX_MIN       = int(os.environ.get("DATA_HEALTH_LAT_MAX_MIN", str(72*60)))  # ignore latence > 72h

    # Allègement anomalies
    ANOM_TOPK          = int(os.environ.get("DATA_HEALTH_ANOM_TOPK", "200"))   # 0 = no topK
    ANOM_SAMPLE        = float(os.environ.get("DATA_HEALTH_ANOM_SAMPLE", "0")) # 0..1 fraction (après tri)
    ROUND_ND           = int(os.environ.get("DATA_HEALTH_ROUND", "3"))
    ANOM_INCLUDE_NAMES = os.environ.get("DATA_HEALTH_ANOM_INCLUDE_NAMES", "true").lower() in ("1","true","yes","y")

    # Fenêtre de lecture: au moins CURRENT_DAYS (on ajoute 1 jour de marge)
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=max(CURRENT_DAYS, 7))).replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"[data.health] window UTC: {start.date()} → {now.date()} (days≥{CURRENT_DAYS})")

    blobs = _list_event_blobs(EXPORTS_PREFIX, start, now)
    if not blobs:
        print("[data.health] no event blobs in window — nothing to do")
        return 0

    frames: List[pd.DataFrame] = []
    for bl in blobs:
        print(f"[read] {bl.name}")
        try:
            frames.append(_read_parquet_blob_to_df(bl))
        except Exception as e:
            print(f"[warn] failed to read {bl.name}: {e}")

    if not frames:
        print("[data.health] no readable data — nothing to do")
        return 0

    df = pd.concat(frames, ignore_index=True)
    cols = _detect_columns(df)
    ts_col, sid_col, bikes_col = cols["ts"], cols["station"], cols["bikes"]

    # Typage
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dt.tz_convert(None)
    df[sid_col] = df[sid_col].astype("string")
    df[bikes_col] = pd.to_numeric(df[bikes_col], errors="coerce")
    if cols["docks"] and cols["docks"] in df.columns:
        df[cols["docks"]] = pd.to_numeric(df[cols["docks"]], errors="coerce")
    if cols["capacity"] and cols["capacity"] in df.columns:
        df[cols["capacity"]] = pd.to_numeric(df[cols["capacity"]], errors="coerce")
    if cols["ingested_at"] and cols["ingested_at"] in df.columns:
        df[cols["ingested_at"]] = pd.to_datetime(df[cols["ingested_at"]], utc=True, errors="coerce").dt.tz_convert(None)

    df = df.dropna(subset=[ts_col, sid_col]).copy()

    # KPIs
    kfresh = kpi_freshness(df, sid_col, ts_col, FRESH_SLO_MIN, BIN_MIN)
    kcov, by_station_cov, cov_by_hour = kpi_completeness(df, sid_col, ts_col, CURRENT_DAYS, BIN_MIN, TZNAME)

    # Latence (plus robuste)
    klat, lat_df = kpi_latency(df, ts_col, cols["ingested_at"], cap_min=LAT_MAX_MIN)

    # Duplications / flats
    flats = flat_sequences(df, sid_col, ts_col, bikes_col, FLAT_STEPS, CURRENT_DAYS, BIN_MIN)
    dups = duplication_stats(df, sid_col, ts_col)
    dup_total = int(dups["dups"].sum()) if len(dups) else 0
    dups_pct = (dup_total / max(1, len(df))) * 100.0 if len(df) else 0.0

    # Missing bins (fenêtre récente)
    tmax = df[ts_col].max(); tmin = tmax - pd.Timedelta(days=CURRENT_DAYS)
    exp = _expected_bins(tmin, tmax, BIN_MIN)
    got = (df[(df[ts_col] > tmin) & (df[ts_col] <= tmax)]
             .groupby(sid_col)[ts_col].nunique()
             .rename_axis(sid_col).reset_index(name="obs"))
    miss = got.copy()
    miss["missing"] = (exp - miss["obs"]).clip(lower=0)
    miss["expected"] = int(exp)
    missing_stations = int((miss["missing"] > 0).sum()) if len(miss) else 0

    # ── 1) table station_health (corrigée + noms éventuels)
    station_health_rows = (
        by_station_cov.merge(miss[[sid_col, "missing"]], on=sid_col, how="left")
        .rename(columns={sid_col: "station_id"})
        .sort_values("coverage_pct", ascending=True)
        .reset_index(drop=True)
    )
    station_health_rows["expected"] = int(exp)
    station_health_rows["missing"] = station_health_rows["missing"].where(
        station_health_rows["missing"].notna(),
        other=(station_health_rows["expected"] - station_health_rows["obs"]).clip(lower=0)
    )

    # Map 'station_id' → 'name' (dernier nom non nul connu)
    if ANOM_INCLUDE_NAMES and cols.get("name") and cols["name"] in df.columns:
        name_col = cols["name"]
        name_map_df = (
            df.dropna(subset=[name_col])
            .sort_values([sid_col, ts_col])
            .groupby(sid_col, as_index=False)
            .tail(1)[[sid_col, name_col]]
            .drop_duplicates(subset=[sid_col])
            .rename(columns={sid_col: "station_id", name_col: "name"})
        )
        if not name_map_df.empty:
            station_health_rows = station_health_rows.merge(name_map_df, on="station_id", how="left")
    else:
        station_health_rows["name"] = None

    cols_out = ["station_id", "name", "obs", "expected", "coverage_pct", "missing"]
    for c in cols_out:
        if c not in station_health_rows.columns:
            station_health_rows[c] = None
    station_health_rows = station_health_rows[cols_out]

    # ── 2) coverage by hour
    cov_by_hour_rows = cov_by_hour.rename(columns={"coverage_pct": "coverage_pct"}).to_dict(orient="records")

    # ── 3) Anomalies (shardées & allégées)
    # nom lookup (optionnel)
    name_lookup: Optional[Dict[str, str]] = None
    if ANOM_INCLUDE_NAMES and cols.get("name") and cols["name"] in df.columns:
        name_col = cols["name"]
        name_lookup_df = (
            df.dropna(subset=[name_col])
            .sort_values([sid_col, ts_col])
            .groupby(sid_col, as_index=False)
            .tail(1)[[sid_col, name_col]]
            .drop_duplicates(subset=[sid_col])
            .rename(columns={sid_col: "station_id", name_col: "name"})
        )
        name_lookup = dict(zip(name_lookup_df["station_id"].astype(str), name_lookup_df["name"].astype(str)))

    def _nm(sid: str) -> Optional[str]:
        if not name_lookup: return None
        return name_lookup.get(str(sid))

    # 3.a flats
    an_flat: List[Dict[str, object]] = []
    if len(flats):
        for _, r in flats.iterrows():
            sid = str(r[sid_col])
            an_flat.append({
                "type": "flat_sequence",
                "station_id": sid,
                **({"name": _nm(sid)} if ANOM_INCLUDE_NAMES else {}),
                "start": r["start"].isoformat(),
                "end": r["end"].isoformat(),
                "steps": int(r["steps"]),
                "duration_min": int(r["duration_min"]),
            })

    # 3.b duplicates
    an_dups: List[Dict[str, object]] = []
    if len(dups):
        # garde-fou: tri uniquement si colonne disponible
        dups_sorted = dups.copy()
        if "dups" in dups_sorted.columns:
            dups_sorted = dups_sorted.sort_values("dups", ascending=False)
        for _, r in dups_sorted.iterrows():
            sid = str(r[sid_col])
            val = int(r["dups"]) if "dups" in r and pd.notna(r["dups"]) else 0
            if val <= 0:  # on ne stocke que s'il y a duplication
                continue
            an_dups.append({
                "type": "duplicates",
                "station_id": sid,
                **({"name": _nm(sid)} if ANOM_INCLUDE_NAMES else {}),
                "dups": val,
            })

    # 3.c missing (top stations les plus manquantes)
    an_missing: List[Dict[str, object]] = []
    if len(miss):
        worst_miss = miss.sort_values("missing", ascending=False)
        for _, r in worst_miss.iterrows():
            if r["missing"] > 0:
                sid = str(r[sid_col])
                an_missing.append({
                    "type": "missing_bins",
                    "station_id": sid,
                    **({"name": _nm(sid)} if ANOM_INCLUDE_NAMES else {}),
                    "missing": int(r["missing"]),
                    "expected": int(r["expected"]),
                })

    # Post-traitement anomalies: topK → sample → round
    def _postproc(an_rows: List[Dict[str, object]], sort_key: Optional[str] = None) -> List[Dict[str, object]]:
        rows = list(an_rows)
        if not rows:
            return rows
        if sort_key and all((sort_key in r) for r in rows):
            rows = sorted(rows, key=lambda r: r.get(sort_key, 0), reverse=True)
        if ANOM_TOPK > 0 and len(rows) > ANOM_TOPK:
            rows = rows[:ANOM_TOPK]
        if 0.0 < ANOM_SAMPLE < 1.0 and len(rows) > 0:
            # échantillonnage simple et déterministe: on prend un pas régulier
            step = max(1, int(round(1.0 / ANOM_SAMPLE)))
            rows = rows[::step]
        # arrondis légers
        for r in rows:
            for k, v in list(r.items()):
                if isinstance(v, (float, np.floating)):
                    r[k] = _r(float(v), ROUND_ND)
        return rows

    an_flat     = _postproc(an_flat, sort_key="duration_min")
    an_dups     = _postproc(an_dups, sort_key="dups")
    an_missing  = _postproc(an_missing, sort_key="missing")

    # Alertes globales
    alerts = {
        "freshness_p95_ok": (None if kfresh.get("freshness_p95_ok") is None else bool(kfresh["freshness_p95_ok"])),
        "coverage_ok": bool(float(kcov.get("coverage_global_pct", 0.0)) >= float(COMPL_ALERT_PCT)),
        "duplication_alert": bool(dups_pct >= float(DUP_ALERT_PCT)),
        "flat_sequences_found": bool(len(an_flat)),
    }

    # KPI document principal (avec en-tête params lisible)
    params_header = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "tz": TZNAME or "UTC",
        "bin_min": int(BIN_MIN),
        "current_days": int(CURRENT_DAYS),
        "thresholds": {
            "fresh_slo_min": FRESH_SLO_MIN,
            "compl_alert_pct": COMPL_ALERT_PCT,
            "dup_alert_pct": DUP_ALERT_PCT,
            "flat_steps": FLAT_STEPS,
            "lat_max_min": LAT_MAX_MIN,
        },
        "anomaly_params": {
            "topk": ANOM_TOPK,
            "sample_frac": ANOM_SAMPLE,
            "round_nd": ROUND_ND,
            "include_names": ANOM_INCLUDE_NAMES,
        },
        "sources": {
            "exports_prefix": EXPORTS_PREFIX,
        },
    }

    data_health = {
        **params_header,
        "rows": int(len(df)),
        "stations": int(df[sid_col].nunique()),
        "span": [
            df[ts_col].min().isoformat() if len(df) else None,
            df[ts_col].max().isoformat() if len(df) else None
        ],
        **kfresh, **kcov, **klat,
        "dups_pct": _r(dups_pct, 3),
        "missing_stations": missing_stations,
        "alerts": alerts,
    }

    # ─────────────────────── Uploads ONLY → latest ───────────────────────
    mon_base = os.environ.get("GCS_MONITORING_PREFIX", "").rstrip("/")
    if not mon_base.endswith("/monitoring"):
        mon_base = mon_base + "/monitoring"
    base_alias = f"{mon_base}/data/health/latest"

    # core KPI + tables
    _upload_json_gs(data_health,                               f"{base_alias}/kpis.json")
    _upload_json_gs(station_health_rows.to_dict("records"),    f"{base_alias}/station_health.json")
    _upload_json_gs(cov_by_hour_rows,                          f"{base_alias}/coverage_by_hour.json")
    _upload_json_gs(alerts,                                    f"{base_alias}/alerts.json")

    # anomalies shardées + manifeste
    anomalies_prefix = f"{base_alias}/anomalies"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "params": params_header["anomaly_params"],
        "files": [],
        "counts": {
            "flat": len(an_flat),
            "duplicates": len(an_dups),
            "missing": len(an_missing),
        }
    }

    def _upload_anom(name: str, rows: List[Dict[str, object]]):
        uri = f"{anomalies_prefix}/{name}.json"
        _upload_json_gs(rows, uri)
        manifest["files"].append({"name": name, "uri": uri, "count": len(rows)})

    _upload_anom("flat", an_flat)
    _upload_anom("duplicates", an_dups)
    _upload_anom("missing", an_missing)
    _upload_json_gs(manifest, f"{anomalies_prefix}/manifest.json")

    print("[data.health] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
