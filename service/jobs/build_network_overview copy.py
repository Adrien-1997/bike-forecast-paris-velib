# service/jobs/build_network_overview.py
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
    raise RuntimeError("pyarrow requis") from e

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage requis") from e

SCHEMA_VERSION = "1.2"  # ajout day_for_today_utc + JSON safe

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

def _upload_json_gs(obj: dict, gs_uri: str):
    # JSON "safe": remplace NaN/±Inf par null
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
    print(f"[network.overview] wrote → {gs_uri} ({len(data):,} bytes)")

# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────

def _detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    lower = {c.lower(): c for c in df.columns}
    def any_of(*cands):
        for c in cands:
            if c in lower:
                return lower[c]
        return None
    ts = any_of("ts","tbin_utc","timestamp","datetime")
    st = any_of("station_id","stationcode","id","station")
    bikes = any_of("bikes","num_bikes_available","velos","velos_disponibles")
    cap = any_of("capacity","num_docks_total","dock_count","cap")
    docks = any_of("docks_avail","num_docks_available","places_disponibles","free_docks")
    name = any_of("name","station_name","nom")
    lat = any_of("lat","latitude")
    lon = any_of("lon","lng","longitude")
    if not ts or not st or not bikes:
        raise KeyError(f"[overview] Colonnes minimales absentes (ts={ts}, station={st}, bikes={bikes})")
    return dict(ts=ts, station=st, bikes=bikes, capacity=cap, docks=docks, name=name, lat=lat, lon=lon)

def _to_local(s_utc_like: pd.Series, tzname: str) -> pd.Series:
    s = pd.to_datetime(s_utc_like, utc=True, errors="coerce")
    return s.dt.tz_convert(tzname)

def _today_bounds_local(series_local: pd.Series) -> Tuple[pd.Timestamp, pd.Timestamp]:
    tmax = series_local.max()
    start = tmax.normalize()
    end = start + pd.Timedelta(days=1)
    return start, end

def _safe_ratio(n: float, d: float) -> float:
    return float("nan") if d == 0 else round(100.0 * n / d, 2)

def _part_bool(x: pd.Series) -> float:
    if x.size == 0:
        return float("nan")
    return float((x.mean() * 100.0).round(2))

def _safe_num(v: object) -> Optional[float]:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except Exception:
        return None

def _compute_snapshot(df: pd.DataFrame, cols: Dict[str, Optional[str]], tzname: str, station_universe: List[str]) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    last_ts = pd.to_datetime(df[cols["ts"]], utc=True, errors="coerce").max()
    snap = df[df[cols["ts"]] == last_ts].copy()

    # dérive docks_avail si absent
    docks_col = cols["docks"]
    if docks_col is None and cols["capacity"]:
        docks = (pd.to_numeric(snap[cols["capacity"]], errors="coerce") - pd.to_numeric(snap[cols["bikes"]], errors="coerce")).clip(lower=0)
        snap["__docks_avail"] = docks
        docks_col = "__docks_avail"

    bikes = pd.to_numeric(snap[cols["bikes"]], errors="coerce")
    has_bike = bikes > 0
    has_dock = None
    sat = pen = None
    if docks_col and docks_col in snap.columns:
        docks = pd.to_numeric(snap[docks_col], errors="coerce")
        has_dock = docks > 0
        sat = docks == 0
    pen = bikes == 0

    active = snap[cols["station"]].astype(str).nunique()
    universe = len(station_universe)
    offline = max(universe - active, 0)

    kpis = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "snapshot_ts_utc": pd.Timestamp(last_ts).tz_convert("UTC").isoformat().replace("+00:00","Z"),
        "snapshot_ts_local": str(_to_local(pd.Series([last_ts]), tzname).iloc[0]),
        "stations_universe": universe,
        "stations_active": int(active),
        "stations_offline": int(offline),
        "availability_bike_pct": _part_bool(has_bike) if has_bike is not None else float("nan"),
        "availability_dock_pct": _part_bool(has_dock) if has_dock is not None else float("nan"),
        "penury_pct": _part_bool(pen) if pen is not None else float("nan"),
        "saturation_pct": _part_bool(sat) if sat is not None else float("nan"),
    }

    # Distribution instantanée
    dist = pd.DataFrame({
        "metric": ["bike_avail","dock_avail","penury","saturation"],
        "count": [
            int(has_bike.sum()) if has_bike is not None else 0,
            int(has_dock.sum()) if has_dock is not None else 0,
            int(pen.sum()) if pen is not None else 0,
            int(sat.sum()) if sat is not None else 0,
        ],
    })
    dist["total_active"] = active
    dist["pct"] = dist.apply(lambda r: _safe_ratio(r["count"], r["total_active"]), axis=1)

    # Index carte snapshot
    map_rows = []
    latc = cols["lat"]; lonc = cols["lon"]; namec = cols["name"]
    for _, row in snap.iterrows():
        sid = str(row[cols["station"]])
        lat = float(row[latc]) if latc and pd.notna(row[latc]) else None
        lon = float(row[lonc]) if lonc and pd.notna(row[lonc]) else None
        name = str(row[namec]) if namec and pd.notna(row[namec]) else sid
        b = int(pd.to_numeric(row[cols["bikes"]], errors="coerce")) if pd.notna(row[cols["bikes"]]) else None
        d = None
        if docks_col and docks_col in snap.columns:
            d = int(pd.to_numeric(row[docks_col], errors="coerce")) if pd.notna(row[docks_col]) else None
        map_rows.append({
            "station_id": sid, "name": name, "lat": lat, "lon": lon,
            "bikes": b, "docks_avail": d,
            "is_penury": (1 if (b == 0 if b is not None else False) else 0),
            "is_saturation": (1 if (d == 0 if d is not None else False) else 0),
        })
    map_df = pd.DataFrame(map_rows)

    return kpis, dist, map_df

def _coverage_volatility(df: pd.DataFrame, cols: Dict[str, Optional[str]], tzname: str, last_days: int) -> Tuple[float,float]:
    if last_days <= 0:
        return float("nan"), float("nan")
    ts_local = _to_local(df[cols["ts"]], tzname)
    tmax = ts_local.max()
    start = tmax - pd.Timedelta(days=last_days)
    win = df.loc[(ts_local >= start) & (ts_local <= tmax)].copy()
    if win.empty:
        return float("nan"), float("nan")
    total_ts = win[cols["ts"]].nunique()
    per_station = (win.groupby(cols["station"])[cols["ts"]].nunique() / max(total_ts,1)).reindex(win[cols["station"]].unique()).fillna(0.0)
    coverage_pct = float((per_station.mean() * 100.0).round(2))

    start_day, end_day = _today_bounds_local(ts_local)
    today = df[(ts_local >= start_day) & (ts_local < end_day)].copy()
    vol = (today.groupby(cols["station"])[cols["bikes"]].std(ddof=0)).median()
    return coverage_pct, float(0.0 if pd.isna(vol) else round(vol, 2))

def _today_vs_median_utc(df: pd.DataFrame, cols: Dict[str, Optional[str]], tzname: str, ref_days: int, day_for_today_utc: datetime) -> Tuple[dict, dict]:
    """
    Construit:
      - today_curve: % stations avec ≥1 vélo par 5 min DU JOUR UTC CHOISI (day_for_today_utc)
      - ref_median_curve: médiane sur les 'ref_days' jours précédents, même weekday, en 5 min (UTC)
    """
    df = df.copy()
    ts_utc = pd.to_datetime(df[cols["ts"]], utc=True, errors="coerce")
    df["_ts_utc"] = ts_utc
    df["_hhmm"] = df["_ts_utc"].dt.strftime("%H:%M")
    df["_weekday"] = df["_ts_utc"].dt.weekday

    bikes_all = pd.to_numeric(df[cols["bikes"]], errors="coerce")
    df["__has_bike"] = bikes_all > 0

    start_utc = pd.Timestamp(day_for_today_utc)
    end_utc = start_utc + pd.Timedelta(days=1)

    today = df[(ts_utc >= start_utc) & (ts_utc < end_utc)].copy()

    ref_start = start_utc - pd.Timedelta(days=ref_days)
    ref = df[(ts_utc >= ref_start) & (ts_utc < start_utc) & (df["_weekday"] == start_utc.weekday())].copy()

    def agg_part(d: pd.DataFrame) -> pd.DataFrame:
        if d.empty:
            return pd.DataFrame(columns=["_hhmm","pct"])
        out = (d.groupby(["_ts_utc","_hhmm"])[["__has_bike"]]
                 .agg(has_bike=("__has_bike","mean"), n=("__has_bike","size"))
                 .reset_index())
        out["pct"] = out["has_bike"] * 100.0
        return out

    cur = agg_part(today)
    ref_curve = agg_part(ref)

    # médiane par hh:mm sur la référence (UTC)
    bins = pd.Index(pd.date_range("00:00","23:55",freq="5min").strftime("%H:%M"))
    ref_med = ref_curve.groupby("_hhmm")["pct"].median().reindex(bins)

    today_doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "day_for_today_utc": start_utc.strftime("%Y-%m-%d"),
        "points": [{"hhmm": str(h), "pct": _safe_num(v)} for h, v in (cur[["_hhmm","pct"]].itertuples(index=False, name=None) if not cur.empty else [])]
    }
    ref_doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "day_for_today_utc": start_utc.strftime("%Y-%m-%d"),
        "median": [{"hhmm": str(h), "pct_median": _safe_num(v)} for h, v in ref_med.items()]
    }
    return today_doc, ref_doc

def _kpis_today_vs_lags_utc(df: pd.DataFrame, cols: Dict[str, Optional[str]], day_for_today_utc: datetime) -> dict:
    ts_utc_all = pd.to_datetime(df[cols["ts"]], utc=True, errors="coerce")

    def day_kpis(day_start_utc: pd.Timestamp) -> dict:
        day_end = day_start_utc + pd.Timedelta(days=1)
        d = df[(ts_utc_all >= day_start_utc) & (ts_utc_all < day_end)].copy()
        if d.empty:
            return {"avail_bike": np.nan, "avail_dock": np.nan, "pen": np.nan, "sat": np.nan}

        b = pd.to_numeric(d[cols["bikes"]], errors="coerce")
        has_bike = (b > 0)

        docks = None
        if cols["docks"] and cols["docks"] in d.columns:
            docks = pd.to_numeric(d[cols["docks"]], errors="coerce")
        elif cols["capacity"] and cols["capacity"] in d.columns:
            docks = (pd.to_numeric(d[cols["capacity"]], errors="coerce") - b).clip(lower=0)

        has_dock = (docks > 0) if docks is not None else None
        pen = (b == 0)
        sat = (docks == 0) if docks is not None else None

        return {
            "avail_bike": float((has_bike.mean() * 100.0).round(2)),
            "avail_dock": float((has_dock.mean() * 100.0).round(2)) if has_dock is not None else np.nan,
            "pen":        float((pen.mean() * 100.0).round(2)),
            "sat":        float((sat.mean() * 100.0).round(2)) if sat is not None else np.nan,
        }

    start_today = pd.Timestamp(day_for_today_utc)
    k_today = day_kpis(start_today)
    k_lags = {
        "J-7":  day_kpis(start_today - pd.Timedelta(days=7)),
        "J-14": day_kpis(start_today - pd.Timedelta(days=14)),
        "J-21": day_kpis(start_today - pd.Timedelta(days=21)),
    }

    def pack(d: dict) -> dict:
        return {k: (_safe_num(v) if v is not None else None) for k, v in d.items()}

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "day_for_today_utc": start_today.strftime("%Y-%m-%d"),
        "today": pack(k_today),
        "lags": {k: pack(v) for k, v in k_lags.items()},
    }

def _stations_tension(df: pd.DataFrame, cols: Dict[str, Optional[str]], tzname: str, last_days: int) -> dict:
    ts_loc = _to_local(df[cols["ts"]], tzname)
    tmax = ts_loc.max()
    start_ld = tmax - pd.Timedelta(days=last_days)
    win = df[(ts_loc >= start_ld) & (ts_loc <= tmax)].copy()
    if win.empty:
        return {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"), "rows": []}

    pen_rate = win.groupby(cols["station"])[cols["bikes"]].apply(lambda s: (pd.to_numeric(s, errors="coerce") == 0).mean())
    # saturation
    if cols["docks"] and cols["docks"] in win.columns:
        sat_rate = win.groupby(cols["station"])[cols["docks"]].apply(lambda s: (pd.to_numeric(s, errors="coerce") == 0).mean())
    elif cols["capacity"] and cols["capacity"] in win.columns:
        cap = pd.to_numeric(win[cols["capacity"]], errors="coerce")
        bks = pd.to_numeric(win[cols["bikes"]], errors="coerce")
        sat_rate = ((cap - bks) <= 0).groupby(win[cols["station"]]).mean()
    else:
        sat_rate = pd.Series(np.nan, index=pen_rate.index)

    out = (pd.DataFrame({
        "station_id": pen_rate.index.astype(str),
        "penury_rate": pen_rate.values,
        "saturation_rate": sat_rate.values,
    }).sort_values(["penury_rate","saturation_rate"], ascending=False))
    rows = out.to_dict(orient="records")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "rows": rows
    }

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    EXPORTS_PREFIX = os.environ.get("GCS_EXPORTS_PREFIX")    # gs://bucket/velib/exports
    MON_PREFIX     = os.environ.get("GCS_MONITORING_PREFIX") # gs://bucket/monitoring  (ou gs://bucket/velib)
    if not (EXPORTS_PREFIX and EXPORTS_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_EXPORTS_PREFIX manquant ou invalide")
    if not (MON_PREFIX and MON_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX manquant ou invalide")

    TZNAME       = os.environ.get("OVERVIEW_TZ", "Europe/Paris")
    LAST_DAYS    = int(os.environ.get("OVERVIEW_LAST_DAYS", "7"))
    REF_DAYS     = int(os.environ.get("OVERVIEW_REF_DAYS", "28"))

    now = datetime.now(timezone.utc)
    # Fenêtre de lecture : max(LAST_DAYS, REF_DAYS, 7) pour tout calcul
    WINDOW_DAYS  = max(LAST_DAYS, REF_DAYS, 7)
    start = (now - timedelta(days=WINDOW_DAYS - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"[network.overview] window UTC: {start.date()} → {now.date()} (days={WINDOW_DAYS})")

    # 1) Lire les parquets évènementiels dans la fenêtre
    blobs = _list_event_blobs(EXPORTS_PREFIX, start, now)
    if not blobs:
        print("[network.overview] no event blobs in window — nothing to do")
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
        print("[network.overview] no readable data — nothing to do")
        return 0

    ev = pd.concat(frames, ignore_index=True)
    if ev.empty:
        print("[network.overview] events empty — nothing to do")
        return 0

    # 2) Colonnes & typage
    cols = _detect_columns(ev)
    for c in [cols["ts"], cols["station"]]:
        if c is None: raise RuntimeError("colonnes minimales manquantes")

    # TZ-AWARE UTC parsing (fix tz-naive vs tz-aware mix)
    ev[cols["ts"]] = pd.to_datetime(ev[cols["ts"]], utc=True, errors="coerce")
    ev[cols["station"]] = ev[cols["station"]].astype("string")

    # numeric columns
    for c in [cols["bikes"], cols["capacity"], cols["docks"], cols["lat"], cols["lon"]]:
        if c and c in ev.columns:
            ev[c] = pd.to_numeric(ev[c], errors="coerce")
    if cols["name"] and cols["name"] in ev.columns:
        ev[cols["name"]] = ev[cols["name"]].astype("string")

    ev = ev.dropna(subset=[cols["ts"], cols["station"]]).copy()

    # fenêtre stricte (UTC aware)
    ev = ev[(ev[cols["ts"]] >= pd.Timestamp(start)) &
            (ev[cols["ts"]] <= pd.Timestamp(now))].copy()

    # 3) Univers de stations (sur la fenêtre)
    ts_loc_all = _to_local(ev[cols["ts"]], TZNAME)
    tmax = ts_loc_all.max()
    uni_start = tmax - pd.Pandas.Timedelta(days=WINDOW_DAYS) if hasattr(pd, "Pandas") else tmax - pd.Timedelta(days=WINDOW_DAYS)
    station_universe = (
        ev.loc[(ts_loc_all >= uni_start) & (ts_loc_all <= tmax), cols["station"]]
        .astype(str).dropna().unique().tolist()
    )

    # 4) Calculs clés
    # Jour opérationnel basé sur le DERNIER fichier events_* disponible
    m = re.search(r"events_(\d{4}-\d{2}-\d{2})\.parquet$", blobs[-1].name)
    if not m:
        raise RuntimeError("Impossible de déterminer la date du dernier fichier events")
    day_for_today_utc = datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    print(f"[network.overview] day_for_today_utc auto-detected → {day_for_today_utc.date()} (from last events_*.parquet)")

    kpis, dist, map_df = _compute_snapshot(ev, cols, TZNAME, station_universe)
    cov, vol = _coverage_volatility(ev, cols, TZNAME, LAST_DAYS)
    kpis["coverage_pct"] = cov
    kpis["volatility_today"] = vol
    kpis["last_days"] = LAST_DAYS
    kpis["ref_days"] = REF_DAYS

    # Courbes & comparatifs en UTC (jour choisi)
    today_curve_doc, ref_median_doc = _today_vs_median_utc(ev, cols, TZNAME, REF_DAYS, day_for_today_utc)
    kpi_bars_doc = _kpis_today_vs_lags_utc(ev, cols, day_for_today_utc)
    tension_doc = _stations_tension(ev, cols, TZNAME, LAST_DAYS)

    # 5) Uploads (latest + versionnés) — avec normalisation du préfixe
    mon_base = MON_PREFIX.rstrip("/")
    if not mon_base.endswith("/monitoring"):
        mon_base = mon_base + "/monitoring"

    base_alias = f"{mon_base}/network/overview/latest"
    base_ver   = f"{mon_base}/network/overview/{now.strftime('%Y-%m-%dT%H-%M-%SZ')}"

    # KPIs & distributions
    _upload_json_gs(kpis,                                   f"{base_alias}/kpis.json")
    _upload_json_gs(dist.to_dict(orient="records"),         f"{base_alias}/snapshot_distribution.json")

    # Courbes / docs
    _upload_json_gs(today_curve_doc,                        f"{base_alias}/today_curve.json")
    _upload_json_gs(ref_median_doc,                         f"{base_alias}/ref_median_curve.json")
    _upload_json_gs(kpi_bars_doc,                           f"{base_alias}/kpis_today_vs_lags.json")

    # Carte snapshot (rows)
    _upload_json_gs({
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "rows": map_df.to_dict(orient="records"),
    }, f"{base_alias}/snapshot_map.json")

    # Stations en tension
    _upload_json_gs(tension_doc,                            f"{base_alias}/stations_tension.json")

    # Versionnés
    _upload_json_gs(kpis,                                   f"{base_ver}/kpis.json")
    _upload_json_gs(dist.to_dict(orient="records"),         f"{base_ver}/snapshot_distribution.json")
    _upload_json_gs(today_curve_doc,                        f"{base_ver}/today_curve.json")
    _upload_json_gs(ref_median_doc,                         f"{base_ver}/ref_median_curve.json")
    _upload_json_gs(kpi_bars_doc,                           f"{base_ver}/kpis_today_vs_lags.json")
    _upload_json_gs({
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "rows": map_df.to_dict(orient="records"),
    }, f"{base_ver}/snapshot_map.json")
    _upload_json_gs(tension_doc,                            f"{base_ver}/stations_tension.json")

    print("[network.overview] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())