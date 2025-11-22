# service/jobs/build_network_dynamics.py

"""
Vélib’ Forecast — Network Dynamics monitoring job (LATEST ONLY).

Role
----
This job reads the *event* parquet files:

    gs://.../velib/exports/events_YYYY-MM-DD.parquet

over a sliding window of `MON_LAST_DAYS` days (in UTC), and builds several
JSON artifacts used by the **Network Dynamics** page of the Monitoring UI:

- heatmaps & profiles (dow × hour) for:
    * mean occupancy
    * penury rate
    * saturation rate
- hourly penury/saturation rates (0–23h)
- penury / saturation episodes by station (≥ ~1 hour)
- aggregates "by zone" (lat/lon coarse grid)
- tension index by station (penury_rate + saturation_rate)
- regularity_today: correlation between today's occupancy curve and
  the "typical" curve for the same weekday over the last ~90 days.

Outputs (LATEST only)
---------------------
All artifacts are published under:

    <GCS_MONITORING_PREFIX>/monitoring/network/dynamics/latest/

with a `manifest.json` describing the window, thresholds, and produced files.

Environment
-----------
Required:
    GCS_EXPORTS_PREFIX    = gs://bucket/velib/exports
    GCS_MONITORING_PREFIX = gs://bucket/velib  (or .../monitoring)

Optional (only MON_*):
    MON_TZ                = Europe/Paris  (timezone for local calendars)
    MON_LAST_DAYS         = 7            (window length, in days)
    MON_PENURY_THRESH     = 2            (bikes <= T → penury candidate)
    MON_SATURATION_THRESH = 2            (capacity - bikes <= T → saturation candidate)

Notes
-----
- All outputs are *LATEST ONLY* (no dated folders); the UI always reads
  from `.../network/dynamics/latest/...`.
- JSON is sanitized: NaN/±Inf converted to null.
- Penury / saturation flags are made **exclusive** at row level:
  a row cannot be both penury and saturation.
"""

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

SCHEMA_VERSION = "1.3"  # ENV unifié MON_*, fenêtre stricte, manifest, LATEST only

# ──────────────────────────────────────────────────────────────────────────────
# ENV — unifié (pas de fallbacks exotiques)
# ──────────────────────────────────────────────────────────────────────────────

def _env(name: str, default=None):
    """
    Read an environment variable with a default fallback.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : Any
        Fallback value if the variable is missing or empty.

    Returns
    -------
    Any
        Raw string value from the environment, or the default.
    """
    v = os.environ.get(name)
    return v if (v is not None and v != "") else default

def _env_int(name: str, default: int) -> int:
    """
    Read an integer-valued environment variable.

    If parsing fails, the provided default is returned.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : int
        Fallback integer value.

    Returns
    -------
    int
        Parsed integer or default.
    """
    try:
        return int(_env(name, default))
    except Exception:
        return default

# Requis
GCS_EXPORTS_PREFIX    = _env("GCS_EXPORTS_PREFIX")    # gs://bucket/velib/exports
GCS_MONITORING_PREFIX = _env("GCS_MONITORING_PREFIX") # gs://bucket/velib   (ou .../monitoring)

# Optionnels (seulement MON_*)
MON_TZ                = _env("MON_TZ", "Europe/Paris")
MON_LAST_DAYS         = _env_int("MON_LAST_DAYS", 7)
MON_PENURY_THRESH     = _env_int("MON_PENURY_THRESH", 2)
MON_SATURATION_THRESH = _env_int("MON_SATURATION_THRESH", 2)

if not (GCS_EXPORTS_PREFIX and str(GCS_EXPORTS_PREFIX).startswith("gs://")):
    raise RuntimeError("GCS_EXPORTS_PREFIX manquant ou invalide (attendu gs://...)")
if not (GCS_MONITORING_PREFIX and str(GCS_MONITORING_PREFIX).startswith("gs://")):
    raise RuntimeError("GCS_MONITORING_PREFIX manquant ou invalide (attendu gs://...)")

# ──────────────────────────────────────────────────────────────────────────────
# GCS helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split(gs: str) -> Tuple[str, str]:
    """
    Split a GCS URI `gs://bucket/path` into (bucket, key).

    Parameters
    ----------
    gs : str
        GCS URI.

    Returns
    -------
    (str, str)
        Bucket name and object key (without trailing slash).

    Raises
    ------
    AssertionError
        If the URI does not start with `gs://`.
    """
    assert gs.startswith("gs://"), f"bad GCS uri: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k.rstrip("/")

def _read_parquet_blob_to_df(blob: "storage.Blob") -> pd.DataFrame:
    """
    Read a Parquet blob from GCS into a pandas DataFrame.

    Parameters
    ----------
    blob : google.cloud.storage.Blob
        GCS blob pointing to a parquet file.

    Returns
    -------
    pandas.DataFrame
        DataFrame loaded from the parquet content.
    """
    buf = BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    tbl = pq.read_table(buf)
    return tbl.to_pandas()

def _list_event_blobs(exports_prefix: str, start_date: datetime, end_date: datetime) -> List["storage.Blob"]:
    """
    List `events_YYYY-MM-DD.parquet` blobs in a given UTC date window.

    The filter is done on the date embedded in the file name.

    Parameters
    ----------
    exports_prefix : str
        Base GCS prefix for exports (GCS_EXPORTS_PREFIX).
    start_date : datetime
        Start of the window (inclusive, UTC).
    end_date : datetime
        End of the window (inclusive, UTC).

    Returns
    -------
    list[google.cloud.storage.Blob]
        Sorted list of blobs matching `events_YYYY-MM-DD.parquet`
        within the requested date range.
    """
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

def _upload_json_gs(obj: dict | list, gs_uri: str, log_prefix: str = "network.dynamics"):
    """JSON safe: remplace NaN/±Inf par null."""
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
    print(f"[{log_prefix}] wrote → {gs_uri} ({len(data):,} bytes)")

# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────

def _detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Auto-detect core columns in the events dataframe.

    The function is tolerant to different naming conventions and returns a
    mapping with the following keys (values may be None if not found):

        ts, station, bikes, docks, capacity, name, lat, lon, is_pen, is_sat

    Minimal required columns are: timestamp, station_id, bikes.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw events dataframe.

    Returns
    -------
    dict
        Mapping from logical column names to actual DataFrame column names.

    Raises
    ------
    KeyError
        If minimal columns are not present.
    """
    lower = {c.lower(): c for c in df.columns}
    def any_of(*cands):
        for c in cands:
            if c in lower:
                return lower[c]
        return None
    ts = any_of("ts","tbin_utc","timestamp","datetime")
    st = any_of("station_id","stationcode","id","station")
    bikes = any_of("bikes","num_bikes_available","velos","velos_disponibles")
    docks = any_of("docks_avail","num_docks_available","places_disponibles","free_docks")
    cap = any_of("capacity","num_docks_total","dock_count","cap","capacity_src","capacity_est")
    name = any_of("name","station_name","nom")
    lat  = any_of("lat","latitude")
    lon  = any_of("lon","lng","longitude")
    is_pen = any_of("is_penury","penury","pen")
    is_sat = any_of("is_saturation","saturation","sat")
    if not ts or not st or not bikes:
        raise KeyError(f"[dynamics] Colonnes minimales absentes (ts={ts}, station={st}, bikes={bikes})")
    return dict(ts=ts, station=st, bikes=bikes, docks=docks, capacity=cap,
                name=name, lat=lat, lon=lon, is_pen=is_pen, is_sat=is_sat)

def _to_local(s_utc_like: pd.Series, tzname: str) -> pd.Series:
    """
    Convert a timestamp-like Series to a localized datetime (tz-aware).

    Parameters
    ----------
    s_utc_like : pandas.Series
        Series convertible to datetime (assumed UTC).
    tzname : str
        Target timezone name (e.g. "Europe/Paris").

    Returns
    -------
    pandas.Series
        Timezone-aware series converted to the target timezone.
    """
    s = pd.to_datetime(s_utc_like, utc=True, errors="coerce")
    return s.dt.tz_convert(tzname)

def _safe_num(v: object) -> Optional[float]:
    """
    Safely cast a value to float, returning None for NaN/Inf or errors.

    Parameters
    ----------
    v : Any
        Input value.

    Returns
    -------
    float | None
        Finite float or None.
    """
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Occupation + masques EXCLUSIFS
# ──────────────────────────────────────────────────────────────────────────────

def _estimate_capacity_window(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.Series:
    """Capacité estimée par station : priorité capacité fournie ; sinon q98(bikes+docks) ; sinon q98(bikes)."""
    sid = cols["station"]; bikes = cols["bikes"]; docks = cols["docks"]; cap = cols["capacity"]

    def est(g: pd.DataFrame) -> float:
        if cap and cap in g.columns:
            capv = pd.to_numeric(g[cap], errors="coerce").max()
            if pd.notna(capv) and capv > 0:
                return float(capv)
        b = pd.to_numeric(g[bikes], errors="coerce").clip(lower=0)
        if docks and docks in g.columns:
            d = pd.to_numeric(g[docks], errors="coerce").clip(lower=0)
            s = (b + d).dropna()
            if len(s):
                return float(s.quantile(0.98))
        if len(b):
            return float(b.quantile(0.98))
        return float("nan")

    cols_for_group = [c for c in [cap, docks, bikes] if c and c in df.columns]
    if not cols_for_group:
        return df.groupby(cols["station"]).size().rename("cap_est").astype(float) * float("nan")
    return df.groupby(sid)[cols_for_group].apply(est).rename("cap_est")

def _compute_occ_and_masks(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    """
    Ajoute : occ (0..1 ou NaN), valid (bool), is_pen, is_sat (EXCLUSIFS) pour CHAQUE LIGNE.

    Important : on utilise toujours df['cap_est'] alignée ligne à ligne (merge par station).

    Règles
    ------
    - cap_est estimée par station si absente (voir `_estimate_capacity_window`)
    - occ = bikes / cap_est, borné à [0,1], NaN si cap_est non valide
    - valid = isfinite(occ)
    - penury brute :
        * bikes <= MON_PENURY_THRESH
    - saturation brute :
        * si docks dispo : docks == 0
        * sinon : capacity - bikes <= MON_SATURATION_THRESH
    - is_sat = saturation_brute & valid
    - is_pen = penury_brute & valid & ~is_sat   (exclusivité)
    """
    bikes = cols["bikes"]; docks = cols["docks"]; sid = cols["station"]

    # seuils (uniquement MON_*)
    _PEN_T = float(MON_PENURY_THRESH)
    _SAT_T = float(MON_SATURATION_THRESH)

    # 1) s'assurer que df possède cap_est (colonne alignée ligne à ligne)
    if "cap_est" not in df.columns:
        cap_per_station = _estimate_capacity_window(df, cols)
        df = df.merge(cap_per_station.to_frame(), left_on=sid, right_index=True, how="left")

    # 2) occ et validité
    b = pd.to_numeric(df[bikes], errors="coerce").clip(lower=0)
    c = pd.to_numeric(df["cap_est"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        occ = b / c
    occ = np.where((~np.isnan(c)) & (c > 0), np.clip(occ, 0.0, 1.0), np.nan)
    valid = np.isfinite(occ)

    # 3) règles brutes
    pen_raw = (b <= _PEN_T)
    if docks and docks in df.columns:
        d = pd.to_numeric(df[docks], errors="coerce")
        sat_raw = (d == 0)
    else:
        sat_raw = (pd.notna(c) & pd.notna(b) & ((c - b) <= _SAT_T))

    # 4) exclusivité + validité
    is_sat = (sat_raw & valid)
    is_pen = (pen_raw & valid & ~is_sat)

    out = df.copy()
    out["occ"] = occ
    out["valid"] = valid
    out["is_pen"] = is_pen
    out["is_sat"] = is_sat
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Dynamics computations (JSON-oriented)
# ──────────────────────────────────────────────────────────────────────────────

def _heatmap_and_profiles(ev: pd.DataFrame, cols: Dict[str, Optional[str]], tzname: str) -> dict:
    """
    Build dow×hour heatmaps and daily profiles of occupancy.

    Heatmaps:
      - occ_mean[dow][hour]
      - penury_rate[dow][hour]
      - saturation_rate[dow][hour]

    Profiles:
      - for each dow, the list of 24 occ_mean values (0..1 or None).
    """
    ts = cols["ts"]
    ldt = _to_local(ev[ts], tzname)
    df = ev.assign(dow=ldt.dt.dayofweek, hour=ldt.dt.hour).copy()
    df = _compute_occ_and_masks(df, cols)

    agg = (df.groupby(["dow","hour"])
             .agg(occ_mean=("occ","mean"),
                  penury_rate=("is_pen","mean"),
                  saturation_rate=("is_sat","mean"),
                  n_obs=("valid","sum"))
             .reset_index())

    def mat(col):
        piv = agg.pivot(index="dow", columns="hour", values=col).reindex(index=range(7), columns=range(24))
        return [[ _safe_num(v) for v in row ] for row in piv.to_numpy(dtype=float)]

    prof = agg.pivot(index="dow", columns="hour", values="occ_mean").reindex(index=range(7), columns=range(24))
    profiles = {str(d): [ _safe_num(x) for x in prof.loc[d].to_numpy(dtype=float) ] for d in range(7)}

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "heatmap": {
            "occ_mean": mat("occ_mean"),
            "penury_rate": mat("penury_rate"),
            "saturation_rate": mat("saturation_rate"),
        },
        "profiles_occ_by_dow": profiles,
    }

def _hourly_pen_sat(ev: pd.DataFrame, cols: Dict[str, Optional[str]], tzname: str) -> dict:
    """
    Compute hourly penury/saturation rates aggregated over the full window.

    Output: one row per hour 0..23 with mean penury_rate and saturation_rate.
    """
    ts = cols["ts"]
    ldt = _to_local(ev[ts], tzname)
    df = ev.assign(hour=ldt.dt.hour).copy()
    df = _compute_occ_and_masks(df, cols)

    agg = (df.groupby("hour")
             .agg(penury_rate=("is_pen","mean"),
                  saturation_rate=("is_sat","mean"))
             .reindex(range(24)))

    rows = [{"hour": int(h),
             "penury_rate": _safe_num(agg.loc[h,"penury_rate"]),
             "saturation_rate": _safe_num(agg.loc[h,"saturation_rate"])}
            for h in range(24)]
    return {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"), "rows": rows}

def _episodes(ev: pd.DataFrame, cols: Dict[str, Optional[str]], tzname: str, last_days: int) -> dict:
    """
    Detect penury / saturation episodes per station over the last N days.

    Definition
    ----------
    - The window is restricted to the last `last_days` (local time).
    - Sampling step (minutes) is estimated from median inter-timestamp diff.
    - An episode is a consecutive run of True flags (penury or saturation)
      of at least ~1 hour (computed as 60 / step_min rounded).

    Result
    ------
    A list of episodes with:
      station_id, type ("penury"/"saturation"), start_utc, end_utc,
      steps, duration_min.
    """
    ts = cols["ts"]; sid = cols["station"]
    if last_days <= 0 or ev.empty:
        return {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"), "rows": []}

    ts_loc = _to_local(ev[ts], tzname)
    tmax = ts_loc.max()
    tmin = tmax - pd.Timedelta(days=last_days)
    w = ev[(ts_loc >= tmin) & (ts_loc <= tmax)].copy()
    if w.empty:
        return {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"), "rows": []}

    w = w.copy()
    w["_ts"] = pd.to_datetime(w[ts], utc=True, errors="coerce")
    w = _compute_occ_and_masks(w, cols)  # exclusivité

    diffs = (w.sort_values([sid,"_ts"]).groupby(sid)["_ts"].diff().dropna().dt.total_seconds()/60.0)
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    step_min = float(np.median(diffs)) if len(diffs) else 15.0
    min_steps = max(1, int(round(60.0/step_min)))  # ~1h par défaut

    rows: List[dict] = []
    for st, sub in w.sort_values([sid,"_ts"]).groupby(sid):
        def runs(mask: pd.Series) -> List[Tuple[pd.Timestamp,pd.Timestamp,int]]:
            start = None; last_t = None; k = 0; out = []
            for i_, flag in zip(sub.index, mask):
                if bool(flag):
                    if start is None:
                        start = sub["_ts"].loc[i_]; k = 1
                    else:
                        k += 1
                    last_t = sub["_ts"].loc[i_]
                else:
                    if start is not None:
                        out.append((start, last_t, k))
                        start = None; last_t = None; k = 0
            if start is not None:
                out.append((start, last_t, k))
            return [r for r in out if r[2] >= min_steps]

        for a,b_,k in runs(sub["is_pen"]):
            rows.append({"station_id": str(st), "type": "penury", "start_utc": a.isoformat().replace("+00:00","Z"),
                         "end_utc": b_.isoformat().replace("+00:00","Z"), "steps": int(k), "duration_min": _safe_num(k*step_min)})
        for a,b_,k in runs(sub["is_sat"]):
            rows.append({"station_id": str(st), "type": "saturation", "start_utc": a.isoformat().replace("+00:00","Z"),
                         "end_utc": b_.isoformat().replace("+00:00","Z"), "steps": int(k), "duration_min": _safe_num(k*step_min)})

    return {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "last_days": last_days, "rows": sorted(rows, key=lambda r: (r["station_id"], r["start_utc"]))}

def _by_zone(ev: pd.DataFrame, cols: Dict[str, Optional[str]], tzname: str, last_days: int) -> dict:
    """
    Aggregate dynamics metrics by coarse spatial zones (lat/lon grid).

    - Window restricted to last `last_days` (local time).
    - Each zone is defined by rounding (lat, lon) to 3 decimals and
      concatenating as "<lat>|<lon>".
    - For each zone:
        * occ_mean
        * penury_rate
        * saturation_rate
        * cap_sum   (sum of station capacities in the zone)
        * n_obs     (number of valid occupancy observations)
    """
    ts = cols["ts"]; sid = cols["station"]; latc = cols["lat"]; lonc = cols["lon"]
    ldt = _to_local(ev[ts], tzname)
    tmax = ldt.max(); tmin = tmax - pd.Timedelta(days=last_days)
    win = ev[(ldt >= tmin) & (ldt <= tmax)].copy()
    if win.empty:
        return {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"), "rows": []}

    def zone_label(a, b):
        try:
            a = float(a); b = float(b)
            return f"{round(a,3)}|{round(b,3)}"
        except Exception:
            return "NA"

    win["_zone"] = [zone_label(a,b) for a,b in zip(win.get(latc, np.nan), win.get(lonc, np.nan))]
    win = _compute_occ_and_masks(win, cols)

    # Moyennes temporelles (sur pas valides)
    agg_time = (win.groupby("_zone", dropna=False).agg(
        occ_mean=("occ","mean"),
        penury_rate=("is_pen","mean"),
        saturation_rate=("is_sat","mean"),
        n_obs=("valid","sum"),
    ).reset_index())

    # Capacité Σ par zone = somme des capacités uniques PAR STATION dans la zone
    cap_by_station = (win.dropna(subset=["cap_est"])
                        .groupby([sid, "_zone"], dropna=False)["cap_est"].last()
                        .reset_index())
    cap_zone = cap_by_station.groupby("_zone", dropna=False)["cap_est"].sum().rename("cap_sum").reset_index()

    agg = agg_time.merge(cap_zone, on="_zone", how="left")

    rows = []
    for _, r in agg.iterrows():
        rows.append({
            "zone": str(r["_zone"]),
            "occ_mean": _safe_num(r["occ_mean"]),
            "penury_rate": _safe_num(r["penury_rate"]),
            "saturation_rate": _safe_num(r["saturation_rate"]),
            "cap_sum": _safe_num(r["cap_sum"]),
            "n_obs": int(r["n_obs"] if np.isfinite(r["n_obs"]) else 0)
        })
    rows.sort(key=lambda x: (x["occ_mean"] if x["occ_mean"] is not None else 1e9))
    return {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "last_days": last_days, "rows": rows}

def _tension_by_station(ev: pd.DataFrame, cols: Dict[str, Optional[str]], tzname: str, last_days: int) -> dict:
    """
    Compute a tension index per station over the last `last_days`.

    For each station, we aggregate:
      - penury_rate
      - saturation_rate
      - occ_mean
      - name, lat, lon
      - n_obs (valid occupancy observations)
      - tension_index = penury_rate + saturation_rate

    The resulting rows are sorted by decreasing tension_index.
    """
    ts = cols["ts"]; sid = cols["station"]; namec = cols["name"]; latc = cols["lat"]; lonc = cols["lon"]
    ldt = _to_local(ev[ts], tzname)
    tmax = ldt.max(); tmin = tmax - pd.Timedelta(days=last_days)
    win = ev[(ldt >= tmin) & (ldt <= tmax)].copy()
    if win.empty:
        return {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"), "rows": []}

    win = _compute_occ_and_masks(win, cols)

    name_col = namec if namec else "name"
    df_agg = pd.DataFrame({
        sid: win[sid].astype(str),
        "pen": win["is_pen"].astype(bool),
        "sat": win["is_sat"].astype(bool),
        "occ": win["occ"],
        "valid": win["valid"].astype(bool),
        name_col: (win[namec] if namec and namec in win.columns else win[sid].astype(str)),
        "lat": (pd.to_numeric(win[latc], errors="coerce") if latc and latc in win.columns else np.nan),
        "lon": (pd.to_numeric(win[lonc], errors="coerce") if lonc and lonc in win.columns else np.nan),
    })

    agg = (df_agg.groupby(sid, dropna=False).agg(
        penury_rate=("pen","mean"),
        saturation_rate=("sat","mean"),
        occ_mean=("occ","mean"),
        name=(name_col,"last"),
        lat=("lat","last"),
        lon=("lon","last"),
        n_obs=("valid","sum")
    ).reset_index())

    rows = []
    for _, r in agg.iterrows():
        ti = ((r["penury_rate"] if pd.notna(r["penury_rate"]) else 0.0) +
              (r["saturation_rate"] if pd.notna(r["saturation_rate"]) else 0.0))
        rows.append({
            "station_id": str(r[sid]),
            "name": (None if pd.isna(r["name"]) else str(r["name"])),
            "lat": _safe_num(r["lat"]),
            "lon": _safe_num(r["lon"]),
            "penury_rate": _safe_num(r["penury_rate"]),
            "saturation_rate": _safe_num(r["saturation_rate"]),
            "occ_mean": _safe_num(r["occ_mean"]),
            "tension_index": _safe_num(ti),
            "n_obs": int(r["n_obs"] if np.isfinite(r["n_obs"]) else 0)
        })
    rows.sort(key=lambda x: (-(x["tension_index"] if x["tension_index"] is not None else -1)))
    return {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "last_days": last_days, "rows": rows}

def _regularity_today(ev: pd.DataFrame, cols: Dict[str, Optional[str]], tzname: str) -> dict:
    """
    Measure how "regular" today's occupancy curve is, per station.

    Idea
    ----
    For each station:
      1. Extract today's occupancy curve (local date of last event file).
      2. Build a "typical" occupancy curve for the same weekday:
         median occupancy by hh:mm across past days (same dow, last ~90 days).
      3. Compute the Pearson correlation between:
         - today's curve
         - the typical curve
         Only if there are ≥ 8 overlapping time points and both curves have
         non-zero standard deviation.

    Output
    ------
    A row per station:
      - station_id
      - regularity_corr_today_vs_typical (float in [-1, 1] or None)
    """
    ts = cols["ts"]; sid = cols["station"]; bikes = cols["bikes"]
    if ev.empty:
        return {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"), "rows": []}

    ldt = _to_local(ev[ts], tzname)
    df = ev.assign(date_local=ldt.dt.date, dow=ldt.dt.dayofweek, hhmm=ldt.dt.strftime("%H:%M")).copy()
    last_local_day = df.loc[df[ts].idxmax(), "date_local"]
    today = df[df["date_local"] == last_local_day].copy()
    if today.empty:
        return {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"), "rows": []}

    target_dow = int(today["dow"].iloc[0])
    ref = df[(df["date_local"] < last_local_day) & (df["dow"] == target_dow)].copy()
    if ref.empty:
        return {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"), "rows": []}

    def _cap_est_local(dfg: pd.DataFrame) -> float:
        ce = _estimate_capacity_window(dfg, cols)
        return float(ce.iloc[0]) if len(ce) else float("nan")

    def occ_curve(dfg: pd.DataFrame) -> pd.Series:
        cap = _cap_est_local(dfg)
        b = pd.to_numeric(dfg[bikes], errors="coerce").clip(lower=0)
        if np.isfinite(cap) and cap > 0:
            o = (b / cap).clip(0,1)
        else:
            q = b.quantile(0.98) if len(b) else float("nan")
            o = (b / q).clip(0,1) if np.isfinite(q) and q > 0 else pd.Series(0.0, index=dfg.index)
        return pd.DataFrame({"hhmm": dfg["hhmm"].values, "occ": o.values}).groupby("hhmm")["occ"].mean()

    today_curve = today.groupby(sid).apply(occ_curve)

    rows = []
    for st, cur in today_curve.items():
        sub = ref[ref[sid] == st]
        if sub.empty or cur.empty:
            corr = float("nan")
        else:
            days = sorted(sub["date_local"].unique())[-90:]
            ref_curves = []
            for d in days:
                c = occ_curve(sub[sub["date_local"] == d])
                if not c.empty:
                    ref_curves.append(c)
            if ref_curves:
                ref_med = pd.concat(ref_curves, axis=1).median(axis=1)
                idx = cur.index.intersection(ref_med.index)
                if len(idx) >= 8:
                    a = cur.reindex(idx).astype(float).values
                    b = ref_med.reindex(idx).astype(float).values
                    if np.std(a) == 0 or np.std(b) == 0:
                        corr = float("nan")
                    else:
                        corr = float(np.corrcoef(a,b)[0,1])
                else:
                    corr = float("nan")
            else:
                corr = float("nan")
        rows.append({"station_id": str(st), "regularity_corr_today_vs_typical": _safe_num(corr)})

    return {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "rows": rows}

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    """
    CLI entrypoint for the Network Dynamics job (LATEST ONLY).

    Pipeline
    --------
    1. Define a strict UTC window of `MON_LAST_DAYS` days:
         start = 00:00:00 of (today - (MON_LAST_DAYS - 1))
         end   = now (UTC)
    2. List and read all `events_YYYY-MM-DD.parquet` in the window.
       If none are found, publish a minimal manifest and stop.
    3. Detect columns and normalize dtypes (timestamps, numerics, strings).
       Filter events strictly within [start, end].
    4. Determine `day_for_today_utc` from the last events file name.
    5. Compute dynamics artifacts:
         - heatmaps & profiles,
         - hourly penury/saturation,
         - episodes per station,
         - by_zone aggregates,
         - tension_by_station,
         - regularity_today.
    6. Upload each artifact as JSON under:
         <GCS_MONITORING_PREFIX>/monitoring/network/dynamics/latest/
    7. Build and upload `manifest.json` summarizing:
         - schema version, window_days, tz,
         - thresholds, sources, artifacts list,
         - day_for_today_utc.

    Returns
    -------
    int
        Exit code (0 = success).
    """
    now = datetime.now(timezone.utc)

    # Fenêtre STRICTE = MON_LAST_DAYS
    window_days = int(MON_LAST_DAYS)
    end = now
    start = (end - timedelta(days=window_days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"[network.dynamics] window UTC: {start.date()} → {end.date()} (days={window_days})")

    # 1) Lire les parquets évènementiels dans la fenêtre
    blobs = _list_event_blobs(GCS_EXPORTS_PREFIX, start, end)
    if not blobs:
        print("[network.dynamics] no event blobs in window — nothing to do")
        # publier manifest minimal pour latest
        mon_base = GCS_MONITORING_PREFIX.rstrip("/")
        if not mon_base.endswith("/monitoring"):
            mon_base = mon_base + "/monitoring"
        base_latest = f"{mon_base}/network/dynamics/latest"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": now.isoformat().replace("+00:00","Z"),
            "latest_prefix": base_latest,
            "window_days": window_days,
            "tz": MON_TZ,
            "thresholds": {"penury": int(MON_PENURY_THRESH), "saturation": int(MON_SATURATION_THRESH)},
            "sources": {"exports_prefix": GCS_EXPORTS_PREFIX},
            "artifacts": [],
        }
        _upload_json_gs(manifest, f"{base_latest}/manifest.json")
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
        print("[network.dynamics] no readable data — nothing to do")
        return 0

    ev = pd.concat(frames, ignore_index=True)
    if ev.empty:
        print("[network.dynamics] events empty — nothing to do")
        return 0

    # 2) Colonnes & typage
    cols = _detect_columns(ev)
    ev[cols["ts"]] = pd.to_datetime(ev[cols["ts"]], utc=True, errors="coerce")
    ev[cols["station"]] = ev[cols["station"]].astype("string")
    for c in [cols["bikes"], cols["docks"], cols["capacity"], cols["lat"], cols["lon"]]:
        if c and c in ev.columns:
            ev[c] = pd.to_numeric(ev[c], errors="coerce")
    if cols["name"] and cols["name"] in ev.columns:
        ev[cols["name"]] = ev[cols["name"]].astype("string")

    ev = ev.dropna(subset=[cols["ts"], cols["station"]]).copy()
    ev = ev[(ev[cols["ts"]] >= pd.Timestamp(start)) & (ev[cols["ts"]] <= pd.Timestamp(end))].copy()

    # 3) Jour opérationnel (= dernier events_*.parquet lu)
    m = re.search(r"events_(\d{4}-\d{2}-\d{2})\.parquet$", blobs[-1].name)
    if not m:
        raise RuntimeError("Impossible de déterminer la date du dernier fichier events")
    day_for_today_utc = datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    print(f"[network.dynamics] day_for_today_utc auto-detected → {day_for_today_utc.date()}")

    # 4) Calculs dynamiques
    heatmaps_profiles_doc = _heatmap_and_profiles(ev, cols, MON_TZ)
    hourly_doc            = _hourly_pen_sat(ev, cols, MON_TZ)
    episodes_doc          = _episodes(ev, cols, MON_TZ, MON_LAST_DAYS)
    by_zone_doc           = _by_zone(ev, cols, MON_TZ, MON_LAST_DAYS)
    tension_doc           = _tension_by_station(ev, cols, MON_TZ, MON_LAST_DAYS)
    regularity_doc        = _regularity_today(ev, cols, MON_TZ)

    # 5) Uploads (LATEST only + manifest)
    mon_base = GCS_MONITORING_PREFIX.rstrip("/")
    if not mon_base.endswith("/monitoring"):
        mon_base = mon_base + "/monitoring"
    base_latest = f"{mon_base}/network/dynamics/latest"

    common_meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "day_for_today_utc": day_for_today_utc.strftime("%Y-%m-%d"),
    }

    _upload_json_gs({**common_meta, **{"heatmaps_profiles": heatmaps_profiles_doc}}, f"{base_latest}/heatmaps_profiles.json")
    _upload_json_gs({**common_meta, **{"hourly":             hourly_doc}},            f"{base_latest}/hourly_pen_sat.json")
    _upload_json_gs({**common_meta, **{"episodes":           episodes_doc}},          f"{base_latest}/episodes.json")
    _upload_json_gs({**common_meta, **{"by_zone":            by_zone_doc}},           f"{base_latest}/by_zone.json")
    _upload_json_gs({**common_meta, **{"tension_by_station": tension_doc}},           f"{base_latest}/tension_by_station.json")
    _upload_json_gs({**common_meta, **{"regularity_today":   regularity_doc}},        f"{base_latest}/regularity_today.json")

    # Manifest top-level (pour la page)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "latest_prefix": base_latest,
        "window_days": int(MON_LAST_DAYS),
        "tz": MON_TZ,
        "thresholds": {"penury": int(MON_PENURY_THRESH), "saturation": int(MON_SATURATION_THRESH)},
        "sources": {"exports_prefix": GCS_EXPORTS_PREFIX},
        "artifacts": [
            "heatmaps_profiles.json",
            "hourly_pen_sat.json",
            "episodes.json",
            "by_zone.json",
            "tension_by_station.json",
            "regularity_today.json",
        ],
        "day_for_today_utc": day_for_today_utc.strftime("%Y-%m-%d"),
    }
    _upload_json_gs(manifest, f"{base_latest}/manifest.json")

    print("[network.dynamics] done (latest only)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
