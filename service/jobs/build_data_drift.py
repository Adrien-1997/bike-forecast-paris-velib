# service/jobs/build_data_drift.py
from __future__ import annotations
import os, re, json, sys
from io import BytesIO
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception as e:
    raise RuntimeError("pyarrow is required") from e

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage is required") from e


SCHEMA_VERSION = "1.4"  # 1.4 = exclut coords/temps des features de drift + garde-fous robustes

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

def _upload_json_gs(obj: dict | list, gs_uri: str):
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
    print(f"[data.drift] wrote → {gs_uri} ({len(data):,} bytes)")

# ──────────────────────────────────────────────────────────────────────────────
# Normalisation des events (schéma build_datasets.py)
# ──────────────────────────────────────────────────────────────────────────────

KEY_STR = {"station_id", "status", "name"}
KEY_TIME = {"tbin_utc", "ts", "timestamp"}

# Exclusions explicites pour le drift
EXCLUDE_EXACT = {
    # identifiers & time-like
    "station_id", "date_local", "dow", "hour", "h", "min", "ts_local",
    # geo
    "lat", "lon",
    # we never use raw timestamps as features
    "tbin_utc", "ts", "timestamp",
}
EXCLUDE_PATTERNS = [
    r".*_(sin|cos)$",           # encodages cycliques éventuels
    r"^(hour|minute|dow)(_|$)", # colonnes temporelles dérivées
]

def _is_time_or_coord(col: str) -> bool:
    if col in EXCLUDE_EXACT:
        return True
    return any(re.match(p, col) for p in EXCLUDE_PATTERNS)

def _ensure_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attend au minimum: tbin_utc, station_id.
    - Force tbin_utc en datetime64[ns] UTC (naive).
    - Ne JAMAIS convertir les colonnes datetime en numériques.
    - Convertit en numériques (coerce) les colonnes object/bool pertinentes.
    """
    if not {"tbin_utc", "station_id"}.issubset(df.columns):
        raise RuntimeError("Missing minimal columns: tbin_utc/station_id")

    out = df.copy()

    # station_id en string
    out["station_id"] = out["station_id"].astype("string")

    # tbin_utc en datetime naive (UTC)
    out["tbin_utc"] = pd.to_datetime(out["tbin_utc"], errors="coerce", utc=True).dt.tz_convert(None)

    # Pour chaque colonne:
    for c in out.columns:
        if c in KEY_STR or c in KEY_TIME:
            continue  # ne pas toucher station_id/status/name et timestamps
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            continue  # sécurité: ne touche pas les datetime
        if pd.api.types.is_numeric_dtype(out[c]):
            continue  # déjà numérique
        # object/bool -> to_numeric(coerce)
        try:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        except Exception:
            pass

    return out

def _to_local(df: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    dt = pd.to_datetime(df["tbin_utc"], errors="coerce", utc=True)
    dt_local = dt.dt.tz_convert(tz) if tz else dt
    return df.assign(date_local=dt_local.dt.date, dow=dt_local.dt.dayofweek, hour=dt_local.dt.hour, ts_local=dt_local)

# ──────────────────────────────────────────────────────────────────────────────
# Drift metrics
# ──────────────────────────────────────────────────────────────────────────────

def _valid_series(x) -> pd.Series:
    """Force Series 1D numérique sans NaN; retourne Series vide si impossible."""
    try:
        s = pd.to_numeric(pd.Series(x, copy=False), errors="coerce").dropna()
        return s if isinstance(s, pd.Series) else pd.Series([], dtype=float)
    except Exception:
        return pd.Series([], dtype=float)

def _psi_continuous(ref: pd.Series, cur: pd.Series, bins: int = 20, eps: float = 1e-9) -> float:
    a = _valid_series(ref)
    b = _valid_series(cur)
    if len(a) < 5 or len(b) < 5:
        return np.nan
    if a.min() == a.max() or b.min() == b.max():
        return np.nan
    try:
        q = np.unique(np.nanquantile(a, np.linspace(0, 1, bins + 1)))
    except Exception:
        return np.nan
    if q.size < 3:
        return np.nan
    ca, _ = np.histogram(a, bins=q)
    cb, _ = np.histogram(b, bins=q)
    if ca.sum() == 0 or cb.sum() == 0:
        return np.nan
    pa = (ca / ca.sum()).astype(float) + eps
    pb = (cb / cb.sum()).astype(float) + eps
    return float(np.sum((pa - pb) * np.log(pa / pb)))

def _ks_stat(ref: pd.Series, cur: pd.Series) -> float:
    a = _valid_series(ref)
    b = _valid_series(cur)
    if len(a) < 5 or len(b) < 5:
        return np.nan
    both = pd.concat([a, b], ignore_index=True)
    if both.empty or both.min() == both.max():
        return np.nan
    try:
        q = np.unique(np.nanquantile(both, np.linspace(0, 1, 201)))
    except Exception:
        return np.nan
    if q.size < 3:
        return np.nan
    ca, _ = np.histogram(a, bins=q)
    cb, _ = np.histogram(b, bins=q)
    if ca.sum() == 0 or cb.sum() == 0:
        return np.nan
    cdfa = np.cumsum(ca) / ca.sum()
    cdfb = np.cumsum(cb) / cb.sum()
    return float(np.max(np.abs(cdfa - cdfb)))

def _delta_mean_var(ref: pd.Series, cur: pd.Series) -> tuple[float, float]:
    a = _valid_series(ref)
    b = _valid_series(cur)
    if len(a) < 5 or len(b) < 5:
        return (np.nan, np.nan)
    if a.std(ddof=1) == 0:
        return (np.nan, np.nan)
    dm = (b.mean() - a.mean()) / (a.std(ddof=1) + 1e-9)
    avar = a.var(ddof=1)
    dv = (b.var(ddof=1) - avar) / (avar + 1e-9)
    return (float(dm), float(dv))

def _split_windows(df: pd.DataFrame, current_days: int, reference_days: int) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Timestamp]]:
    if df.empty:
        return df.iloc[0:0].copy(), df.iloc[0:0].copy(), {}
    tmax = df["tbin_utc"].max()
    t_cur_start = tmax - pd.Timedelta(days=current_days)
    t_ref_end = t_cur_start
    t_ref_start = t_ref_end - pd.Timedelta(days=reference_days)
    ref = df[(df["tbin_utc"] >= t_ref_start) & (df["tbin_utc"] < t_ref_end)].copy()
    cur = df[(df["tbin_utc"] >= t_cur_start) & (df["tbin_utc"] <= tmax)].copy()
    bounds = {"tmax": tmax, "t_cur_start": t_cur_start, "t_ref_start": t_ref_start, "t_ref_end": t_ref_end}
    return ref, cur, bounds

def _assign_zone(df: pd.DataFrame) -> pd.Series:
    for c in ("arrondissement", "arr", "zone", "district"):
        if c in df.columns:
            return df[c].astype(str)
    lat = pd.to_numeric(df.get("lat"), errors="coerce")
    lon = pd.to_numeric(df.get("lon"), errors="coerce")
    mask = lat.notna() & lon.notna()
    z = pd.Series(index=df.index, dtype=object, name="zone")
    lat_r = (lat[mask] * 100).round() / 100.0
    lon_r = (lon[mask] * 100).round() / 100.0
    z.loc[mask] = lat_r.astype(str) + "," + lon_r.astype(str)
    return z

def _compute_drift(events: pd.DataFrame, current_days: int, reference_days: int, tz: Optional[str]) -> dict:
    df = _to_local(events, tz)
    ref, cur, bounds = _split_windows(df, current_days=current_days, reference_days=reference_days)

    # agrégation par (date_local, station) — moyenne journalière
    def agg(d: pd.DataFrame):
        # Colonnes numériques candidates (on exclut coord/temps ici aussi)
        num_cols = [
            c for c in d.columns
            if c not in {"station_id","tbin_utc","status","name","date_local","dow","hour","ts_local"}
            and pd.api.types.is_numeric_dtype(d[c])
            and not _is_time_or_coord(c)
        ]
        # On garde lat/lon séparément pour la carte des zones, mais on ne les met pas dans num_cols
        keep = ["date_local","station_id"] + num_cols + [c for c in ("lat","lon") if c in d.columns]
        return d[keep].groupby(["date_local","station_id"], dropna=True).mean(numeric_only=True).reset_index()

    ref = agg(ref); cur = agg(cur)

    # auto features: numériques communes, hors coord/temps
    common_num = [
        c for c in ref.columns
        if c not in {"date_local","station_id"} and c in cur.columns
        and pd.api.types.is_numeric_dtype(ref[c]) and not _is_time_or_coord(c)
    ]

    rows_psi, rows_ks, rows_delta = [], [], []
    for f in sorted(common_num):
        try:
            rf = _valid_series(ref[f]); cf = _valid_series(cur[f])
            if len(rf) < 5 or len(cf) < 5:
                print(f"[data.drift][info] skip feature '{f}' (insufficient data)")
                continue
            psi = _psi_continuous(rf, cf)
            ks  = _ks_stat(rf, cf)
            dm, dv = _delta_mean_var(rf, cf)
            rows_psi.append({"feature": f, "psi": float(psi) if np.isfinite(psi) else None})
            rows_ks.append({"feature": f, "ks":  float(ks)  if np.isfinite(ks)  else None})
            rows_delta.append({"feature": f,
                               "delta_mean": float(dm) if np.isfinite(dm) else None,
                               "delta_var":  float(dv) if np.isfinite(dv) else None})
        except Exception as e:
            print(f"[data.drift][warn] metrics failed for feature '{f}': {e}")

    psi_df = pd.DataFrame(rows_psi)
    ks_df = pd.DataFrame(rows_ks)
    d_df = pd.DataFrame(rows_delta)

    # EMA quotidienne sur occ_ratio (si dispo)
    ema_df = pd.DataFrame(columns=["date_local","psi_ema"])
    if "occ_ratio" in events.columns:
        by_day = df.groupby("date_local")["occ_ratio"].apply(lambda s: float(np.nanmean(pd.to_numeric(s, errors='coerce')))).reset_index()
        by_day = by_day.sort_values("date_local")
        alpha = 2 / (7 + 1.0)
        ema, last = [], None
        for _, r in by_day.iterrows():
            x = r["occ_ratio"]
            if pd.isna(x):
                ema.append(np.nan); continue
            last = x if last is None else (alpha * x + (1 - alpha) * last)
            ema.append(last)
        ema_df = pd.DataFrame({"date_local": by_day["date_local"].astype(str), "psi_ema": ema})

    # Résumé + alertes
    psi_global = None
    if not psi_df.empty:
        if "occ_ratio" in list(psi_df["feature"]):
            v = psi_df.loc[psi_df["feature"]=="occ_ratio","psi"].values[0]
            psi_global = float(v) if v is not None and np.isfinite(v) else None
        else:
            v = np.nanmedian([p for p in psi_df["psi"] if p is not None])
            psi_global = float(v) if np.isfinite(v) else None

    alerts = []
    if psi_global is not None:
        if psi_global >= 0.25:
            alerts.append({"level": "high", "code": "psi_global_high", "text": f"High global PSI ({psi_global:.3f})"})
        elif psi_global >= 0.10:
            alerts.append({"level": "medium", "code": "psi_global_medium", "text": f"Moderate global PSI ({psi_global:.3f})"})

    summary = {
        "psi_global": psi_global,
        "top_feature": (psi_df.sort_values("psi", ascending=False).iloc[0]["feature"] if not psi_df.empty else None),
        "top_feature_psi": (float(psi_df.sort_values("psi", ascending=False).iloc[0]["psi"]) if not psi_df.empty and psi_df.sort_values("psi", ascending=False).iloc[0]["psi"] is not None else None),
    }

    # Zones PSI (occ_ratio si dispo, sinon bikes) — robuste
    zones_doc = {"rows": []}
    try:
        rows = []
        if {"lat", "lon"}.issubset(df.columns):
            ref_z = ref.assign(zone=_assign_zone(ref))
            cur_z = cur.assign(zone=_assign_zone(cur))
            for z, rsub in ref_z.groupby("zone", dropna=True):
                csub = cur_z[cur_z["zone"] == z]
                if csub.empty:
                    continue
                metric = None
                for m in ("occ_ratio", "bikes"):
                    if m in rsub.columns and m in csub.columns:
                        metric = m
                        break
                if not metric:
                    continue

                rvals = _valid_series(rsub[metric])
                cvals = _valid_series(csub[metric])
                if len(rvals) < 5 or len(cvals) < 5:
                    continue

                psi = _psi_continuous(rvals, cvals)
                lat = float(rsub["lat"].median()) if "lat" in rsub.columns and rsub["lat"].notna().any() else None
                lon = float(rsub["lon"].median()) if "lon" in rsub.columns and rsub["lon"].notna().any() else None
                zname = None if (z is None or (isinstance(z, float) and np.isnan(z))) else str(z)
                rows.append({"zone": zname, "psi": float(psi) if np.isfinite(psi) else None, "lat": lat, "lon": lon})
        zones_doc = {"rows": rows}
    except Exception as e:
        print(f"[data.drift] zones PSI failed: {e}")

    return {
        "psi_df": psi_df, "ks_df": ks_df, "deltas_df": d_df,
        "psi_daily_ema": ema_df,
        "summary": summary, "alerts": alerts,
        "bounds": {
            "tmax_utc": pd.Timestamp(bounds.get("tmax")).tz_localize("UTC").isoformat().replace("+00:00","Z") if bounds else None,
            "cur_start_utc": pd.Timestamp(bounds.get("t_cur_start")).tz_localize("UTC").isoformat().replace("+00:00","Z") if bounds else None,
            "ref_start_utc": pd.Timestamp(bounds.get("t_ref_start")).tz_localize("UTC").isoformat().replace("+00:00","Z") if bounds else None,
            "ref_end_utc": pd.Timestamp(bounds.get("t_ref_end")).tz_localize("UTC").isoformat().replace("+00:00","Z") if bounds else None,
        },
        "zones": zones_doc,
        "feature_list": sorted(common_num),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    EXPORTS_PREFIX = os.environ.get("GCS_EXPORTS_PREFIX")
    MON_PREFIX     = os.environ.get("GCS_MONITORING_PREFIX")
    if not (EXPORTS_PREFIX and EXPORTS_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_EXPORTS_PREFIX missing or invalid")
    if not (MON_PREFIX and MON_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX missing or invalid")

    CUR_DAYS = int(os.environ.get("DRIFT_CURRENT_DAYS", "7"))
    REF_DAYS = int(os.environ.get("DRIFT_REFERENCE_DAYS", "28"))
    TZNAME   = os.environ.get("DRIFT_TZ", "Europe/Paris")

    now = datetime.now(timezone.utc)
    WINDOW_DAYS = max(CUR_DAYS, REF_DAYS, 7)
    start = (now - timedelta(days=WINDOW_DAYS - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"[data.drift] window UTC: {start.date()} → {now.date()} (days={WINDOW_DAYS})")

    blobs = _list_event_blobs(EXPORTS_PREFIX, start, now)
    if not blobs:
        print("[data.drift] no event blobs in window — nothing to do]")
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
        print("[data.drift] no readable data — nothing to do")
        return 0

    ev = pd.concat(frames, ignore_index=True)
    if ev.empty:
        print("[data.drift] events empty — nothing to do")
        return 0

    ev_norm = _ensure_events(ev)
    res = _compute_drift(ev_norm, current_days=CUR_DAYS, reference_days=REF_DAYS, tz=TZNAME)

    out_prefix = MON_PREFIX.rstrip("/") + "/monitoring/data/drift/latest"
    files = {
        "psi_by_feature.json":       res["psi_df"].to_dict(orient="records"),
        "ks_by_feature.json":        res["ks_df"].to_dict(orient="records"),
        "deltas_by_feature.json":    res["deltas_df"].to_dict(orient="records"),
        "psi_global_daily_ema.json": res["psi_daily_ema"].to_dict(orient="records"),
        "summary.json":              {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"), **res["summary"]},
        "alerts.json":               res["alerts"],
        "bounds.json":               res["bounds"],
        "zones.json":                res["zones"],
        "features_detected.json":    res["feature_list"],
    }
    for fname, payload in files.items():
        _upload_json_gs(payload, f"{out_prefix}/{fname}")

    print(f"[data.drift] done → {out_prefix}/ (LATEST only)")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[data.drift][fatal] {e}", file=sys.stderr)
        sys.exit(2)
