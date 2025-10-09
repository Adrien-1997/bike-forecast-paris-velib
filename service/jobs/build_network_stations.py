# service/jobs/build_network_stations.py
from __future__ import annotations
import os, json, sys
from io import BytesIO
from typing import List, Dict
from datetime import datetime, timezone

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

SCHEMA_VERSION = "1.0"

# ─────────────── GCS helpers ───────────────
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
    storage.Client().bucket(bkt).blob(key).upload_from_string(data, content_type="application/json")
    print(f"[network.stations] wrote → {gs_uri} ({len(data):,} bytes)")

# ─────────────── Core helpers ───────────────
def _recent_cut(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if days <= 0:
        return df
    ts = pd.to_datetime(df["tbin_utc"], errors="coerce")
    max_ts = ts.max()
    if pd.isna(max_ts):
        return df
    start = (max_ts.normalize() - pd.Timedelta(days=days-1))
    return df[ts >= start].copy()

def _safe_occ_ratio(df: pd.DataFrame) -> pd.Series:
    """Toujours renvoyer une pandas.Series alignée sur df.index."""
    if "occ_ratio" in df.columns and not df["occ_ratio"].isna().all():
        return pd.to_numeric(df["occ_ratio"], errors="coerce")
    cap = pd.to_numeric(df.get("capacity"), errors="coerce")
    bk  = pd.to_numeric(df.get("bikes"), errors="coerce")
    out = np.where((cap > 0) & bk.notna(), bk / cap, np.nan)
    return pd.Series(out, index=df.index, dtype="float64")

def _coverage_pct(n_bins: int, n_days: float) -> float:
    # attendu ≈ 288 bins / jour
    expected = max(1.0, 288.0 * max(0.0, n_days))
    return float(100.0 * min(1.0, n_bins / expected))

def _profile24(df: pd.DataFrame, occ: pd.Series) -> List[float | None]:
    d = df.copy()
    d["occ_ratio"] = pd.to_numeric(occ, errors="coerce")
    d["hour"] = pd.to_datetime(d["tbin_utc"], errors="coerce").dt.hour
    vals: List[float | None] = []
    for h in range(24):
        s = pd.to_numeric(d.loc[d["hour"] == h, "occ_ratio"], errors="coerce")
        s = s[(s >= 0) & (s <= 1)]
        if len(s) == 0:
            vals.append(None)
        else:
            vals.append(float(s.mean()))
    return vals

def _days_span(df: pd.DataFrame) -> float:
    ts = pd.to_datetime(df["tbin_utc"], errors="coerce")
    if ts.isna().all():
        return 0.0
    dmin = ts.min().normalize()
    dmax = ts.max().normalize()
    return float((dmax - dmin).days + 1)

# ─────────────── Main ───────────────
def main() -> int:
    EXPORTS_PREFIX = os.environ.get("GCS_EXPORTS_PREFIX")
    MON_PREFIX     = os.environ.get("GCS_MONITORING_PREFIX")
    if not (EXPORTS_PREFIX and EXPORTS_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_EXPORTS_PREFIX manquant ou invalide")
    if not (MON_PREFIX and MON_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX manquant ou invalide")

    WINDOW_DAYS   = int(os.environ.get("NETWORK_WINDOW_DAYS","14"))  # même param que dynamics par défaut
    STATIONS_MAX  = int(os.environ.get("STATIONS_MAX","3000"))       # limite sécurité (taille JSON)
    MIN_BINS_KEEP = int(os.environ.get("MIN_BINS_KEEP","50"))        # filtre stations trop vides

    anchor = datetime.now(timezone.utc)
    anchor_tag = anchor.strftime("%Y%m%d")

    events_uri = f"{EXPORTS_PREFIX.rstrip('/')}/events.parquet"
    print(f"[network.stations] read: {events_uri}")
    ev = _read_parquet_gs(events_uri)
    if ev.empty:
        print("[network.stations] events empty — nothing to do")
        return 0

    # colonnes minimales et types
    need = {"tbin_utc","station_id","bikes","capacity","lat","lon","name","is_penury","is_saturation","occ_ratio"}
    for c in need:
        if c not in ev.columns:
            ev[c] = pd.NA
    ev["tbin_utc"]   = pd.to_datetime(ev["tbin_utc"], errors="coerce")
    ev["station_id"] = ev["station_id"].astype("string")
    ev["is_penury"]  = pd.to_numeric(ev["is_penury"], errors="coerce")
    ev["is_saturation"] = pd.to_numeric(ev["is_saturation"], errors="coerce")
    ev = ev.dropna(subset=["tbin_utc","station_id"]).copy()

    # focus fenêtre récente
    ev_win = _recent_cut(ev, WINDOW_DAYS)
    if ev_win.empty:
        print("[network.stations] window empty — nothing to do")
        return 0

    # calculs par station
    recs: List[Dict] = []
    g = ev_win.groupby("station_id", dropna=True)
    n_days = _days_span(ev_win)

    for sid, grp in g:
        occ = _safe_occ_ratio(grp)                    # Series float64 alignée
        occ = pd.to_numeric(occ, errors="coerce")
        occ_valid = occ[(occ >= 0) & (occ <= 1)]
        n_bins = int(occ_valid.notna().sum())

        if n_bins < MIN_BINS_KEEP:
            continue

        # champs "statiques" robustes (dernier non-null si dispo)
        _name = grp["name"].dropna().astype("string").tail(1)
        name  = None if _name.empty else str(_name.iloc[0])

        _lat = pd.to_numeric(grp["lat"], errors="coerce").dropna().tail(1)
        _lon = pd.to_numeric(grp["lon"], errors="coerce").dropna().tail(1)
        lat  = None if _lat.empty else float(_lat.iloc[0])
        lon  = None if _lon.empty else float(_lon.iloc[0])

        # stats
        capacity_est = float(pd.to_numeric(grp["capacity"], errors="coerce").median(skipna=True)) if grp["capacity"].notna().any() else None
        volatility   = float(np.nanstd(occ_valid)) if occ_valid.notna().sum() > 1 else None
        penury_rate  = float(np.nanmean(pd.to_numeric(grp["is_penury"], errors="coerce"))) if grp["is_penury"].notna().any() else None
        sat_rate     = float(np.nanmean(pd.to_numeric(grp["is_saturation"], errors="coerce"))) if grp["is_saturation"].notna().any() else None
        coverage     = _coverage_pct(n_bins=n_bins, n_days=n_days)

        profile = _profile24(grp, occ)                # 24 valeurs moyennes par heure

        recs.append({
            "station_id": str(sid),
            "name": name,
            "lat": lat,
            "lon": lon,
            "capacity_est": capacity_est,
            "volatility": volatility,
            "penury_rate": penury_rate,
            "saturation_rate": sat_rate,
            "bins_seen": n_bins,
            "coverage_pct": coverage,
            "profile": profile
            # "cluster": "C?"  # à ajouter plus tard si on calcule un clustering
        })

    # sécurité taille JSON : trier et couper si besoin
    recs.sort(key=lambda r: (-r["bins_seen"], r["station_id"]))
    if len(recs) > STATIONS_MAX:
        recs = recs[:STATIONS_MAX]

    # --- Global health summary (intégration de la logique 7j ici) ---
    stations_n = len(recs)
    bins_sum = int(sum(int(r.get("bins_seen") or 0) for r in recs))
    expected_sum = int(max(1, round(288 * max(0.0, n_days))) * stations_n)
    completeness_avg = float(100.0 * bins_sum / expected_sum) if expected_sum > 0 else 0.0
    # bornage léger
    completeness_avg = round(min(100.0, max(0.0, completeness_avg)), 2)

    health_summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "window_days": WINDOW_DAYS,
        "from_date": (ev_win["tbin_utc"].min().date().isoformat() if not ev_win.empty else None),
        "to_date":   (ev_win["tbin_utc"].max().date().isoformat() if not ev_win.empty else None),
        "stations": stations_n,
        "bins_present_sum": bins_sum,
        "bins_expected_sum": expected_sum,
        "completeness_pct_avg": completeness_avg,
    }

    # payload stations
    out = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "window_days": WINDOW_DAYS,
        "stations": recs
    }

    # uploads
    out_alias = f"{MON_PREFIX.rstrip('/')}/network/stations.json"
    out_ver   = f"{MON_PREFIX.rstrip('/')}/network/stations_{anchor_tag}.json"
    _upload_json_gs(out, out_alias)
    _upload_json_gs(out, out_ver)

    # health summary (alias + versionné)
    health_base = f"{MON_PREFIX.rstrip('/')}/health"
    _upload_json_gs(health_summary, f"{health_base}/summary.json")
    _upload_json_gs(health_summary, f"{health_base}/summary_{anchor_tag}.json")

    print("[network.stations] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
