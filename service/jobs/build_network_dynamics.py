# service/jobs/build_network_dynamics.py
from __future__ import annotations
import os, json, sys
from io import BytesIO
from typing import List, Dict
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
    print(f"[network.dynamics] wrote → {gs_uri} ({len(data):,} bytes)")

# ─────────────── Params & time ───────────────
def _anchor_day_utc() -> datetime:
    ad = os.environ.get("ANCHOR_DAY")
    if ad:
        return datetime.fromisoformat(ad).replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)

def _recent_cut(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if days <= 0:
        return df
    max_ts = pd.to_datetime(df["tbin_utc"], errors="coerce").max()
    if pd.isna(max_ts):
        return df
    start = (max_ts.normalize() - pd.Timedelta(days=days-1))
    return df[pd.to_datetime(df["tbin_utc"], errors="coerce") >= start].copy()

# ─────────────── Core builders ───────────────
def _build_daily_profiles(events: pd.DataFrame) -> List[dict]:
    d = events.copy()
    d["tbin_utc"] = pd.to_datetime(d["tbin_utc"], errors="coerce")
    d["hour"] = d["tbin_utc"].dt.hour
    d["dow"]  = d["tbin_utc"].dt.dayofweek  # 0=Mon
    # On calcule sur occ_ratio (fallback sur bikes/cap si besoin)
    if "occ_ratio" not in d.columns or d["occ_ratio"].isna().all():
        cap = pd.to_numeric(d["capacity"], errors="coerce")
        bk  = pd.to_numeric(d["bikes"], errors="coerce")
        d["occ_ratio"] = np.where((cap > 0) & bk.notna(), bk/cap, np.nan)
    recs: List[dict] = []
    grp = d.groupby(["dow","hour"], dropna=True)
    for (dw, hr), g in grp:
        s = pd.to_numeric(g["occ_ratio"], errors="coerce")
        s = s[(s >= 0) & (s <= 1)]
        if len(s) == 0:
            p50 = p90 = None
            n = 0
        else:
            p50 = float(np.nanpercentile(s, 50))
            p90 = float(np.nanpercentile(s, 90))
            n = int(len(s))
        recs.append({"dow": int(dw), "hour": int(hr), "occ_p50": p50, "occ_p90": p90, "n": n})
    recs.sort(key=lambda r: (r["dow"], r["hour"]))
    return recs

def _build_last_day_heatmap(events: pd.DataFrame, max_points: int) -> List[dict]:
    d = events.copy()
    d["tbin_utc"] = pd.to_datetime(d["tbin_utc"], errors="coerce")
    last_day = d["tbin_utc"].dt.normalize().max()
    if pd.isna(last_day):
        return []
    d = d[d["tbin_utc"].dt.normalize() == last_day]

    if "occ_ratio" not in d.columns or d["occ_ratio"].isna().all():
        cap = pd.to_numeric(d["capacity"], errors="coerce")
        bk  = pd.to_numeric(d["bikes"], errors="coerce")
        d["occ_ratio"] = np.where((cap > 0) & bk.notna(), bk/cap, np.nan)

    d = d[["tbin_utc","station_id","occ_ratio"]].copy()
    d = d.dropna(subset=["tbin_utc","station_id"])
    # downsample si trop gros
    if max_points and len(d) > max_points:
        frac = max_points / float(len(d))
        d = d.sample(frac=frac, random_state=42).sort_values(["tbin_utc","station_id"])
    # sérialisation
    out = []
    for r in d.itertuples(index=False):
        val = None
        if pd.notna(r.occ_ratio):
            # clamp [0,1]
            v = float(r.occ_ratio)
            if v < 0 or v > 1:
                val = None
            else:
                val = v
        out.append({
            "tbin_utc": pd.to_datetime(r.tbin_utc, utc=True).isoformat().replace("+00:00","Z"),
            "station_id": str(r.station_id),
            "occ": val
        })
    return out

# ─────────────── Main ───────────────
def main() -> int:
    EXPORTS_PREFIX = os.environ.get("GCS_EXPORTS_PREFIX")
    MON_PREFIX     = os.environ.get("GCS_MONITORING_PREFIX")
    if not (EXPORTS_PREFIX and EXPORTS_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_EXPORTS_PREFIX manquant ou invalide")
    if not (MON_PREFIX and MON_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX manquant ou invalide")

    WINDOW_DAYS   = int(os.environ.get("NETWORK_WINDOW_DAYS","14"))  # fenêtre d’analyse
    HEAT_MAX_PT   = int(os.environ.get("HEATMAP_MAX_POINTS","200000"))  # downsample sécurité
    anchor        = datetime.now(timezone.utc)
    anchor_tag    = anchor.strftime("%Y%m%d")

    events_uri = f"{EXPORTS_PREFIX.rstrip('/')}/events.parquet"
    print(f"[network.dynamics] read: {events_uri}")
    ev = _read_parquet_gs(events_uri)

    if ev.empty:
        print("[network.dynamics] events empty — nothing to do")
        return 0

    # filtres / types
    if "tbin_utc" not in ev.columns:
        print("[network.dynamics] missing tbin_utc — abort")
        return 1
    ev["tbin_utc"] = pd.to_datetime(ev["tbin_utc"], errors="coerce")
    ev = ev.dropna(subset=["tbin_utc","station_id"]).copy()

    # fenêtre récente pour profils
    ev_win = _recent_cut(ev, WINDOW_DAYS)

    daily_profiles   = _build_daily_profiles(ev_win)
    last_day_heatmap = _build_last_day_heatmap(ev, max_points=HEAT_MAX_PT)

    out = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "window_days": WINDOW_DAYS,
        "daily_profiles": daily_profiles,
        "last_day_heatmap": last_day_heatmap
        # "episodes": []    # (optionnel) à ajouter plus tard si besoin
    }

    out_alias = f"{MON_PREFIX.rstrip('/')}/network/dynamics.json"
    out_ver   = f"{MON_PREFIX.rstrip('/')}/network/dynamics_{anchor_tag}.json"
    _upload_json_gs(out, out_alias)
    _upload_json_gs(out, out_ver)
    print("[network.dynamics] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
