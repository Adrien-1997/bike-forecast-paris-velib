# quick_cov_check.py
from __future__ import annotations
import re
from io import BytesIO
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

# Dépendances: google-cloud-storage, pyarrow
from google.cloud import storage
import pyarrow.parquet as pq

GS_URI = "gs://velib-forecast-472820_cloudbuild/velib/exports/events_2025-10-24.parquet"
BIN_MIN = 5  # pas temporel attendu (min)
TZNAME = "Europe/Paris"  # pour la couverture par heure locale

def _split(gs: str):
    assert gs.startswith("gs://")
    b, k = gs[5:].split("/", 1)
    return b, k

def _r(x, nd=3):
    try:
        x = float(x)
        return float(np.round(x, nd))
    except Exception:
        return None

def _bins_between(a: pd.Timestamp, b: pd.Timestamp, bin_min: int) -> int:
    """Nb de bins (strict à gauche, inclusif à droite) alignés sur bin_min pour (a, b]."""
    if pd.isna(a) or pd.isna(b) or b <= a:
        return 1
    return int(np.floor((b - a).total_seconds() / 60.0 / max(1, int(bin_min))) + 1)

def _expected_bins_per_hour(days: int, bin_min: int) -> int:
    return max(0, (60 // max(1, int(bin_min))) * max(0, int(days)))

def _detect_columns(df: pd.DataFrame):
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
    name  = any_of("name","station_name","nom")
    if not ts or not sid or not bikes:
        raise KeyError(f"Colonnes minimales absentes (ts={ts}, station={sid}, bikes={bikes})")
    return dict(ts=ts, station=sid, bikes=bikes, docks=docks, capacity=cap, ingested_at=ing, name=name)

def read_parquet_gs(gs_uri: str) -> pd.DataFrame:
    bkt, key = _split(gs_uri)
    buf = BytesIO()
    storage.Client().bucket(bkt).blob(key).download_to_file(buf)
    buf.seek(0)
    tbl = pq.read_table(buf)
    return tbl.to_pandas()

def main():
    df = read_parquet_gs(GS_URI)
    cols = _detect_columns(df)
    ts_col, sid_col = cols["ts"], cols["station"]

    # Typage & nettoyage
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dt.tz_convert(None)
    df[sid_col] = df[sid_col].astype("string")
    # déduplication stricte (ts, station)
    df = df.dropna(subset=[ts_col, sid_col]).drop_duplicates(subset=[ts_col, sid_col])

    # Fenêtre du jour (UTC) déduite du nom de fichier
    m = re.search(r"events_(\d{4}-\d{2}-\d{2})\.parquet$", GS_URI)
    if not m:
        raise RuntimeError("Impossible d'inférer la date depuis l'URI.")
    day = datetime.strptime(m.group(1), "%Y-%m-%d").date()
    tmin = pd.Timestamp(day, tz="UTC").tz_convert(None)                     # 00:00:00 UTC
    tmax = (pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1)
            - pd.Timedelta(minutes=BIN_MIN)).tz_convert(None)               # 23:55:00 UTC si pas=5min

    # Limite la fenêtre au contenu réel (au cas où le fichier ne couvre pas 24h)
    if not df.empty:
        real_min, real_max = df[ts_col].min(), df[ts_col].max()
        # on garde l'intention "jour complet", mais on note la réalité
    exp_global = _bins_between(tmin, tmax, BIN_MIN)  # 288 attendu pour 24h/5min

    # Couverture naïve (même attente pour tous)
    got = (df[(df[ts_col] >= tmin) & (df[ts_col] <= tmax)]
             .groupby(sid_col)[ts_col].nunique()
             .rename_axis(sid_col).reset_index(name="obs"))
    if got.empty:
        print("Aucune donnée dans la fenêtre.")
        return 0
    got["expected_global"] = int(exp_global)
    got["coverage_pct_global"] = got["obs"] / got["expected_global"] * 100.0
    coverage_naive_mean = float(got["coverage_pct_global"].mean())

    # Presence-aware (expected_i = entre 1er et dernier point dans la fenêtre)
    pres = (df[(df[ts_col] >= tmin) & (df[ts_col] <= tmax)]
            .groupby(sid_col)[ts_col].agg(first="min", last="max").reset_index())
    pres["expected_i"] = pres.apply(
        lambda r: _bins_between(max(tmin, r["first"]), min(tmax, r["last"]), BIN_MIN),
        axis=1,
    ).astype(int)

    pa = got.merge(pres[[sid_col, "expected_i"]], on=sid_col, how="left")
    pa["obs_clamped"] = np.minimum(pa["obs"].astype(int), pa["expected_i"].astype(int))
    pa["coverage_pct_i"] = pa["obs_clamped"] / pa["expected_i"].replace(0, np.nan) * 100.0
    coverage_pa_mean = float(np.nanmean(pa["coverage_pct_i"]))

    # Couverture par heure locale
    win = df[(df[ts_col] >= tmin) & (df[ts_col] <= tmax)].copy()
    if not win.empty:
        win["_hour"] = pd.to_datetime(win[ts_col]).dt.tz_localize("UTC").dt.tz_convert(TZNAME).dt.hour
        exp_hour = _expected_bins_per_hour(1, BIN_MIN)  # 12 si 5 min
        per_hour = (win.groupby(["_hour", sid_col])[ts_col].nunique().rename("obs").reset_index())
        per_hour["coverage_pct"] = per_hour["obs"].clip(upper=exp_hour) / float(exp_hour or 1) * 100.0
        cov_by_hour = (per_hour.groupby("_hour")["coverage_pct"].mean()
                                .rename_axis("hour").reset_index())
    else:
        cov_by_hour = pd.DataFrame(columns=["hour","coverage_pct"])

    # Stats rapides
    stations = df[sid_col].nunique()
    span = (df[ts_col].min(), df[ts_col].max())

    print("\n=== Quick coverage check (events file, single day) ===")
    print(f"File day (UTC): {day}")
    print(f"Rows: {len(df):,} | Stations: {stations}")
    print(f"Span in file (UTC): {span[0]} → {span[1]}")
    print(f"Expected bins (global, 24h @ {BIN_MIN}min): {exp_global}")
    print(f"Naive coverage mean:   {_r(coverage_naive_mean, 2)} %")
    print(f"Presence-aware mean:   {_r(coverage_pa_mean, 2)} %")
    print("\nTop 5 worst stations (presence-aware):")
    worst = pa.sort_values("coverage_pct_i").head(5)
    for _, r in worst.iterrows():
        print(f"  {r[sid_col]} : obs={int(r['obs'])} / exp_i={int(r['expected_i'])} → {_r(r['coverage_pct_i'],2)} %")
    print("\nCoverage by local hour (mean across stations):")
    if len(cov_by_hour):
        for _, r in cov_by_hour.sort_values("hour").iterrows():
            print(f"  h={int(r['hour']):02d} → {_r(r['coverage_pct'],2)} %")
    else:
        print("  (no data)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
