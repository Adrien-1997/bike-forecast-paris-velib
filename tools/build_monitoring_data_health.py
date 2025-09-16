# tools/build_monitoring_data_health.py
# Page builder — "Monitoring / Santé des données"
#
# Produit :
# - KPIs de fraîcheur, complétude, latence d’ingestion (si dispo)
# - Rapport de schéma & contraintes (pass/warn/fail)
# - Tables : data_health.csv (KPIs), station_health.csv (par station),
#            completeness_by_station_day.csv, coverage_by_hour.csv,
#            schema_report.csv, anomalies.csv
# - Figures : gauges simples (barres) fraîcheur/complétude, heatmap complétude (stations × jours),
#             top stations problématiques (barres), distribution de latence (si dispo)
#
# CLI :
#   python tools/build_monitoring_data_health.py --events docs/exports/events.parquet \
#       --current-days 7 --tz Europe/Paris --fresh-slo-min 5 --flat-steps 8
#
# Sorties (docs/assets/… et docs/exports/…):
# - assets/figs/monitoring/data_health/gauges.png
# - assets/figs/monitoring/data_health/heatmap_completeness.png
# - assets/figs/monitoring/data_health/top_issues.png
# - assets/figs/monitoring/data_health/latency_hist.png (si latence dispo)
# - assets/tables/monitoring/data_health/data_health.csv
# - assets/tables/monitoring/data_health/station_health.csv
# - assets/tables/monitoring/data_health/completeness_by_station_day.csv
# - assets/tables/monitoring/data_health/coverage_by_hour.csv
# - assets/tables/monitoring/data_health/schema_report.csv
# - assets/tables/monitoring/data_health/anomalies.csv
#
# Hypothèses colonnes (souples) :
# - ts | tbin_utc | timestamp     → horodatage (sera converti UTC naïf)
# - station_id | stationcode      → identifiant station (string)
# - bikes | nb_velos_bin | …      → nombre de vélos dispos (float/int)
# - docks_avail | nb_docks_bin    → bornes libres (optionnel)
# - capacity                       → capacité estimée (optionnel)
# - ingested_at | ingest_ts | …    → timestamp d’ingestion (optionnel)
# - lat, lon, name                 → métadonnées (optionnel)

from __future__ import annotations

import os, math
from pathlib import Path
import argparse
import json
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------- Paths ---------------------------

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
FIGS_DIR = ASSETS / "figs" / "monitoring" / "data_health"
TABLES_DIR = ASSETS / "tables" / "monitoring" / "data_health"
EXPORTS = DOCS / "exports"

plt.rcParams["figure.figsize"] = (9, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

# --------------------------- JSON helper ---------------------------

def _json_default(o):
    # Normalize NumPy/Pandas types → Python natives for json
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    # Pandas NA/NaT → None
    try:
        if pd.isna(o):
            return None
    except Exception:
        pass
    return str(o)

# --------------------------- Utils ---------------------------

def _mkdirs() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

def _save_fig(path: Path) -> None:
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()

def _now_utc_floor15() -> pd.Timestamp:
    # UTC → drop tz ⇒ UTC naïf (compatibles avec df["ts"])
    return pd.Timestamp.now(tz="UTC").floor("15min").tz_localize(None)

def _expected_bins(tmin: pd.Timestamp, tmax: pd.Timestamp) -> int:
    return max(1, math.ceil((tmax - tmin).total_seconds() / 60 / 15.0) + 1)

def _read_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[data-health] Introuvable: {path}")
    df = pd.read_parquet(path)

    # ts → UTC naïf
    tcol = None
    for c in ("ts", "tbin_utc", "timestamp"):
        if c in df.columns:
            tcol = c
            break
    if tcol is None:
        raise KeyError("[data-health] Colonne temporelle manquante (ts/tbin_utc/timestamp)")
    df["ts"] = (
        pd.to_datetime(df[tcol], errors="coerce", utc=True)
          .dt.floor("15min")
          .dt.tz_localize(None)
    )

    # station_id
    sid = None
    for c in ("station_id", "stationcode", "station"):
        if c in df.columns:
            sid = c
            break
    if sid is None:
        raise KeyError("[data-health] Identifiant station manquant (station_id/stationcode)")
    df["station_id"] = df[sid].astype(str)

    # core numeric
    bikes_col = None
    for c in ("bikes", "nb_velos_bin", "velos_disponibles", "numBikesAvailable"):
        if c in df.columns:
            bikes_col = c; break
    if bikes_col is None:
        raise KeyError("[data-health] Colonne vélos manquante (bikes/nb_velos_bin/velos_disponibles)")
    df["bikes"] = pd.to_numeric(df[bikes_col], errors="coerce")

    docks_col = None
    for c in ("docks_avail", "nb_docks_bin", "numDocksAvailable"):
        if c in df.columns:
            docks_col = c; break
    df["docks_avail"] = pd.to_numeric(df.get(docks_col, np.nan), errors="coerce")

    cap_col = "capacity" if "capacity" in df.columns else None
    df["capacity"] = pd.to_numeric(df.get(cap_col, np.nan), errors="coerce")

    # ingestion timestamp (optionnel) → si absent, série de NaT
    ing_col = None
    for c in ("ingested_at", "ingest_ts", "ingest_time", "received_at", "created_at", "etl_ts", "load_ts"):
        if c in df.columns:
            ing_col = c; break
    if ing_col is None:
        df["ingested_at"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    else:
        df["ingested_at"] = (
            pd.to_datetime(df[ing_col], errors="coerce", utc=True)
              .dt.tz_localize(None)
        )

    # coords (optionnelles)
    if "lat" in df.columns: df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    if "lon" in df.columns: df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    keep = ["ts", "station_id", "bikes", "docks_avail", "capacity", "ingested_at"]
    for c in ("lat", "lon", "name"):
        if c in df.columns: keep.append(c)
    return df[keep].copy()

# --------------------------- KPIs ---------------------------

def kpi_freshness(df: pd.DataFrame, slo_min: int) -> dict:
    """Âge du dernier point par station, puis P50/P95 (minutes) vs SLO."""
    now = _now_utc_floor15()
    last_ts = df.groupby("station_id")["ts"].max().reset_index(name="last_ts")
    last_ts["age_min"] = (now - last_ts["last_ts"]).dt.total_seconds() / 60.0
    p50 = float(np.nanpercentile(last_ts["age_min"], 50)) if len(last_ts) else np.nan
    p95 = float(np.nanpercentile(last_ts["age_min"], 95)) if len(last_ts) else np.nan
    return {
        "now_utc": now.isoformat(),
        "ts_global_max": (df["ts"].max().isoformat() if len(df) else None),
        "freshness_age_p50_min": round(p50, 2),
        "freshness_age_p95_min": round(p95, 2),
        "freshness_slo_min": float(slo_min),
        "freshness_p95_ok": (p95 <= slo_min) if np.isfinite(p95) else None
    }

def kpi_completeness(df: pd.DataFrame, current_days: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Complétude globale & par station sur fenêtre récente; heatmap station×jour."""
    if current_days <= 0 or df.empty:
        return {"coverage_global_pct": np.nan}, pd.DataFrame(), pd.DataFrame()

    tmax = df["ts"].max()
    tmin = tmax - pd.Timedelta(days=current_days)
    win = df[(df["ts"] > tmin) & (df["ts"] <= tmax)].copy()
    if win.empty:
        return {"coverage_global_pct": np.nan}, pd.DataFrame(), pd.DataFrame()

    exp = _expected_bins(tmin, tmax)
    per_station = (win.groupby("station_id")["ts"].nunique()
                      .clip(upper=exp)
                      .rename_axis("station_id").reset_index(name="obs"))
    per_station["expected"] = exp
    per_station["coverage_pct"] = per_station["obs"] / per_station["expected"] * 100.0
    coverage_global = float(per_station["coverage_pct"].mean()) if len(per_station) else np.nan

    # coverage by hour of day (0..23) across stations
    per_hour = win.copy()
    per_hour["hour"] = per_hour["ts"].dt.hour
    per_hour = (per_hour.groupby(["hour", "station_id"])["ts"].nunique()
                         .rename("obs").reset_index())
    # expected bins per hour per station: #days * 4 bins/hour
    exp_hour = current_days * 4
    per_hour["coverage_pct"] = per_hour["obs"].clip(upper=exp_hour) / float(exp_hour) * 100.0
    cov_by_hour = (per_hour.groupby("hour")["coverage_pct"].mean()
                          .rename_axis("hour").reset_index())

    # station×day heatmap: fraction of present bins per day
    win["date"] = win["ts"].dt.date
    per_sd = win.groupby(["station_id","date"])["ts"].nunique().reset_index(name="obs")
    per_sd["expected"] = 24 * 4  # 96 bins per day
    per_sd["coverage_pct"] = per_sd["obs"].clip(upper=per_sd["expected"]) / per_sd["expected"] * 100.0
    heat = per_sd.pivot(index="station_id", columns="date", values="coverage_pct").fillna(0.0)

    return {"coverage_global_pct": round(coverage_global, 2)}, per_station, heat

def kpi_latency(df: pd.DataFrame) -> tuple[dict, Optional[pd.DataFrame]]:
    """Latence d’ingestion: (ingested_at - ts) en minutes. Retourne KPIs + distribution brute."""
    if "ingested_at" not in df.columns or df["ingested_at"].isna().all():
        return {"latency_p95_min": np.nan, "latency_p50_min": np.nan}, None
    lat = (df.dropna(subset=["ts", "ingested_at"]).copy())
    lat["latency_min"] = (lat["ingested_at"] - lat["ts"]).dt.total_seconds() / 60.0
    if lat.empty:
        return {"latency_p95_min": np.nan, "latency_p50_min": np.nan}, None
    p50 = float(np.nanpercentile(lat["latency_min"], 50))
    p95 = float(np.nanpercentile(lat["latency_min"], 95))
    return {"latency_p50_min": round(p50,2), "latency_p95_min": round(p95,2)}, lat[["ts","station_id","latency_min"]]

def schema_report(df: pd.DataFrame) -> pd.DataFrame:
    """Vérifie un schéma minimal & contraintes simples."""
    rep = []
    def add(name, status, detail=""):
        rep.append({"check": name, "status": status, "detail": detail})

    # colonnes obligatoires
    for col in ("ts", "station_id", "bikes"):
        add(f"col:{col}", "pass" if col in df.columns else "fail")

    # types
    add("type:ts:datetime64", "pass" if np.issubdtype(df["ts"].dtype, np.datetime64) else "fail")
    add("type:station_id:str", "pass" if df["station_id"].dtype == object or str(df["station_id"].dtype).startswith("string") else "warn")
    add("type:bikes:numeric", "pass" if np.issubdtype(df["bikes"].dtype, np.number) else "fail")

    # valeurs plausibles
    if "bikes" in df.columns:
        neg = int((df["bikes"] < 0).sum())
        add("bikes>=0", "pass" if neg == 0 else "fail", f"negatives={neg}")
    if "docks_avail" in df.columns:
        neg = int((df["docks_avail"] < 0).sum())
        add("docks_avail>=0", "pass" if neg == 0 else "fail", f"negatives={neg}")
    if "capacity" in df.columns and df["capacity"].notna().any():
        neg = int((df["capacity"] <= 0).sum())
        add("capacity>0", "pass" if neg == 0 else "fail", f"nonpositive={neg}")

    # doublons exacts (ts, station_id)
    dups = int(df.duplicated(subset=["ts","station_id"]).sum())
    add("no_duplicates(ts,station)", "pass" if dups == 0 else "warn", f"dups={dups}")

    # couverture temporelle grossière
    if len(df):
        tmin, tmax = df["ts"].min(), df["ts"].max()
        exp = _expected_bins(tmin, tmax)
        got = int(df["ts"].nunique())
        ratio = got / float(exp)
        add("temporal_coverage", "pass" if ratio >= 0.9 else "warn", f"bins={got}/{exp} ({ratio:.2%})")

    return pd.DataFrame(rep)

def flat_sequences(df: pd.DataFrame, min_steps: int, current_days: int) -> pd.DataFrame:
    """
    Detect flat sequences (constant `bikes`) over the recent window.
    Returns columns: station_id, steps, start, end.
    """
    if df.empty:
        return pd.DataFrame(columns=["station_id", "steps", "start", "end"])

    tmax = df["ts"].max()
    tmin = tmax - pd.Timedelta(days=current_days)
    win = df[(df["ts"] > tmin) & (df["ts"] <= tmax)].copy()
    if win.empty:
        return pd.DataFrame(columns=["station_id", "steps", "start", "end"])

    win = win.sort_values(["station_id", "ts"])
    win["delta"] = win.groupby("station_id")["bikes"].diff().fillna(0.0).abs()
    win["is_flat"] = (win["delta"] == 0.0)

    # group consecutive flat points per station
    win["grp"] = (~win["is_flat"]).groupby(win["station_id"]).cumsum()
    agg = (
        win[win["is_flat"]]
        .groupby(["station_id", "grp"])
        .agg(steps=("is_flat", "size"), start=("ts", "min"), end=("ts", "max"))
        .reset_index()  # KEEP station_id
    )

    out = agg[agg["steps"] >= min_steps][["station_id", "steps", "start", "end"]]
    return out.sort_values(["steps"], ascending=False).reset_index(drop=True)

def duplication_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Comptage de doublons (ts, station_id)."""
    if df.empty: return pd.DataFrame(columns=["station_id","dups"])
    dups = (df.duplicated(subset=["ts","station_id"])
              .groupby(df["station_id"]).sum().astype(int)
              .rename_axis("station_id").reset_index(name="dups"))
    return dups.sort_values("dups", ascending=False)

def missing_bins(df: pd.DataFrame, current_days: int) -> pd.DataFrame:
    """Trous temporels par station (#bins manquants) sur la fenêtre récente."""
    if df.empty: return pd.DataFrame()
    tmax = df["ts"].max()
    tmin = tmax - pd.Timedelta(days=current_days)
    exp = _expected_bins(tmin, tmax)
    got = (df[(df["ts"] > tmin) & (df["ts"] <= tmax)]
             .groupby("station_id")["ts"].nunique()
             .rename_axis("station_id").reset_index(name="obs"))
    got["missing"] = (exp - got["obs"]).clip(lower=0)
    got["expected"] = exp
    return got.sort_values("missing", ascending=False)

# --------------------------- Plots ---------------------------

def plot_gauges(kpi_fresh: dict, kpi_cov: dict, slo_min: int) -> None:
    fig, ax = plt.subplots()
    labels = ["Fresh P50", "Fresh P95", "Coverage"]
    vals = [kpi_fresh.get("freshness_age_p50_min", np.nan),
            kpi_fresh.get("freshness_age_p95_min", np.nan),
            kpi_cov.get("coverage_global_pct", np.nan)]
    ax.bar(labels, vals)
    ax.axhline(slo_min, linestyle="--", linewidth=1)
    ax.set_ylabel("Minutes / Percent")
    ax.set_title("Data Health — Freshness & Coverage")
    _save_fig(FIGS_DIR / "gauges.png")

def plot_heatmap(heat: pd.DataFrame, max_rows: int = 120) -> None:
    if heat.empty:
        return
    # couper pour lisibilité
    if len(heat) > max_rows:
        heat = heat.iloc[:max_rows]
    fig, ax = plt.subplots(figsize=(10, max(4, len(heat)*0.08)))
    im = ax.imshow(heat.values, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels([str(c) for c in heat.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=7)
    ax.set_title("Completeness Heatmap — station × day")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    _save_fig(FIGS_DIR / "heatmap_completeness.png")

def plot_top_issues(st_health: pd.DataFrame) -> None:
    if st_health.empty: return
    df = st_health.copy()
    df["score"] = (100.0 - df["coverage_pct"]).fillna(0) + df.get("missing", 0).fillna(0)
    df = df.sort_values("score", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df["station_id"].astype(str), df["score"])
    ax.invert_yaxis()
    ax.set_xlabel("Issue score (higher=worse)")
    ax.set_title("Top issue stations")
    _save_fig(FIGS_DIR / "top_issues.png")

def plot_latency_hist(lat_df: Optional[pd.DataFrame]) -> None:
    if lat_df is None or lat_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(lat_df["latency_min"].dropna().values, bins=50)
    ax.set_xlabel("Latency (min)")
    ax.set_ylabel("Count")
    ax.set_title("Ingestion latency distribution")
    _save_fig(FIGS_DIR / "latency_hist.png")

# --------------------------- Main ---------------------------

def main(events_path: Path, current_days: int, tz: Optional[str], fresh_slo_min: int,
         dup_alert_pct: float, flat_steps: int, compl_alert_pct: float) -> None:
    _mkdirs()
    df = _read_events(events_path)

    # Info basic
    info = {
        "rows": int(len(df)),
        "stations": int(df["station_id"].nunique()),
        "span": [df["ts"].min().isoformat() if len(df) else None,
                 df["ts"].max().isoformat() if len(df) else None],
    }

    # KPIs
    kfresh = kpi_freshness(df, slo_min=fresh_slo_min)
    kcov, by_station_cov, heat = kpi_completeness(df, current_days=current_days)
    klat, lat_df = kpi_latency(df)

    # Anomalies : séquences plates
    flats = flat_sequences(df, min_steps=flat_steps, current_days=current_days)

    # Duplications
    dups = duplication_stats(df)

    # Missing bins (fenêtre récente)
    miss = missing_bins(df, current_days=current_days)

    # Station health table
    st_health = by_station_cov.merge(miss, on="station_id", how="left")
    st_health["missing"] = st_health["missing"].fillna(0).astype(int)
    st_health = st_health.sort_values("coverage_pct", ascending=True)

    # Plots
    try:
        plot_gauges(kfresh, kcov, fresh_slo_min)
        plot_heatmap(heat)
        plot_top_issues(st_health)
        plot_latency_hist(lat_df)
    except Exception as e:
        print(f"[warn] plotting failed: {e}")

    # Tables export
    data_health = {
        **info, **kfresh, **kcov, **klat,
        "current_days": int(current_days),
        "tz": tz or "UTC",
    }
    pd.DataFrame([data_health]).to_csv(TABLES_DIR / "data_health.csv", index=False)
    st_health.to_csv(TABLES_DIR / "station_health.csv", index=False)
    heat.to_csv(TABLES_DIR / "completeness_by_station_day.csv")
    # coverage by hour (depuis kpi_completeness)
    if current_days > 0 and len(df):
        tmax = df["ts"].max()
        tmin = tmax - pd.Timedelta(days=current_days)
        win = df[(df["ts"] > tmin) & (df["ts"] <= tmax)].copy()
        per_hour = win.copy()
        per_hour["hour"] = per_hour["ts"].dt.hour
        per_hour = (per_hour.groupby(["hour", "station_id"])["ts"].nunique()
                             .rename("obs").reset_index())
        exp_hour = current_days * 4
        per_hour["coverage_pct"] = per_hour["obs"].clip(upper=exp_hour) / float(exp_hour) * 100.0
        cov_by_hour = (per_hour.groupby("hour")["coverage_pct"].mean()
                              .rename_axis("hour").reset_index())
        cov_by_hour.to_csv(TABLES_DIR / "coverage_by_hour.csv", index=False)
    else:
        pd.DataFrame(columns=["hour","coverage_pct"]).to_csv(TABLES_DIR / "coverage_by_hour.csv", index=False)

    # Schema report
    sch = schema_report(df)
    sch.to_csv(TABLES_DIR / "schema_report.csv", index=False)

    # anomalies.csv (combine flats + dups + worst missing)
    anomalies = []

    if len(flats):
        for _, r in flats.iterrows():
            anomalies.append({
                "type": "flat_sequence",
                "station_id": r["station_id"],
                "start": r["start"].isoformat(),
                "end": r["end"].isoformat(),
                "steps": int(r["steps"]),
                "detail": f"flat for {int(r['steps'])}×15min"
            })

    if len(dups):
        top_dups = dups[dups["dups"] > 0].copy()
        for _, r in top_dups.iterrows():
            anomalies.append({
                "type": "duplicates",
                "station_id": r["station_id"],
                "dups": int(r["dups"]),
                "detail": f"{int(r['dups'])} duplicate rows on (ts,station_id)"
            })

    if len(miss):
        worst_miss = miss.sort_values("missing", ascending=False).head(50)
        for _, r in worst_miss.iterrows():
            if r["missing"] > 0:
                anomalies.append({
                    "type": "missing_bins",
                    "station_id": r["station_id"],
                    "missing": int(r["missing"]),
                    "expected": int(r["expected"]),
                    "detail": f"missing={int(r['missing'])}/{int(r['expected'])} bins in window"
                })

    pd.DataFrame(anomalies).to_csv(TABLES_DIR / "anomalies.csv", index=False)

    # seuils d’alerte globaux (cast → types natifs)
    alerts = {
        "freshness_p95_ok": (None if data_health.get("freshness_p95_ok") is None
                             else bool(data_health["freshness_p95_ok"])),
        "coverage_ok": bool(float(data_health.get("coverage_global_pct", 0.0)) >= float(compl_alert_pct)),
        "duplication_alert": (bool(float(dups["dups"].sum()) / max(1, len(df)) * 100.0 >= float(dup_alert_pct))
                              if len(dups) else False),
        "flat_sequences_found": bool(len(flats)),
    }
    with open(TABLES_DIR / "alerts.json", "w", encoding="utf-8") as f:
        json.dump(alerts, f, ensure_ascii=False, indent=2, default=_json_default)

    # Log récap
    print("[data-health]\n",
          json.dumps({
              "info": info,
              "kpis": {"freshness": kfresh, "coverage": kcov, "latency": klat},
              "alerts": alerts
          }, ensure_ascii=False, indent=2, default=_json_default))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True, help="Chemin du Parquet events")
    ap.add_argument("--current-days", type=int, default=7, help="Fenêtre récente (jours) pour la complétude")
    ap.add_argument("--tz", type=str, default="Europe/Paris", help="Fuseau pour l’affichage (info)")
    ap.add_argument("--fresh-slo-min", type=int, default=5, help="SLO fraîcheur (min, P95)")
    ap.add_argument("--dup-alert-pct", type=float, default=1.0, help="Seuil d’alerte duplications (%)")
    ap.add_argument("--flat-steps", type=int, default=8, help="Séquence min (pas de 15min) pour considérer une série plate")
    ap.add_argument("--compl-alert-pct", type=float, default=98.0, help="Seuil d’alerte de complétude globale (%)")
    args = ap.parse_args()

    main(
        events_path=args.events,
        current_days=args.current_days,
        tz=args.tz,
        fresh_slo_min=args.fresh_slo_min,
        dup_alert_pct=args.dup_alert_pct,
        flat_steps=args.flat_steps,
        compl_alert_pct=args.compl_alert_pct,
    )
