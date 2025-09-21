# tools/build_monitoring_data_health.py
# Page builder — "Monitoring / Santé des données"
#
# Produit :
# - KPIs de fraîcheur, complétude, latence d’ingestion (si dispo)
# - Rapport de schéma & contraintes (pass/warn/fail)
# - Tables : data_health.csv (KPIs), station_health.csv (par station),
#            completeness_by_station_day.csv, coverage_by_hour.csv,
#            schema_report.csv, anomalies.csv, alerts.json
# - Figures : gauges (barres) fraîcheur/complétude, heatmap complétude (stations × jours),
#             top stations problématiques (barres), distribution de latence (si dispo)
# - Markdown : docs/monitoring/data-health.md (rendu complet)
#
# CLI :
#   python tools/build_monitoring_data_health.py --events docs/exports/events.parquet \
#       --current-days 7 --tz Europe/Paris --fresh-slo-min 5 --flat-steps 6 --bin-min 5
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

import math
from pathlib import Path
import argparse
import json
from typing import Optional, Tuple

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
OUT_MD = DOCS / "monitoring" / "data-health.md"

plt.rcParams["figure.figsize"] = (9, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

for d in (FIGS_DIR, TABLES_DIR, OUT_MD.parent):
    d.mkdir(parents=True, exist_ok=True)

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

def rel_from_md(md_path: Path, target: Path) -> str:
    """Chemin relatif (POSIX) depuis md_path vers target, compatible MkDocs (use_directory_urls: true)."""
    md_rel = Path(md_path).resolve().relative_to(DOCS.resolve())
    parts = md_rel.with_suffix("").parts
    depth = len(parts) if parts[-1] != "index" else len(parts) - 1
    prefix = "../" * max(depth, 0)
    rel_from_docs = Path(target).resolve().relative_to(DOCS.resolve()).as_posix()
    return (prefix + rel_from_docs).replace("//", "/")

def _mkdirs() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

def _save_fig(path: Path) -> None:
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()

def _now_utc_floor(bin_min: int) -> pd.Timestamp:
    # UTC → drop tz ⇒ UTC naïf (compatibles avec df["ts"])
    rule = f"{int(bin_min)}min"
    return pd.Timestamp.now(tz="UTC").floor(rule).tz_localize(None)

def _expected_bins(tmin: pd.Timestamp, tmax: pd.Timestamp, bin_min: int) -> int:
    return max(1, math.ceil((tmax - tmin).total_seconds() / 60 / float(bin_min)) + 1)

def _floor_series(s: pd.Series, bin_min: int) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True).dt.floor(f"{int(bin_min)}min").dt.tz_localize(None)

def _read_events(path: Path, bin_min: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[data-health] Introuvable: {path}")
    df = pd.read_parquet(path)

    # ts → UTC naïf
    tcol = next((c for c in ("ts", "tbin_utc", "timestamp") if c in df.columns), None)
    if tcol is None:
        raise KeyError("[data-health] Colonne temporelle manquante (ts/tbin_utc/timestamp)")
    df["ts"] = _floor_series(df[tcol], bin_min)

    # station_id
    sid = next((c for c in ("station_id", "stationcode", "station") if c in df.columns), None)
    if sid is None:
        raise KeyError("[data-health] Identifiant station manquant (station_id/stationcode)")
    df["station_id"] = df[sid].astype(str)

    # core numeric
    bikes_col = next((c for c in ("bikes", "nb_velos_bin", "velos_disponibles", "numBikesAvailable") if c in df.columns), None)
    if bikes_col is None:
        raise KeyError("[data-health] Colonne vélos manquante (bikes/nb_velos_bin/velos_disponibles)")
    df["bikes"] = pd.to_numeric(df[bikes_col], errors="coerce")

    docks_col = next((c for c in ("docks_avail", "nb_docks_bin", "numDocksAvailable") if c in df.columns), None)
    df["docks_avail"] = pd.to_numeric(df.get(docks_col, np.nan), errors="coerce")

    df["capacity"] = pd.to_numeric(df.get("capacity", np.nan), errors="coerce")

    # ingestion timestamp (optionnel)
    ing_col = next((c for c in ("ingested_at", "ingest_ts", "ingest_time", "received_at", "created_at", "etl_ts", "load_ts") if c in df.columns), None)
    if ing_col is None:
        df["ingested_at"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    else:
        df["ingested_at"] = pd.to_datetime(df[ing_col], errors="coerce", utc=True).dt.tz_localize(None)

    # coords (optionnelles)
    if "lat" in df.columns: df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    if "lon" in df.columns: df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    keep = ["ts", "station_id", "bikes", "docks_avail", "capacity", "ingested_at"]
    for c in ("lat", "lon", "name"):
        if c in df.columns: keep.append(c)
    return df[keep].copy()

# --------------------------- KPIs ---------------------------

def kpi_freshness(df: pd.DataFrame, slo_min: int, bin_min: int) -> dict:
    """Âge du dernier point par station, puis P50/P95 (minutes) vs SLO."""
    now = _now_utc_floor(bin_min)
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
        "bin_min": int(bin_min),
        "freshness_p95_ok": (p95 <= slo_min) if np.isfinite(p95) else None
    }

def kpi_completeness(
    df: pd.DataFrame,
    current_days: int,
    bin_min: int,
    tz: Optional[str]
) -> Tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Complétude globale & par station sur fenêtre récente; heatmap station×jour; couverture par heure."""
    if current_days <= 0 or df.empty:
        return {"coverage_global_pct": np.nan}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    tmax = df["ts"].max()
    tmin = tmax - pd.Timedelta(days=current_days)
    win = df[(df["ts"] > tmin) & (df["ts"] <= tmax)].copy()
    if win.empty:
        return {"coverage_global_pct": np.nan}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    exp = _expected_bins(tmin, tmax, bin_min)
    per_station = (win.groupby("station_id")["ts"].nunique()
                      .clip(upper=exp)
                      .rename_axis("station_id").reset_index(name="obs"))
    per_station["expected"] = exp
    per_station["coverage_pct"] = per_station["obs"] / per_station["expected"] * 100.0
    coverage_global = float(per_station["coverage_pct"].mean()) if len(per_station) else np.nan

    # coverage by hour of day (0..23) across stations
    if tz:
        # ts est naïf UTC → on le localise en UTC puis convertit en tz souhaité
        win["hour"] = pd.to_datetime(win["ts"]).dt.tz_localize("UTC").dt.tz_convert(tz).dt.hour
    else:
        # Sinon, rester en UTC
        win["hour"] = pd.to_datetime(win["ts"]).dt.hour

    per_hour = (win.groupby(["hour", "station_id"])["ts"].nunique()
                  .rename("obs").reset_index())
    # expected bins per hour per station: #days * (60/bin_min) bins/hour
    exp_hour = current_days * (60 // max(1, int(bin_min)))
    per_hour["coverage_pct"] = per_hour["obs"].clip(upper=exp_hour) / float(exp_hour) * 100.0
    cov_by_hour = (per_hour.groupby("hour")["coverage_pct"].mean()
                          .rename_axis("hour").reset_index())

    # station×day heatmap: fraction of present bins per day
    win["date"] = win["ts"].dt.date
    per_sd = win.groupby(["station_id","date"])["ts"].nunique().reset_index(name="obs")
    per_sd["expected"] = 24 * (60 // max(1, int(bin_min)))  # bins per day
    per_sd["coverage_pct"] = per_sd["obs"].clip(upper=per_sd["expected"]) / per_sd["expected"] * 100.0
    heat = per_sd.pivot(index="station_id", columns="date", values="coverage_pct").fillna(0.0)

    return {"coverage_global_pct": round(coverage_global, 2)}, per_station, heat, cov_by_hour

def kpi_latency(df: pd.DataFrame) -> Tuple[dict, Optional[pd.DataFrame]]:
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

def schema_report(df: pd.DataFrame, bin_min: int) -> pd.DataFrame:
    """Vérifie un schéma minimal & contraintes simples."""
    rep = []
    def add(name, status, detail=""):
        rep.append({"check": name, "status": status, "detail": detail})

    # colonnes obligatoires
    for col in ("ts", "station_id", "bikes"):
        add(f"col:{col}", "pass" if col in df.columns else "fail")

    # types
    add("type:ts:datetime64", "pass" if np.issubdtype(df["ts"].dtype, np.datetime64) else "fail")
    add("type:station_id:str", "pass" if (df["station_id"].dtype == object or str(df["station_id"].dtype).startswith("string")) else "warn")
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
        exp = _expected_bins(tmin, tmax, bin_min=bin_min)
        got = int(df["ts"].nunique())
        ratio = got / float(exp)
        add("temporal_coverage", "pass" if ratio >= 0.9 else "warn", f"bins={got}/{exp} ({ratio:.2%})")

    return pd.DataFrame(rep)

def flat_sequences(df: pd.DataFrame, min_steps: int, current_days: int, bin_min: int) -> pd.DataFrame:
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
        .reset_index()
    )

    out = agg[agg["steps"] >= min_steps][["station_id", "steps", "start", "end"]]
    out = out.sort_values(["steps"], ascending=False).reset_index(drop=True)
    # Ajoute la durée en minutes pour lecture rapide
    out["duration_min"] = (out["steps"] * bin_min).astype(int)
    return out

def duplication_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Comptage de doublons (ts, station_id)."""
    if df.empty: return pd.DataFrame(columns=["station_id","dups"])
    dups = (df.duplicated(subset=["ts","station_id"])
              .groupby(df["station_id"]).sum().astype(int)
              .rename_axis("station_id").reset_index(name="dups"))
    return dups.sort_values("dups", ascending=False)

def missing_bins(df: pd.DataFrame, current_days: int, bin_min: int) -> pd.DataFrame:
    """Trous temporels par station (#bins manquants) sur la fenêtre récente."""
    if df.empty: return pd.DataFrame()
    tmax = df["ts"].max()
    tmin = tmax - pd.Timedelta(days=current_days)
    exp = _expected_bins(tmin, tmax, bin_min)
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

# --------------------------- Markdown template ---------------------------

MD_TEMPLATE = """# Santé des données

Cette page vérifie que le **pipeline d’ingestion** fournit des données **fraîches, complètes et conformes** au contrat attendu.

## Objectif
Vérifier que le **pipeline d’ingestion** fournit des données **fraîches, complètes et conformes** au contrat attendu.

## Questions auxquelles la page répond
- Les données sont-elles **fraîches** (latence nominale respectée) ?
- Quel est le **taux de complétude** (stations × timestamps) et où sont les trous ?
- Le **schéma** (colonnes/types/unités) est-il conforme ? Y a-t-il des champs anormaux (valeurs négatives, hors bornes, constantes) ?
- Observe-t-on des **ruptures** (station figée, série plate, duplication, horodatages non monotones) ?

## Indicateurs clés (KPIs)
- **Fraîcheur** : P50={fresh_p50:.2f} min · P95={fresh_p95:.2f} min vs SLO={slo_min:.0f} min → {fresh_status}
- **Complétude (fenêtre {current_days} j)** : {coverage_global:.2f}% (moyenne stations)
- **Latence d’ingestion** : médiane={lat_p50} min · P95={lat_p95} min
- **Schéma & contraintes** : {schema_pass} pass · {schema_warn} warn · {schema_fail} fail
- **Anomalies** : doublons={dups_pct:.2f}% · séquences plates≥{flat_steps} pas ({bin_min} min/pt)={flat_count} · stations avec trous={missing_stations}

> Snapshot UTC : **{now_utc}** · Fenêtre récente : **{current_days}** jours · **Pas** : **{bin_min} min**

---

## Visualisations
### Jauges fraîcheur & complétude
![gauges]({gauges_rel})

### Heatmap de complétude par station/jour
![heatmap]({heatmap_rel})

### Top stations problématiques
![top issues]({top_issues_rel})

### Distribution de la latence d’ingestion
{latency_block}

---

## Seuils & Alertes
- **Fraîcheur P95 ≤ SLO** : {fresh_alert}
- **Complétude ≥ {compl_alert_pct:.0f}%** : {cov_alert}
- **Timestamps dupliqués ≤ {dup_alert_pct:.1f}%** : {dup_alert}
- **Séries plates détectées** : {flat_alert}

Fichier JSON : `{alerts_rel}`

---

## Méthodes & règles
- **Contrat de schéma** : colonnes minimales, types et bornes souples (warnings) / dures (erreurs).
- **Détection plateaux** : séquence ≥ **{flat_steps}** pas de **{bin_min} min** → alerte station.
- **Contrôle temps** : pas manquants, pas dupliqués, dérive/retard d’horloge.
- **Cartographie** : heatmap complétude (stations en lignes × jours en colonnes).
- **Couverture par heure** : moyenne de la couverture (toutes stations) par heure locale sur {current_days} jours.

---

## Tables d’appui
- KPIs globaux : `{data_health_csv_rel}`
- Détail par station (complétude, trous) : `{station_health_csv_rel}`
- Heatmap station×jour (CSV) : `{heatmap_csv_rel}`
- Couverture moyenne par heure : `{cov_by_hour_csv_rel}`
- Rapport de schéma : `{schema_report_csv_rel}`
- Anomalies : `{anomalies_csv_rel}`

---

## Limites
La qualité “technique” n’implique pas la **représentativité** (couverte par la page *Drift*). Les estimations de latence ne sont disponibles que si `ingested_at` est fourni.

"""

# --------------------------- Main ---------------------------

def main(events_path: Path, current_days: int, tz: Optional[str], fresh_slo_min: int,
         dup_alert_pct: float, flat_steps: int, compl_alert_pct: float, bin_min: int) -> None:
    _mkdirs()
    df = _read_events(events_path, bin_min=bin_min)

    # Info basic
    info = {
        "rows": int(len(df)),
        "stations": int(df["station_id"].nunique()),
        "span": [df["ts"].min().isoformat() if len(df) else None,
                 df["ts"].max().isoformat() if len(df) else None],
        "bin_min": int(bin_min),
    }

    # KPIs
    kfresh = kpi_freshness(df, slo_min=fresh_slo_min, bin_min=bin_min)
    kcov, by_station_cov, heat, cov_by_hour = kpi_completeness(
        df, current_days=current_days, bin_min=bin_min, tz=tz
    )
    klat, lat_df = kpi_latency(df)

    # Anomalies : séquences plates
    flats = flat_sequences(df, min_steps=flat_steps, current_days=current_days, bin_min=bin_min)

    # Duplications
    dups = duplication_stats(df)
    dup_total = int(dups["dups"].sum()) if len(dups) else 0
    dups_pct = (dup_total / max(1, len(df))) * 100.0 if len(df) else 0.0

    # Missing bins (fenêtre récente)
    miss = missing_bins(df, current_days=current_days, bin_min=bin_min)
    missing_stations = int((miss["missing"] > 0).sum()) if len(miss) else 0

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
    (TABLES_DIR / "dummy").parent.mkdir(parents=True, exist_ok=True)
    data_health_csv = TABLES_DIR / "data_health.csv"
    pd.DataFrame([data_health]).to_csv(data_health_csv, index=False)

    station_health_csv = TABLES_DIR / "station_health.csv"
    st_health.to_csv(station_health_csv, index=False)

    heatmap_csv = TABLES_DIR / "completeness_by_station_day.csv"
    heat.to_csv(heatmap_csv)

    cov_by_hour_csv = TABLES_DIR / "coverage_by_hour.csv"
    if not cov_by_hour.empty:
        cov_by_hour.to_csv(cov_by_hour_csv, index=False)
    else:
        pd.DataFrame(columns=["hour","coverage_pct"]).to_csv(cov_by_hour_csv, index=False)

    # Schema report
    sch = schema_report(df, bin_min=bin_min)
    schema_report_csv = TABLES_DIR / "schema_report.csv"
    sch.to_csv(schema_report_csv, index=False)

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
                "duration_min": int(r["duration_min"]),
                "detail": f"flat for {int(r['steps'])}×{bin_min}min"
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

    anomalies_csv = TABLES_DIR / "anomalies.csv"
    pd.DataFrame(anomalies).to_csv(anomalies_csv, index=False)

    # seuils d’alerte globaux (cast → types natifs)
    alerts = {
        "freshness_p95_ok": (None if data_health.get("freshness_p95_ok") is None
                             else bool(data_health["freshness_p95_ok"])),
        "coverage_ok": bool(float(data_health.get("coverage_global_pct", 0.0)) >= float(compl_alert_pct)),
        "duplication_alert": (bool(float(dups["dups"].sum()) / max(1, len(df)) * 100.0 >= float(dup_alert_pct))
                              if len(dups) else False),
        "flat_sequences_found": bool(len(flats)),
    }
    alerts_json = TABLES_DIR / "alerts.json"
    with open(alerts_json, "w", encoding="utf-8") as f:
        json.dump(alerts, f, ensure_ascii=False, indent=2, default=_json_default)

    # Log récap
    print("[data-health]\n",
          json.dumps({
              "info": info,
              "kpis": {"freshness": kfresh, "coverage": kcov, "latency": klat},
              "alerts": alerts
          }, ensure_ascii=False, indent=2, default=_json_default))

    # --------------------- Rendu Markdown ---------------------
    schema_pass = int((sch["status"] == "pass").sum())
    schema_warn = int((sch["status"] == "warn").sum())
    schema_fail = int((sch["status"] == "fail").sum())

    gauges_rel = rel_from_md(OUT_MD, FIGS_DIR / "gauges.png")
    heatmap_rel = rel_from_md(OUT_MD, FIGS_DIR / "heatmap_completeness.png")
    top_issues_rel = rel_from_md(OUT_MD, FIGS_DIR / "top_issues.png")
    latency_rel = rel_from_md(OUT_MD, FIGS_DIR / "latency_hist.png")

    data_health_csv_rel = rel_from_md(OUT_MD, data_health_csv)
    station_health_csv_rel = rel_from_md(OUT_MD, station_health_csv)
    heatmap_csv_rel = rel_from_md(OUT_MD, heatmap_csv)
    cov_by_hour_csv_rel = rel_from_md(OUT_MD, cov_by_hour_csv)
    schema_report_csv_rel = rel_from_md(OUT_MD, schema_report_csv)
    anomalies_csv_rel = rel_from_md(OUT_MD, anomalies_csv)
    alerts_rel = rel_from_md(OUT_MD, alerts_json)

    fresh_status = "✅ OK" if alerts["freshness_p95_ok"] is True else ("❌ Hors SLO" if alerts["freshness_p95_ok"] is False else "n.d.")
    fresh_alert = "✅ Conforme" if alerts["freshness_p95_ok"] else ("❌ Non conforme" if alerts["freshness_p95_ok"] is False else "n.d.")
    cov_alert = "✅ Conforme" if alerts["coverage_ok"] else "❌ Non conforme"
    dup_alert = "✅ OK" if not alerts["duplication_alert"] else "❌ Trop de doublons"
    flat_alert = "❌ Oui" if alerts["flat_sequences_found"] else "✅ Non"

    latency_block = (f"![latency]({latency_rel})"
                     if (FIGS_DIR / "latency_hist.png").exists()
                     else "_Latence non disponible (pas de `ingested_at`)._")

    md = MD_TEMPLATE.format(
        fresh_p50=(kfresh.get("freshness_age_p50_min", float("nan")) or float("nan")),
        fresh_p95=(kfresh.get("freshness_age_p95_min", float("nan")) or float("nan")),
        slo_min=(kfresh.get("freshness_slo_min", float("nan")) or float("nan")),
        fresh_status=fresh_status,
        coverage_global=(kcov.get("coverage_global_pct", float("nan")) or float("nan")),
        lat_p50=("n.d." if pd.isna(klat.get("latency_p50_min", np.nan)) else f"{klat['latency_p50_min']:.2f}"),
        lat_p95=("n.d." if pd.isna(klat.get("latency_p95_min", np.nan)) else f"{klat['latency_p95_min']:.2f}"),
        schema_pass=schema_pass, schema_warn=schema_warn, schema_fail=schema_fail,
        dups_pct=dups_pct,
        flat_steps=flat_steps,
        bin_min=bin_min,
        flat_count=int(len(flats)) if len(flats) else 0,
        missing_stations=missing_stations,
        now_utc=kfresh.get("now_utc", ""),
        current_days=current_days,
        gauges_rel=gauges_rel,
        heatmap_rel=heatmap_rel,
        top_issues_rel=top_issues_rel,
        latency_block=latency_block,
        fresh_alert=fresh_alert,
        compl_alert_pct=compl_alert_pct,
        cov_alert=cov_alert,
        dup_alert_pct=dup_alert_pct,
        dup_alert=dup_alert,
        flat_alert=flat_alert,
        alerts_rel=alerts_rel,
        data_health_csv_rel=data_health_csv_rel,
        station_health_csv_rel=station_health_csv_rel,
        heatmap_csv_rel=heatmap_csv_rel,
        cov_by_hour_csv_rel=cov_by_hour_csv_rel,
        schema_report_csv_rel=schema_report_csv_rel,
        anomalies_csv_rel=anomalies_csv_rel,
    )

    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)

    print(f"[data-health] markdown -> {OUT_MD}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True, help="Chemin du Parquet events")
    ap.add_argument("--current-days", type=int, default=7, help="Fenêtre récente (jours) pour la complétude")
    ap.add_argument("--tz", type=str, default="Europe/Paris", help="Fuseau pour l’affichage (info)")
    ap.add_argument("--fresh-slo-min", type=int, default=5, help="SLO fraîcheur (min, P95)")
    ap.add_argument("--dup-alert-pct", type=float, default=1.0, help="Seuil d’alerte duplications (%)")
    ap.add_argument("--flat-steps", type=int, default=6, help="Séquence min (en pas) pour considérer une série plate")
    ap.add_argument("--compl-alert-pct", type=float, default=98.0, help="Seuil d’alerte de complétude globale (%)")
    ap.add_argument("--bin-min", type=int, default=5, help="Taille du pas temporel (minutes)")
    args = ap.parse_args()

    main(
        events_path=args.events,
        current_days=args.current_days,
        tz=args.tz,
        fresh_slo_min=args.fresh_slo_min,
        dup_alert_pct=args.dup_alert_pct,
        flat_steps=args.flat_steps,
        compl_alert_pct=args.compl_alert_pct,
        bin_min=args.bin_min,
    )
