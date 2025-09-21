# tools/build_data_dictionary.py
# -----------------------------------------------------------------------------
# Données — Page "Data / Dictionnaire & schéma"
#
# Rôle
# ----
# Produire un **dictionnaire formel** (noms, types pandas/SQL, unités/domaines,
# description, obligatoire, défaut, exemples), comparer aux **schémas réels**,
# exécuter des **règles de validation** et générer la page Markdown.
#
# Entrées
# -------
# - `docs/exports/events.parquet`
# - `docs/exports/perf.parquet` (optionnel)
#
# Sorties
# -------
# - `docs/assets/tables/data/dictionary/*.csv|json`
# - `docs/assets/figs/data/dictionary/*.png`
# - `docs/data/dictionary.md`
#
# Notes
# -----
# - Cadence temporelle : pas **5 minutes** (arrondi des timestamps).
# - Clé primaire attendue : `(ts, station_id)` unique dans les deux exports.
#
# CLI
# ---
# python tools/build_data_dictionary.py \
#   --exports-dir docs/exports --occ-tol 0.05 --ts-mode aware
# # ou
# python tools/build_data_dictionary.py \
#   --events docs/exports/events.parquet --perf docs/exports/perf.parquet \
#   --occ-tol 0.05 --ts-mode aware
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------- Paths ---------------------------

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
TABLES_DIR = ASSETS / "tables" / "data" / "dictionary"
FIGS_DIR = ASSETS / "figs" / "data" / "dictionary"
OUT_MD = DOCS / "data" / "dictionary.md"

# --------------------------- Utils ---------------------------

def _mkdirs() -> None:
    for d in (TABLES_DIR, FIGS_DIR, OUT_MD.parent):
        d.mkdir(parents=True, exist_ok=True)

def _now_iso() -> str:
    # horodatage UTC conscient du fuseau (pour traçabilité)
    return pd.Timestamp.now(tz="UTC").isoformat()

def _read_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[dictionary] Fichier introuvable: {path}")
    return pd.read_parquet(path)

def _coerce_ts_5m(df: pd.DataFrame, col: str, ts_mode: str) -> pd.DataFrame:
    """Normalise la colonne temporelle au pas 5 min.
       - ts_mode='aware'  → datetime64[ns, UTC]
       - ts_mode='naive'  → datetime64[ns] (UTC implicite)"""
    if col not in df.columns:
        return df
    df = df.copy()
    s = pd.to_datetime(df[col], errors="coerce", utc=True).dt.floor("5min")
    if ts_mode == "naive":
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    df[col] = s
    return df

def _ensure_station_str(df: pd.DataFrame) -> pd.DataFrame:
    for alt in ("stationcode", "stationCode", "id"):
        if "station_id" not in df.columns and alt in df.columns:
            df = df.rename(columns={alt: "station_id"})
    if "station_id" in df.columns:
        df["station_id"] = df["station_id"].astype(str)
    return df

def _schema(df: pd.DataFrame) -> pd.DataFrame:
    return (pd.DataFrame({"column": df.columns, "dtype_pandas": [str(t) for t in df.dtypes]})
            .sort_values("column").reset_index(drop=True))

def _pandas_to_sql(dtype_str: str) -> str:
    d = dtype_str.lower()
    if "datetime64" in d:
        return "timestamp with time zone" if "utc" in d else "timestamp without time zone"
    if "int" in d:
        return "bigint"
    if "float" in d or "double" in d:
        return "double precision"
    if "bool" in d:
        return "boolean"
    return "text"

def _examples(df: pd.DataFrame, cols: List[str], n: int = 5) -> pd.DataFrame:
    keep = [c for c in cols if c in df.columns]
    if not keep:
        return pd.DataFrame(columns=cols)
    return df[keep].head(n).copy()

def rel_from_md(md_path: Path, target: Path) -> str:
    """Chemin relatif POSIX depuis md_path vers target (compatible MkDocs)."""
    md_rel = Path(md_path).resolve().relative_to(DOCS.resolve())
    parts = md_rel.with_suffix("").parts
    depth = len(parts) if parts[-1] != "index" else len(parts) - 1
    prefix = "../" * max(depth, 0)
    rel_from_docs = Path(target).resolve().relative_to(DOCS.resolve()).as_posix()
    return (prefix + rel_from_docs).replace("//", "/")

# --------------------------- Canonical dictionaries ---------------------------

def _ts_desc(ts_mode: str) -> tuple[str, str]:
    if ts_mode == "naive":
        return ("datetime64[ns]", "timestamp without time zone")
    return ("datetime64[ns, UTC]", "timestamp with time zone")

def _dictionary_events(ts_mode: str) -> pd.DataFrame:
    ts_pandas, ts_sql = _ts_desc(ts_mode)
    rows = [
        ("ts", ts_pandas, ts_sql,
         "UTC, pas 5 min (xx:00, :05, :10, …)",
         f"Horodatage du *bin source* ({'UTC naïf' if ts_mode=='naive' else 'UTC tz-aware'}), arrondi 5 min.",
         True, None, "2025-01-02 08:15"),
        ("station_id", "string", "text",
         "Identifiant station stable", "Identifiant canonique de la station (chaîne).",
         True, None, "3005"),
        ("bikes", "int64", "bigint",
         "≥ 0, vélos disponibles", "Nombre de vélos disponibles à T.", True, 0, "0, 7, 15"),
        ("capacity", "int64", "bigint",
         "≥ 0, docks totaux (estimés)", "Capacité totale estimée (si connue).", False, None, "20, 35"),
        ("occ", "float64", "double precision",
         "[0,1], ratio d’occupation", "Rapport *bikes/capacity* si *capacity*>0.", False, None, "0.35, 0.80"),
        ("name", "string", "text", "Texte libre", "Nom d’affichage (optionnel).", False, None, "Rivoli - Pont Neuf"),
        ("lat", "float64", "double precision", "WGS84", "Latitude (WGS84).", False, None, "48.859"),
        ("lon", "float64", "double precision", "WGS84", "Longitude (WGS84).", False, None, "2.347"),
        ("hour_utc", "int64", "bigint", "Entier 0–23", "Heure UTC extraite de ts (optionnel).", False, None, "8,17,23"),
        ("temp_C", "float64", "double precision", "°C", "Température (si intégrée).", False, None, "12.3"),
        ("precip_mm", "float64", "double precision", "mm", "Précipitations (si intégrées).", False, None, "0.4"),
        ("wind_mps", "float64", "double precision", "m/s", "Vent (si intégré).", False, None, "3.2"),
    ]
    return pd.DataFrame(rows, columns=[
        "name","type_pandas","type_sql","unit_or_domain",
        "description","required","default","examples"
    ])

def _dictionary_perf(ts_mode: str) -> pd.DataFrame:
    ts_pandas, ts_sql = _ts_desc(ts_mode)
    rows = [
        ("ts", ts_pandas, ts_sql,
         "UTC, pas 5 min (xx:00, :05, :10, …)",
         f"Horodatage *T* (même bin que events, {'UTC naïf' if ts_mode=='naive' else 'UTC tz-aware'}).",
         True, None, "2025-01-02 08:15"),
        ("station_id", "string", "text",
         "Identifiant station stable", "Identifiant canonique (chaîne).", True, None, "3005"),
        ("y_true", "float64", "double precision",
         "≥ 0", "Cible observée à T+h, ramenée à T (shift(-steps)).", True, None, "5, 12"),
        ("y_pred_baseline", "float64", "double precision",
         "≥ 0", "Baseline persistance (valeur observée à T).", True, None, "7, 9"),
        ("y_pred", "float64", "double precision",
         "≥ 0", "Prédiction du modèle alignée sur T (optionnelle).", False, None, "8.4"),
        ("horizon_min", "int64", "bigint",
         "> 0", "Horizon en minutes (ex. 15).", False, None, "15"),
    ]
    return pd.DataFrame(rows, columns=[
        "name","type_pandas","type_sql","unit_or_domain",
        "description","required","default","examples"
    ])

# --------------------------- Validation rules & run ---------------------------

def _validation_rules(ts_mode: str) -> pd.DataFrame:
    ts_expect = "datetime64 (naïf UTC)" if ts_mode=="naive" else "datetime64 (UTC tz-aware)"
    rows = [
        ("E_COL_TS","events","hard","column_exists(ts)","Colonne ts présente"),
        ("E_COL_STID","events","hard","column_exists(station_id)","Colonne station_id présente"),
        ("E_COL_BIKES","events","hard","column_exists(bikes)","Colonne bikes présente"),
        ("E_TYPE_TS","events","hard",f"dtype(ts) is {ts_expect}", f"ts de type attendu ({ts_expect})"),
        ("E_NUM_BIKES","events","hard","numeric(bikes) & bikes>=0","bikes numérique, non négatif"),
        ("E_RANGE_OCC","events","soft","0<=occ<=1 (si présent)","occ borné à [0,1]"),
        ("E_RANGE_CAP","events","soft","capacity>=0 (si présent)","capacity non négatif"),
        ("E_OCC_FORMULA","events","soft","median(|occ-(bikes/capacity)|) <= tol (95% lignes avec capacity>0)","occ≈bikes/capacity (tolérance)"),
        ("E_KEY_UNIQ","events","hard","unique(ts,station_id)","Clé (ts,station_id) unique"),
        ("E_STEP_5M","events","soft","ts minute%5==0","Horodatages alignés (pas 5 min)"),

        ("P_COL_TS","perf","hard","column_exists(ts)","Colonne ts présente"),
        ("P_COL_STID","perf","hard","column_exists(station_id)","Colonne station_id présente"),
        ("P_COL_YTRUE","perf","hard","column_exists(y_true)","Colonne y_true présente"),
        ("P_COL_BASE","perf","hard","column_exists(y_pred_baseline)","Colonne y_pred_baseline présente"),
        ("P_TYPE_TS","perf","hard",f"dtype(ts) is {ts_expect}", f"ts de type attendu ({ts_expect})"),
        ("P_NUM_NONNEG","perf","soft","y_true,y_pred_baseline,y_pred >= 0 (si présents)","Non négatifs"),
        ("P_HORIZON_GT0","perf","soft","horizon_min>0 (si présent)","Horizon positif"),
        ("P_KEY_UNIQ","perf","hard","unique(ts,station_id)","Clé (ts,station_id) unique"),
        ("P_STEP_5M","perf","soft","ts minute%5==0","Horodatages alignés (pas 5 min)"),
    ]
    return pd.DataFrame(rows, columns=["id","applies_to","severity","rule","description"])

def _validate_events(df: pd.DataFrame, occ_tol: float, ts_mode: str) -> pd.DataFrame:
    checks = []
    def add(id_, status, detail=None): checks.append({"id": id_, "status": status, "detail": detail})

    add("E_COL_TS", "pass" if "ts" in df.columns else "fail")
    add("E_COL_STID", "pass" if "station_id" in df.columns else "fail")
    add("E_COL_BIKES", "pass" if "bikes" in df.columns else "fail")

    if "ts" in df.columns and pd.api.types.is_datetime64_any_dtype(df["ts"]):
        tz = getattr(df["ts"].dt.tz, "zone", None)
        ok_ts = (tz is None) if ts_mode=="naive" else (tz is not None)
        add("E_TYPE_TS", "pass" if ok_ts else "fail", detail=("tz-aware" if tz else "tz-naive"))
        mis = (df["ts"].dt.minute % 5 != 0).mean() * 100.0
        add("E_STEP_5M", "pass" if mis == 0 else "warn", detail=f"misaligned_pct={mis:.3f}%")
    else:
        add("E_TYPE_TS","fail","non-datetime")

    if "bikes" in df.columns:
        bikes_numeric = pd.api.types.is_numeric_dtype(df["bikes"])
        nonneg_share = (df["bikes"].dropna() >= 0).mean() * 100.0 if bikes_numeric else 0.0
        add("E_NUM_BIKES", "pass" if (bikes_numeric and nonneg_share == 100.0) else "fail",
            detail=f"share_nonneg={nonneg_share:.2f}%")

    if "occ" in df.columns and pd.api.types.is_numeric_dtype(df["occ"]):
        share_in_range = ((df["occ"].between(0.0, 1.0)) | (df["occ"].isna())).mean() * 100.0
        add("E_RANGE_OCC", "pass" if share_in_range >= 99.0 else "warn", detail=f"share_in_range={share_in_range:.2f}%")
    else:
        add("E_RANGE_OCC","n/a","colonne absente")

    if "capacity" in df.columns and pd.api.types.is_numeric_dtype(df["capacity"]):
        cap_nonneg = (df["capacity"].dropna() >= 0).mean() * 100.0
        add("E_RANGE_CAP", "pass" if cap_nonneg >= 99.0 else "warn", detail=f"share_nonneg={cap_nonneg:.2f}%")
        sub = df.dropna(subset=["bikes","capacity"]).copy()
        sub = sub[sub["capacity"] > 0]
        if "occ" in df.columns and not sub.empty:
            occ_calc = (sub["bikes"].clip(lower=0) / sub["capacity"]).clip(0,1)
            med_abs = float(np.nanmedian(np.abs(occ_calc - sub["occ"])))
            share_ok = float((np.abs(occ_calc - sub["occ"]) <= occ_tol).mean() * 100.0)
            status = "pass" if (med_abs <= occ_tol and share_ok >= 95.0) else "warn"
            add("E_OCC_FORMULA", status, detail=f"median_abs_diff={med_abs:.3f}; share<=tol={share_ok:.1f}%")
        else:
            add("E_OCC_FORMULA","n/a","données insuffisantes")
    else:
        add("E_RANGE_CAP","n/a","colonne absente")
        add("E_OCC_FORMULA","n/a","capacity absente")

    if set(("ts","station_id")).issubset(df.columns):
        dup_pct = df.duplicated(subset=["ts","station_id"]).mean() * 100.0
        add("E_KEY_UNIQ", "pass" if dup_pct == 0 else "fail", detail=f"duplicated_pct={dup_pct:.3f}%")
    else:
        add("E_KEY_UNIQ","fail","colonnes manquantes")
    return pd.DataFrame(checks)

def _validate_perf(df: pd.DataFrame, ts_mode: str) -> pd.DataFrame:
    checks = []
    def add(id_, status, detail=None): checks.append({"id": id_, "status": status, "detail": detail})

    add("P_COL_TS", "pass" if "ts" in df.columns else "fail")
    add("P_COL_STID", "pass" if "station_id" in df.columns else "fail")
    add("P_COL_YTRUE", "pass" if "y_true" in df.columns else "fail")
    add("P_COL_BASE", "pass" if "y_pred_baseline" in df.columns else "fail")

    if "ts" in df.columns and pd.api.types.is_datetime64_any_dtype(df["ts"]):
        tz = getattr(df["ts"].dt.tz, "zone", None)
        ok_ts = (tz is None) if ts_mode=="naive" else (tz is not None)
        add("P_TYPE_TS", "pass" if ok_ts else "fail", detail=("tz-aware" if tz else "tz-naive"))
        mis = (df["ts"].dt.minute % 5 != 0).mean() * 100.0
        add("P_STEP_5M", "pass" if mis == 0 else "warn", detail=f"misaligned_pct={mis:.3f}%")
    else:
        add("P_TYPE_TS","fail","non-datetime")

    for c in ("y_true","y_pred_baseline","y_pred"):
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            nonneg = (df[c].dropna() >= 0).mean() * 100.0
            add("P_NUM_NONNEG", "pass" if nonneg >= 99.0 else "warn", detail=f"{c}: share_nonneg={nonneg:.2f}%")
    if "horizon_min" in df.columns:
        pos = (pd.to_numeric(df["horizon_min"], errors="coerce") > 0).mean() * 100.0
        add("P_HORIZON_GT0", "pass" if pos >= 99.0 else "warn", detail=f"share>0={pos:.2f}%")
    else:
        add("P_HORIZON_GT0", "warn", "colonne absente")

    if set(("ts","station_id")).issubset(df.columns):
        dup_pct = df.duplicated(subset=["ts","station_id"]).mean() * 100.0
        add("P_KEY_UNIQ", "pass" if dup_pct == 0 else "fail", detail=f"duplicated_pct={dup_pct:.3f}%")
    else:
        add("P_KEY_UNIQ","fail","colonnes manquantes")
    return pd.DataFrame(checks)

# --------------------------- Keys & Figures ---------------------------

def _keys_report(events: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if set(("ts","station_id")).issubset(events.columns):
        dup_pct = events.duplicated(subset=["ts","station_id"]).mean() * 100.0
        rows.append({"file":"events.parquet","dup_pct":round(float(dup_pct),5),"rows":int(len(events))})
    if set(("ts","station_id")).issubset(perf.columns):
        dup_pct = perf.duplicated(subset=["ts","station_id"]).mean() * 100.0
        rows.append({"file":"perf.parquet","dup_pct":round(float(dup_pct),5),"rows":int(len(perf))})
    return pd.DataFrame(rows)

def _bar(value: float, title: str, fname: str) -> None:
    plt.figure(figsize=(4.2,2.6))
    plt.bar([title], [value])
    plt.title(title)
    plt.tight_layout()
    (FIGS_DIR / fname).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGS_DIR / fname, dpi=150)
    plt.close()

# --------------------------- Markdown ---------------------------

MD_TEMPLATE = """# Dictionnaire & schéma

**Objectif.** Fournir un **contrat formel**: noms, types, unités, domaines, contraintes et **règles de validation**, pour intégrer les données sans lire le code.

> **Build (UTC)** : `{build_utc}`  
> **Fenêtre `events`** : `{ev_ts_min}` → `{ev_ts_max}` · **rows** = {ev_rows:,}  
> **Fenêtre `perf`** : `{pf_ts_min}` → `{pf_ts_max}` · **rows** = {pf_rows:,}

---

## Coup d’œil visuel
![Duplication clés]({dup_rel})  ![Horodatages non alignés]({mis_rel})

---

## Contenu détaillé

### Champs canoniques (par fichier)
- Dictionnaire `events` : {dict_events_rel}  
- Dictionnaire `perf` : {dict_perf_rel}  
- Schéma réel `events` : {schema_events_actual_rel}  
- Schéma réel `perf` : {schema_perf_actual_rel}

Chaque table documente : **Nom** · **Type (pandas/SQL)** · **Unité/domaine** · **Description** · **Obligatoire ?** · **Valeur par défaut** · **Exemples**.

### Clés & unicité
- `(ts, station_id)` est la **clé primaire unique** dans `events.parquet` et `perf.parquet`.  
- Rapport : {keys_uniqueness_rel}

### Règles de cohérence
- `bikes >= 0`, `capacity >= 0`, `0 ≤ occ ≤ 1`.  
- Si `capacity` connue, alors `occ ≈ bikes / capacity` (tolérance).  
- `y_true`, `y_pred`, `y_pred_baseline` **non négatifs** ; `horizon_min > 0`.  
- **Horodatage** : `ts` en {ts_human}, arrondi *:00, :05, :10, …*.

Règles listées : {validation_rules_rel} — Rapport d’exécution : {validation_report_events_rel} · {validation_report_perf_rel}

### Exemples
- Échantillon `events` : {examples_events_rel}  
- Échantillon `perf` : {examples_perf_rel}
"""

def _ts_human(ts_mode: str) -> str:
    return "**UTC (naïf)**" if ts_mode=="naive" else "**UTC tz-aware**"

def _render_md(ev_stats: Dict[str, object], pf_stats: Dict[str, object], ts_mode: str) -> str:
    return MD_TEMPLATE.format(
        build_utc=_now_iso(),
        ev_ts_min=ev_stats.get("ts_min"), ev_ts_max=ev_stats.get("ts_max"), ev_rows=ev_stats.get("rows",0),
        pf_ts_min=pf_stats.get("ts_min"), pf_ts_max=pf_stats.get("ts_max"), pf_rows=pf_stats.get("rows",0),
        ts_human=_ts_human(ts_mode),
        dup_rel=rel_from_md(OUT_MD, FIGS_DIR / "dup_pct.png"),
        mis_rel=rel_from_md(OUT_MD, FIGS_DIR / "misaligned_pct.png"),
        dict_events_rel=rel_from_md(OUT_MD, TABLES_DIR / "dictionary_events.csv"),
        dict_perf_rel=rel_from_md(OUT_MD, TABLES_DIR / "dictionary_perf.csv"),
        schema_events_actual_rel=rel_from_md(OUT_MD, TABLES_DIR / "schema_events_actual.csv"),
        schema_perf_actual_rel=rel_from_md(OUT_MD, TABLES_DIR / "schema_perf_actual.csv"),
        validation_rules_rel=rel_from_md(OUT_MD, TABLES_DIR / "validation_rules.csv"),
        validation_report_events_rel=rel_from_md(OUT_MD, TABLES_DIR / "validation_report_events.csv"),
        validation_report_perf_rel=rel_from_md(OUT_MD, TABLES_DIR / "validation_report_perf.csv"),
        keys_uniqueness_rel=rel_from_md(OUT_MD, TABLES_DIR / "keys_uniqueness_report.csv"),
        examples_events_rel=rel_from_md(OUT_MD, TABLES_DIR / "examples_events.csv"),
        examples_perf_rel=rel_from_md(OUT_MD, TABLES_DIR / "examples_perf.csv"),
    )

# --------------------------- Main ---------------------------

def main(events_path: Path, perf_path: Path, occ_tol: float, ts_mode: str) -> None:
    _mkdirs()

    # 1) Dictionnaires canoniques (paramétrés par ts_mode)
    dict_events = _dictionary_events(ts_mode)
    dict_perf = _dictionary_perf(ts_mode)
    dict_events.to_csv(TABLES_DIR / "dictionary_events.csv", index=False)
    dict_perf.to_csv(TABLES_DIR / "dictionary_perf.csv", index=False)

    # 2) Charger exports & normaliser
    events = _ensure_station_str(_coerce_ts_5m(_read_parquet_safe(events_path), "ts", ts_mode))
    if perf_path.exists():
        perf = _ensure_station_str(_coerce_ts_5m(_read_parquet_safe(perf_path), "ts", ts_mode))
    else:
        perf = pd.DataFrame(columns=["ts","station_id"])

    # 3) Schémas réels
    schema_events = _schema(events).rename(columns={"dtype_pandas":"dtype_pandas_detected"})
    schema_events["dtype_sql_guess"] = [ _pandas_to_sql(t) for t in schema_events["dtype_pandas_detected"] ]
    schema_events.to_csv(TABLES_DIR / "schema_events_actual.csv", index=False)

    schema_perf = _schema(perf).rename(columns={"dtype_pandas":"dtype_pandas_detected"})
    schema_perf["dtype_sql_guess"] = [ _pandas_to_sql(t) for t in schema_perf["dtype_pandas_detected"] ]
    schema_perf.to_csv(TABLES_DIR / "schema_perf_actual.csv", index=False)

    # 4) Règles (catalogue)
    rules = _validation_rules(ts_mode)
    rules.to_csv(TABLES_DIR / "validation_rules.csv", index=False)

    # 5) Validation (exécution)
    v_events = _validate_events(events, occ_tol=occ_tol, ts_mode=ts_mode)
    v_perf = _validate_perf(perf, ts_mode=ts_mode)
    v_events.to_csv(TABLES_DIR / "validation_report_events.csv", index=False)
    v_perf.to_csv(TABLES_DIR / "validation_report_perf.csv", index=False)

    # 6) Unicité des clés
    keys_rep = _keys_report(events, perf)
    keys_rep.to_csv(TABLES_DIR / "keys_uniqueness_report.csv", index=False)

    # 7) Exemples
    ex_events = _examples(events, dict_events["name"].tolist(), n=5)
    ex_perf = _examples(perf, dict_perf["name"].tolist(), n=5)
    ex_events.to_csv(TABLES_DIR / "examples_events.csv", index=False)
    ex_perf.to_csv(TABLES_DIR / "examples_perf.csv", index=False)

    # 8) Build info + stats
    ev_stats = {
        "rows": int(len(events)),
        "ts_min": events["ts"].min().isoformat() if "ts" in events and len(events) else None,
        "ts_max": events["ts"].max().isoformat() if "ts" in events and len(events) else None,
    }
    pf_stats = {
        "rows": int(len(perf)),
        "ts_min": perf["ts"].min().isoformat() if "ts" in perf and len(perf) else None,
        "ts_max": perf["ts"].max().isoformat() if "ts" in perf and len(perf) else None,
    }
    info = {
        "build_utc": _now_iso(),
        "inputs": {"events": str(events_path), "perf": str(perf_path)},
        "rows": {"events": ev_stats["rows"], "perf": pf_stats["rows"]},
        "time_bounds": {"events": {"ts_min": ev_stats["ts_min"], "ts_max": ev_stats["ts_max"]},
                        "perf": {"ts_min": pf_stats["ts_min"], "ts_max": pf_stats["ts_max"]}},
        "tolerances": {"occ_abs_tolerance": occ_tol, "occ_share_within_tol_min_pct": 95.0},
        "primary_key": ["ts","station_id"],
        "timestamp_granularity": f"5min ({'UTC naïf' if ts_mode=='naive' else 'UTC tz-aware'})"
    }
    (TABLES_DIR / "build_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    # 9) Petites jauges visuelles
    dup_events = events.duplicated(subset=["ts","station_id"]).mean()*100.0 if set(("ts","station_id")).issubset(events.columns) else 100.0
    mis_events = (events["ts"].dt.minute % 5 != 0).mean()*100.0 if "ts" in events.columns and len(events) else 100.0
    _bar(dup_events, "Duplication clés (events) %", "dup_pct.png")
    _bar(mis_events, "Misaligned 5min (events) %", "misaligned_pct.png")

    # 10) Page Markdown
    md = _render_md(ev_stats, pf_stats, ts_mode)
    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)

    print("[data/dictionary] Done.")
    print(f"[data/dictionary] Page   → {OUT_MD}")
    print(f"[data/dictionary] Tables → {TABLES_DIR}")

# --------------------------- CLI ---------------------------

def _resolve_paths_from_cli(args) -> Tuple[Path, Path]:
    base = Path(args.exports_dir) if args.exports_dir else None
    default_events = (base / "events.parquet") if base else None
    default_perf = (base / "perf.parquet") if base else None
    events = Path(args.events) if args.events else default_events
    perf = Path(args.perf) if args.perf else default_perf
    if events is None:
        raise SystemExit("CLI error: fournir --events <path> ou --exports-dir <dir> contenant events.parquet")
    if perf is None:
        perf = (base / "perf.parquet") if base else DOCS / "exports" / "perf.parquet"
    return events, perf

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build 'Data / Dictionnaire & schéma' assets + Markdown")
    ap.add_argument("--events", type=Path, required=False, help="Path to docs/exports/events.parquet")
    ap.add_argument("--perf", type=Path, required=False, help="Path to docs/exports/perf.parquet (peut ne pas exister)")
    ap.add_argument("--exports-dir", type=Path, required=False, help="Dossier contenant events.parquet et perf.parquet")
    ap.add_argument("--occ-tol", type=float, default=0.05, help="Tolérance absolue pour la règle occ≈bikes/capacity")
    ap.add_argument("--ts-mode", choices=["naive","aware"], default="aware",
                    help="Format cible de ts: naive (UTC sans tz) ou aware (UTC tz-aware)")
    args = ap.parse_args()

    events_path, perf_path = _resolve_paths_from_cli(args)
    main(events_path=events_path, perf_path=perf_path, occ_tol=args.occ_tol, ts_mode=args.ts_mode)
