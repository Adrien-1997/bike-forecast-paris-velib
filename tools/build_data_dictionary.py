# tools/build_data_dictionary.py
# Page builder — "Data / Dictionnaire & schéma"
#
# But
# ----
# Générer un dictionnaire de données formel (par fichier), aligné avec le
# contrat décrit dans `dictionary.md`, et produire :
#   - Dictionnaires canoniques : noms, types attendus (pandas/SQL), unités/domaines,
#     description, caractère obligatoire, valeur par défaut, exemples.
#   - Schémas *réels* lus dans les exports (types pandas détectés).
#   - Vérifications de cohérence (règles) et rapport pass/warn/fail.
#
# Sorties (docs/assets/*)
# ----------------------
# tables/data/dictionary/
#   - dictionary_events.csv
#   - dictionary_perf.csv
#   - schema_events_actual.csv
#   - schema_perf_actual.csv
#   - validation_rules.csv
#   - validation_report_events.csv
#   - validation_report_perf.csv
#   - keys_uniqueness_report.csv
#   - examples_events.csv
#   - examples_perf.csv
#   - build_info.json
#
# CLI
# ---
# python tools/build_data_dictionary.py \
#   --events docs/exports/events.parquet \
#   --perf docs/exports/perf.parquet \
#   --occ-tol 0.05
#
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------- Paths ---------------------------

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
TABLES_DIR = ASSETS / "tables" / "data" / "dictionary"
FIGS_DIR = ASSETS / "figs" / "data" / "dictionary"


# --------------------------- Utils ---------------------------

def _mkdirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

def _now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()

def _read_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[dictionary] Fichier introuvable: {path}")
    return pd.read_parquet(path)

def _coerce_ts_15m(df: pd.DataFrame, col: str = "ts") -> pd.DataFrame:
    if col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.floor("15min")
    return df

def _ensure_station_str(df: pd.DataFrame) -> pd.DataFrame:
    # normaliser le nom si besoin puis caster str
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
        return "timestamp without time zone"
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
    out = df[keep].head(n).copy()
    return out


# --------------------------- Canonical dictionaries ---------------------------

def _dictionary_events() -> pd.DataFrame:
    rows = [
        # name, type_pandas, type_sql, unit/domain, description, required, default, examples
        ("ts", "datetime64[ns]", "timestamp without time zone",
         "UTC, pas 15 min (xx:00, :15, :30, :45)",
         "Horodatage du *bin source* (UTC naïf), arrondi 15 min.",
         True, None, "2025-01-02 08:15"),
        ("station_id", "string", "text",
         "Identifiant station stable",
         "Identifiant canonique de la station (chaîne).",
         True, None, "3005"),
        ("bikes", "int64", "bigint",
         "≥ 0, vélos disponibles",
         "Nombre de vélos disponibles à T.",
         True, 0, "0, 7, 15"),
        ("capacity", "int64", "bigint",
         "≥ 0, docks totaux (estimés)",
         "Capacité totale estimée de la station (si connue).",
         False, None, "20, 35"),
        ("occ", "float64", "double precision",
         "[0,1], ratio d’occupation",
         "Rapport *bikes/capacity* lorsque *capacity* est disponible.",
         False, None, "0.35, 0.80"),
        ("name", "string", "text",
         "Texte libre",
         "Nom d’affichage de la station (optionnel).",
         False, None, "Rivoli - Pont Neuf"),
        ("lat", "float64", "double precision",
         "WGS84",
         "Latitude (WGS84).", False, None, "48.859"),
        ("lon", "float64", "double precision",
         "WGS84",
         "Longitude (WGS84).", False, None, "2.347"),
        ("hour_utc", "int64", "bigint",
         "Entier 0–23",
         "Heure UTC extraite de ts (facultatif).",
         False, None, "8, 17, 23"),
        ("temp_C", "float64", "double precision",
         "°C", "Température (si intégrée).", False, None, "12.3"),
        ("precip_mm", "float64", "double precision",
         "mm", "Précipitations (si intégrées).", False, None, "0.4"),
        ("wind_mps", "float64", "double precision",
         "m/s", "Vent (si intégré).", False, None, "3.2"),
    ]
    return pd.DataFrame(rows, columns=[
        "name", "type_pandas", "type_sql", "unit_or_domain",
        "description", "required", "default", "examples"
    ])

def _dictionary_perf() -> pd.DataFrame:
    rows = [
        ("ts", "datetime64[ns]", "timestamp without time zone",
         "UTC, pas 15 min (xx:00, :15, :30, :45)",
         "Horodatage *T* (même bin que events).",
         True, None, "2025-01-02 08:15"),
        ("station_id", "string", "text",
         "Identifiant station stable",
         "Identifiant canonique de la station (chaîne).",
         True, None, "3005"),
        ("y_true", "float64", "double precision",
         "≥ 0", "Cible observée à T+h, ramenée à T (shift(-steps)).",
         True, None, "5, 12"),
        ("y_pred_baseline", "float64", "double precision",
         "≥ 0", "Baseline persistance (valeur observée à T).",
         True, None, "7, 9"),
        ("y_pred", "float64", "double precision",
         "≥ 0", "Prédiction du modèle alignée sur T (optionnelle).",
         False, None, "8.4"),
        ("horizon_min", "int64", "bigint",
         "> 0", "Horizon en minutes (ex. 60).",
         False, None, "60"),
    ]
    return pd.DataFrame(rows, columns=[
        "name", "type_pandas", "type_sql", "unit_or_domain",
        "description", "required", "default", "examples"
    ])


# --------------------------- Validation rules ---------------------------

def _validation_rules() -> pd.DataFrame:
    rows = [
        # id, applies_to, severity, rule, description
        ("E_COL_TS", "events", "hard", "column_exists(ts)",
         "Colonne ts présente"),
        ("E_COL_STID", "events", "hard", "column_exists(station_id)",
         "Colonne station_id présente"),
        ("E_COL_BIKES", "events", "hard", "column_exists(bikes)",
         "Colonne bikes présente"),
        ("E_TYPE_TS", "events", "hard", "dtype(ts) is datetime64",
         "ts de type datetime64[ns] (naïf UTC attendu)"),
        ("E_NUM_BIKES", "events", "hard", "numeric(bikes) & bikes>=0",
         "bikes numérique, non négatif"),
        ("E_RANGE_OCC", "events", "soft", "0<=occ<=1 (si présent)",
         "occ borné à [0,1]"),
        ("E_RANGE_CAP", "events", "soft", "capacity>=0 (si présent)",
         "capacity non négatif"),
        ("E_OCC_FORMULA", "events", "soft", "median(|occ-(bikes/capacity)|) <= tol (95% lignes avec capacity>0)",
         "occ≈bikes/capacity (tolérance)"),
        ("E_KEY_UNIQ", "events", "hard", "unique(ts,station_id)",
         "Clé (ts,station_id) unique"),
        ("E_STEP_15M", "events", "soft", "ts minute%15==0",
         "Horodatages alignés (pas 15 min)"),

        ("P_COL_TS", "perf", "hard", "column_exists(ts)", "Colonne ts présente"),
        ("P_COL_STID", "perf", "hard", "column_exists(station_id)", "Colonne station_id présente"),
        ("P_COL_YTRUE", "perf", "hard", "column_exists(y_true)", "Colonne y_true présente"),
        ("P_COL_BASE", "perf", "hard", "column_exists(y_pred_baseline)", "Colonne y_pred_baseline présente"),
        ("P_TYPE_TS", "perf", "hard", "dtype(ts) is datetime64", "ts de type datetime64[ns]"),
        ("P_NUM_NONNEG", "perf", "soft", "y_true,y_pred_baseline,y_pred >= 0 (si présents)", "Non négatifs"),
        ("P_HORIZON_GT0", "perf", "soft", "horizon_min>0 (si présent)", "Horizon positif"),
        ("P_KEY_UNIQ", "perf", "hard", "unique(ts,station_id)", "Clé (ts,station_id) unique"),
        ("P_STEP_15M", "perf", "soft", "ts minute%15==0", "Horodatages alignés (pas 15 min)"),
    ]
    return pd.DataFrame(rows, columns=["id", "applies_to", "severity", "rule", "description"])


def _validate_events(df: pd.DataFrame, occ_tol: float) -> pd.DataFrame:
    checks = []

    def add(id_, status, detail=None):
        checks.append({"id": id_, "status": status, "detail": detail})

    # Existence
    add("E_COL_TS", "pass" if "ts" in df.columns else "fail")
    add("E_COL_STID", "pass" if "station_id" in df.columns else "fail")
    add("E_COL_BIKES", "pass" if "bikes" in df.columns else "fail")

    # Types & intervalles
    if "ts" in df.columns:
        dtype_is_dt = pd.api.types.is_datetime64_any_dtype(df["ts"])
        tz_aware = False
        try:
            tz_aware = getattr(df["ts"].dt.tz, "zone", None) is not None
        except Exception:
            tz_aware = False
        add("E_TYPE_TS", "pass" if dtype_is_dt and not tz_aware else "fail",
            detail="tz-aware" if tz_aware else None)

        mis = (df["ts"].dt.minute % 15 != 0).mean() * 100.0
        add("E_STEP_15M", "pass" if mis == 0 else "warn", detail=f"misaligned_pct={mis:.3f}%")

    if "bikes" in df.columns:
        bikes_numeric = pd.api.types.is_numeric_dtype(df["bikes"])
        nonneg_share = (df["bikes"].dropna() >= 0).mean() * 100.0 if bikes_numeric else 0.0
        add("E_NUM_BIKES", "pass" if (bikes_numeric and nonneg_share == 100.0) else "fail",
            detail=f"share_nonneg={nonneg_share:.2f}%")

    if "occ" in df.columns and pd.api.types.is_numeric_dtype(df["occ"]):
        share_in_range = ((df["occ"].between(0.0, 1.0)) | (df["occ"].isna())).mean() * 100.0
        add("E_RANGE_OCC", "pass" if share_in_range >= 99.0 else "warn", detail=f"share_in_range={share_in_range:.2f}%")
    else:
        add("E_RANGE_OCC", "n/a", detail="colonne absente")

    if "capacity" in df.columns and pd.api.types.is_numeric_dtype(df["capacity"]):
        cap_nonneg = (df["capacity"].dropna() >= 0).mean() * 100.0
        add("E_RANGE_CAP", "pass" if cap_nonneg >= 99.0 else "warn", detail=f"share_nonneg={cap_nonneg:.2f}%")
        # occ consistency
        sub = df.dropna(subset=["bikes", "capacity"]).copy()
        sub = sub[sub["capacity"] > 0]
        if "occ" in df.columns and not sub.empty:
            occ_calc = (sub["bikes"].clip(lower=0) / sub["capacity"]).clip(0, 1)
            med_abs = float(np.nanmedian(np.abs(occ_calc - sub["occ"])))
            share_ok = float((np.abs(occ_calc - sub["occ"]) <= occ_tol).mean() * 100.0)
            status = "pass" if (med_abs <= occ_tol and share_ok >= 95.0) else "warn"
            add("E_OCC_FORMULA", status, detail=f"median_abs_diff={med_abs:.3f}; share<=tol={share_ok:.1f}%")
        else:
            add("E_OCC_FORMULA", "n/a", detail="données insuffisantes")
    else:
        add("E_RANGE_CAP", "n/a", detail="colonne absente")
        add("E_OCC_FORMULA", "n/a", detail="capacity absente")

    # Unicité clé
    if set(("ts", "station_id")).issubset(df.columns):
        dup_pct = df.duplicated(subset=["ts", "station_id"]).mean() * 100.0
        add("E_KEY_UNIQ", "pass" if dup_pct == 0 else "fail", detail=f"duplicated_pct={dup_pct:.3f}%")
    else:
        add("E_KEY_UNIQ", "fail", detail="colonnes manquantes")

    return pd.DataFrame(checks)


def _validate_perf(df: pd.DataFrame) -> pd.DataFrame:
    checks = []

    def add(id_, status, detail=None):
        checks.append({"id": id_, "status": status, "detail": detail})

    # Existence
    add("P_COL_TS", "pass" if "ts" in df.columns else "fail")
    add("P_COL_STID", "pass" if "station_id" in df.columns else "fail")
    add("P_COL_YTRUE", "pass" if "y_true" in df.columns else "fail")
    add("P_COL_BASE", "pass" if "y_pred_baseline" in df.columns else "fail")

    # Types & bornes
    if "ts" in df.columns:
        dtype_is_dt = pd.api.types.is_datetime64_any_dtype(df["ts"])
        tz_aware = False
        try:
            tz_aware = getattr(df["ts"].dt.tz, "zone", None) is not None
        except Exception:
            tz_aware = False
        add("P_TYPE_TS", "pass" if dtype_is_dt and not tz_aware else "fail",
            detail="tz-aware" if tz_aware else None)

        mis = (df["ts"].dt.minute % 15 != 0).mean() * 100.0
        add("P_STEP_15M", "pass" if mis == 0 else "warn", detail=f"misaligned_pct={mis:.3f}%")

    for c in ("y_true", "y_pred_baseline", "y_pred"):
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            nonneg = (df[c].dropna() >= 0).mean() * 100.0
            add("P_NUM_NONNEG", "pass" if nonneg >= 99.0 else "warn", detail=f"{c}: share_nonneg={nonneg:.2f}%")

    if "horizon_min" in df.columns:
        pos = (pd.to_numeric(df["horizon_min"], errors="coerce") > 0).mean() * 100.0
        add("P_HORIZON_GT0", "pass" if pos >= 99.0 else "warn", detail=f"share>0={pos:.2f}%")
    else:
        add("P_HORIZON_GT0", "warn", detail="colonne absente")

    # Unicité clé
    if set(("ts", "station_id")).issubset(df.columns):
        dup_pct = df.duplicated(subset=["ts", "station_id"]).mean() * 100.0
        add("P_KEY_UNIQ", "pass" if dup_pct == 0 else "fail", detail=f"duplicated_pct={dup_pct:.3f}%")
    else:
        add("P_KEY_UNIQ", "fail", detail="colonnes manquantes")

    return pd.DataFrame(checks)


# --------------------------- Keys report ---------------------------

def _keys_report(events: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if set(("ts", "station_id")).issubset(events.columns):
        dup_pct = events.duplicated(subset=["ts", "station_id"]).mean() * 100.0
        rows.append({"file": "events.parquet", "dup_pct": round(float(dup_pct), 5), "rows": int(len(events))})
    if set(("ts", "station_id")).issubset(perf.columns):
        dup_pct = perf.duplicated(subset=["ts", "station_id"]).mean() * 100.0
        rows.append({"file": "perf.parquet", "dup_pct": round(float(dup_pct), 5), "rows": int(len(perf))})
    return pd.DataFrame(rows)


# --------------------------- Main ---------------------------

def main(events_path: Path, perf_path: Path, occ_tol: float) -> None:
    _mkdirs()

    # 1) Dictionnaires canoniques
    dict_events = _dictionary_events()
    dict_perf = _dictionary_perf()
    dict_events.to_csv(TABLES_DIR / "dictionary_events.csv", index=False)
    dict_perf.to_csv(TABLES_DIR / "dictionary_perf.csv", index=False)

    # 2) Charger exports & normaliser
    events = _read_parquet_safe(events_path)
    events = _ensure_station_str(_coerce_ts_15m(events, "ts"))

    if perf_path.exists():
        perf = _read_parquet_safe(perf_path)
        perf = _ensure_station_str(_coerce_ts_15m(perf, "ts"))
    else:
        perf = pd.DataFrame(columns=["ts", "station_id"])

    # 3) Schémas réels
    schema_events = _schema(events).rename(columns={"dtype_pandas": "dtype_pandas_detected"})
    schema_events["dtype_sql_guess"] = [ _pandas_to_sql(t) for t in schema_events["dtype_pandas_detected"] ]
    schema_events.to_csv(TABLES_DIR / "schema_events_actual.csv", index=False)

    schema_perf = _schema(perf).rename(columns={"dtype_pandas": "dtype_pandas_detected"})
    schema_perf["dtype_sql_guess"] = [ _pandas_to_sql(t) for t in schema_perf["dtype_pandas_detected"] ]
    schema_perf.to_csv(TABLES_DIR / "schema_perf_actual.csv", index=False)

    # 4) Règles (catalogue)
    rules = _validation_rules()
    rules.to_csv(TABLES_DIR / "validation_rules.csv", index=False)

    # 5) Validation (exécution)
    v_events = _validate_events(events, occ_tol=occ_tol)
    v_perf = _validate_perf(perf)
    v_events.to_csv(TABLES_DIR / "validation_report_events.csv", index=False)
    v_perf.to_csv(TABLES_DIR / "validation_report_perf.csv", index=False)

    # 6) Unicité des clés
    keys_rep = _keys_report(events, perf)
    keys_rep.to_csv(TABLES_DIR / "keys_uniqueness_report.csv", index=False)

    # 7) Exemples (quelques lignes)
    ex_events = _examples(events, dict_events["name"].tolist(), n=5)
    ex_perf = _examples(perf, dict_perf["name"].tolist(), n=5)
    ex_events.to_csv(TABLES_DIR / "examples_events.csv", index=False)
    ex_perf.to_csv(TABLES_DIR / "examples_perf.csv", index=False)

    # 8) Build info
    info = {
        "build_utc": _now_iso(),
        "inputs": {
            "events": str(events_path),
            "perf": str(perf_path),
        },
        "rows": {
            "events": int(len(events)),
            "perf": int(len(perf)),
        },
        "time_bounds": {
            "events": {
                "ts_min": events["ts"].min().isoformat() if "ts" in events and len(events) else None,
                "ts_max": events["ts"].max().isoformat() if "ts" in events and len(events) else None,
            },
            "perf": {
                "ts_min": perf["ts"].min().isoformat() if "ts" in perf and len(perf) else None,
                "ts_max": perf["ts"].max().isoformat() if "ts" in perf and len(perf) else None,
            },
        },
        "tolerances": {
            "occ_abs_tolerance": occ_tol,
            "occ_share_within_tol_min_pct": 95.0
        },
        "primary_key": ["ts", "station_id"],
        "timestamp_granularity": "15min (UTC naïf, xx:00/:15/:30/:45)"
    }
    (TABLES_DIR / "build_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[data/dictionary] Done.")
    print(f"[data/dictionary] Dictionnaires → {TABLES_DIR / 'dictionary_events.csv'} ; {TABLES_DIR / 'dictionary_perf.csv'}")
    print(f"[data/dictionary] Rapports → {TABLES_DIR / 'validation_report_events.csv'} ; {TABLES_DIR / 'validation_report_perf.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build 'Data / Dictionnaire & schéma' assets")
    ap.add_argument("--events", type=Path, required=True, help="Path to docs/exports/events.parquet")
    ap.add_argument("--perf", type=Path, required=True, help="Path to docs/exports/perf.parquet (peut ne pas exister)")
    ap.add_argument("--occ-tol", type=float, default=0.05, help="Tolérance absolue pour la règle occ≈bikes/capacity")
    args = ap.parse_args()

    main(events_path=args.events, perf_path=args.perf, occ_tol=args.occ_tol)
