# tools/build_data_exports.py
# Page builder — "Data / Exports" + page Markdown visuelle (docs/data/exports.md)
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------- Paths ---------------------------

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
TABLES_DIR = ASSETS / "tables" / "exports"
FIGS_DIR = ASSETS / "figs" / "data" / "export"
OUT_MD = DOCS / "data" / "exports.md"  # <-- page Markdown générée

def _mkdirs() -> None:
    for d in (TABLES_DIR, FIGS_DIR, OUT_MD.parent):
        d.mkdir(parents=True, exist_ok=True)

# --------------------------- Utils ---------------------------

def _save_fig(path: Path) -> None:
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()

def _now_iso() -> str:
    # Always return an aware UTC ISO timestamp
    return pd.Timestamp.now(tz="UTC").isoformat()

def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[exports] Introuvable: {path}")
    return pd.read_parquet(path)

def _schema(df: pd.DataFrame) -> pd.DataFrame:
    return (pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})
            .sort_values("column")
            .reset_index(drop=True))

def _align_ts(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    if ts_col not in df.columns:
        raise KeyError(f"[exports] Colonne '{ts_col}' manquante")
    df = df.copy()
    # Parse in UTC and keep timezone-aware timestamps; align to 15-minute grid
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True).dt.floor("15min")
    return df

def rel_from_md(md_path: Path, target: Path) -> str:
    """Chemin relatif (POSIX) depuis md_path vers target, compatible MkDocs
    (use_directory_urls: true). Ex. docs/data/exports.md -> ../../assets/..."""
    md_rel = Path(md_path).resolve().relative_to(DOCS.resolve())
    parts = md_rel.with_suffix("").parts  # ('data','exports') ou ('index',)
    depth = len(parts) if parts[-1] != "index" else len(parts) - 1
    prefix = "../" * max(depth, 0)
    rel_from_docs = Path(target).resolve().relative_to(DOCS.resolve()).as_posix()
    return (prefix + rel_from_docs).replace("//", "/")

# --------------------------- Contract checks ---------------------------

def _events_contract(df: pd.DataFrame) -> pd.DataFrame:
    checks = []
    def add(name: str, status: str, level: str = "hard", detail: Optional[str] = None) -> None:
        checks.append({"check": name, "status": status, "level": level, "detail": detail})

    # Colonnes minimales
    for col in ("ts", "station_id", "bikes"):
        add(f"column:{col}:exists", "pass" if col in df.columns else "fail", level="hard")

    # Types
    if "ts" in df.columns:
        add("type:ts:datetime64", "pass" if is_datetime64_any_dtype(df["ts"]) else "fail")
    if "station_id" in df.columns:
        add("type:station_id:str", "pass" if pd.api.types.is_string_dtype(df["station_id"]) else "warn", level="soft")
    if "bikes" in df.columns:
        add("type:bikes:numeric", "pass" if pd.api.types.is_numeric_dtype(df["bikes"]) else "fail")

    # Bornes / règles
    if "bikes" in df.columns and pd.api.types.is_numeric_dtype(df["bikes"]):
        add("rule:bikes>=0", "pass" if (df["bikes"].dropna() >= 0).all() else "fail")

    if "capacity" in df.columns and pd.api.types.is_numeric_dtype(df["capacity"]):
        add("rule:capacity>=0", "pass" if (df["capacity"].dropna() >= 0).all() else "fail")

    if "occ" in df.columns and pd.api.types.is_numeric_dtype(df["occ"]):
        ok_share = ((df["occ"].between(0.0, 1.0)) | (df["occ"].isna())).mean() * 100.0
        add("rule:occ∈[0,1]", "pass" if ok_share >= 99.0 else "warn", level="soft",
            detail=f"share_in_range={ok_share:.2f}%")

    # Cohérence occ ≈ bikes/capacity (souple)
    if set(("bikes", "capacity", "occ")).issubset(df.columns):
        sub = df.dropna(subset=["bikes", "capacity", "occ"]).copy()
        sub = sub[sub["capacity"] > 0]
        if not sub.empty:
            occ_calc = (sub["bikes"].clip(lower=0) / sub["capacity"]).clip(0, 1)
            diff_med = float(np.nanmedian(np.abs(occ_calc - sub["occ"])))
            add("rule:occ≈bikes/capacity (tol. médiane ≤ 0,05)",
                "pass" if diff_med <= 0.05 else "warn", level="soft",
                detail=f"median_abs_diff={diff_med:.3f}")
        else:
            add("rule:occ≈bikes/capacity", "n/a", level="info", detail="données insuffisantes")

    # Unicité & pas 15 min
    if set(("ts", "station_id")).issubset(df.columns) and is_datetime64_any_dtype(df["ts"]):
        dup_pct = df.duplicated(subset=["ts", "station_id"]).mean() * 100.0
        add("key:(ts,station_id):unique", "pass" if dup_pct == 0 else "fail",
            detail=f"duplicated_pct={dup_pct:.3f}%")
        mis = (df["ts"].dt.minute % 15 != 0).mean() * 100.0
        add("time:aligned_15min", "pass" if mis == 0 else "warn", level="soft",
            detail=f"misaligned_pct={mis:.3f}%")

    return pd.DataFrame(checks)

def _perf_contract(df: pd.DataFrame) -> pd.DataFrame:
    checks = []
    def add(name: str, status: str, level: str = "hard", detail: Optional[str] = None) -> None:
        checks.append({"check": name, "status": status, "level": level, "detail": detail})

    # Colonnes minimales
    for col in ("ts", "station_id", "y_true"):
        add(f"column:{col}:exists", "pass" if col in df.columns else "fail", level="hard")

    # Types
    if "ts" in df.columns:
        add("type:ts:datetime64", "pass" if is_datetime64_any_dtype(df["ts"]) else "fail")
    if "station_id" in df.columns:
        add("type:station_id:str", "pass" if pd.api.types.is_string_dtype(df["station_id"]) else "warn", level="soft")

    # y_true / y_pred* must be numeric if present
    for c in ("y_true", "y_pred_baseline", "y_pred"):
        if c in df.columns:
            add(f"type:{c}:numeric", "pass" if pd.api.types.is_numeric_dtype(df[c]) else "fail")

    # Unicité & pas 15 min
    if set(("ts", "station_id")).issubset(df.columns) and is_datetime64_any_dtype(df["ts"]):
        dup_pct = df.duplicated(subset=["ts", "station_id"]).mean() * 100.0
        add("key:(ts,station_id):unique", "pass" if dup_pct == 0 else "fail",
            detail=f"duplicated_pct={dup_pct:.3f}%")
        mis = (df["ts"].dt.minute % 15 != 0).mean() * 100.0
        add("time:aligned_15min", "pass" if mis == 0 else "warn", level="soft",
            detail=f"misaligned_pct={mis:.3f}%")

    return pd.DataFrame(checks)

# --------------------------- Catalog & gauges ---------------------------

def _file_stats(path: Path, df: pd.DataFrame) -> Dict[str, object]:
    size = path.stat().st_size if path.exists() else None
    ts_min = df["ts"].min().isoformat() if "ts" in df.columns and len(df) else None
    ts_max = df["ts"].max().isoformat() if "ts" in df.columns and len(df) else None
    stations = int(df["station_id"].astype(str).nunique()) if "station_id" in df.columns else None
    key_dup_pct = df.duplicated(subset=["ts", "station_id"]).mean() * 100.0 if set(("ts","station_id")).issubset(df.columns) else None
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": int(size) if size is not None else None,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "ts_min": ts_min,
        "ts_max": ts_max,
        "stations": stations,
        "dup_pct": float(round(key_dup_pct, 4)) if key_dup_pct is not None else None,
    }

def _plot_gauges(events_stats: Dict[str, object], perf_stats: Dict[str, object]) -> None:
    # Freshness ~ écart entre now et ts_max de events (minutes)
    if events_stats.get("ts_max"):
        now = pd.Timestamp.now(tz="UTC")                 # aware
        tmax = pd.to_datetime(events_stats["ts_max"], utc=True)  # aware UTC
        age_min = (now - tmax).total_seconds() / 60.0
        plt.figure(figsize=(4.2, 2.6))
        plt.bar(["Fresh (min)"], [age_min])
        plt.title("Fraîcheur events (âge du dernier point)")
        _save_fig(FIGS_DIR / "gauge_freshness.png")

    # Volumes gauge (events vs perf)
    plt.figure(figsize=(5.5, 2.6))
    plt.bar(["events", "perf"], [events_stats.get("rows", 0), perf_stats.get("rows", 0)])
    plt.title("Volumes (rows)")
    _save_fig(FIGS_DIR / "gauge_rows.png")

# --------------------------- Markdown rendering ---------------------------

MD_TEMPLATE = """# Exports

Cette page liste **tout ce qui est publié** (fichiers de données et tables dérivées) et rappelle le **contrat minimal** pour les consommer sans surprise.

> **Build (UTC)** : `{build_utc}`  
> **Fenêtre `events`** : `{ev_ts_min}` → `{ev_ts_max}` · **rows** = {ev_rows:,} · **stations** ≈ {ev_stations}  
> **Fenêtre `perf`** : `{pf_ts_min}` → `{pf_ts_max}` · **rows** = {pf_rows:,}

---

## Coup d’œil visuel
![Fraîcheur]({gauge_fresh_rel})  ![Volumes]({gauge_rows_rel})

---

## Fichiers principaux
- **`docs/exports/events.parquet`**  
  - **Clé** : `(ts, station_id)` unique.  
  - **Colonnes canoniques (si présentes)** :  
    - `ts` *(UTC, 15 min)* — horodatage du **bin source**.  
    - `station_id` *(str)* — identifiant station stable.  
    - `bikes` *(int ≥ 0)* — vélos disponibles.  
    - `capacity` *(int ≥ 0)* — capacité estimée du dock.  
    - `occ` *(float ∈ [0,1])* — ratio d’occupation (ex. `bikes / capacity`, si capacité connue).  
    - **Métadonnées** : `name` *(str)*, `lat` *(float)*, `lon` *(float)*, `hour_utc` *(0–23)*.  
    - **Météo (optionnelles)** : `temp_C` *(°C)*, `precip_mm` *(mm)*, `wind_mps` *(m/s)*.
- **`docs/exports/perf.parquet`**  
  - **Clé** : `(ts, station_id)` unique.  
  - **Colonnes canoniques** :  
    - `ts` *(UTC, 15 min)* — **même bin source T** que `events`.  
    - `station_id` *(str)*.  
    - `y_true` *(float/int ≥ 0)* — cible observée à **T+h** (ramenée à T via `shift(-steps)`).  
    - `y_pred_baseline` *(float ≥ 0)* — **persistance** (valeur observée à T).  
    - `y_pred` *(float ≥ 0, optionnel)* — prédiction **modèle** alignée sur T (injectée après coup).  
    - `horizon_min` *(int > 0, ex. 60)* — horizon en minutes.

---

## Tables secondaires (lecture & monitoring)
Exportées sous `docs/assets/tables/exports/` :
- Schémas : `{schema_events_rel}` · `{schema_perf_rel}`
- Contrats : `{contract_events_rel}` · `{contract_perf_rel}`
- Catalog : `{catalog_rel}` · Build info : `{build_info_rel}`

---

## Cadence & fraîcheur
- Ingestion et normalisation **toutes les 15 minutes** (ou au rythme de la source).  
- Les assets analytiques (tables/figures) sont régénérés **plusieurs fois par jour**.  
- Cette page précise **la date/heure du dernier build** et la **fenêtre couverte**.

---

## Garanties minimales
- Pas de **fusion en avance** (aucune fuite de futur) ; `perf.parquet` est strictement aligné **à T**.  
- **Aucune imputation lourde** dans les exports (ni interpolation) : les trous reflètent l’état réel de l’ingestion.  
- Clés `(ts, station_id)` **sans doublons** ; horodatages **arrondis 15 min**.
"""

def _render_md(ev_stats: Dict[str, object], pf_stats: Dict[str, object]) -> str:
    return MD_TEMPLATE.format(
        build_utc=_now_iso(),
        ev_ts_min=ev_stats.get("ts_min"),
        ev_ts_max=ev_stats.get("ts_max"),
        ev_rows=ev_stats.get("rows", 0),
        ev_stations=ev_stats.get("stations", 0),
        pf_ts_min=pf_stats.get("ts_min"),
        pf_ts_max=pf_stats.get("ts_max"),
        pf_rows=pf_stats.get("rows", 0),

        gauge_fresh_rel=rel_from_md(OUT_MD, FIGS_DIR / "gauge_freshness.png"),
        gauge_rows_rel=rel_from_md(OUT_MD, FIGS_DIR / "gauge_rows.png"),

        schema_events_rel=rel_from_md(OUT_MD, TABLES_DIR / "schema_events.csv"),
        schema_perf_rel=rel_from_md(OUT_MD, TABLES_DIR / "schema_perf.csv"),
        contract_events_rel=rel_from_md(OUT_MD, TABLES_DIR / "contract_events_report.csv"),
        contract_perf_rel=rel_from_md(OUT_MD, TABLES_DIR / "contract_perf_report.csv"),
        catalog_rel=rel_from_md(OUT_MD, TABLES_DIR / "catalog.csv"),
        build_info_rel=rel_from_md(OUT_MD, TABLES_DIR / "build_info.json"),
    )

# --------------------------- Main flow ---------------------------

def main(events_path: Path, perf_path: Path) -> None:
    _mkdirs()

    # Charger events
    events = _read_parquet(events_path)
    # Normaliser types clés
    if "station_id" not in events.columns:
        for c in ("stationcode", "stationCode", "id"):
            if c in events.columns:
                events = events.rename(columns={c: "station_id"})
                break
    events = _align_ts(events, "ts")
    events["station_id"] = events["station_id"].astype(str)

    # Charger perf (si dispo)
    if perf_path.exists():
        perf = _read_parquet(perf_path)
        if "station_id" not in perf.columns:
            for c in ("stationcode", "stationCode", "id"):
                if c in perf.columns:
                    perf = perf.rename(columns={c: "station_id"})
                    break
        perf = _align_ts(perf, "ts")
        perf["station_id"] = perf["station_id"].astype(str)
    else:
        perf = pd.DataFrame(columns=["ts", "station_id", "y_true", "y_pred_baseline"])

    # Schémas
    _schema(events).to_csv(TABLES_DIR / "schema_events.csv", index=False)
    _schema(perf).to_csv(TABLES_DIR / "schema_perf.csv", index=False)

    # Contrats
    ev_rep = _events_contract(events)
    pf_rep = _perf_contract(perf)
    ev_rep.to_csv(TABLES_DIR / "contract_events_report.csv", index=False)
    pf_rep.to_csv(TABLES_DIR / "contract_perf_report.csv", index=False)

    # Catalog
    ev_stats = _file_stats(events_path, events)
    pf_stats = _file_stats(perf_path, perf) if perf_path.exists() else {
        "path": str(perf_path), "exists": False, "size_bytes": None, "rows": 0, "cols": 0,
        "ts_min": None, "ts_max": None, "stations": None, "dup_pct": None,
    }
    catalog = pd.DataFrame([ev_stats, pf_stats])
    catalog.to_csv(TABLES_DIR / "catalog.csv", index=False)

    # Build info
    info = {
        "build_utc": _now_iso(),
        "events": {"path": str(events_path), "ts_min": ev_stats["ts_min"], "ts_max": ev_stats["ts_max"], "rows": ev_stats["rows"]},
        "perf": {"path": str(perf_path), "ts_min": pf_stats["ts_min"], "ts_max": pf_stats["ts_max"], "rows": pf_stats["rows"]},
    }
    (TABLES_DIR / "build_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # Jauges
    _plot_gauges(ev_stats, pf_stats)

    # Page Markdown
    md = _render_md(ev_stats, pf_stats)
    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)

    print("[exports] Done.")
    print(f"[exports] Page   → {OUT_MD}")
    print(f"[exports] Catalog → {TABLES_DIR / 'catalog.csv'}")
    print(f"[exports] Contracts → {TABLES_DIR / 'contract_events_report.csv'} ; {TABLES_DIR / 'contract_perf_report.csv'}")

# --------------------------- CLI compatibility ---------------------------

def _resolve_paths_from_cli(args) -> Tuple[Path, Path]:
    """
    Compatibilité CLI :
      - ancien mode : --events ... --perf ...
      - nouveau mode : --exports-dir <dir>  (déduit events/perf)
    Les arguments explicites (--events/--perf) priment sur --exports-dir.
    """
    base = Path(args.exports_dir) if args.exports_dir else None
    default_events = (base / "events.parquet") if base else None
    default_perf = (base / "perf.parquet") if base else None

    events = Path(args.events) if args.events else default_events
    perf = Path(args.perf) if args.perf else default_perf

    if events is None:
        raise SystemExit("CLI error: fournir --events <path> ou --exports-dir <dir> contenant events.parquet")
    if perf is None:
        # On passe quand même un chemin (même s'il n'existe pas) : main gère l'absence.
        perf = (base / "perf.parquet") if base else DOCS / "exports" / "perf.parquet"

    return events, perf

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build 'Data / Exports' inventory, contract checks & Markdown page")
    ap.add_argument("--events", type=Path, required=False, help="Path to events.parquet")
    ap.add_argument("--perf", type=Path, required=False, help="Path to perf.parquet (peut ne pas exister)")
    ap.add_argument("--exports-dir", type=Path, required=False, help="Dossier contenant events.parquet et perf.parquet")
    args = ap.parse_args()

    events_path, perf_path = _resolve_paths_from_cli(args)
    main(events_path=events_path, perf_path=perf_path)
