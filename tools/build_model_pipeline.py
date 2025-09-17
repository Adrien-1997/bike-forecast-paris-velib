# tools/build_model_pipeline.py
# Page builder — "Modèle / Pipeline d’entraînement & features"
#
# Produit des artefacts décrivant le pipeline :
# - schemas des fichiers d’entrée (events/perf)
# - résumé pipeline (JSON) : fenêtres, horizon, bornes temporelles, nb stations
# - inventaire des features attendues par le modèle (si dispo) + classification par familles
# - figures simples : diagramme du dataflow, histogramme des familles de features
#
# CLI :
#   python tools/build_model_pipeline.py --events docs/exports/events.parquet --perf docs/exports/perf.parquet --horizon 60
#
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

# Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
# Écritures sous docs/assets pour la publication MkDocs
TABLES_OUT = DOCS / "assets" / "tables" / "model" / "pipeline"
FIGS_OUT   = DOCS / "assets" / "figs"   / "model" / "pipeline"
MD_OUT     = DOCS / "model" / "pipeline.md"

TABLES_OUT.mkdir(parents=True, exist_ok=True)
FIGS_OUT.mkdir(parents=True, exist_ok=True)
MD_OUT.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------- utils --------------------------------------

def _log(msg: str) -> None:
    print(f"[model/pipeline] {msg}")

def _df_schema(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        non_null = int(s.notna().sum())
        pct = round(100.0 * non_null / max(n, 1), 2)
        ex = None
        if non_null:
            try:
                ex = s.dropna().iloc[0]
                if isinstance(ex, (pd.Timestamp, np.datetime64)):
                    ex = pd.to_datetime(ex).isoformat()
            except Exception:
                ex = None
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "non_null_pct": pct,
            "example": ex,
        })
    return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)

def rel_from_md(md_path: Path, target_under_docs: Path) -> str:
    """
    Chemin relatif (POSIX) depuis md_path vers un fichier situé sous docs/.
    Compatible MkDocs (use_directory_urls: true).
    """
    md_rel = md_path.resolve().relative_to(DOCS.resolve())
    parts = md_rel.with_suffix("").parts          # ex: ('model','pipeline')
    depth = len(parts) if parts[-1] != "index" else len(parts) - 1
    prefix = "../" * max(depth, 0)
    rel_from_docs = target_under_docs.resolve().relative_to(DOCS.resolve()).as_posix()
    return (prefix + rel_from_docs).replace("//", "/")

def _md_table(df: pd.DataFrame, cols: List[str]) -> str:
    if df is None or df.empty:
        return "_(aucune donnée)_"
    df2 = df[cols].copy()
    head = "| " + " | ".join(cols) + " |\n"
    sep  = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    rows = ""
    for _, r in df2.iterrows():
        rows += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return head + sep + rows

# ------------------------ feature detection logic ---------------------------

def _try_import_features_direct() -> Tuple[List[str], str]:
    """Try to import src.features.feature_cols. Returns (features, source_label) or ([], "")."""
    try:
        import importlib
        sf2 = importlib.import_module("src.features")
        cols = list(getattr(sf2, "feature_cols"))
        return [str(x) for x in cols], "src.features.feature_cols"
    except Exception:
        return [], ""

def _try_import_from_model(horizon: Optional[int]) -> Tuple[List[str], str]:
    """Try to import src.forecast and load model bundle (or feature_names_in_)."""
    try:
        import importlib
        sf = importlib.import_module("src.forecast")
    except Exception:
        return [], ""

    feat_cols: List[str] = []
    source = ""

    # Preferred: load_model_bundle(horizon_minutes=...)
    try:
        if hasattr(sf, "load_model_bundle"):
            model, maybe_feats = sf.load_model_bundle(horizon_minutes=horizon)  # type: ignore[attr-defined]
            if maybe_feats and len(maybe_feats):
                feat_cols = [str(x) for x in maybe_feats]
                source = "src.forecast.load_model_bundle"
            else:
                fni = getattr(model, "feature_names_in_", None)
                if fni is not None and len(fni):
                    feat_cols = [str(x) for x in list(fni)]
                    source = "model.feature_names_in_"
    except Exception:
        pass

    # Secondary: model from get_model()
    if not feat_cols:
        try:
            if hasattr(sf, "get_model"):
                model = sf.get_model()
                fni = getattr(model, "feature_names_in_", None)
                if fni is not None and len(fni):
                    feat_cols = [str(x) for x in list(fni)]
                    source = "model.feature_names_in_"
        except Exception:
            pass

    return feat_cols, source

def _try_load_feature_list(horizon: Optional[int]) -> Tuple[List[str], dict]:
    meta = {"source": None}

    # 1) Preferred: from model / bundle
    feats, src = _try_import_from_model(horizon)
    if feats:
        meta["source"] = src
        return feats, meta

    # 2) Fallback: static feature list from src.features
    feats2, src2 = _try_import_features_direct()
    if feats2:
        meta["source"] = src2
        return feats2, meta

    # 3) Nothing found
    return [], meta

# --------------------------- feature reporting ------------------------------

_WEATHER = {
    "temp_air","temp_feels_like","precip_mm","rain_intensity","wind_speed",
    "wind_gust","wind_dir_sin","wind_dir_cos","humidity","pressure"
}
_SEASONAL = {"sin_hour","cos_hour","sin_doy","cos_doy","dow","is_weekend","is_holiday"}

def _family_of(name: str) -> str:
    n = name.lower()
    if n.startswith("lag_"): return "lags"
    if n.startswith("roll_"): return "rolling"
    if n.startswith("diff_") or "trend" in n: return "trend"
    if n in _SEASONAL: return "seasonality"
    if n in _WEATHER: return "weather"
    if "ratio" in n or n in {"capacity"}: return "normalization"
    if n.startswith("station_") or n in {"cluster_id","elevation"}: return "station_meta"
    return "other"

def _write_features_contract(features: List[str]) -> None:
    if not features:
        # placeholders (sous docs/)
        pd.DataFrame(columns=["feature","family"]).to_csv(TABLES_OUT / "features_contract.csv", index=False)
        pd.DataFrame(columns=["family","count"]).to_csv(TABLES_OUT / "features_by_family.csv", index=False)
        fig = plt.figure(figsize=(6, 4))
        plt.title("Features by family — (none detected)")
        plt.xlabel("family"); plt.ylabel("count")
        fig.savefig(FIGS_OUT / "features_by_family.png", bbox_inches="tight")
        plt.close(fig)
        return

    rows = [{"feature": f, "family": _family_of(f)} for f in features]
    df = pd.DataFrame(rows).sort_values(["family","feature"])
    df.to_csv(TABLES_OUT / "features_contract.csv", index=False)

    # Comptage robuste par famille
    families = [_family_of(f) for f in features]
    fam_counts = pd.Series(families, name="family").value_counts().reset_index()
    fam_counts.columns = ["family", "count"]
    fam_counts = fam_counts.sort_values(["count","family"], ascending=[False, True]).reset_index(drop=True)
    fam_counts.to_csv(TABLES_OUT / "features_by_family.csv", index=False)

    fig = plt.figure(figsize=(7, 4.5))
    plt.bar(fam_counts["family"], fam_counts["count"])  # no explicit colors
    plt.title("Features by family"); plt.xlabel("family"); plt.ylabel("count")
    plt.xticks(rotation=20, ha="right")
    fig.savefig(FIGS_OUT / "features_by_family.png", bbox_inches="tight")
    plt.close(fig)

# ----------------------------- extra visuals --------------------------------

def _draw_pipeline_architecture() -> None:
    """Schéma global pipeline."""
    fig = plt.figure(figsize=(10.5, 4.4))
    ax = plt.gca(); ax.axis("off")

    def box(x, y, w, h, text):
        rect = Rectangle((x, y), w, h, fill=False)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10)

    # Top: sources -> features -> training -> artefact
    box(0.03, 0.55, 0.18, 0.30, "Sources\n(events / calendar / météo / geo)")
    box(0.28, 0.55, 0.18, 0.30, "Feature\nEngineering")
    box(0.53, 0.55, 0.18, 0.30, "Training\n+ Validation")
    box(0.78, 0.55, 0.18, 0.30, "Artefact\n(model + feature\ncontract)")

    # Bottom: inference -> predictions -> monitoring (loop)
    box(0.28, 0.10, 0.18, 0.30, "Inference\n(online/offline)")
    box(0.53, 0.10, 0.18, 0.30, "Predictions\n(y_pred @ t)")
    box(0.78, 0.10, 0.18, 0.30, "Monitoring\n(PSI, drift,\nmetrics)")

    # Arrows (top)
    ax.annotate("", xy=(0.28, 0.70), xytext=(0.21, 0.70), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.53, 0.70), xytext=(0.46, 0.70), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.78, 0.70), xytext=(0.71, 0.70), arrowprops=dict(arrowstyle="->"))
    # Inference path
    ax.annotate("", xy=(0.37, 0.40), xytext=(0.37, 0.55), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.62, 0.25), xytext=(0.46, 0.25), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.88, 0.25), xytext=(0.71, 0.25), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(0.88, 0.55), xytext=(0.88, 0.40), arrowprops=dict(arrowstyle="->"))

    fig.savefig(FIGS_OUT / "pipeline_architecture.png", bbox_inches="tight")
    plt.close(fig)

def _draw_rolling_origin() -> None:
    """Schéma de validation à origine glissante."""
    fig = plt.figure(figsize=(10, 2.8))
    ax = plt.gca(); ax.axis("off")

    base_x = 0.05; y = 0.45
    train_w, val_w, test_w = 0.25, 0.08, 0.10
    gap = 0.03

    for i in range(4):
        x = base_x + i * (train_w + val_w + test_w + gap)
        ax.add_patch(Rectangle((x, y), train_w, 0.18, fill=False))
        ax.text(x + train_w/2, y + 0.09, "Train", ha="center", va="center", fontsize=9)
        xv = x + train_w
        ax.add_patch(Rectangle((xv, y), val_w, 0.18, fill=False))
        ax.text(xv + val_w/2, y + 0.09, "Val", ha="center", va="center", fontsize=9)
        xt = xv + val_w
        ax.add_patch(Rectangle((xt, y), test_w, 0.18, fill=False))
        ax.text(xt + test_w/2, y + 0.09, "Test", ha="center", va="center", fontsize=9)
        if i < 3:
            ax.annotate("", xy=(xt + test_w + 0.01, y + 0.09),
                        xytext=(xt + test_w + gap - 0.01, y + 0.09),
                        arrowprops=dict(arrowstyle="->"))

    ax.text(0.5, 0.12, "Temps →", ha="center", va="center", fontsize=10)
    fig.savefig(FIGS_OUT / "rolling_origin.png", bbox_inches="tight")
    plt.close(fig)

def _plot_events_coverage_daily(ev: pd.DataFrame) -> None:
    """Courbe du nombre de lignes par jour dans events.parquet (couverture)."""
    fig = plt.figure(figsize=(9, 3.6))
    if "ts" in ev.columns and not ev["ts"].isna().all():
        ts = pd.to_datetime(ev["ts"], errors="coerce").dropna()
        if len(ts):
            s = pd.Series(1, index=ts)
            daily = s.resample("D").sum().fillna(0)
            plt.plot(daily.index, daily.values)
            plt.title("Couverture quotidienne — events.parquet")
            plt.xlabel("date"); plt.ylabel("lignes / jour")
            fig.autofmt_xdate()
        else:
            plt.axis("off")
            plt.text(0.5, 0.5, "Aucune date exploitable dans events.ts", ha="center", va="center")
    else:
        plt.axis("off")
        plt.text(0.5, 0.5, "Colonne 'ts' absente ou vide dans events.parquet", ha="center", va="center")

    fig.savefig(FIGS_OUT / "events_coverage_daily.png", bbox_inches="tight")
    plt.close(fig)

def _draw_dataflow_doc() -> None:
    """Petit schéma de dataflow du builder (doc)."""
    fig = plt.figure(figsize=(7.5, 3.2))
    plt.axis("off")
    plt.text(0.05, 0.6, "events.parquet", fontsize=11, bbox=dict(boxstyle="round", fc="white", ec="black"))
    plt.text(0.35, 0.6, "build_model_pipeline.py", fontsize=11, bbox=dict(boxstyle="round", fc="white", ec="black"))
    plt.text(0.72, 0.7, "tables/…", fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="black"))
    plt.text(0.72, 0.45, "figs/…", fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="black"))
    plt.annotate("", xy=(0.33, 0.63), xytext=(0.20, 0.63), arrowprops=dict(arrowstyle="->"))
    plt.annotate("", xy=(0.70, 0.70), xytext=(0.53, 0.63), arrowprops=dict(arrowstyle="->"))
    plt.annotate("", xy=(0.70, 0.47), xytext=(0.53, 0.63), arrowprops=dict(arrowstyle="->"))
    fig.savefig(FIGS_OUT / "dataflow.png", bbox_inches="tight")
    plt.close(fig)

# --------------------------- markdown template ------------------------------

SECTION_2_PIPELINE = """## 6) Explications détaillées

### Objectif
Décrire précisément **d’où viennent les données**, **comment on fabrique les features**, **comment on entraîne** et **versionne** le modèle.

### Données d’entrée
- **Séries réseau** : vélos/docks disponibles par station (`events.parquet`).  
- **Calendrier** : heure/minute, jour de semaine, jours fériés, vacances (si disponibles).  
- **Météo** (optionnel) : température, pluie, vent (si intégrée et historisée).  
- **Géographie légère** (optionnelle) : arrondissement/quartier, altitude, distance centre.

### Construction des features (exemples typiques)
- **Retards (lags)** : valeurs H−15, −30, −45, −60… (pas d’info future).  
- **Fenêtres glissantes** : moyennes/medians/écarts-types sur 1–6 h, indicateurs de volatilité.  
- **Saisonnalité** : heure **sin/cos**, jour de semaine encodé, période scolaire.  
- **Interactions légères** : lags × heure, météo × heure (si pertinent).  
- **Capacité & normalisation** : ratios (`y/capacité`) lorsque la comparaison inter-stations est requise.

### Entraînement & validation
- **Découpes temporelles** : train → val → test **dans l’ordre du temps**.  
- **Validation à origine glissante** (rolling origin) pour évaluer la robustesse.  
- **Objectif** : minimiser MAE (primat opérationnel), contrôle RMSE/biais.  
- **Anti-fuite** : toutes les features utilisent **exclusivement** des informations ≤ t (aucun futur).

### Artefacts & versioning
- Modèle sérialisé (ex. `joblib`) + **signature** des features attendues.  
- **Version sémantique** (ex. `vX.Y.Z`) liée au schéma de features & à l’horizon.  
- Journal de **reproductibilité** : seed, plage temporelle d’entraînement, métriques de val/test.  
- **Planification** : ré-entraînement périodique (ex. quotidien/hebdo) ou à l’alerte monitoring.

### Déploiement & prédiction
- **Chargement** depuis l’artefact, vérification du **contrat de features** (colonnes, types, ordre).  
- Production de `y_pred` à chaque pas pour chaque station prévue, **timestampée à t** (et non t+h).
"""

MD_TEMPLATE = """# Modèle — Pipeline d’entraînement & features

> _Vue d’ensemble visuelle et documentée du pipeline (sources → features → entraînement → artefact → inférence → monitoring)._

## 1) Aperçu visuel
<p>
  <img src="{pipeline_arch_rel}" alt="Architecture du pipeline" width="100%"/>
</p>

## 2) Quick stats
- **Période events** : {ts_min_events} → {ts_max_events}
- **Période perf**   : {ts_min_perf} → {ts_max_perf}
- **Stations**       : events={events_stations} · perf={perf_stations}
- **Horizon (min)**  : {horizon_min}
- **Features**       : {features_count} (source: {features_source})

### Artefacts (liens techniques)
- **Schémas** : [events]({schema_events_rel}) · [perf]({schema_perf_rel})
- **Contrat des features** : [features_contract.csv]({features_contract_rel}) · [features_by_family.csv]({features_by_family_rel})
- **Overview JSON** : [pipeline_overview.json]({overview_json_rel})

---

## 3) Validation & Données — en un coup d’œil
<div style="display:flex; gap: 1rem; flex-wrap: wrap;">
  <figure style="flex:1; min-width: 320px;">
    <img src="{rolling_origin_rel}" alt="Rolling origin" width="100%"/>
    <figcaption>Validation à origine glissante (robustesse temporelle).</figcaption>
  </figure>
  <figure style="flex:1; min-width: 320px;">
    <img src="{coverage_daily_rel}" alt="Couverture quotidienne events" width="100%"/>
    <figcaption>Couverture quotidienne des events (volumétrie & trous).</figcaption>
  </figure>
</div>

---

## 4) Features — répartition & contrat
<figure>
  <img src="{features_fig_rel}" alt="Répartition des features par famille" width="70%"/>
  <figcaption>Répartition des features par famille.</figcaption>
</figure>

**Tableau des familles**
{family_table_md}

**Aperçu des features** (extrait) :
{features_preview_md}

---

## 5) Dataflow du builder (doc)
<figure>
  <img src="{dataflow_fig_rel}" alt="Dataflow du builder" width="70%"/>
  <figcaption>De `events.parquet` → builder → artefacts docs/assets/…</figcaption>
</figure>

---

{section2}

---

## 7) Annexes & liens
- `{schema_events_rel}`
- `{schema_perf_rel}`
- `{features_contract_rel}`
- `{features_by_family_rel}`
- `{overview_json_rel}`
"""

# ------------------------------- main ---------------------------------------

def main(events_path: str, perf_path: str, horizon: Optional[int]) -> None:
    _log("Starting…")

    # Load datasets
    ev = pd.read_parquet(events_path)
    pf = pd.read_parquet(perf_path)

    # Write schemas (sous docs/)
    _df_schema(ev).to_csv(TABLES_OUT / "schema_events.csv", index=False)
    _df_schema(pf).to_csv(TABLES_OUT / "schema_perf.csv", index=False)
    _log(f"Schémas → {TABLES_OUT / 'schema_events.csv'} ; {TABLES_OUT / 'schema_perf.csv'}")

    # Overview JSON (dates, stations, horizon)
    def _safe_ts_bounds(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        if "ts" not in df.columns or df["ts"].isna().all():
            return None, None
        ts = pd.to_datetime(df["ts"], errors="coerce").dropna()
        if not len(ts):
            return None, None
        return ts.min().isoformat(), ts.max().isoformat()

    ts_min_ev, ts_max_ev = _safe_ts_bounds(ev)
    ts_min_pf, ts_max_pf = _safe_ts_bounds(pf)

    overview = {
        "events_rows": int(len(ev)),
        "perf_rows": int(len(pf)),
        "events_stations": int(ev["station_id"].nunique()) if "station_id" in ev.columns else None,
        "perf_stations": int(pf["station_id"].nunique()) if "station_id" in pf.columns else None,
        "ts_min_events": ts_min_ev,
        "ts_max_events": ts_max_ev,
        "ts_min_perf": ts_min_pf,
        "ts_max_perf": ts_max_pf,
        "horizon_min": int(horizon) if horizon is not None else None,
        "outputs": {"tables": str(TABLES_OUT), "figs": str(FIGS_OUT)},
    }
    (TABLES_OUT / "pipeline_overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")

    # Feature list detection
    feats, meta = _try_load_feature_list(horizon)
    src = meta.get("source") or "(none)"
    feat_count = len(feats)

    if feats:
        _log(f"Features: {feat_count} (source: {src})")
    else:
        _log("Features non détectées — tables placeholders écrites.")

    _write_features_contract(feats)

    # Génère tous les visuels (embedés ensuite dans la page)
    _draw_pipeline_architecture()
    _draw_rolling_origin()
    _plot_events_coverage_daily(ev)
    _draw_dataflow_doc()

    # ---------- Build feature families table & preview list for MD ----------
    if feats:
        families = [_family_of(f) for f in feats]
        fam_counts = pd.Series(families, name="family").value_counts().reset_index()
        fam_counts.columns = ["family", "count"]
        fam_counts = fam_counts.sort_values(["count","family"], ascending=[False, True]).reset_index(drop=True)
        family_table_md = _md_table(fam_counts, ["family", "count"])
        preview_n = min(40, len(feats))
        preview_lines = "\n".join(f"- `{x}`" for x in feats[:preview_n])
        if len(feats) > preview_n:
            preview_lines += f"\n… (+{len(feats) - preview_n} autres)"
        features_preview_md = preview_lines
    else:
        family_table_md = "_(aucune feature détectée)_"
        features_preview_md = "_(aucune feature détectée)_"

    # ----------------------------- Markdown ---------------------------------
    # Liens relatifs depuis docs/model/pipeline.md
    schema_events_rel      = rel_from_md(MD_OUT, TABLES_OUT / "schema_events.csv")
    schema_perf_rel        = rel_from_md(MD_OUT, TABLES_OUT / "schema_perf.csv")
    features_contract_rel  = rel_from_md(MD_OUT, TABLES_OUT / "features_contract.csv")
    features_by_family_rel = rel_from_md(MD_OUT, TABLES_OUT / "features_by_family.csv")
    overview_json_rel      = rel_from_md(MD_OUT, TABLES_OUT / "pipeline_overview.json")
    features_fig_rel       = rel_from_md(MD_OUT, FIGS_OUT   / "features_by_family.png")
    dataflow_fig_rel       = rel_from_md(MD_OUT, FIGS_OUT   / "dataflow.png")
    pipeline_arch_rel      = rel_from_md(MD_OUT, FIGS_OUT   / "pipeline_architecture.png")
    rolling_origin_rel     = rel_from_md(MD_OUT, FIGS_OUT   / "rolling_origin.png")
    coverage_daily_rel     = rel_from_md(MD_OUT, FIGS_OUT   / "events_coverage_daily.png")

    md = MD_TEMPLATE.format(
        ts_min_events=ts_min_ev,
        ts_max_events=ts_max_ev,
        ts_min_perf=ts_min_pf,
        ts_max_perf=ts_max_pf,
        events_stations=overview["events_stations"],
        perf_stations=overview["perf_stations"],
        horizon_min=overview["horizon_min"],
        features_count=feat_count,
        features_source=src,
        schema_events_rel=schema_events_rel,
        schema_perf_rel=schema_perf_rel,
        features_contract_rel=features_contract_rel,
        features_by_family_rel=features_by_family_rel,
        overview_json_rel=overview_json_rel,
        rolling_origin_rel=rolling_origin_rel,
        coverage_daily_rel=coverage_daily_rel,
        features_fig_rel=features_fig_rel,
        dataflow_fig_rel=dataflow_fig_rel,
        pipeline_arch_rel=pipeline_arch_rel,
        family_table_md=family_table_md,
        features_preview_md=features_preview_md,
        section2=SECTION_2_PIPELINE.strip(),
    )
    MD_OUT.write_text(md, encoding="utf-8", newline="\n")
    _log(f"Markdown → {MD_OUT}")

    _log("Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="Chemin vers docs/exports/events.parquet")
    ap.add_argument("--perf", required=True, help="Chemin vers docs/exports/perf.parquet")
    ap.add_argument("--horizon", type=int, default=None, help="Horizon minutes (pour charger le bundle si nécessaire)")
    # Accepté pour homogénéité CLI, non utilisé par ce script
    ap.add_argument("--tz", default="Europe/Paris", help="IANA timezone (optionnel)")
    args = ap.parse_args()

    try:
        main(events_path=args.events, perf_path=args.perf, horizon=args.horizon)
    except KeyboardInterrupt:
        _log("Interrupted")
        sys.exit(130)
    except Exception as e:
        _log(f"ERROR: {type(e).__name__}: {e}")
        raise

