# tools/build_data_methodology.py
# Page builder — "Data / Méthodologie & licences"
#
# Ce script matérialise la méthodologie et produit :
# - Tables CSV/JSON de vérifications
# - Figures (dataflow, scores d'horizon, barres d'alignement)
# - La page Markdown complète: docs/data/methodology.md
#
# Entrées: docs/exports/events.parquet, docs/exports/perf.parquet
# Sorties:
#   docs/assets/tables/data/methodology/*.csv|json
#   docs/assets/figs/data/methodology/*.png
#   docs/data/methodology.md

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import pandas as pd

# Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Console (évite les erreurs d'encodage Windows)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# --------------------------- Chemins ---------------------------

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS_DIR = DOCS / "assets"
TABLES_DIR = ASSETS_DIR / "tables" / "data" / "methodology"
FIGS_DIR = ASSETS_DIR / "figs" / "data" / "methodology"
OUT_MD = DOCS / "data" / "methodology.md"

for d in (TABLES_DIR, FIGS_DIR, OUT_MD.parent):
    d.mkdir(parents=True, exist_ok=True)


def rel_from_md(md_path: Path, target: Path) -> str:
    """
    Chemin relatif POSIX depuis md_path vers target (compatible MkDocs use_directory_urls:true)
    """
    md_rel = Path(md_path).resolve().relative_to(DOCS.resolve())
    parts = md_rel.with_suffix("").parts     # ('data','methodology') ou ('index',)
    depth = len(parts) if parts[-1] != "index" else len(parts) - 1
    prefix = "../" * max(depth, 0)
    rel_from_docs = Path(target).resolve().relative_to(DOCS.resolve()).as_posix()
    return (prefix + rel_from_docs).replace("//", "/")


# --------------------------- Utilitaires génériques ---------------------------

def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z"


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet introuvable: {path}")
    return pd.read_parquet(path)


def _coerce_ts_15(df: pd.DataFrame, col: str = "ts") -> pd.DataFrame:
    if col not in df.columns:
        return df
    out = df.copy()
    out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    out[col] = out[col].dt.floor("15min")
    return out


def _ensure_station_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "station_id" in out.columns:
        try:
            out["station_id"] = pd.to_numeric(out["station_id"], errors="coerce").astype("Int64")
        except Exception:
            pass
    return out


def _schema(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        dtype = str(s.dtype)
        nnull = int(s.isna().sum())
        nunique = int(s.nunique(dropna=True))
        sample = s.dropna().head(3).tolist()
        rows.append({"column": c, "dtype": dtype, "nulls": nnull, "nunique": nunique, "sample": sample})
    return pd.DataFrame(rows)


def _hash_contract(columns: List[str]) -> str:
    m = hashlib.sha256()
    payload = ",".join(sorted(columns)).encode("utf-8")
    m.update(payload)
    return m.hexdigest()[:16]


def _guess_license_name(text: str) -> str:
    t = text.strip().lower()
    if "mit license" in t:
        return "MIT"
    if "apache license" in t or "apache-2.0" in t:
        return "Apache-2.0"
    if "gnu general public license" in t or "gpl" in t:
        return "GPL"
    if "creative commons" in t or "cc-by" in t or "cc0" in t:
        return "Creative Commons"
    if "odbl" in t or "open database license" in t:
        return "ODbL"
    return "Custom/Unknown"


def _read_text_if_exists(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")


# --------------------------- Normalisation ---------------------------

def normalization_report(events: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Events attendu: ts, station_id, bikes, docks (status optionnel)
    ev_cols = ["ts", "station_id", "bikes", "docks"]
    ev_ok = all(c in events.columns for c in ev_cols)
    rows.append({"table": "events", "check": "columns_presence", "ok": bool(ev_ok), "expected": ev_cols})

    # Perf minimal: ts, station_id, y_true (y_pred, baseline, horizon_min optionnels)
    pf_cols_min = ["ts", "station_id", "y_true"]
    pf_ok_min = all(c in perf.columns for c in pf_cols_min)
    rows.append({"table": "perf", "check": "columns_presence_min", "ok": bool(pf_ok_min), "expected": pf_cols_min})

    def _dtype_col(df, col):
        return str(df[col].dtype) if col in df.columns else None

    rows.extend([
        {"table": "events", "check": "dtype_ts", "value": _dtype_col(events, "ts")},
        {"table": "events", "check": "dtype_station_id", "value": _dtype_col(events, "station_id")},
        {"table": "events", "check": "dtype_bikes", "value": _dtype_col(events, "bikes")},
        {"table": "perf", "check": "dtype_ts", "value": _dtype_col(perf, "ts")},
        {"table": "perf", "check": "dtype_station_id", "value": _dtype_col(perf, "station_id")},
        {"table": "perf", "check": "dtype_y_true", "value": _dtype_col(perf, "y_true")},
        {"table": "perf", "check": "dtype_y_pred", "value": _dtype_col(perf, "y_pred") if "y_pred" in perf.columns else None},
        {"table": "perf", "check": "dtype_y_pred_baseline", "value": _dtype_col(perf, "y_pred_baseline") if "y_pred_baseline" in perf.columns else None},
        {"table": "perf", "check": "dtype_horizon_min", "value": _dtype_col(perf, "horizon_min") if "horizon_min" in perf.columns else None},
    ])
    return pd.DataFrame(rows)


# --------------------------- Cible & baseline ---------------------------

def _best_horizon_from_events(perf: pd.DataFrame, events: pd.DataFrame) -> Tuple[Optional[int], Dict[int, float]]:
    """
    Si perf.horizon_min absent, on infère l’horizon qui rend y_true ≈ events.bikes(ts+h).
    On teste une grille et on prend le max du % d’égalité.
    """
    candidates = [15, 30, 45, 60, 90, 120, 150, 180]
    scores = {}
    for h in candidates:
        ev = events[["ts", "station_id", "bikes"]].copy()
        ev["ts_shift_back"] = ev["ts"] - pd.Timedelta(minutes=h)  # ts_shift_back == T (perf)
        joined = perf.merge(
            ev[["ts_shift_back", "station_id", "bikes"]],
            left_on=["ts", "station_id"], right_on=["ts_shift_back", "station_id"], how="left"
        )
        if "y_true" not in joined.columns:
            scores[h] = np.nan
            continue
        ok = (np.abs(joined["y_true"] - joined["bikes"]) <= 1e-6).mean()
        try:
            scores[h] = float(ok)
        except Exception:
            scores[h] = np.nan

    valid = {k: v for k, v in scores.items() if pd.notna(v)}
    if not valid:
        return None, scores
    best = max(valid, key=lambda k: valid[k])
    return int(best), scores


def plot_horizon_scores(scores: Dict[int, float], out_png: Path) -> None:
    plt.figure(figsize=(8, 3.2))
    ks = sorted(scores.keys())
    vs = [scores[k] * 100.0 if pd.notna(scores[k]) else np.nan for k in ks]
    plt.bar([str(k) for k in ks], vs)
    plt.ylabel("% d’alignement y_true ≈ bikes@T+h")
    plt.xlabel("Horizon (minutes)")
    plt.title("Scores d’alignement par horizon (inférence)")
    plt.ylim(0, 100)
    _save_fig(out_png)


def plot_alignment_bars(target_share: float, baseline_share: float, out_png: Path) -> None:
    plt.figure(figsize=(6, 3.2))
    labels = ["Cible @ T+h vs events", "Baseline @ T vs events"]
    vals = [
        float(target_share) * 100.0 if pd.notna(target_share) else np.nan,
        float(baseline_share) * 100.0 if pd.notna(baseline_share) else np.nan,
    ]
    xs = np.arange(len(labels))
    plt.bar(xs, vals)
    plt.xticks(xs, labels, rotation=10)
    plt.ylabel("% d’égalité exacte")
    plt.title("Vérifications d’alignement (part d’égalité)")
    plt.ylim(0, 100)
    _save_fig(out_png)


def target_and_baseline_checks(events: pd.DataFrame, perf: pd.DataFrame, override_hz: Optional[int] = None) -> pd.DataFrame:
    rows = []

    # 1) Horizon retenu
    if override_hz is not None:
        hz = int(override_hz); hz_src = "cli:--horizon"
        _, scores = _best_horizon_from_events(perf, events)  # pour la figure d’appoint
    elif "horizon_min" in perf.columns:
        horizon = pd.to_numeric(perf["horizon_min"], errors="coerce").dropna()
        hz = int(horizon.iloc[0]) if len(horizon) else None
        hz_src = "perf:horizon_min"
        _, scores = _best_horizon_from_events(perf, events)  # pour la figure
    else:
        hz, scores = _best_horizon_from_events(perf, events)
        hz_src = "inferred"

    rows.append({"check": "horizon_minutes", "value": hz, "source": hz_src})

    # 2) y_true ≈ events.bikes(ts+h)
    if hz is not None:
        ev = events[["ts", "station_id", "bikes"]].copy()
        ev["ts_shift_back"] = ev["ts"] - pd.Timedelta(minutes=hz)
        j = perf.merge(
            ev[["ts_shift_back", "station_id", "bikes"]],
            left_on=["ts", "station_id"], right_on=["ts_shift_back", "station_id"], how="left"
        )
        prop_target = float((np.abs(j["y_true"] - j["bikes"]) <= 1e-6).mean()) if "y_true" in j.columns else np.nan
        rows.append({"check": "target_alignment_share", "value": prop_target, "source": f"events.bikes @ T+{hz}min"})
    else:
        prop_target = np.nan
        rows.append({"check": "target_alignment_share", "value": np.nan, "source": "unknown_horizon"})

    # 3) y_pred_baseline ≈ events.bikes(ts)
    e2 = events[["ts", "station_id", "bikes"]].copy()
    j2 = perf.merge(e2, on=["ts", "station_id"], how="left", suffixes=("", "_events"))
    if "y_pred_baseline" in j2.columns:
        prop_base = float((np.abs(j2["y_pred_baseline"] - j2["bikes"]) <= 1e-6).mean())
    else:
        prop_base = np.nan
    rows.append({"check": "baseline_persistence_share", "value": prop_base, "source": "events.bikes @ T"})

    # On attache les valeurs utiles pour les figures (stockées au niveau objet via attributs)
    df = pd.DataFrame(rows)
    df._hz = hz
    df._hz_src = hz_src
    df._scores = scores
    df._prop_target = prop_target
    df._prop_base = prop_base
    return df


# --------------------------- Injection modèle ---------------------------

def model_injection_report(perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Couverture des prédictions
    cov = float(perf["y_pred"].notna().mean()) if "y_pred" in perf.columns else np.nan
    rows.append({"check": "model_prediction_coverage", "value": cov})

    # Sanity: |y_pred - y_true| (NB: y_true est à T+h)
    if "y_pred" in perf.columns and "y_true" in perf.columns:
        try:
            err = float(np.nanmean(np.abs(perf["y_pred"] - perf["y_true"])))
        except Exception:
            err = np.nan
    else:
        err = np.nan
    rows.append({"check": "model_abs_error_sanity", "value": err})
    return pd.DataFrame(rows)


# --------------------------- Versionnage & compatibilité ---------------------------

def versioning(events: pd.DataFrame, perf: pd.DataFrame) -> Dict[str, object]:
    ev_cols = list(events.columns)
    pf_cols = list(perf.columns)
    return {
        "contract": {
            "events": {"columns": ev_cols, "hash": _hash_contract(ev_cols)},
            "perf": {"columns": pf_cols, "hash": _hash_contract(pf_cols)},
        }
    }


# --------------------------- Licences ---------------------------

def licenses_report(code_license_path: Optional[Path], data_license_path: Optional[Path]) -> pd.DataFrame:
    code_text = _read_text_if_exists(code_license_path)
    data_text = _read_text_if_exists(data_license_path)
    rows = []
    rows.append({"artifact": "code",
                 "license_name": _guess_license_name(code_text) if code_text else None,
                 "length": len(code_text) if code_text else 0})
    rows.append({"artifact": "data",
                 "license_name": _guess_license_name(data_text) if data_text else None,
                 "length": len(data_text) if data_text else 0})
    return pd.DataFrame(rows)


# --------------------------- Figures ---------------------------

def plot_dataflow(out_png: Path) -> None:
    plt.figure(figsize=(10, 6))
    ax = plt.gca(); ax.axis("off")

    def box(x, y, w, h, text, fc="#f0f4ff"):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=8",
                              ec="#1f3a93", fc=fc, lw=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10)

    def arrow(xy1, xy2):
        arr = FancyArrowPatch(xy1, xy2, arrowstyle="->", mutation_scale=12, lw=1.2, ec="#333")
        ax.add_patch(arr)

    box(0.05, 0.65, 0.25, 0.2, "Normalisation\n(events)\nts, station_id,\nbikes, docks")
    box(0.05, 0.30, 0.25, 0.2, "Construction perf\n(ts, station_id,\ny_true @T+h, baseline,\ny_pred?, horizon_min)")
    arrow((0.175, 0.65), (0.175, 0.52)); arrow((0.175, 0.30), (0.175, 0.17))

    box(0.40, 0.60, 0.28, 0.22, "Contrôles\n(schéma, types,\narrondi 15min)")
    box(0.40, 0.25, 0.28, 0.22, "Cible & baseline\n(alignements,\nhorizon)")
    arrow((0.30, 0.70), (0.40, 0.70)); arrow((0.30, 0.35), (0.40, 0.35))

    box(0.75, 0.60, 0.20, 0.22, "Versionnage\n(contrats, hash)")
    box(0.75, 0.25, 0.20, 0.22, "Licences\n(code & data)")
    arrow((0.68, 0.70), (0.75, 0.70)); arrow((0.68, 0.35), (0.75, 0.35))

    ax.text(0.5, 0.03, "Dataflow — Méthodologie", ha="center", fontsize=11)
    _save_fig(out_png)


# --------------------------- Markdown ---------------------------

MD_TEMPLATE = """# Données — Méthodologie & licences

Cette page documente **comment** les exports sont produits et **dans quel cadre** ils peuvent être utilisés.

---

## 1) Méthodologie de fabrication (vue d’ensemble)

- **Normalisation** : renommage des colonnes source → **schéma canonique**, arrondi des timestamps à 15 min, harmonisation des types.  
- **Séparation** en deux vues :  
  - **`events.parquet`** (état instantané) ;  
  - **`perf.parquet`** (vérité à T+h ramenée à T, baseline, et prédictions si injectées).  
- **Cible & baseline** : `y_true` = `bikes` à **T+h** par station ; `y_pred_baseline` = persistance (`bikes` à **T**).  
- **Injection modèle** : `y_pred` est ajoutée après normalisation, en garantissant l’alignement sur **T** et le **mapping station** robuste (nom/lat/lon → `station_id`).

![Dataflow]({dataflow_rel})

---

## 2) Qualité & monitoring

- Contrôles automatiques **à chaque build** : fraîcheur, complétude, schéma, anomalies (séries plates, doublons).  
- **Drift** suivi par PSI/K–S sur les features clés (si disponibles) et dérive de cible.  
- **Traçabilité** : horodatages de build et métriques exposés dans `docs/assets/tables/`.

### Cible & baseline — vérifications clés

- **Horizon retenu** : **{hz} min** ({hz_src}).  
- **Part d’égalité** cible vs events @T+{hz} : **{target_share_pct:.2f}%**  
- **Part d’égalité** baseline vs events @T : **{baseline_share_pct:.2f}%**

![Barres d’alignement]({align_bars_rel})

**Scores d’alignement par horizon (inférence)**  
*(utile si `horizon_min` n’est pas présent dans `perf`)*

![Scores d’horizon]({hz_scores_rel})

---

## 3) Versionnage & compatibilité

- Le **contrat de schéma** est **stable** ; toute rupture sera annoncée via **bump de version** et *release notes*.  
- Les colonnes optionnelles peuvent **apparaître/disparaître** sans rompre le contrat (elles sont marquées *optionnelles* dans le dictionnaire).

- Empreintes actuelles (hash):  
  - `events` → `{events_hash}`  
  - `perf` → `{perf_hash}`

---

## 4) Licences & usages

- **Données dérivées** : les exports restent soumis à la **licence de la source originale** (respecter attribution/partage).  
- **Code** : licence du dépôt (ex. MIT).  
- **Usages** : pas de tentative de **ré-identification** ; pas d’usage contraire aux CGU de la source ; indiquer l’**UTC** lors de toute republication de chronologies.

> Licences détectées : code = **{code_license}**, data = **{data_license}**.

---

## 5) Limites & transparence

- Les exports **reflètent l’état réel** de l’ingestion : pas d’imputation lourde ; les trous sont signalés, pas “réparés”.  
- Les capacités peuvent évoluer ; `occ` est une approximation lorsque la capacité n’est pas officiellement publiée.

---

## 6) Tables & artefacts

- **Schémas effectifs** :  
  - `{events_schema_rel}`  
  - `{perf_schema_rel}`  
- **Contrôles de normalisation** : `{norm_rel}`  
- **Cible & baseline (checks)** : `{tbc_rel}`  
- **Injection modèle (couverture/erreur)** : `{inj_rel}`  
- **Versioning (JSON)** : `{versioning_rel}`  
- **Licences (CSV)** : `{licenses_rel}`  
- **Résumé de build** : `{summary_rel}`

*Fuseau d’affichage (paramètre) : `{tz_display}` — Build UTC : `{build_utc}`.*
"""

# --------------------------- Main ---------------------------

def main(
    events_path: Path,
    perf_path: Path,
    tz: Optional[str],
    data_license_path: Optional[Path],
    code_license_path: Optional[Path],
    horizon_override: Optional[int] = None,
) -> None:

    # Charger & normaliser
    events = _ensure_station_id(_coerce_ts_15(_read_parquet(events_path), "ts"))
    perf = _ensure_station_id(_coerce_ts_15(_read_parquet(perf_path), "ts"))

    # Schémas effectifs
    events_schema_csv = TABLES_DIR / "events_schema_actual.csv"
    perf_schema_csv = TABLES_DIR / "perf_schema_actual.csv"
    _schema(events).to_csv(events_schema_csv, index=False)
    _schema(perf).to_csv(perf_schema_csv, index=False)

    # Normalisation
    norm_csv = TABLES_DIR / "normalization_report.csv"
    norm = normalization_report(events, perf)
    norm.to_csv(norm_csv, index=False)

    # Cible & baseline (+ figures)
    tbc = target_and_baseline_checks(events, perf, override_hz=horizon_override)
    tbc_csv = TABLES_DIR / "target_baseline_checks.csv"
    tbc.to_csv(tbc_csv, index=False)

    hz = getattr(tbc, "_hz", None)
    hz_src = getattr(tbc, "_hz_src", "unknown")
    scores = getattr(tbc, "_scores", {})
    prop_target = getattr(tbc, "_prop_target", np.nan)
    prop_base = getattr(tbc, "_prop_base", np.nan)

    # Figures
    dataflow_png = FIGS_DIR / "dataflow.png"
    plot_dataflow(dataflow_png)

    hz_scores_png = FIGS_DIR / "horizon_alignment_scores.png"
    if scores:
        plot_horizon_scores(scores, hz_scores_png)
    else:
        # figure de fallback
        plt.figure(figsize=(6, 2.2))
        plt.text(0.5, 0.5, "Scores d’horizon indisponibles", ha="center", va="center")
        plt.axis("off")
        _save_fig(hz_scores_png)

    align_bars_png = FIGS_DIR / "alignment_checks.png"
    plot_alignment_bars(prop_target, prop_base, align_bars_png)

    # Injection modèle
    inj_csv = TABLES_DIR / "model_injection_report.csv"
    inj = model_injection_report(perf)
    inj.to_csv(inj_csv, index=False)

    # Versionnage
    ver_json = TABLES_DIR / "versioning.json"
    ver = versioning(events, perf)
    ver_json.write_text(json.dumps(ver, ensure_ascii=False, indent=2), encoding="utf-8")
    events_hash = ver["contract"]["events"]["hash"]
    perf_hash = ver["contract"]["perf"]["hash"]

    # Licences
    lic_csv = TABLES_DIR / "licenses.csv"
    lic = licenses_report(code_license_path=code_license_path, data_license_path=data_license_path)
    lic.to_csv(lic_csv, index=False)
    code_license = lic.loc[lic["artifact"] == "code", "license_name"].iloc[0] if not lic.empty else None
    data_license = lic.loc[lic["artifact"] == "data", "license_name"].iloc[0] if not lic.empty else None

    # Summary JSON
    summary_json = TABLES_DIR / "summary.json"
    summary = {
        "build_utc": _now_iso(),
        "inputs": {
            "events": {
                "path": str(events_path),
                "rows": int(len(events)),
                "ts_min": events["ts"].min().isoformat() if len(events) else None,
                "ts_max": events["ts"].max().isoformat() if len(events) else None,
            },
            "perf": {
                "path": str(perf_path),
                "rows": int(len(perf)),
                "ts_min": perf["ts"].min().isoformat() if len(perf) else None,
                "ts_max": perf["ts"].max().isoformat() if len(perf) else None,
            },
        },
        "timezone_display": tz,
        "artifacts": {"tables_dir": str(TABLES_DIR), "figs_dir": str(FIGS_DIR)},
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # --------------------- Rendu Markdown ---------------------
    md = MD_TEMPLATE.format(
        # Visuels
        dataflow_rel=rel_from_md(OUT_MD, dataflow_png),
        hz_scores_rel=rel_from_md(OUT_MD, hz_scores_png),
        align_bars_rel=rel_from_md(OUT_MD, align_bars_png),
        # Valeurs dynamiques
        hz=hz if hz is not None else "N/A",
        hz_src=hz_src,
        target_share_pct=(prop_target * 100.0) if pd.notna(prop_target) else float("nan"),
        baseline_share_pct=(prop_base * 100.0) if pd.notna(prop_base) else float("nan"),
        events_hash=events_hash, perf_hash=perf_hash,
        code_license=code_license, data_license=data_license,
        # Tables
        events_schema_rel=rel_from_md(OUT_MD, events_schema_csv),
        perf_schema_rel=rel_from_md(OUT_MD, perf_schema_csv),
        norm_rel=rel_from_md(OUT_MD, norm_csv),
        tbc_rel=rel_from_md(OUT_MD, tbc_csv),
        inj_rel=rel_from_md(OUT_MD, inj_csv),
        versioning_rel=rel_from_md(OUT_MD, ver_json),
        licenses_rel=rel_from_md(OUT_MD, lic_csv),
        summary_rel=rel_from_md(OUT_MD, summary_json),
        # Métadonnées
        tz_display=tz,
        build_utc=summary["build_utc"],
    )

    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)

    print(f"[data/methodology] OK -> {OUT_MD}")
    print(f"[data/methodology] figs -> {FIGS_DIR}")
    print(f"[data/methodology] tables -> {TABLES_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build 'Data / Méthodologie & licences' (tables + figures + MD)")
    ap.add_argument("--events", type=Path, required=True, help="Path to docs/exports/events.parquet")
    ap.add_argument("--perf", type=Path, required=True, help="Path to docs/exports/perf.parquet")
    ap.add_argument("--tz", type=str, default="Europe/Paris", help="Fuseau d’affichage (ex. Europe/Paris)")
    ap.add_argument("--data-license", type=Path, default=None, help="Chemin fichier licence données (optionnel)")
    ap.add_argument("--code-license", type=Path, default=None, help="Chemin fichier licence code (optionnel)")
    ap.add_argument("--horizon", type=int, default=None, help="Horizon (minutes) pour forcer l’alignement T→T+h")
    args = ap.parse_args()

    main(
        events_path=args.events,
        perf_path=args.perf,
        tz=args.tz,
        data_license_path=args.data_license,
        code_license_path=args.code_license,
        horizon_override=args.horizon,
    )
