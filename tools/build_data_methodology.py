# tools/build_data_methodology.py
# Page builder — "Data / Méthodologie & licences"
#
# Ce script matérialise, sous forme de tableaux/JSON/figures,
# la méthodologie décrite dans `methodology.md` :
# - Normalisation (schéma canonique, arrondi 15 min, types)
# - Séparation events/perf
# - Cible & baseline (vérifications d’alignement)
# - Injection modèle (couverture, alignement à T)
# - Versionnage & compatibilité (empreinte de contrat)
# - Licences (code + données si fichier fourni)
#
# Sorties (docs/assets/*)
# ----------------------
# tables/data/methodology/
#   - normalization_report.csv
#   - events_schema_actual.csv
#   - perf_schema_actual.csv
#   - target_baseline_checks.csv
#   - model_injection_report.csv
#   - versioning.json
#   - licenses.csv
#
# figs/data/methodology/
#   - dataflow.png
#
# NOTE : Ce fichier ne dépend pas d’Internet. Les entrées sont les exports locaux
#        Parquet `events.parquet` et `perf.parquet`.

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --------------------------- Constantes chemins ---------------------------

ASSETS_DIR = Path("docs/assets")
TABLES_DIR = ASSETS_DIR / "tables" / "data" / "methodology"
FIGS_DIR = ASSETS_DIR / "figs" / "data" / "methodology"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------- Utilitaires génériques ---------------------------

def _mkdirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)


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
    # On quantise à 15 minutes par sécurité (les exports sont censés l’être)
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


def _present(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    # Pour écriture "humaine" (tri de colonnes)
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy()


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
    if path is None:
        return None
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")


# --------------------------- Normalisation ---------------------------

def normalization_report(events: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Events attendu: ts, station_id, bikes, docks, status (optionnel)
    ev_cols = ["ts", "station_id", "bikes", "docks"]
    ev_ok = all(c in events.columns for c in ev_cols)
    rows.append({"table": "events", "check": "columns_presence", "ok": bool(ev_ok), "expected": ev_cols})

    # Perf attendu: ts, station_id, y_true, y_pred, y_pred_baseline, horizon_min
    pf_cols_min = ["ts", "station_id", "y_true"]
    pf_ok_min = all(c in perf.columns for c in pf_cols_min)
    rows.append({"table": "perf", "check": "columns_presence_min", "ok": bool(pf_ok_min), "expected": pf_cols_min})

    # Types de base
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

    # Alignements temporels
    # Hypothèse : ts(Perf) = T et y_true(T) est la vérité à T+h ; events est observé à T+h.
    # On vérifiera ça plus loin dans la section "Cible & baseline".
    return pd.DataFrame(rows)


# --------------------------- Cible & baseline ---------------------------

def _best_horizon_from_events(perf: pd.DataFrame, events: pd.DataFrame) -> Tuple[Optional[int], Dict[int, float]]:
    """
    Si perf.horizon_min absent, on tente d'inférer l'horizon qui rend y_true ≈ events.bikes(ts+h).
    On teste quelques candidats et on prend celui qui maximise le % d'égalité (tolérance 1e-6).
    """
    candidates = [15, 30, 45, 60, 90, 120, 150, 180]
    scores = {}
    for h in candidates:
        ev = events[["ts","station_id","bikes"]].copy()
        ev["ts_shift_back"] = ev["ts"] - pd.Timedelta(minutes=h)  # ts_shift_back == T (perf)
        joined = perf.merge(ev[["ts_shift_back","station_id","bikes"]],
                            left_on=["ts","station_id"], right_on=["ts_shift_back","station_id"], how="left")
        if "y_true" not in joined.columns:
            scores[h] = np.nan
            continue
        ok = (np.abs(joined["y_true"] - joined["bikes"]) <= 1e-6).mean()
        try:
            scores[h] = float(ok)
        except Exception:
            scores[h] = np.nan

    # Choix (max du score)
    valid = {k:v for k,v in scores.items() if pd.notna(v)}
    if not valid:
        return None, scores
    best = max(valid, key=lambda k: valid[k])
    return int(best), scores


def target_and_baseline_checks(events: pd.DataFrame, perf: pd.DataFrame, override_hz: Optional[int] = None) -> pd.DataFrame:
    rows = []

    # 1) Horizon
    if override_hz is not None:
        hz = int(override_hz)
        hz_src = "cli:--horizon"
    elif "horizon_min" in perf.columns:
        horizon = pd.to_numeric(perf["horizon_min"], errors="coerce").dropna()
        hz = int(horizon.iloc[0]) if len(horizon) else None
        hz_src = "perf:horizon_min"
    else:
        hz, scores = _best_horizon_from_events(perf, events)
        hz_src = "inferred"
    rows.append({"check": "horizon_minutes", "value": hz, "source": hz_src})

    # 2) y_true ≈ events.bikes(ts+h)
    if hz is not None:
        ev = events[["ts","station_id","bikes"]].copy()
        ev["ts_shift_back"] = ev["ts"] - pd.Timedelta(minutes=hz)
        j = perf.merge(ev[["ts_shift_back","station_id","bikes"]],
                       left_on=["ts","station_id"], right_on=["ts_shift_back","station_id"], how="left")
        if "y_true" in j.columns:
            prop = float((np.abs(j["y_true"] - j["bikes"]) <= 1e-6).mean())
        else:
            prop = np.nan
        rows.append({"check": "target_alignment_share", "value": prop, "source": f"events.bikes @ T+{hz}min"})
    else:
        rows.append({"check": "target_alignment_share", "value": np.nan, "source": "unknown_horizon"})

    # 3) y_pred_baseline ≈ events.bikes(ts)
    e2 = events[["ts","station_id","bikes"]].copy()
    j2 = perf.merge(e2, on=["ts","station_id"], how="left", suffixes=("","_events"))
    if "y_pred_baseline" in j2.columns:
        prop_base = float((np.abs(j2["y_pred_baseline"] - j2["bikes"]) <= 1e-6).mean())
    else:
        prop_base = np.nan
    rows.append({"check": "baseline_persistence_share", "value": prop_base, "source": "events.bikes @ T"})

    return pd.DataFrame(rows)


# --------------------------- Injection modèle ---------------------------

def model_injection_report(perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(perf)
    # Couverture des prédictions
    if "y_pred" in perf.columns:
        cov = float(perf["y_pred"].notna().mean())
    else:
        cov = np.nan
    rows.append({"check": "model_prediction_coverage", "value": cov})

    # Alignement aux observations à T (naif)
    if "y_pred" in perf.columns and "y_true" in perf.columns:
        # NB: y_true est l’observation à T+h ; l’alignement à T est juste un sanity check
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
    # Empreintes simples basées sur les colonnes (contrat de données)
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
    rows.append({
        "artifact": "code",
        "license_name": _guess_license_name(code_text) if code_text else None,
        "length": len(code_text) if code_text else 0
    })
    rows.append({
        "artifact": "data",
        "license_name": _guess_license_name(data_text) if data_text else None,
        "length": len(data_text) if data_text else 0
    })
    return pd.DataFrame(rows)


# --------------------------- Schéma de flux (figure) ---------------------------

def plot_dataflow(out_png: Path) -> None:
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f0f4ff"):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=8",
                              ec="#1f3a93", fc=fc, lw=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10)

    def arrow(xy1, xy2):
        arr = FancyArrowPatch(xy1, xy2, arrowstyle="->", mutation_scale=12, lw=1.2, ec="#333")
        ax.add_patch(arr)

    # Layout simple
    box(0.05, 0.65, 0.25, 0.2, "Raw → Events\n(ts, station_id,\nbikes, docks)")
    box(0.05, 0.30, 0.25, 0.2, "Raw → Perf\n(ts, station_id,\ny_true, y_pred, baseline,\nhorizon_min)")
    arrow((0.175, 0.65), (0.175, 0.52))
    arrow((0.175, 0.30), (0.175, 0.17))

    box(0.40, 0.60, 0.28, 0.22, "Normalisation\n(quantize 15min,\ntypes, schémas)")
    box(0.40, 0.25, 0.28, 0.22, "Cible & baseline\n(alignements,\nhorizon)")
    arrow((0.30, 0.70), (0.40, 0.70))
    arrow((0.30, 0.35), (0.40, 0.35))

    box(0.75, 0.60, 0.20, 0.22, "Versionnage\n(contrats, hash)")
    box(0.75, 0.25, 0.20, 0.22, "Licences\n(code & data)")
    arrow((0.68, 0.70), (0.75, 0.70))
    arrow((0.68, 0.35), (0.75, 0.35))

    ax.text(0.5, 0.03, "Dataflow — Méthodologie", ha="center", fontsize=11)
    _save_fig(out_png)


# --------------------------- Main ---------------------------

def main(events_path: Path, perf_path: Path, tz: Optional[str],
         data_license_path: Optional[Path], code_license_path: Optional[Path],
         horizon_override: Optional[int] = None) -> None:
    _mkdirs()

    # Charger & normaliser
    events = _ensure_station_id(_coerce_ts_15(_read_parquet(events_path), "ts"))
    perf = _ensure_station_id(_coerce_ts_15(_read_parquet(perf_path), "ts"))

    # Schémas effectifs
    _schema(events).to_csv(TABLES_DIR / "events_schema_actual.csv", index=False)
    _schema(perf).to_csv(TABLES_DIR / "perf_schema_actual.csv", index=False)

    # Normalisation
    norm = normalization_report(events, perf)
    norm.to_csv(TABLES_DIR / "normalization_report.csv", index=False)

    # Cible & baseline
    tbc = target_and_baseline_checks(events, perf, override_hz=horizon_override)
    tbc.to_csv(TABLES_DIR / "target_baseline_checks.csv", index=False)

    # Injection modèle
    inj = model_injection_report(perf)
    inj.to_csv(TABLES_DIR / "model_injection_report.csv", index=False)

    # Versionnage
    ver = versioning(events, perf)
    (TABLES_DIR / "versioning.json").write_text(json.dumps(ver, ensure_ascii=False, indent=2), encoding="utf-8")

    # Licences
    lic = licenses_report(code_license_path=code_license_path, data_license_path=data_license_path)
    lic.to_csv(TABLES_DIR / "licenses.csv", index=False)

    # Figure dataflow
    plot_dataflow(FIGS_DIR / "dataflow.png")

    # Build summary
    summary = {
        "build_utc": _now_iso(),
        "inputs": {
            "events": {"path": str(events_path), "rows": int(len(events)),
                       "ts_min": events["ts"].min().isoformat() if len(events) else None,
                       "ts_max": events["ts"].max().isoformat() if len(events) else None},
            "perf": {"path": str(perf_path), "rows": int(len(perf)),
                     "ts_min": perf["ts"].min().isoformat() if len(perf) else None,
                     "ts_max": perf["ts"].max().isoformat() if len(perf) else None},
        },
        "timezone_display": tz,
        "artifacts": {
            "tables_dir": str(TABLES_DIR),
            "figs_dir": str(FIGS_DIR),
        },
    }
    (TABLES_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build 'Data / Méthodologie & licences' assets")
    ap.add_argument("--events", type=Path, required=True, help="Path to docs/exports/events.parquet")
    ap.add_argument("--perf", type=Path, required=True, help="Path to docs/exports/perf.parquet")
    ap.add_argument("--tz", type=str, default=None, help="Fuseau pour l’affichage (ex. Europe/Paris)")
    ap.add_argument("--data-license", type=Path, default=None, help="Chemin d’un fichier de licence des données (optionnel)")
    ap.add_argument("--code-license", type=Path, default=None, help="Chemin d’un fichier de licence du code (optionnel)")
    ap.add_argument("--horizon", type=int, default=None, help="Horizon (minutes) pour les vérifications d’alignement")
    args = ap.parse_args()

    main(
        events_path=args.events,
        perf_path=args.perf,
        tz=args.tz,
        data_license_path=args.data_license,
        code_license_path=args.code_license,
        horizon_override=args.horizon,
    )
