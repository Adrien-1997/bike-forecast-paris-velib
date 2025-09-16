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
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

# Matplotlib is used only to draw a couple of simple figures
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TABLES_OUT = ROOT / "docs" / "assets" / "tables" / "model" / "pipeline"
FIGS_OUT   = ROOT / "docs" / "assets" / "figs"   / "model" / "pipeline"
TABLES_OUT.mkdir(parents=True, exist_ok=True)
FIGS_OUT.mkdir(parents=True, exist_ok=True)


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


# ------------------------ feature detection logic ---------------------------

def _try_import_features_direct() -> Tuple[List[str], str]:
    """Try to import src.features.feature_cols.
    Returns (features, source_label) or ([], "")."""
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
                model = sf.get_model()  # optional user function
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
        # write placeholders to keep outputs consistent
        pd.DataFrame(columns=["feature","family"]).to_csv(TABLES_OUT / "features_contract.csv", index=False)
        pd.DataFrame(columns=["family","count"]).to_csv(TABLES_OUT / "features_by_family.csv", index=False)
        fig = plt.figure(figsize=(6, 4))
        plt.title("Features by family — (none detected)")
        plt.xlabel("family")
        plt.ylabel("count")
        fig.savefig(FIGS_OUT / "features_by_family.png", bbox_inches="tight")
        plt.close(fig)
        return

    rows = [{"feature": f, "family": _family_of(f)} for f in features]
    df = pd.DataFrame(rows).sort_values(["family","feature"])
    df.to_csv(TABLES_OUT / "features_contract.csv", index=False)

    fam = df.groupby("family").size().reset_index(name="count").sort_values("count", ascending=False)
    fam.to_csv(TABLES_OUT / "features_by_family.csv", index=False)

    fig = plt.figure(figsize=(7, 4.5))
    plt.bar(fam["family"], fam["count"])  # do not set explicit colors
    plt.title("Features by family")
    plt.xlabel("family")
    plt.ylabel("count")
    plt.xticks(rotation=20, ha="right")
    fig.savefig(FIGS_OUT / "features_by_family.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------- main ---------------------------------------

def main(events_path: str, perf_path: str, horizon: Optional[int]) -> None:
    _log("Starting…")

    # Load datasets
    ev = pd.read_parquet(events_path)
    pf = pd.read_parquet(perf_path)

    # Write schemas
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

    if feats:
        _log(f"Features: {len(feats)} (source: {src})")
    else:
        _log("Features non détectées — tables placeholders écrites.")

    _write_features_contract(feats)

    # Minimal dataflow placeholder figure (for the doc)
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

