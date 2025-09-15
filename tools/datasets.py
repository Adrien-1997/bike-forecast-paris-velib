# tools/datasets.py
# Normalise les données Vélib' en deux jeux :
#   - events.parquet : ts (UTC, 15 min), station_id, bikes, capacity, occ[, lat, lon, name, meteo...]
#   - perf.parquet   : ts (UTC, 15 min), station_id, y_true (T+h), y_pred (optionnel), y_pred_baseline (persistance), horizon_min
#
# Usage principal (appelé par generate_monitoring.py) :
#   python tools/datasets.py --input docs/exports/velib.parquet --horizon 60 \
#          --out-events docs/exports/events.parquet --out-perf docs/exports/perf.parquet --lag-steps 0
#
# Notes :
# - On ne produit PAS de y_pred modèle ici (il sera injecté par tools/apply_model.py).
# - y_true est calculé comme bikes(T+steps) via shift(-steps) par station.
# - y_pred_baseline = bikes(T) (persistance).
# - ts est conservé en UTC et arrondi au pas 15 minutes (ou pas détecté).
# - --lag-steps permet de décaler une éventuelle colonne y_pred déjà présente (rare).

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXPORTS = DOCS / "exports"


# --------------------------- Mapping colonnes source -> canonique ---------------------------

DEFAULT_MAP: Dict[str, str] = {
    "ts": "tbin_utc",
    "station_id": "stationcode",
    "bikes": "nb_velos_bin",
    "capacity": "capacity_bin",
    "occ": "occ_ratio_bin",
    "lat": "lat",
    "lon": "lon",
    "name": "name",
    "hour_utc": "hour_utc",
    # météo (optionnelles)
    "temp_C": "temp_C",
    "precip_mm": "precip_mm",
    "wind_mps": "wind_mps",
    # prédictions éventuelles présentes en entrée (rare)
    "y_pred": "y_pred",
}


# --------------------------- IO util ---------------------------

def _read_any(path: Path) -> pd.DataFrame:
    """Lit parquet ou csv (auto-détection)."""
    if not path.exists():
        raise FileNotFoundError(f"[datasets] Introuvable: {path}")
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf in (".csv", ".txt"):
        return pd.read_csv(path)
    # fallback parquet
    return pd.read_parquet(path)


def _write_any(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suf = path.suffix.lower()
    if suf in (".csv", ".txt"):
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


# --------------------------- Helpers temps / pas ---------------------------

def _coerce_ts_utc_15min(s: pd.Series) -> pd.Series:
    """Convertit en datetime (UTC) puis floor 15 minutes."""
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    # si déjà timezone-naive mais censé être UTC, on le localise en UTC
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    return ts.dt.floor("15min").dt.tz_convert("UTC").dt.tz_localize(None)


def _infer_step_minutes(ts: pd.Series) -> int:
    """Détecte (grossièrement) le pas dominant en minutes (par ex. 15)."""
    t = pd.to_datetime(ts, utc=False, errors="coerce").sort_values().dropna()
    if len(t) < 3:
        return 15
    diffs = (t.diff().dropna().value_counts(normalize=True).index)
    if len(diffs) == 0:
        return 15
    # on prend la différence la plus fréquente
    step = int(round(diffs[0] / np.timedelta64(1, "m")))
    # clamp raisonnable
    if step <= 0 or step > 120:
        return 15
    return step


# --------------------------- Normalisation ---------------------------

def load_normalized(input_path: Path,
                    horizon_minutes: int = 60,
                    last_days: Optional[int] = None
                    ) -> Tuple[pd.DataFrame, pd.DataFrame, int, Dict[str, Any], Dict[str, str]]:
    """
    Normalise la table source → events + perf (sans y_pred modèle).
    Retourne : (events, perf, steps, dtypes_raw, mapping_utilisé)
    """
    raw = _read_any(input_path)

    # mapping dynamique : garder uniquement les colonnes présentes
    mapping = {k: v for k, v in DEFAULT_MAP.items() if v in raw.columns}

    df = raw.rename(columns={v: k for k, v in mapping.items()}).copy()

    # ts en UTC, floor pas 15 min
    if "ts" not in df.columns:
        raise KeyError("[datasets] La colonne temporelle 'ts' (ex. tbin_utc) est absente de la source.")
    df["ts"] = _coerce_ts_utc_15min(df["ts"])

    # filtre fenêtre récente si demandé
    if last_days and last_days > 0:
        tmax = df["ts"].max()
        if pd.notna(tmax):
            df = df[df["ts"] >= (tmax - pd.Timedelta(days=last_days))].copy()

    # station_id au format str
    if "station_id" not in df.columns:
        raise KeyError("[datasets] La colonne 'station_id' (ex. stationcode) est requise.")
    df["station_id"] = df["station_id"].astype(str)

    # compléments numériques
    for c in ("bikes", "capacity", "occ"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # météo optionnelle
    for c in ("temp_C", "precip_mm", "wind_mps"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # pas & steps
    base_step = _infer_step_minutes(df["ts"])
    steps = max(int(round(horizon_minutes / max(base_step, 1))), 1)

    # EVENTS (colonnes utiles si présentes)
    event_cols = ["ts", "station_id", "bikes", "capacity", "occ",
                  "lat", "lon", "name", "hour_utc", "temp_C", "precip_mm", "wind_mps"]
    events = df[[c for c in event_cols if c in df.columns]].copy()

    # PERF
    # y_true = bikes(T+steps) si bikes dispo
    perf_keys = ["ts", "station_id"]
    perf = df[perf_keys].drop_duplicates().sort_values(perf_keys).copy()

    if "bikes" in df.columns:
        # pour calculer y_true, on a besoin des bikes groupés par station au pas natif
        tmp = df[["ts", "station_id", "bikes"]].sort_values(perf_keys).copy()
        tmp["y_true"] = tmp.groupby("station_id")["bikes"].shift(-steps)
        tmp["y_pred_baseline"] = tmp.groupby("station_id")["bikes"].shift(0)  # persistance
        perf = perf.merge(tmp.drop(columns=["bikes"]), on=perf_keys, how="left")
    else:
        perf["y_true"] = np.nan
        perf["y_pred_baseline"] = np.nan

    perf["horizon_min"] = int(horizon_minutes)

    # si la source contenait déjà une colonne y_pred (rare) → l'aligner optionnellement plus tard
    if "y_pred" in df.columns:
        yp = (df[perf_keys + ["y_pred"]]
              .dropna(subset=["y_pred"])
              .drop_duplicates(perf_keys)
              .copy())
        perf = perf.merge(yp, on=perf_keys, how="left")

    # types clean
    for c in ("y_true", "y_pred", "y_pred_baseline"):
        if c in perf.columns:
            perf[c] = pd.to_numeric(perf[c], errors="coerce")

    dtypes_raw = {c: str(t) for c, t in raw.dtypes.items()}
    return events, perf, steps, dtypes_raw, mapping


# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Normalise la source Vélib' en events/perf (UTC, 15min).")
    ap.add_argument("--input", type=Path, required=True, help="Fichier source (parquet/csv).")
    ap.add_argument("--horizon", type=int, default=60, help="Horizon en minutes (ex. 60).")
    ap.add_argument("--last-days", type=int, default=None, help="Filtrer sur N derniers jours (optionnel).")
    ap.add_argument("--out-events", type=Path, default=EXPORTS / "events.parquet")
    ap.add_argument("--out-perf", type=Path, default=EXPORTS / "perf.parquet")
    ap.add_argument("--lag-steps", type=int, default=0, help="Décalage à appliquer à y_pred si présent (ex. -4).")
    ap.add_argument("--as-csv", action="store_true", help="Écrit en CSV au lieu de parquet.")
    args = ap.parse_args()

    events, perf, steps, dtypes_raw, mapping = load_normalized(
        input_path=args.input,
        horizon_minutes=args.horizon,
        last_days=args.last_days
    )

    # appliquer un décalage y_pred si demandé (rare, utile si y_pred a été posée à T+h)
    if args.lag_steps != 0 and "y_pred" in perf.columns:
        perf["y_pred"] = perf.groupby("station_id")["y_pred"].shift(args.lag_steps)

    # métrique de couverture simple (pour logs)
    def _cov(d: pd.DataFrame, cols):
        return {c: round(d[c].notna().mean() * 100, 2) for c in cols if c in d.columns}

    print("[COVERAGE %]", _cov(perf, ["y_true", "y_pred", "y_pred_baseline"]))

    # écriture fichiers
    args.out_events.parent.mkdir(parents=True, exist_ok=True)
    args.out_perf.parent.mkdir(parents=True, exist_ok=True)

    if args.as_csv:
        _write_any(events, args.out_events.with_suffix(".csv"))
        _write_any(perf, args.out_perf.with_suffix(".csv"))
    else:
        _write_any(events, args.out_events)
        _write_any(perf, args.out_perf)

    # logs finaux
    print("[OK] Datasets prêt.")
    print(f"- Stations: {events['station_id'].nunique()}")
    print(f"- Pas: ~{_infer_step_minutes(events['ts'])} min × steps={steps} (horizon={args.horizon} min)")
    print(f"- mapping source → canonique : {mapping}")
    print(f"[events] → {args.out_events}")
    print(f"[perf]   → {args.out_perf}")


if __name__ == "__main__":
    main()
