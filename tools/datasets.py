# tools/datasets.py
# Point d'entrée unique pour lire docs/exports/velib.parquet,
# normaliser le schéma, reconstruire y_true / y_pred si absents,
# et fournir des vues prêtes pour "Usage" et "Performance".
#
# Usage (depuis la racine du repo) :
#   python tools/datasets.py --input docs/exports/velib.parquet --horizon 60 \
#       --out-events docs/exports/events.parquet \
#       --out-perf docs/exports/perf.parquet
#
# Options :
#   --horizon 60            # minutes (par défaut 60)
#   --last-days 7           # filtre sur les X derniers jours (optionnel)
#   --csv                   # exporte en CSV au lieu de Parquet
#
# Sorties :
#   events: [ts, station_id, bikes, capacity, (lat, lon, name optionnels)]
#   perf:   [ts, station_id, y_true, y_pred, horizon_min]
#
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from pathlib import Path


@dataclass
class Meta:
    bin_minutes: int
    steps_for_horizon: int
    horizon_minutes: int
    n_stations: int
    ts_min: pd.Timestamp
    ts_max: pd.Timestamp


# --------- Helpers ---------

def _first_present(cols: List[str], present: set) -> Optional[str]:
    for c in cols:
        if c in present:
            return c
    return None


def _to_naive_utc(s: pd.Series) -> pd.Series:
    """Parse datetimes as UTC then drop tz to keep them naive (compatible plotting)."""
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    return dt.dt.tz_localize(None)


def _infer_bin_minutes(df: pd.DataFrame, ts_col: str, sid_col: str) -> int:
    """Infère la granularité (minutes) à partir de la médiane des pas temporels par station."""
    g = df.sort_values([sid_col, ts_col]).copy()
    dmins = g.groupby(sid_col)[ts_col].diff().dt.total_seconds().div(60.0)
    med = float(dmins[dmins > 0].median())
    # Arrondir à un pas standard raisonnable
    candidates = np.array([1, 5, 10, 15, 30, 60, 120])
    if np.isnan(med) or med <= 0:
        return 15  # défaut raisonnable
    idx = np.argmin(np.abs(candidates - med))
    return int(candidates[idx])


def _rename_and_normalize(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Renomme en colonnes canoniques : ts, station_id, bikes, capacity (+ lat, lon, name si présents).
    Retourne (df, mapping d'origine->canonique).
    """
    cols = set(df.columns)

    station_col = _first_present(["stationcode", "station_id", "station", "code"], cols)
    time_col    = _first_present(["tbin_utc", "tbin", "timestamp_utc", "timestamp"], cols)
    bikes_col   = _first_present(["nb_velos_bin", "bikes", "available_bikes", "n_bikes"], cols)
    cap_col     = _first_present(["capacity_bin", "nb_bornes_bin", "capacity", "dock_count", "num_docks"], cols)
    lat_col     = _first_present(["lat", "latitude"], cols)
    lon_col     = _first_present(["lon", "longitude"], cols)
    name_col    = _first_present(["name", "station_name", "nom"], cols)

    required = [station_col, time_col, bikes_col]
    if any(c is None for c in required):
        missing = ["station_id" if station_col is None else None,
                   "ts" if time_col is None else None,
                   "bikes" if bikes_col is None else None]
        missing = [m for m in missing if m]
        raise ValueError(f"Colonnes indispensables manquantes dans velib.parquet: {missing}")

    out = pd.DataFrame()
    out["station_id"] = df[station_col].astype(str)
    out["ts"] = _to_naive_utc(df[time_col])
    out["bikes"] = pd.to_numeric(df[bikes_col], errors="coerce")

    if cap_col is not None:
        out["capacity"] = pd.to_numeric(df[cap_col], errors="coerce")
    else:
        out["capacity"] = np.nan

    if lat_col is not None:  out["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    if lon_col is not None:  out["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    if name_col is not None: out["name"] = df[name_col].astype(str)

    mapping = {
        "station_id": station_col,
        "ts": time_col,
        "bikes": bikes_col,
        "capacity": cap_col or "",
        "lat": lat_col or "",
        "lon": lon_col or "",
        "name": name_col or "",
    }
    return out, mapping


def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["station_id", "ts"]).reset_index(drop=True)


def _make_truth_pred(
    df: pd.DataFrame,
    horizon_minutes: int,
    bin_minutes: int,
    y_true_col_candidates: List[str],
    y_pred_col_candidates: List[str],
    original: pd.DataFrame,
) -> Tuple[pd.DataFrame, bool, bool]:
    """
    Construit y_true / y_pred si absents, en décalant 'bikes' par ± steps.
    steps = round(horizon_minutes / bin_minutes).
    Si y_true/y_pred existent déjà dans le parquet original, on les réutilise.
    Retourne (df_out, used_existing_y_true, used_existing_y_pred).
    """
    steps = max(1, int(round(horizon_minutes / bin_minutes)))

    # Reprendre y_true/y_pred si déjà présents dans le fichier original
    cols = set(original.columns)
    y_true_col = _first_present(y_true_col_candidates, cols)
    y_pred_col = _first_present(y_pred_col_candidates, cols)

    out = df.copy()
    used_true = False
    used_pred = False

    if y_true_col:
        out["y_true"] = pd.to_numeric(original[y_true_col], errors="coerce").values
        used_true = True
    else:
        out["y_true"] = (
            out.groupby("station_id")["bikes"].shift(-steps)
        )

    if y_pred_col:
        out["y_pred"] = pd.to_numeric(original[y_pred_col], errors="coerce").values
        used_pred = True
    else:
        # Baseline naïve : valeur actuelle pour t+H => shift(+steps)
        out["y_pred"] = (
            out.groupby("station_id")["bikes"].shift(steps)
        )

    return out, used_true, used_pred


# --------- Public API ---------

def load_normalized(path: Path, horizon_minutes: int = 60, last_days: Optional[int] = None) -> Tuple[pd.DataFrame, Meta, Dict[str, str]]:
    """
    Charge docs/exports/velib.parquet (ou .csv), normalise les colonnes
    et ajoute y_true/y_pred si absents. Retourne (df, meta, mapping).
    """
    path = Path(path)
    if not path.exists():
        alt = path.with_suffix(".csv")
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Fichier introuvable: {path}")

    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    else:
        raise ValueError("Format non supporté (attendu .parquet ou .csv)")

    base, mapping = _rename_and_normalize(raw)
    base = _ensure_sorted(base)

    # Filtre temporel optionnel
    if last_days is not None and last_days > 0:
        tmax = base["ts"].max()
        tmin = tmax - pd.Timedelta(days=last_days)
        base = base[base["ts"].between(tmin, tmax)].copy()

    # Granularité
    bin_minutes = _infer_bin_minutes(base, "ts", "station_id")

    # Construit y_true / y_pred (ou les réutilise si présents)
    ytrue_candidates = ["y_true", "nb_velos_tplus", "ytrue"]
    ypred_candidates = ["y_pred", "nb_velos_pred", "ypred"]

    perf, used_true, used_pred = _make_truth_pred(
        base, horizon_minutes, bin_minutes, ytrue_candidates, ypred_candidates, raw
    )

    # Nettoyage & indicateurs
    perf["horizon_min"] = int(horizon_minutes)
    # Clamp y_* si capacité dispo
    if "capacity" in perf.columns:
        cap_safe = perf["capacity"].replace(0, np.nan)
        perf["y_true"] = np.clip(perf["y_true"], 0, cap_safe)
        perf["y_pred"] = np.clip(perf["y_pred"], 0, cap_safe)

    # Méta
    meta = Meta(
        bin_minutes=bin_minutes,
        steps_for_horizon=max(1, int(round(horizon_minutes / bin_minutes))),
        horizon_minutes=int(horizon_minutes),
        n_stations=int(perf["station_id"].nunique()),
        ts_min=pd.to_datetime(perf["ts"]).min(),
        ts_max=pd.to_datetime(perf["ts"]).max(),
    )

    # Tables prêtes
    events_cols = ["ts", "station_id", "bikes", "capacity"]
    extra_cols = [c for c in ["lat", "lon", "name"] if c in perf.columns]
    events = perf[events_cols + extra_cols].copy()

    perf_out = perf[["ts", "station_id", "y_true", "y_pred", "horizon_min"]].dropna().reset_index(drop=True)

    return (events.reset_index(drop=True),
            perf_out.reset_index(drop=True),
            meta,
            mapping,
            {"used_existing_y_true": used_true, "used_existing_y_pred": used_pred})


def export_df(df: pd.DataFrame, path: Path, as_csv: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if as_csv or path.suffix.lower() == ".csv":
        df.to_csv(path, index=False, encoding="utf-8")
    else:
        df.to_parquet(path, index=False)


def sanity_report(meta: Meta, stats: Dict[str, bool]) -> str:
    lines = [
        f"- Stations: {meta.n_stations}",
        f"- Période:  {meta.ts_min}  →  {meta.ts_max}",
        f"- Pas temps: {meta.bin_minutes} min",
        f"- Horizon:   {meta.horizon_minutes} min  (steps={meta.steps_for_horizon})",
        f"- y_true existant: {stats['used_existing_y_true']}",
        f"- y_pred existant: {stats['used_existing_y_pred']}",
    ]
    return "\n".join(lines)


# --------- CLI ---------

def main():
    ap = argparse.ArgumentParser(description="Normalize velib exports and build events/perf tables.")
    ap.add_argument("--input", type=Path, default=Path("docs/exports/velib.parquet"))
    ap.add_argument("--horizon", type=int, default=60, help="Horizon de prédiction, en minutes (défaut 60)")
    ap.add_argument("--last-days", type=int, default=None, help="Filtrer sur les N derniers jours")
    ap.add_argument("--out-events", type=Path, default=Path("docs/exports/events.parquet"))
    ap.add_argument("--out-perf", type=Path, default=Path("docs/exports/perf.parquet"))
    ap.add_argument("--csv", action="store_true", help="Exporter en CSV")
    args = ap.parse_args()

    events, perf, meta, mapping, used = load_normalized(
        args.input, horizon_minutes=args.horizon, last_days=args.last_days
    )

    export_df(events, args.out_events, as_csv=args.csv)
    export_df(perf, args.out_perf, as_csv=args.csv)

    print("[OK] Datasets prêt.")
    print(sanity_report(meta, used))
    print(f"[events] → {args.out_events}")
    print(f"[perf]   → {args.out_perf}")
    print(f"[mapping] colonnes source → canonique : {mapping}")

if __name__ == "__main__":
    main()