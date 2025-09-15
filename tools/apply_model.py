# tools/apply_model.py
# Applique le modèle entraîné et injecte `y_pred` dans docs/exports/perf.parquet.
# Aligne les prédictions à l'instant T (pas T+h), mappe station_id de façon stable,
# et garantit des clés uniques (ts, station_id).

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# --- Repo root importable ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forecast import load_model_bundle                   # (model, feat_cols)
from src.features import _load_base_15min                    # base 15 min (tbin_utc, +meta)
from src.cal_features import add_calendar_features, feature_cols as calfeat_cols

DOCS = ROOT / "docs"
EXPORTS = DOCS / "exports"
EVENTS_PATH = EXPORTS / "events.parquet"
PERF_PATH = EXPORTS / "perf.parquet"


# --------------------------- Features d'inférence ---------------------------

def build_inference_frame(horizon_minutes: int, lookback_days: int) -> pd.DataFrame:
    """
    Construit les features comme à l'entraînement, en préservant tbin_utc/name/lat/lon.
    """
    base = _load_base_15min().copy()

    # fenêtre temporelle
    if lookback_days and lookback_days > 0:
        tmin = pd.to_datetime(base["tbin_utc"]).min()
        tmax = pd.to_datetime(base["tbin_utc"]).max()
        cutoff = max(tmax.floor("15min") - pd.Timedelta(days=lookback_days), tmin)
        base = base[base["tbin_utc"] >= cutoff].copy()

    # tri
    order_cols = [c for c in ["stationcode", "tbin_utc"] if c in base.columns]
    if order_cols:
        base = base.sort_values(order_cols)

    # lags/rollings identiques au train
    def add_lags_rollings(dfg: pd.DataFrame) -> pd.DataFrame:
        g = dfg.copy()
        for b in (1, 2, 3, 4, 8, 16):  # 15,30,45,60,120,240 min
            g[f"lag_nb_{b}b"]  = g["nb_velos_bin"].shift(b)
            g[f"lag_occ_{b}b"] = g["occ_ratio_bin"].shift(b)
        g["roll_nb_4b"]  = g["nb_velos_bin"].rolling(4, min_periods=1).mean()
        g["roll_nb_8b"]  = g["nb_velos_bin"].rolling(8, min_periods=1).mean()
        g["roll_occ_4b"] = g["occ_ratio_bin"].rolling(4, min_periods=1).mean()
        g["roll_occ_8b"] = g["occ_ratio_bin"].rolling(8, min_periods=1).mean()
        g["trend_nb_4b"]  = (g["nb_velos_bin"] - g["nb_velos_bin"].shift(4)) / 4.0
        g["trend_occ_4b"] = (g["occ_ratio_bin"] - g["occ_ratio_bin"].shift(4)) / 4.0
        return g

    if "stationcode" in base.columns:
        try:
            base = base.groupby("stationcode", group_keys=False).apply(
                add_lags_rollings, include_groups=False
            )
        except TypeError:  # pandas < 2.2
            base = base.groupby("stationcode", group_keys=False).apply(add_lags_rollings)
    else:
        base = add_lags_rollings(base)

    # features calendaires
    base = add_calendar_features(base, tz="Europe/Paris")

    # jeu de colonnes numériques attendu (remplissage 0.0 si manquant)
    num_cols = [
        "nb_velos_bin","nb_bornes_bin","capacity_bin","occ_ratio_bin",
        "temp_C","precip_mm","wind_mps",
        *[f"lag_nb_{b}b" for b in (1,2,3,4,8,16)],
        *[f"lag_occ_{b}b" for b in (1,2,3,4,8,16)],
        "roll_nb_4b","roll_nb_8b","roll_occ_4b","roll_occ_8b",
        "trend_nb_4b","trend_occ_4b",
        *calfeat_cols(base),
    ]
    seen=set(); num_cols=[c for c in num_cols if not (c in seen or seen.add(c))]
    for c in num_cols:
        if c not in base.columns:
            base[c] = 0.0
    base[num_cols] = (
        base[num_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
    )

    # garder tbin_utc/name/lat/lon pour mapping station
    for k in ("tbin_utc", "name", "lat", "lon"):
        if k not in base.columns:
            base[k] = np.nan

    return base


# --------------------------- Main ---------------------------

def main(horizon: int, lookback_days: int):
    # fichiers requis
    if not PERF_PATH.exists():
        raise FileNotFoundError(f"[apply_model] Introuvable: {PERF_PATH}")
    if not EVENTS_PATH.exists():
        raise FileNotFoundError(f"[apply_model] Introuvable: {EVENTS_PATH}")

    # modèle + colonnes de features
    model, feat_cols = load_model_bundle(horizon_minutes=horizon)

    # features d'inférence
    df = build_inference_frame(horizon_minutes=horizon, lookback_days=lookback_days)

    # s'assurer que toutes les colonnes attendues existent
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = (df[feat_cols]
         .apply(pd.to_numeric, errors="coerce")
         .replace([np.inf, -np.inf], np.nan)
         .fillna(0.0)
         .astype("float32"))

    # prédiction (cible T+h), mais on va horodater à T (voir plus bas)
    df["y_pred"] = model.predict(X).astype(float)

    # ------------------ PREDS (ts, y_pred, name, lat, lon) ------------------
    ts_src = "tbin_utc" if "tbin_utc" in df.columns else ("ts" if "ts" in df.columns else None)
    if ts_src is None:
        raise KeyError("[apply_model] Aucune colonne temporelle trouvée (tbin_utc/ts).")

    preds = df[[ts_src, "y_pred", "name", "lat", "lon"]].copy()
    preds.rename(columns={ts_src: "ts"}, inplace=True)
    preds["ts"] = pd.to_datetime(preds["ts"], utc=False, errors="coerce").dt.floor("15min")

    # ❗ Correction d’offset : tbin_utc est le bin cible (T+h). On ramène l’horodatage au bin source T.
    preds["ts"] = preds["ts"] - pd.Timedelta(minutes=horizon)

    # normalisation pour mapping
    preds["name"] = preds["name"].astype(str).str.strip().str.lower()
    preds["lat"]  = pd.to_numeric(preds["lat"], errors="coerce").round(5)
    preds["lon"]  = pd.to_numeric(preds["lon"], errors="coerce").round(5)

    # ------------------ Mapping station_id (one-to-one) ------------------
    events = pd.read_parquet(EVENTS_PATH, columns=["ts","station_id","name","lat","lon"]).copy()
    events["ts"] = pd.to_datetime(events["ts"], utc=False, errors="coerce").dt.floor("15min")
    events["station_id"] = events["station_id"].astype(str)
    events["name"] = events["name"].astype(str).str.strip().str.lower()
    events["lat"]  = pd.to_numeric(events["lat"], errors="coerce").round(5)
    events["lon"]  = pd.to_numeric(events["lon"], errors="coerce").round(5)

    # fenêtre des events limitée à preds (évite volumétrie)
    tmin, tmax = preds["ts"].min(), preds["ts"].max()
    events = events[(events["ts"] >= tmin) & (events["ts"] <= tmax)].copy()

    # clé combinée
    preds["__k"]  = preds["name"] + "|" + preds["lat"].astype(str) + "|" + preds["lon"].astype(str)
    events["__k"] = events["name"] + "|" + events["lat"].astype(str) + "|" + events["lon"].astype(str)

    # un seul station_id par (ts, __k)
    events = events.sort_values(["ts"]).drop_duplicates(subset=["ts","__k"], keep="last")

    # merge many-to-one (pas de duplication)
    try:
        preds = preds.merge(events[["ts","__k","station_id"]], on=["ts","__k"], how="left", validate="many_to_one")
    except Exception:
        preds = preds.merge(events[["ts","__k","station_id"]], on=["ts","__k"], how="left")

    preds = preds.dropna(subset=["station_id"]).drop(columns=["__k"]).copy()
    preds["station_id"] = preds["station_id"].astype(str)

    # ------------------ Alignement aux clés de perf & dédup ------------------
    perf_keys = pd.read_parquet(PERF_PATH, columns=["ts","station_id"])
    perf_keys["ts"] = pd.to_datetime(perf_keys["ts"], utc=False, errors="coerce").dt.floor("15min")
    perf_keys["station_id"] = perf_keys["station_id"].astype(str)

    print("[apply_model] preds bruts:", len(preds))
    preds = preds.merge(perf_keys.drop_duplicates(), on=["ts","station_id"], how="inner")
    print("[apply_model] preds alignés (dans perf):", len(preds))

    # une ligne unique par (ts, station_id)
    preds = preds.groupby(["ts","station_id"], as_index=False)["y_pred"].mean()

    # ------------------ Merge dans perf.parquet ------------------
    perf = pd.read_parquet(PERF_PATH)
    perf["ts"] = pd.to_datetime(perf["ts"], utc=False, errors="coerce").dt.floor("15min")
    perf["station_id"] = perf["station_id"].astype(str)

    out = perf.merge(preds, on=["ts","station_id"], how="left", suffixes=("", "_new"))

    # Combinaison robuste (si une ancienne y_pred existe déjà)
    if "y_pred" in out.columns and "y_pred_new" in out.columns:
        out["y_pred"] = out["y_pred"].combine_first(out["y_pred_new"])
        out.drop(columns=["y_pred_new"], inplace=True)
    elif "y_pred_new" in out.columns and "y_pred" not in out.columns:
        out.rename(columns={"y_pred_new": "y_pred"}, inplace=True)

    out.to_parquet(PERF_PATH, index=False)

    cov = (out[["y_true", "y_pred"]].notna().mean() * 100).round(2).to_dict()
    print(f"[apply_model] OK → {PERF_PATH.resolve()}")
    print("[apply_model] couverture après merge:", cov)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Appliquer le modèle et injecter y_pred (aligné T) dans perf.parquet")
    ap.add_argument("--horizon", type=int, default=60, help="Horizon minutes (doit matcher le modèle)")
    ap.add_argument("--lookback-days", type=int, default=30, help="Fenêtre de reconstruction des features")
    args = ap.parse_args()
    main(horizon=args.horizon, lookback_days=args.lookback_days)
