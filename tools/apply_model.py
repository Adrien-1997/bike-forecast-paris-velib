# tools/apply_model.py
# Applique le modèle entraîné et injecte y_pred dans docs/exports/perf.parquet.

from __future__ import annotations
import argparse
from pathlib import Path
import sys, os
import pandas as pd
import numpy as np

# --- UTF-8 stdout (Windows safe) ---
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- Repo root importable ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forecast import load_model_bundle                   # (model, feat_cols)
from src.features import _load_base_15min                    # base 15 min (tbin_utc, +meta)
from src.cal_features import add_calendar_features           # calendrier

# Tente d'importer feature_cols (liste) OU feature_cols() (fonction)
def _load_calfeat_cols() -> list[str]:
    try:
        from src.cal_features import feature_cols as _fc
    except Exception:
        return []
    try:
        return list(_fc() if callable(_fc) else _fc)
    except Exception:
        return []

CALFEAT_COLS = _load_calfeat_cols()

DOCS = ROOT / "docs"
EXPORTS = DOCS / "exports"

# --------------------------- Features d'inférence ---------------------------

def build_inference_frame(horizon_minutes: int, lookback_days: int) -> pd.DataFrame:
    """Construit les features comme à l'entraînement, en préservant tbin_utc/name/lat/lon."""
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
    base = add_calendar_features(base, tz="Europe/Paris")  # tz uniquement pour features cal.

    # renommer/garantir colonnes attendues
    if "nb_velos_bin" in base.columns:
        base["y_src"] = base["nb_velos_bin"]

    # compléter colonnes manquantes (calendaires)
    for col in CALFEAT_COLS:
        if col not in base.columns:
            base[col] = 0

    # types cohérents
    for c in base.columns:
        if (
            c.endswith("_bin")
            or c.startswith(("lag_", "roll_", "trend_"))
            or c in CALFEAT_COLS
        ):
            base[c] = pd.to_numeric(base[c], errors="coerce")
    base["occ_ratio_bin"] = pd.to_numeric(base.get("occ_ratio_bin", np.nan), errors="coerce")

    # nettoyer / fillna
    base = (
        base.replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype("float32", errors="ignore")
    )

    # garder tbin_utc/name/lat/lon pour mapping station
    for k in ("tbin_utc", "name", "lat", "lon"):
        if k not in base.columns:
            base[k] = np.nan

    return base


# --------------------------- Main ---------------------------

def main(horizon: int, lookback_days: int, events_path: Path, perf_path: Path) -> None:
    # fichiers requis
    if not perf_path.exists():
        raise FileNotFoundError(f"[apply_model] Introuvable: {perf_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"[apply_model] Introuvable: {events_path}")

    # modèle + colonnes de features
    model, feat_cols = load_model_bundle(horizon_minutes=horizon)

    # features d'inférence
    df = build_inference_frame(horizon_minutes=horizon, lookback_days=lookback_days)

    # s'assurer que toutes les colonnes attendues existent
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = (
        df[feat_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
    )

    # --- Prédiction modèle → colonne standard y_pred -----------------
    df["y_pred"] = model.predict(X).astype(float)

    # ------------------ PREDS (ts, y_pred, name, lat, lon) ------------------
    ts_src = "tbin_utc" if "tbin_utc" in df.columns else ("ts" if "ts" in df.columns else None)
    if ts_src is None:
        raise KeyError("[apply_model] Aucune colonne temporelle trouvée (tbin_utc/ts).")

    # s'assurer que la colonne y_pred existe (renomme y_pred_model au besoin)
    if "y_pred" not in df.columns:
        if "y_pred_model" in df.columns:
            df = df.rename(columns={"y_pred_model": "y_pred"})
        else:
            raise KeyError("[apply_model] y_pred manquante — calcule la prédiction modèle avant ce bloc.")

    preds = df[[ts_src, "y_pred", "name", "lat", "lon"]].copy()
    preds.rename(columns={ts_src: "ts"}, inplace=True)
    preds["ts"] = pd.to_datetime(preds["ts"], utc=False, errors="coerce").dt.floor("15min")

    # normaliser meta pour la clé de mapping (arrondir coords + lower name)
    for k in ("name", "lat", "lon"):
        if k not in preds.columns:
            preds[k] = np.nan
    preds["name"] = preds["name"].astype(str).str.strip().str.lower()
    preds["lat"]  = pd.to_numeric(preds["lat"], errors="coerce").round(5)
    preds["lon"]  = pd.to_numeric(preds["lon"], errors="coerce").round(5)

    events = pd.read_parquet(events_path, columns=["ts","name","lat","lon","station_id"]).copy()
    events["ts"] = pd.to_datetime(events["ts"], utc=False, errors="coerce").dt.floor("15min")
    events["station_id"] = events["station_id"].astype(str)
    events["name"] = events["name"].astype(str).str.strip().str.lower()
    events["lat"]  = pd.to_numeric(events["lat"], errors="coerce").round(5)
    events["lon"]  = pd.to_numeric(events["lon"], errors="coerce").round(5)

    # clé de mapping SANS dépendre de ts (dernière association connue)
    preds["__k"]  = preds["name"] + "|" + preds["lat"].astype(str) + "|" + preds["lon"].astype(str)
    events["__k"] = events["name"] + "|" + events["lat"].astype(str) + "|" + events["lon"].astype(str)
    emap = (events.dropna(subset=["__k","station_id"])
                .sort_values("ts")
                .drop_duplicates("__k", keep="last")[["__k","station_id"]])

    preds = preds.merge(emap, on="__k", how="left").drop(columns="__k")
    preds = preds.dropna(subset=["station_id"]).copy()
    preds["station_id"] = preds["station_id"].astype(str)

    # ------------------ Alignement temporel sur T (fallback si T=0) ------------------
    perf_keys = pd.read_parquet(perf_path, columns=["ts","station_id"]).drop_duplicates()
    perf_keys["ts"] = pd.to_datetime(perf_keys["ts"], utc=False, errors="coerce").dt.floor("15min")
    perf_keys["station_id"] = perf_keys["station_id"].astype(str)

    candidates = {
        "T"  : preds["ts"],
        "T-h": preds["ts"] - pd.Timedelta(minutes=horizon),
        "T+h": preds["ts"] + pd.Timedelta(minutes=horizon),
    }

    def _hits(ts_series: pd.Series) -> int:
        tmp = preds.copy()
        tmp["ts"] = ts_series
        return tmp.merge(perf_keys, on=["ts","station_id"], how="inner").shape[0]

    hits = {k: _hits(v) for k, v in candidates.items()}
    print(f"[apply_model] test alignement (T/T-h/T+h): {hits}")

    if hits["T"] > 0:
        preds["ts"] = candidates["T"]
        chosen = "T"
    else:
        chosen = "T-h" if hits["T-h"] >= hits["T+h"] else "T+h"
        preds["ts"] = candidates[chosen]
    print(f"[apply_model] alignement retenu: {chosen}")

    # garder uniq (ts, station_id)
    preds = preds.groupby(["ts","station_id"], as_index=False)["y_pred"].mean()

    # ------------------ Merge : ÉCRASE TOUJOURS y_pred par la sortie modèle ---------
    perf = pd.read_parquet(perf_path)
    perf["ts"] = pd.to_datetime(perf["ts"], utc=False, errors="coerce").dt.floor("15min")
    perf["station_id"] = perf["station_id"].astype(str)

    out = perf.merge(preds.rename(columns={"y_pred": "y_pred_model"}),
                    on=["ts","station_id"], how="left")

    # on supprime toute ancienne y_pred, puis on pose la nouvelle (modèle)
    if "y_pred" in out.columns:
        out.drop(columns=["y_pred"], inplace=True)
    out.rename(columns={"y_pred_model": "y_pred"}, inplace=True)

    # contrôle de couverture
    matched = out["y_pred"].notna().mean()*100.0
    print(f"[apply_model] couverture modèle après merge: {matched:.2f}%")
    if matched == 0:
        raise RuntimeError("0% de prédictions modèle écrites. Vérifier mapping/horizon/features.")

    out.to_parquet(perf_path, index=False)

    cov = (out[["y_true", "y_pred"]].notna().mean() * 100).round(2).to_dict()
    print(f"[apply_model] OK -> {perf_path.resolve()}")
    print("[apply_model] couverture apres merge:", cov)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Injecte y_pred dans perf.parquet (aligne T). "
                    "⚠️ events/perf doivent avoir été générés localement par datasets.py dans ce même run."
    )
    ap.add_argument("--horizon", type=int, default=60, help="Horizon minutes (doit matcher le modèle)")
    ap.add_argument("--lookback-days", type=int, default=30, help="Fenetre de reconstruction des features")
    ap.add_argument("--events", type=Path, required=True, help="Chemin local vers events.parquet")
    ap.add_argument("--perf", type=Path, required=True, help="Chemin local vers perf.parquet")
    ap.add_argument("--tz", type=str, default=None, help="(Ignore) TZ non utilisée par apply_model")
    args = ap.parse_args()
    main(horizon=args.horizon, lookback_days=args.lookback_days, events_path=args.events, perf_path=args.perf)
