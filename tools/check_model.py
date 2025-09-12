# tools/check_model.py
from pathlib import Path
import sys, json
import pandas as pd
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecast import load_model_bundle, prepare_live_features
# Remplacer l’import depuis l’app par une requête simple
import requests, pandas as pd

def fetch_opendata_v1():
    URL = "https://opendata.paris.fr/api/records/1.0/search/?dataset=velib-disponibilite-en-temps-reel&rows=2000"
    j = requests.get(URL, timeout=20).json()
    rows = []
    for rec in j.get("records", []):
        f = rec.get("fields", {})
        coord = f.get("coordonnees_geo")
        lat, lon = (coord or [None, None])
        rows.append({
            "stationcode": f.get("stationcode"),
            "name": f.get("name"),
            "lat": lat, "lon": lon,
            "capacity": f.get("capacity"),
            "numbikesavailable": f.get("numbikesavailable"),
            "numdocksavailable": f.get("numdocksavailable"),
        })
    df = pd.DataFrame(rows).dropna(subset=["stationcode","lat","lon"])
    for c in ["capacity","numbikesavailable","numdocksavailable"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
def main():
    model_path = ROOT / "models" / "lgb_nbvelos_T+1h.joblib"
    print(f"[check] artefact: {model_path}  (exists={model_path.exists()})")

    try:
        mdl, feats = load_model_bundle(1, model_dir=str(ROOT/"models"))
        print(f"[check] model type: {type(mdl)}")
        print(f"[check] nb features in artefact: {0 if feats is None else len(feats)}")
        if feats:
            print("[check] first 10 features:", feats[:10])
    except Exception as e:
        print(f"[check] load_model_bundle FAILED: {e}")
        return

    # charge un échantillon live
    try:
        df_live = fetch_opendata_v1()
        print(f"[check] df_live shape: {df_live.shape}")
        print(f"[check] df_live columns: {list(df_live.columns)}")
    except Exception as e:
        print(f"[check] fetch_opendata_v1 FAILED: {e}")
        return

    # aligne les features attendues
    if not feats:
        print("[check] artefact ne contient pas la clé 'features' → réentraîner en sauvegardant un bundle dict.")
        return

    missing = [c for c in feats if c not in df_live.columns]
    extra   = [c for c in df_live.columns if c not in feats]
    print(f"[check] missing columns (needed by model): {missing}")
    print(f"[check] extra columns (not used by model): {extra[:10]}{' ...' if len(extra)>10 else ''}")

    X = prepare_live_features(df_live, feats)
    print(f"[check] X shape: {X.shape}, all numeric={all(np.issubdtype(dt, np.number) for dt in X.dtypes)}")

    # test prédiction
    try:
        best_it = getattr(mdl, 'best_iteration', None)
        y = mdl.predict(X, num_iteration=best_it)
        print(f"[check] predict OK, y shape={y.shape}, stats: min={np.min(y):.2f} max={np.max(y):.2f}")
    except Exception as e:
        print(f"[check] predict FAILED: {e}")

if __name__ == "_main_":
    main()