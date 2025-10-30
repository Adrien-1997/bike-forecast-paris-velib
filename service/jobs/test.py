import pandas as pd
import xgboost as xgb
from joblib import load
from google.cloud import storage
from io import BytesIO
from typing import Any, Optional

MODEL_URI = "gs://velib-forecast-472820_cloudbuild/velib/models/h60/latest.joblib"

def load_joblib_from_gcs(gs_uri: str) -> Any:
    assert gs_uri.startswith("gs://")
    bkt, key = gs_uri[5:].split("/", 1)
    buf = BytesIO()
    storage.Client().bucket(bkt).blob(key).download_to_file(buf)
    buf.seek(0)
    return load(buf)

def extract_booster(obj: Any) -> xgb.Booster:
    """Return an xgboost.Booster from various saved formats."""
    # 1) Already a Booster
    if isinstance(obj, xgb.Booster):
        return obj
    # 2) sklearn wrapper
    if hasattr(obj, "get_booster"):
        return obj.get_booster()
    # 3) sklearn Pipeline or dict-like wrappers
    #    - Pipeline: look into named_steps
    if hasattr(obj, "named_steps") and isinstance(obj.named_steps, dict):
        for step in obj.named_steps.values():
            if hasattr(step, "get_booster"):
                return step.get_booster()
            if isinstance(step, xgb.Booster):
                return step
    # 4) Dict bundles: try common keys
    if isinstance(obj, dict):
        for k in ("model", "xgb_model", "estimator", "booster"):
            v = obj.get(k)
            if v is None:
                continue
            if isinstance(v, xgb.Booster):
                return v
            if hasattr(v, "get_booster"):
                return v.get_booster()
    # 5) Last resort: helpful introspection
    raise TypeError(f"Don't know how to extract Booster from type={type(obj)}; "
                    f"attrs={dir(obj)[:20]}")

def main():
    print(f"🔹 Loading model: {MODEL_URI}")
    obj = load_joblib_from_gcs(MODEL_URI)
    booster = extract_booster(obj)

    # Native importances
    gain   = booster.get_score(importance_type="gain")   or {}
    weight = booster.get_score(importance_type="weight") or {}
    cover  = booster.get_score(importance_type="cover")  or {}

    feats = set(gain) | set(weight) | set(cover)
    rows = [{
        "feature": f,
        "gain":   float(gain.get(f, 0.0)),
        "weight": float(weight.get(f, 0.0)),
        "cover":  float(cover.get(f, 0.0)),
    } for f in feats]

    df_imp = pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)
    print("\n📊 Top 20 features (by gain):")
    print(df_imp.head(20))

if __name__ == "__main__":
    main()
