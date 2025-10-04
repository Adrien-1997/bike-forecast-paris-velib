# apps/api/routes/forecast.py
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import fsspec

from api.core.features_live import build_live_features
from api.core.settings import settings

router = APIRouter()

# ---------- Loader tolérant ----------
def _load_model_and_feats():
    """
    Essaie successivement :
      - model_loader.load_trained_model() -> (model, feat_cols, horizon)
      - model_loader.load_model() -> (model, meta)  où meta['feat_cols'], meta['horizon_minutes']
      - lecture directe du joblib settings.models_prefix
    """
    # 1) load_trained_model()
    try:
        from api.core.model_loader import load_trained_model  # type: ignore
        mdl, feat_cols, horizon = load_trained_model()
        return mdl, list(feat_cols or []), int(horizon or 15)
    except Exception:
        pass

    # 2) load_model()
    try:
        from api.core.model_loader import load_model  # type: ignore
        res = load_model()
        if isinstance(res, tuple) and len(res) >= 1:
            mdl = res[0]
            meta = res[1] if len(res) > 1 and isinstance(res[1], dict) else {}
        elif isinstance(res, dict):
            mdl = res.get("model")
            meta = res
        else:
            mdl, meta = res, {}
        feat_cols = meta.get("feat_cols")
        if not feat_cols:
            # tenter d'inférer à partir du modèle
            for attr in ("feature_name_", "feature_names_"):
                if hasattr(mdl, attr):
                    feat_cols = getattr(mdl, attr)
                    break
        if not feat_cols and hasattr(mdl, "booster_"):
            try:
                feat_cols = mdl.booster_.feature_name()
            except Exception:
                pass
        horizon = int(meta.get("horizon_minutes", 15))
        return mdl, list(feat_cols or []), horizon
    except Exception:
        pass

    # 3) Lecture directe du joblib à partir de settings.models_prefix
    uri = getattr(settings, "models_prefix", None) or getattr(settings, "model_uri", None)
    if not uri:
        raise RuntimeError("Aucun modèle trouvé (settings.models_prefix manquant)")

    with fsspec.open(uri, "rb") as f:
        obj = joblib.load(f)

    # le joblib peut être soit un dict {'model','feat_cols','horizon_minutes'} soit directement un modèle
    if isinstance(obj, dict):
        mdl = obj.get("model", obj)
        feat_cols = obj.get("feat_cols")
        horizon = int(obj.get("horizon_minutes", 15))
    else:
        mdl = obj
        feat_cols, horizon = None, 15

    if not feat_cols:
        for attr in ("feature_name_", "feature_names_"):
            if hasattr(mdl, attr):
                feat_cols = getattr(mdl, attr)
                break
    if not feat_cols and hasattr(mdl, "booster_"):
        try:
            feat_cols = mdl.booster_.feature_name()
        except Exception:
            pass

    return mdl, list(feat_cols or []), horizon


# ---------- Schéma d'entrée ----------
class ForecastBatchIn(BaseModel):
    stationcodes: list[str] | None = None
    h: int | None = None  # ignoré si le modèle a un horizon fixé


# ---------- Route ----------
@router.post("/forecast/batch")
def forecast_batch(payload: ForecastBatchIn):
    try:
        # 1) modèle + noms de features
        model, feat_cols, horizon = _load_model_and_feats()

        # 2) construire les features live alignées
        X = build_live_features(feat_cols)  # doit renvoyer feat_cols + ['stationcode','capacity','ts_utc']
        if X is None or X.empty:
            return []

        # 3) filtrage (optionnel)
        if payload.stationcodes:
            want = set(map(str, payload.stationcodes))
            X = X[X["stationcode"].astype(str).isin(want)]
        if X.empty:
            return []

        # 4) prédire
        X_feats = X[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y = model.predict(X_feats)
        y = np.maximum(0, np.round(y)).astype(int)

        # 5) sortie JSON-safe
        out = pd.DataFrame({
            "stationcode": X["stationcode"].astype(str),
            "bikes_pred_t15": y,
            "capacity": pd.to_numeric(X["capacity"], errors="coerce").fillna(0).astype(int),
            "ts_utc": pd.to_datetime(X["ts_utc"], errors="coerce"),
        })
        out["ts_utc"] = out["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        return out.to_dict(orient="records")

    except Exception as e:
        return {"error": str(e), "where": "/forecast/batch"}
