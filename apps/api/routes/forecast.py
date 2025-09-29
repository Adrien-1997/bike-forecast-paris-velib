from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

from ..core.model_loader import load_model_bundle
from ..core.features_live import build_live_features

router = APIRouter(prefix="/forecast", tags=["forecast"])

class BatchIn(BaseModel):
    stationcodes: Optional[List[str]] = None
    h: int = 15

@router.post("/batch")
def forecast_batch(body: BatchIn):
    mb = load_model_bundle()
    if body.h != mb.horizon_minutes:
        # pour l’instant on ne supporte que l’horizon du modèle chargé
        pass
    X = build_live_features(mb.feat_cols)
    if body.stationcodes:
        keep = set(map(str, body.stationcodes))
        X = X[X["stationcode"].isin(keep)]
    if X.empty:
        return []

    sc = X["stationcode"].tolist()
    cap = X["capacity"].to_numpy()
    ts  = X["ts_utc"].tolist()
    Xmat = X.drop(columns=["stationcode","capacity","ts_utc"], errors="ignore")
    best_it = getattr(mb.model, "best_iteration_", None) or getattr(mb.model, "best_iteration", None)
    y = mb.model.predict(Xmat, num_iteration=best_it) if best_it else mb.model.predict(Xmat)
    y = np.clip(y, 0, cap)
    out = [{"stationcode": sc[i], "y_nb_pred": float(y[i]), "capacity": int(cap[i]), "ts_utc": ts[i]} for i in range(len(sc))]
    return out
