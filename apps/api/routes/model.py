from fastapi import APIRouter
from google.cloud import storage
from ..core.settings import settings
import json, io

router = APIRouter(prefix="/model", tags=["model"])

@router.get("")
def model_info():
    if not settings.models_prefix.startswith("gs://"):
        return {"timestamp": None, "mae": None, "rmse": None, "horizon_minutes": 15, "artifact_uri": None}
    bkt, pfx = settings.models_prefix[5:].split("/", 1)
    cli = storage.Client()
    blob = cli.bucket(bkt).blob(f"{pfx}/latest/metrics.json")
    if not blob.exists():
        return {"timestamp": None, "mae": None, "rmse": None, "horizon_minutes": 15, "artifact_uri": None}
    data = json.loads(blob.download_as_bytes())
    return {
        "timestamp": data.get("created_at"),
        "mae": data.get("mae"),
        "rmse": data.get("rmse"),
        "horizon_minutes": data.get("horizon_minutes", 15),
        "artifact_uri": f"gs://{bkt}/{pfx}/latest/lgb_nbvelos_T+15min.joblib",
    }
