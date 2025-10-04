# apps/api/routes/model.py
from fastapi import APIRouter
from google.cloud import storage
from api.core.settings import settings
import json

router = APIRouter(prefix="/model", tags=["model"])

def _target_blobs():
    """
    Retourne (joblib_blob, metrics_blob_ou_None)
    en fonction de settings.models_prefix :
    - fichier direct .joblib → joblib_blob direct, pas de metrics
    - dossier → latest/lgb_nbvelos_T+15min.joblib + metrics.json
    """
    pfx = settings.models_prefix
    if not pfx or not pfx.startswith("gs://"):
        return None, None

    bucket_name, key = pfx[5:].split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    if key.endswith(".joblib"):
        return bucket.blob(key), None

    joblib_blob = bucket.blob(f"{key.rstrip('/')}/latest/lgb_nbvelos_T+15min.joblib")
    metrics_blob = bucket.blob(f"{key.rstrip('/')}/latest/metrics.json")
    return joblib_blob, metrics_blob

@router.get("")
def model_info():
    joblib_blob, metrics_blob = _target_blobs()
    if not joblib_blob or not joblib_blob.exists():
        return {
            "timestamp": None,
            "mae": None,
            "rmse": None,
            "horizon_minutes": 15,
            "artifact_uri": None,
        }

    info = {
        "timestamp": None,
        "mae": None,
        "rmse": None,
        "horizon_minutes": 15,
        "artifact_uri": f"gs://{joblib_blob.bucket.name}/{joblib_blob.name}",
    }

    if metrics_blob and metrics_blob.exists():
        try:
            data = json.loads(metrics_blob.download_as_bytes())
            info.update({
                "timestamp": data.get("created_at"),
                "mae": data.get("mae"),
                "rmse": data.get("rmse"),
                "horizon_minutes": data.get("horizon_minutes", 15),
            })
        except Exception:
            pass

    return info

# Alias pour compat UI existante
@router.get("/info")
def model_info_alias():
    return model_info()
