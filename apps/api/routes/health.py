# apps/api/routes/health.py
from __future__ import annotations

import os
import datetime as dt
from typing import Dict, Any, Optional

from fastapi import APIRouter

from api.core.settings import settings
from api.core.features_live import _latest_parquet

# essaye d'avoir GCS, sinon on renverra juste l'URI
try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    storage = None  # type: ignore

router = APIRouter(prefix="/health", tags=["health"])


def _gcs_blob_meta(uri: str) -> Optional[Dict[str, Any]]:
    """Retourne {uri, etag, size} pour une URI gs:// si possible (sinon None)."""
    if not (uri and uri.startswith("gs://") and storage is not None):
        return None
    try:
        bucket, key = uri[5:].split("/", 1)
        cli = storage.Client()
        blob = cli.bucket(bucket).get_blob(key)
        if not blob:
            return {"uri": uri, "etag": None, "size": None}
        # assure d'avoir les champs à jour
        blob.reload()
        return {"uri": uri, "etag": getattr(blob, "etag", None), "size": getattr(blob, "size", None)}
    except Exception:
        # en cas d’erreur GCS, on renvoie au moins l’URI
        return {"uri": uri, "etag": None, "size": None}


@router.get("")
def health():
    out: Dict[str, Any] = {
        "status": "ok",
        "image_tag": os.getenv("IMAGE_TAG", ""),
        "utc": dt.datetime.utcnow().isoformat() + "Z",
    }

    # Features (latest.parquet)
    try:
        feat_meta = _latest_parquet()  # -> {"uri": "...", "etag": ..., "size": ...}
        out["features"] = feat_meta
    except Exception as e:
        out["features_error"] = str(e)

    # Modèle (joblib)
    try:
        model_uri = getattr(settings, "models_prefix", None)
        if model_uri:
            meta = _gcs_blob_meta(str(model_uri)) or {"uri": str(model_uri), "etag": None, "size": None}
            out["model"] = meta
    except Exception as e:
        out["model_error"] = str(e)

    return out
