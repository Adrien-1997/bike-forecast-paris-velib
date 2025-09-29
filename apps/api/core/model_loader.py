from __future__ import annotations
from pathlib import Path
import joblib, time
from google.cloud import storage
from .settings import settings

class ModelBundle:
    def __init__(self, model, feat_cols, horizon_minutes:int):
        self.model = model
        self.feat_cols = feat_cols
        self.horizon_minutes = horizon_minutes

_cache = {"etag": None, "bundle": None, "ts": 0.0}

def _gcs_latest_blob(cli: storage.Client):
    pfx = settings.models_prefix
    assert pfx.startswith("gs://"), "MODELS_PREFIX doit commencer par gs://"
    bkt, key = pfx[5:].split("/", 1)
    # latest/…joblib
    return cli.bucket(bkt).blob(f"{key}/latest/lgb_nbvelos_T+15min.joblib")

def load_model_bundle(force: bool=False) -> ModelBundle:
    now = time.time()
    if not force and _cache["bundle"] and (now - _cache["ts"] < 600):
        return _cache["bundle"]

    cli = storage.Client()
    blob = _gcs_latest_blob(cli)
    blob.reload()
    etag = blob.etag
    if not force and _cache["etag"] == etag and _cache["bundle"]:
        _cache["ts"] = now
        return _cache["bundle"]

    local = Path(settings.tmp_dir) / "model_latest.joblib"
    local.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local.as_posix())
    bundle = joblib.load(local)
    mb = ModelBundle(bundle["model"], bundle.get("feat_cols", []), bundle.get("horizon_minutes", 15))

    _cache.update({"etag": etag, "bundle": mb, "ts": now})
    return mb
