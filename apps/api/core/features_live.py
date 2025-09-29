from pathlib import Path
import pandas as pd, numpy as np
from google.cloud import storage
from .settings import settings

def _latest_parquet(cli: storage.Client) -> Path:
    pfx = settings.gcs_serving_prefix  # gs://…/serving/features_4h
    assert pfx.startswith("gs://")
    bkt, p = pfx[5:].split("/", 1)
    blobs = list(cli.list_blobs(bkt, prefix=p+"/"))
    cand = [b for b in blobs if b.name.endswith(".parquet")]
    if not cand: raise RuntimeError("no features_4h parquet found")
    b = max(cand, key=lambda x: x.updated)
    loc = Path("/tmp")/Path(b.name).name
    if not loc.exists(): b.download_to_filename(loc.as_posix())
    return loc

def build_live_features(feat_cols: list[str]) -> pd.DataFrame:
    cli = storage.Client()
    parquet = _latest_parquet(cli)
    df = pd.read_parquet(parquet)
    if df.empty: return pd.DataFrame(columns=feat_cols+["stationcode","capacity","ts_utc"])
    # garder ce qui sert au modèle
    for c in feat_cols:
        if c not in df.columns: df[c]=0.0
    X = df.copy()
    X["stationcode"] = X["stationcode"].astype(str)
    if "capacity_bin" in X: X["capacity"]=X["capacity_bin"].fillna(0).astype(int)
    elif "capacity" not in X: X["capacity"]=0
    if "ts_utc" not in X: X["ts_utc"]=pd.Timestamp.utcnow().isoformat()+"Z"
    cols = feat_cols+["stationcode","capacity","ts_utc"]
    return X[cols].apply(pd.to_numeric, errors="ignore").fillna(0.0)
