# apps/api/core/features_live.py
from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from api.core.settings import settings

# essaye d'avoir fsspec/gcsfs si dispo
try:
    import fsspec  # type: ignore
except Exception:  # pragma: no cover
    fsspec = None  # type: ignore

# fallback GCS officiel (déjà présent chez toi)
try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    storage = None  # type: ignore


# ───────────────────────── helpers URI ─────────────────────────

def _serving_prefix() -> str:
    """
    Récupère le 'serving' depuis settings (maj/min acceptés).
    Peut être soit un dossier gs://.../serving soit une URI .parquet.
    """
    p = getattr(settings, "GCS_SERVING_PREFIX", None) or getattr(settings, "gcs_serving_prefix", None)
    if not p:
        raise RuntimeError("GCS_SERVING_PREFIX / gcs_serving_prefix non défini dans settings")
    return str(p).rstrip("/")


def _ensure_latest_uri(prefix_or_file: str) -> str:
    """Si .parquet → renvoie tel quel, sinon ajoute /features_4h/latest.parquet."""
    s = prefix_or_file.rstrip("/")
    if s.endswith(".parquet"):
        return s
    return f"{s}/features_4h/latest.parquet"


def _split_gs(uri: str) -> tuple[str, str]:
    assert uri.startswith("gs://"), "URI GCS attendue (gs://...)"
    b, k = uri[5:].split("/", 1)
    return b, k


# ───────────────────────── meta & lecture ─────────────────────────

def _fs_info(uri: str) -> Dict[str, Any]:
    """
    Tente d'obtenir {etag,size}:
      1) via fsspec/gcsfs si dispo
      2) sinon, via google-cloud-storage (fallback)
    """
    # 1) fsspec si présent
    if fsspec is not None:
        try:
            fs, _, paths = fsspec.get_fs_token_paths(uri)
            info = fs.info(paths[0])
            return {"etag": info.get("etag"), "size": info.get("size")}
        except Exception:
            pass

    # 2) fallback GCS
    if uri.startswith("gs://") and storage is not None:
        try:
            bucket, key = _split_gs(uri)
            cli = storage.Client()
            blob = cli.bucket(bucket).get_blob(key)
            if blob:
                return {"etag": getattr(blob, "etag", None), "size": getattr(blob, "size", None)}
        except Exception:
            pass

    return {"etag": None, "size": None}


def _read_parquet_any(uri: str, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Lit un parquet:
      1) pd.read_parquet(uri) (fonctionne si gcsfs dispo)
      2) fallback GCS: download bytes → pd.read_parquet(BytesIO)
    """
    # 1) tentative directe (fsspec/gcsfs ou local)
    try:
        return pd.read_parquet(uri, columns=list(columns) if columns is not None else None)
    except Exception:
        pass

    # 2) fallback Google Cloud Storage si gs://
    if uri.startswith("gs://") and storage is not None:
        bucket, key = _split_gs(uri)
        cli = storage.Client()
        blob = cli.bucket(bucket).get_blob(key)
        if not blob:
            raise FileNotFoundError(f"Blob introuvable: {uri}")
        data = blob.download_as_bytes()
        return pd.read_parquet(BytesIO(data), columns=list(columns) if columns is not None else None)

    # si tout a échoué
    raise RuntimeError(f"Impossible de lire le parquet: {uri}")


# ───────────────────────── API publique ─────────────────────────

def _latest_parquet(prefix: Optional[str] = None) -> Dict[str, Any]:
    """
    Retourne {uri, etag, size} vers latest.parquet.
    Pas de listage de blobs: l'URI est déterministe.
    """
    base = (prefix or _serving_prefix()).rstrip("/")
    uri = _ensure_latest_uri(base)
    meta = _fs_info(uri)
    return {"uri": uri, **meta}


def read_latest_parquet(columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Lit latest.parquet (ou l’URI fournie dans settings)."""
    m = _latest_parquet()
    return _read_parquet_any(m["uri"], columns=columns)


def build_live_features(feat_cols: List[str]) -> pd.DataFrame:
    """
    Construit la matrice de features alignée sur le modèle à partir de latest.parquet.
    - Remplit les colonnes manquantes avec 0.0
    - stationcode := station_id / stationcode (string)
    - capacity := capacity | capacity_bin
    - ts_utc := ts_utc | tbin_latest | now UTC
    Retourne: feat_cols + ['stationcode','capacity','ts_utc'].
    """
    m = _latest_parquet()
    uri = m["uri"]

    df = _read_parquet_any(uri)
    if df is None or df.empty:
        return pd.DataFrame(columns=list(feat_cols) + ["stationcode", "capacity", "ts_utc"])

    X = df.copy()

    # stationcode (toujours string)
    if "stationcode" in X.columns:
        X["stationcode"] = X["stationcode"].astype(str)
    elif "station_id" in X.columns:
        X["stationcode"] = X["station_id"].astype(str)
    else:
        X["stationcode"] = ""

    # capacity
    if "capacity" in X.columns:
        cap = pd.to_numeric(X["capacity"], errors="coerce")
    elif "capacity_bin" in X.columns:
        cap = pd.to_numeric(X["capacity_bin"], errors="coerce")
    else:
        cap = pd.Series(0, index=X.index, dtype="float64")
    X["capacity"] = cap.fillna(0).round().astype(int)

    # ts_utc (naive UTC)
    if "ts_utc" in X.columns:
        ts = pd.to_datetime(X["ts_utc"], utc=True, errors="coerce").dt.tz_convert(None)
    elif "tbin_latest" in X.columns:
        ts = pd.to_datetime(X["tbin_latest"], utc=True, errors="coerce").dt.tz_convert(None)
    else:
        now_naive = pd.Timestamp.now(tz="UTC").tz_convert(None)
        ts = pd.Series(now_naive, index=X.index)
    fallback_naive = pd.Timestamp.now(tz="UTC").tz_convert(None)
    X["ts_utc"] = ts.fillna(fallback_naive).astype("datetime64[ns]")

    # assurer toutes les colonnes du modèle (0.0 si manquantes) + conversion numérique soft
    for c in feat_cols:
        if c not in X.columns:
            X[c] = 0.0
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    cols = list(feat_cols) + ["stationcode", "capacity", "ts_utc"]
    return X[cols].copy()


__all__ = [
    "_latest_parquet",
    "read_latest_parquet",
    "build_live_features",
]
