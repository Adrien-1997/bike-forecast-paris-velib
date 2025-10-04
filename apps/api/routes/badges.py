# apps/api/routes/badges.py
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, Response

from api.core.settings import settings
from api.core.features_live import _latest_parquet as _latest_features_parquet

# lecture parquet: fsspec si dispo, sinon client GCS
try:
    import fsspec  # type: ignore
except Exception:  # pragma: no cover
    fsspec = None  # type: ignore

try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    storage = None  # type: ignore


router = APIRouter(prefix="/badges", tags=["badges"])


# ───────────────────────────────────────────────────────────────
# Helpers internes
# ───────────────────────────────────────────────────────────────
def _serving_prefix() -> str:
    return (
        getattr(settings, "GCS_SERVING_PREFIX", None)
        or getattr(settings, "gcs_serving_prefix", None)
        or ""
    ).rstrip("/")


def _infer_bronze_prefix() -> Optional[str]:
    """Déduit le préfixe bronze depuis le prefix serving"""
    s = _serving_prefix()
    if not s.startswith("gs://"):
        return None
    parts = s.split("/")
    try:
        i = parts.index("serving")
        parts[i] = "bronze"
        return "/".join(parts[: i + 1])
    except ValueError:
        return None


def _split_gs(uri: str) -> tuple[str, str]:
    assert uri.startswith("gs://")
    b, k = uri[5:].split("/", 1)
    return b, k


def _read_parquet_any(uri: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Lecture parquet via fsspec > GCS > local"""
    # 1) fsspec
    if fsspec is not None:
        try:
            return pd.read_parquet(uri, columns=columns)
        except Exception:
            pass

    # 2) client GCS
    if uri.startswith("gs://") and storage is not None:
        bkt, key = _split_gs(uri)
        cli = storage.Client()
        blob = cli.bucket(bkt).get_blob(key)
        if not blob:
            raise FileNotFoundError(uri)
        data = blob.download_as_bytes()
        return pd.read_parquet(pd.io.common.BytesIO(data), columns=columns)

    # 3) local
    return pd.read_parquet(uri, columns=columns)


def _latest_blob_under(prefix: str) -> Optional[str]:
    """Renvoie le parquet le plus récent sous un préfixe GCS"""
    if storage is None or not prefix.startswith("gs://"):
        return None
    bkt, key = _split_gs(prefix.rstrip("/") + "/")
    cli = storage.Client()
    blobs = list(cli.list_blobs(bkt, prefix=key))
    cands = [b for b in blobs if b.name.endswith(".parquet")]
    if not cands:
        return None
    latest = max(cands, key=lambda b: b.updated)
    return f"gs://{latest.bucket.name}/{latest.name}"


def _weather_from_bronze() -> Dict[str, Any]:
    """Lit la météo depuis le dernier snapshot live (bronze)"""
    bronze_prefix = getattr(settings, "GCS_RAW_PREFIX", None) or _infer_bronze_prefix()
    if not bronze_prefix or not bronze_prefix.startswith("gs://"):
        return {}

    uri = _latest_blob_under(bronze_prefix)
    if not uri:
        return {}

    df = _read_parquet_any(uri, columns=["tbin_utc", "temp_C", "precip_mm", "wind_mps"])
    if df is None or df.empty:
        return {}

    df["tbin_utc"] = pd.to_datetime(df["tbin_utc"], utc=True, errors="coerce")
    latest_bin = df["tbin_utc"].max()
    if pd.isna(latest_bin):
        return {}

    cut = df[df["tbin_utc"] == latest_bin].copy()

    def _num_mean(col: str) -> Optional[float]:
        if col not in cut.columns:
            return None
        v = pd.to_numeric(cut[col], errors="coerce").dropna()
        if v.empty:
            return None
        return float(v.mean())

    return {
        "ts_utc": latest_bin.isoformat().replace("+00:00", "Z"),
        "temp_C": _num_mean("temp_C"),
        "precip_mm": _num_mean("precip_mm"),
        "wind_mps": _num_mean("wind_mps"),
    }


def _freshness_from_features() -> Dict[str, Any]:
    """Calcule la fraîcheur du dernier parquet de features (serving)."""
    try:
        meta = _latest_features_parquet(_serving_prefix())
        uri = meta.get("uri")
        if not uri:
            return {}

        df = _read_parquet_any(uri, columns=["tbin_latest"])
        if df is None or df.empty or "tbin_latest" not in df.columns:
            return {}

        ts = pd.to_datetime(df["tbin_latest"], utc=True, errors="coerce").max()
        if pd.isna(ts):
            return {}

        now = datetime.now(timezone.utc)
        age_min = (now - ts).total_seconds() / 60.0

        return {
            "parquet_ts_utc": ts.isoformat().replace("+00:00", "Z"),
            "age_minutes": round(age_min, 1),
        }
    except Exception:
        return {}


# ───────────────────────────────────────────────────────────────
# Route principale
# ───────────────────────────────────────────────────────────────
@router.get("")
def get_badges(response: Response):
    weather = _weather_from_bronze()
    freshness = _freshness_from_features()

    updated_at = freshness.get("parquet_ts_utc") or weather.get("ts_utc")
    freshness_min = freshness.get("age_minutes")

    response.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"

    return {
        "weather": weather,
        "freshness": freshness,
        "meta": {
            "updated_at": updated_at,
            "freshness_min": freshness_min,
        },
    }
