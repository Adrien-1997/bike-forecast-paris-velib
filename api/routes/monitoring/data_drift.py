# api/routes/data_drift.py

"""Monitoring — Data Drift endpoints.

/monitoring/data/drift (LATEST only)

Arborescence cible (LATEST only)
--------------------------------
<GCS_MONITORING_PREFIX>/monitoring/data/drift/latest/
    manifest.json
    summary.json
    psi_by_feature.json
    ks_by_feature.json
    deltas_by_feature.json
    psi_global_daily_ema.json
    alerts.json
    bounds.json
    zones.json
    features_detected.json
"""

from __future__ import annotations

import os
import json
import math
from pathlib import Path
from typing import Literal, Tuple, Any, Dict

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    storage = None

router = APIRouter(prefix="/monitoring/data/drift", tags=["monitoring-data"])

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "gcs").lower()
USE_LOCAL = STORAGE_BACKEND == "local"

REPO_ROOT = Path(__file__).resolve().parents[2]  # <repo>/
DATA_ROOT = Path(os.getenv("DATA_ROOT", REPO_ROOT / "data"))

GCS_LOCAL_ROOT_PREFIX = (
    os.getenv("GCS_LOCAL_ROOT_PREFIX", "gs://velib-forecast-472820_cloudbuild/velib/").rstrip("/")
    + "/"
)


def _split_gs(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Bad GCS URI: {uri}")
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def _local_path_from_gs(uri: str) -> Path:
    if not uri.startswith(GCS_LOCAL_ROOT_PREFIX):
        raise ValueError(
            f"Cannot map GCS URI to local path:\n  uri={uri}\n  prefix={GCS_LOCAL_ROOT_PREFIX}"
        )
    rel = uri[len(GCS_LOCAL_ROOT_PREFIX) :]
    path = DATA_ROOT / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _read_json_any(gs_uri: str) -> Any:
    if USE_LOCAL:
        path = _local_path_from_gs(gs_uri)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # pragma: no cover
            raise ValueError(f"Invalid JSON in local file {path}: {e}") from e

    if storage is None:
        raise RuntimeError("google-cloud-storage requis en mode GCS")

    bkt, key = _split_gs(gs_uri)
    client = storage.Client()
    blob = client.bucket(bkt).blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"GCS blob not found: {gs_uri}")
    data = blob.download_as_bytes()
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Invalid JSON in {gs_uri}: {e}") from e


def _mon_prefix_or_500() -> str:
    raw = os.environ.get("GCS_MONITORING_PREFIX", "")
    mon = raw.strip().strip("'\"").rstrip("/")
    if not mon.startswith("gs://"):
        print(f"[data_drift] GCS_MONITORING_PREFIX invalide: '{raw}' → '{mon}' (backend={STORAGE_BACKEND})")
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX manquant ou invalide")
    return mon


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _proxy_json(gs_uri: str, request: Request, ttl: int) -> JSONResponse:
    try:
        payload = _read_json_any(gs_uri)
    except FileNotFoundError:
        resp = JSONResponse({}, status_code=204)
        resp.headers["Cache-Control"] = f"public, max-age={int(ttl)}"
        resp.headers.setdefault("Access-Control-Allow-Origin", "*")
        return resp
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur lecture backend: {e}")

    resp = JSONResponse(_json_sanitize(payload))
    resp.headers["Cache-Control"] = f"public, max-age={int(ttl)}"
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    return resp


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints DRIFT (latest-only) — alignés sur network_overview.py
# ──────────────────────────────────────────────────────────────────────────────

NameDrift = Literal[
    "summary",
    "psi_by_feature",
    "ks_by_feature",
    "deltas_by_feature",
    "psi_global_daily_ema",
    "alerts",
    "bounds",
    "zones",
    "features_detected",
]

_TTL_BY_NAME: Dict[str, int] = {
    "summary": 60,
    "psi_by_feature": 300,
    "ks_by_feature": 300,
    "deltas_by_feature": 300,
    "psi_global_daily_ema": 120,
    "alerts": 60,
    "bounds": 300,
    "zones": 300,
    "features_detected": 120,
}


def _base_latest(mon: str) -> str:
    base = f"{mon}/monitoring" if not mon.endswith("/monitoring") else mon
    return f"{base}/data/drift/latest"


@router.get("")
def data_drift_root(request: Request):
    """Alias root → summary.json (latest-only)."""
    mon = _mon_prefix_or_500()
    latest = _base_latest(mon)
    uri = f"{latest}/summary.json"
    return _proxy_json(uri, request, ttl=_TTL_BY_NAME["summary"])


@router.get("/manifest")
def data_drift_manifest(request: Request):
    """Manifest Data Drift (latest-only)."""
    mon = _mon_prefix_or_500()
    latest = _base_latest(mon)
    uri = f"{latest}/manifest.json"
    return _proxy_json(uri, request, ttl=120)


@router.get("/{name}")
def data_drift_doc(name: NameDrift, request: Request):
    """Accès à un artefact Data Drift (latest-only)."""
    mon = _mon_prefix_or_500()
    latest = _base_latest(mon)
    uri = f"{latest}/{name}.json"
    return _proxy_json(uri, request, ttl=_TTL_BY_NAME.get(str(name), 120))
