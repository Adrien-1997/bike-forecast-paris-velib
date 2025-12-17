# api/routes/monitoring/data_health.py

"""Monitoring — Data Health endpoints.

/monitoring/data/health (LATEST only)

Arborescence cible (LATEST only)
--------------------------------
<GCS_MONITORING_PREFIX>/monitoring/data/health/latest/
    manifest.json
    kpis.json
    station_health.json
    coverage_by_hour.json
    alerts.json
    anomalies/
        manifest.json
        flat.json
        duplicates.json
        missing.json
"""

from __future__ import annotations

import os
import json
import math
from pathlib import Path
from typing import Literal, Tuple, Any, List, Dict

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    storage = None

router = APIRouter(prefix="/monitoring/data/health", tags=["monitoring-data"])

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "gcs").lower()
USE_LOCAL = STORAGE_BACKEND == "local"

REPO_ROOT = Path(__file__).resolve().parents[3]  # <repo>/
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
        print(f"[data_health] GCS_MONITORING_PREFIX invalide: '{raw}' → '{mon}' (backend={STORAGE_BACKEND})")
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
# Endpoints HEALTH (latest-only) — alignés sur network_overview.py
# ──────────────────────────────────────────────────────────────────────────────

NameHealth = Literal[
    "kpis",
    "station_health",
    "coverage_by_hour",
    "alerts",
    "anomalies",            # merge compat (flat+duplicates+missing)
    "anomalies_manifest",
    "anomalies_flat",
    "anomalies_duplicates",
    "anomalies_missing",
]

_TTL_BY_NAME: Dict[str, int] = {
    "kpis": 60,
    "station_health": 300,
    "coverage_by_hour": 300,
    "alerts": 60,
    "anomalies": 120,
    "anomalies_manifest": 120,
    "anomalies_flat": 120,
    "anomalies_duplicates": 120,
    "anomalies_missing": 120,
}


def _base_latest(mon: str) -> str:
    base = f"{mon}/monitoring" if not mon.endswith("/monitoring") else mon
    return f"{base}/data/health/latest"


def _name_to_rel(name: str) -> str:
    # "anomalies_*" live under anomalies/
    if name.startswith("anomalies_"):
        leaf = name[len("anomalies_") :]
        return f"anomalies/{leaf}.json"
    return f"{name}.json"


@router.get("")
def data_health_root(request: Request):
    """Alias root → kpis.json (latest-only)."""
    mon = _mon_prefix_or_500()
    latest = _base_latest(mon)
    uri = f"{latest}/kpis.json"
    return _proxy_json(uri, request, ttl=_TTL_BY_NAME["kpis"])


@router.get("/manifest")
def data_health_manifest(request: Request):
    """Manifest Data Health (latest-only)."""
    mon = _mon_prefix_or_500()
    latest = _base_latest(mon)
    uri = f"{latest}/manifest.json"
    return _proxy_json(uri, request, ttl=120)


@router.get("/{name}")
def data_health_doc(name: NameHealth, request: Request):
    """Accès à un artefact Data Health (latest-only)."""
    mon = _mon_prefix_or_500()
    latest = _base_latest(mon)

    if name == "anomalies":
        merged: List[Any] = []
        for shard in ("flat", "duplicates", "missing"):
            uri = f"{latest}/anomalies/{shard}.json"
            try:
                part = _read_json_any(uri)
            except FileNotFoundError:
                continue
            if isinstance(part, list):
                merged.extend(part)

        resp = JSONResponse(_json_sanitize(merged))
        resp.headers["Cache-Control"] = f"public, max-age={_TTL_BY_NAME['anomalies']}"
        resp.headers.setdefault("Access-Control-Allow-Origin", "*")
        return resp

    rel = _name_to_rel(str(name))
    uri = f"{latest}/{rel}"
    return _proxy_json(uri, request, ttl=_TTL_BY_NAME.get(str(name), 120))
