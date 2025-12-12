# api/routes/model_explainability.py

"""Monitoring — Model Explainability endpoints.

Ce module expose les endpoints :

    /monitoring/model/explainability/...

Ils servent les artefacts JSON produits par le job
`build_model_explainability.py`, organisés comme :

    <GCS_MONITORING_PREFIX>/monitoring/model/explainability/<latest|TS>/h{H}/
        overview.json
        residuals.json
        calibration.json
        uncertainty.json
        feature_importance.json

En mode local (STORAGE_BACKEND=local), ces chemins GCS sont mappés sur :

    <DATA_ROOT>/monitoring/model/explainability/<latest|TS>/h{H}/...

Principes :
- le backend ne calcule rien → simple proxy JSON,
- GCS reste la “source logique” (via GCS_MONITORING_PREFIX),
- lecture soit via GCS, soit via le disque local selon STORAGE_BACKEND,
- `NaN` / `±Inf` → `null` avant renvoi.
"""

from __future__ import annotations

import os
import json
import re
import math
from pathlib import Path
from typing import Optional, Literal, Tuple, Dict, Any

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import JSONResponse

# google.cloud.storage est optionnel : utilisé seulement en mode GCS
try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover - mode local ou lib manquante
    storage = None

# /monitoring/model/explainability/*
router = APIRouter(prefix="/monitoring/model/explainability")

# ───────────────────────── Backend (local vs GCS) ─────────────────────────

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "gcs").lower()
USE_LOCAL = STORAGE_BACKEND == "local"

# Repo root ≈ <repo>/
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("DATA_ROOT", REPO_ROOT / "data"))

# Préfixe GCS commun pour le mapping local :
#   gs://.../velib/monitoring/... → data/monitoring/...
GCS_LOCAL_ROOT_PREFIX = (
    os.getenv(
        "GCS_LOCAL_ROOT_PREFIX",
        "gs://velib-forecast-472820_cloudbuild/velib/",
    ).rstrip("/")
    + "/"
)


def _split_gs(uri: str) -> Tuple[str, str]:
    """Découpe une URI `gs://bucket/key` en `(bucket, key)`."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Bad GCS URI: {uri}")
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def _local_path_from_gs(uri: str) -> Path:
    """Mappe une URI GCS velib/... vers un fichier local sous DATA_ROOT.

    Exemple :
        gs://.../velib/monitoring/model/explainability/latest/h15/overview.json
        → DATA_ROOT / "monitoring/model/explainability/latest/h15/overview.json"
    """
    if not uri.startswith(GCS_LOCAL_ROOT_PREFIX):
        raise ValueError(
            f"Cannot map GCS URI to local path:\n  uri={uri}\n  prefix={GCS_LOCAL_ROOT_PREFIX}"
        )
    rel = uri[len(GCS_LOCAL_ROOT_PREFIX) :]  # "monitoring/model/explainability/..."
    path = DATA_ROOT / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _read_json_any(gs_uri: str) -> Any:
    """Lit un JSON soit en local (DATA_ROOT), soit sur GCS, selon STORAGE_BACKEND."""
    if USE_LOCAL:
        path = _local_path_from_gs(gs_uri)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # pragma: no cover - erreurs disque
            raise ValueError(f"Invalid JSON in local file {path}: {e}") from e

    # Mode GCS
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
    """Retourne `GCS_MONITORING_PREFIX` normalisé (racine, sans forcer /monitoring).

    Étapes :
    - lit la variable d’environnement `GCS_MONITORING_PREFIX`,
    - strip espaces + guillemets, retire le slash final,
    - vérifie que le résultat commence par `gs://`.
    """
    mon_raw = os.environ.get("GCS_MONITORING_PREFIX", "")
    mon = mon_raw.strip().strip("'\"").rstrip("/")
    if not mon.startswith("gs://"):
        print(
            f"[model_explainability] GCS_MONITORING_PREFIX invalide. "
            f"Lu='{mon_raw}' nettoyé='{mon}' (backend={STORAGE_BACKEND})"
        )
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX manquant ou invalide")
    return mon


# Timestamp "version" attendu si on activait le time-travel : 2025-10-23T14-30-00Z
_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$")


def _sanitize_at(at: Optional[str]) -> str:
    """Normalise le paramètre `at` (time-travel) en nom de dossier."""
    if not at or at.strip().lower() in ("", "latest"):
        return "latest"
    s = at.strip()
    if not _AT_RE.match(s):
        return "latest"
    return s


def _json_sanitize(obj):
    """Remplace NaN/±Inf par None pour garantir un JSON valide."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _proxy_json(gs_uri: str, request: Request, ttl: int = 120) -> JSONResponse:
    """Lit un JSON (local ou GCS) et le proxifie avec des headers HTTP adaptés."""
    try:
        payload = _read_json_any(gs_uri)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document non trouvé")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur lecture backend: {e}")

    safe = _json_sanitize(payload)
    resp = JSONResponse(safe)
    resp.headers["Cache-Control"] = f"public, max-age={max(0, int(ttl))}"
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    return resp


# ── Endpoints EXPLAINABILITY (latest only, multi-horizon) ─────────────────────

DocNameExplain = Literal[
    "overview",
    "residuals",
    "calibration",
    "uncertainty",
    "feature_importance",
]

# TTL par type de document (en secondes)
_TTL_BY_DOC: Dict[str, int] = {
    "overview": 60,
    "residuals": 600,            # hist/QQ/ACF peu volatiles
    "calibration": 300,          # binning / heure
    "uncertainty": 300,          # couverture
    "feature_importance": 600,   # SHAP / gain XGB relativement stable
}


@router.get("/available")
def model_explain_available():
    """Expose la liste des documents d'explicabilité disponibles."""
    return {
        "docs": ["overview", "residuals", "calibration", "uncertainty", "feature_importance"],
        "horizons": "utiliser ?h=<minutes> (ex: 15, 60)",
        "time_travel": "latest uniquement (param ?at=… ignoré si non valide)",
        "examples": [
            "/monitoring/model/explainability/overview?h=15",
            "/monitoring/model/explainability/residuals?h=60",
            "/monitoring/model/explainability/calibration?h=15",
            "/monitoring/model/explainability/uncertainty?h=60",
            "/monitoring/model/explainability/feature_importance?h=15",
            "/monitoring/model/explainability/manifest?h=15",
        ],
    }


@router.get("/manifest")
def model_explain_manifest(
    request: Request,
    h: int = Query(15, description="Horizon en minutes (ex: 15, 60)"),
    at: Optional[str] = None,
):
    """Expose un "manifest" minimal pour un horizon donné (overview.json)."""
    if h <= 0:
        raise HTTPException(status_code=400, detail="Paramètre h (minutes) doit être > 0")

    mon = _mon_prefix_or_500()
    folder = _sanitize_at(at)  # 'latest'
    base = f"{mon}/monitoring" if not mon.endswith("/monitoring") else mon
    gs_uri = f"{base}/model/explainability/{folder}/h{int(h)}/overview.json"
    print(f"[model_explainability] manifest reading {gs_uri} (backend={STORAGE_BACKEND})")
    return _proxy_json(gs_uri, request, ttl=60)


@router.get("/{doc}")
def model_explain_doc(
    doc: DocNameExplain,
    request: Request,
    h: int = Query(15, description="Horizon en minutes (ex: 15, 60)"),
    at: Optional[str] = None,
):
    """Proxy JSON pour un document d'explicabilité particulier."""
    if h <= 0:
        raise HTTPException(status_code=400, detail="Paramètre h (minutes) doit être > 0")

    mon = _mon_prefix_or_500()
    folder = _sanitize_at(at)  # 'latest'
    base = f"{mon}/monitoring" if not mon.endswith("/monitoring") else mon
    gs_uri = f"{base}/model/explainability/{folder}/h{int(h)}/{doc}.json"
    ttl = _TTL_BY_DOC.get(doc, 180)
    print(f"[model_explainability] reading {gs_uri} (backend={STORAGE_BACKEND})")
    return _proxy_json(gs_uri, request, ttl=ttl)
