# api/routes/network_overview.py

"""Monitoring — Network Overview endpoints.

Ce module expose les endpoints :

    /monitoring/network/overview/...

Ils servent les artefacts JSON produits par le job
`build_network_overview.py` sur GCS, typiquement organisés comme :

    <GCS_MONITORING_PREFIX>/monitoring/network/overview/<latest|TS>/
        kpis.json
        snapshot_distribution.json
        today_curve.json
        ref_median_curve.json
        kpis_today_vs_lags.json
        snapshot_map.json
        stations_tension.json

En mode local (STORAGE_BACKEND=local), ces chemins GCS sont mappés sur :

    <DATA_ROOT>/monitoring/network/overview/<latest|TS>/...

Principe :
- le backend ne calcule rien → simple proxy JSON,
- GCS reste la “source logique” (via GCS_MONITORING_PREFIX),
- lecture soit via GCS, soit via disque local selon STORAGE_BACKEND,
- `NaN` / `±Inf` → `null` avant renvoi.
"""

from __future__ import annotations

import os
import json
import re
import math
from pathlib import Path
from typing import Optional, Literal, Tuple, Dict, Any

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

# google.cloud.storage est optionnel : utilisé seulement en mode GCS
try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover - mode local ou lib manquante
    storage = None

router = APIRouter(prefix="/monitoring/network")

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
        gs://.../velib/monitoring/network/overview/latest/kpis.json
        → DATA_ROOT / "monitoring/network/overview/latest/kpis.json"
    """
    if not uri.startswith(GCS_LOCAL_ROOT_PREFIX):
        raise ValueError(
            f"Cannot map GCS URI to local path:\n  uri={uri}\n  prefix={GCS_LOCAL_ROOT_PREFIX}"
        )
    rel = uri[len(GCS_LOCAL_ROOT_PREFIX):]  # "monitoring/network/overview/..."
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
        except Exception as e:  # pragma: no cover
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
    """Retourne `GCS_MONITORING_PREFIX` normalisé ou lève une 500."""
    mon_raw = os.environ.get("GCS_MONITORING_PREFIX", "")
    mon = mon_raw.strip().strip("'\"").rstrip("/")
    if not mon.startswith("gs://"):
        print(
            f"[network_overview] GCS_MONITORING_PREFIX invalide. "
            f"Lu='{mon_raw}' nettoyé='{mon}' (backend={STORAGE_BACKEND})"
        )
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX manquant ou invalide")
    print(f"[network_overview] GCS_MONITORING_PREFIX='{mon}' (backend={STORAGE_BACKEND})")
    return mon


# Timestamp "version" attendu pour le time-travel: 2025-10-23T14-30-00Z
_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$")


def _sanitize_at(at: Optional[str]) -> str:
    """Normalise le paramètre `at` (time-travel) en nom de dossier."""
    if not at or at.strip().lower() in ("", "latest"):
        return "latest"
    s = at.strip()
    if not _AT_RE.match(s):
        # force 'latest' si format non reconnu (évite path traversal)
        return "latest"
    return s


def _json_sanitize(obj):
    """Remplace NaN/±Inf par None pour produire un JSON valide."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _proxy_json(gs_uri: str, request: Request, ttl: int = 120) -> JSONResponse:
    """Lit un JSON (local ou GCS) et le proxifie avec des headers de cache adaptés."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────

DocNameOverview = Literal[
    "kpis",
    "snapshot_distribution",
    "today_curve",
    "ref_median_curve",
    "kpis_today_vs_lags",
    "snapshot_map",
    "stations_tension",
]

# TTL (secondes) par type de document
_TTL_BY_DOC: Dict[str, int] = {
    "kpis": 60,
    "snapshot_distribution": 120,
    "today_curve": 60,
    "ref_median_curve": 300,
    "kpis_today_vs_lags": 120,
    "snapshot_map": 60,
    "stations_tension": 120,
}


@router.get("/overview/available")
def network_overview_available():
    """Petit index des documents *network overview* disponibles."""
    docs = list(_TTL_BY_DOC.keys())
    return {
        "docs": docs,
        "time_travel": "utiliser ?at=YYYY-MM-DDTHH-MM-SSZ ou sans param pour latest",
    }


@router.get("/overview/{doc}")
def network_overview_doc(doc: DocNameOverview, request: Request, at: Optional[str] = None):
    """Proxy JSON pour les documents générés par le job Overview."""
    mon = _mon_prefix_or_500()  # ex: gs://velib-forecast-472820_cloudbuild/velib
    folder = _sanitize_at(at)   # 'latest' ou timestamp normalisé

    # si l'ENV finit déjà par /monitoring, on ne le redouble pas
    base = f"{mon}/monitoring" if not mon.endswith("/monitoring") else mon
    gs_uri = f"{base}/network/overview/{folder}/{doc}.json"
    ttl = _TTL_BY_DOC.get(doc, 120)
    print(f"[network_overview] reading {gs_uri} (backend={STORAGE_BACKEND})")
    return _proxy_json(gs_uri, request, ttl=ttl)
