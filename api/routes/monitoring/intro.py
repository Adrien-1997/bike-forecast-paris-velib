# api/routes/monitoring/intro.py

"""Monitoring — Intro (landing/overview) endpoint.

Ce module sert un document JSON *consolidé* produit par le job
`build_monitoring_intro.py`, utilisé pour alimenter :

- l’intro de la page Monitoring,
- et/ou des blocs de synthèse sur la landing (couverture, fraîcheur, drift…).

Arborescence logique (time-travel supporté)
------------------------------------------
<GCS_MONITORING_PREFIX>/monitoring/intro/<latest|YYYY-MM-DDTHH-MM-SSZ>/intro.json

En mode local (STORAGE_BACKEND=local), ces chemins sont mappés vers :

<DATA_ROOT>/monitoring/intro/<latest|YYYY-MM-DDTHH-MM-SSZ>/intro.json

Principe :
- le backend ne calcule rien → simple proxy JSON,
- GCS reste la source de vérité logique (GCS_MONITORING_PREFIX),
- lecture soit via GCS (STORAGE_BACKEND=gcs), soit sur disque (local),
- NaN / ±Inf → null avant renvoi.
"""

from __future__ import annotations

import os
import json
import re
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

# google.cloud.storage est optionnel : utilisé seulement en mode GCS
try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover - mode local ou lib manquante
    storage = None

router = APIRouter(prefix="/monitoring")

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
    """Mappe une URI GCS velib/… vers un fichier local sous DATA_ROOT.

    Exemple :
        gs://.../velib/monitoring/intro/latest/intro.json
        → DATA_ROOT / "monitoring/intro/latest/intro.json"
    """
    if not uri.startswith(GCS_LOCAL_ROOT_PREFIX):
        raise ValueError(
            f"Cannot map GCS URI to local path:\n  uri={uri}\n  prefix={GCS_LOCAL_ROOT_PREFIX}"
        )
    rel = uri[len(GCS_LOCAL_ROOT_PREFIX) :]  # "monitoring/intro/latest/intro.json"
    path = DATA_ROOT / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _read_json_any(gs_uri: str) -> Dict[str, Any]:
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
    """
    Retourne le préfixe `.../monitoring` basé sur GCS_MONITORING_PREFIX.

    - lit GCS_MONITORING_PREFIX,
    - strip espaces + guillemets + slash final,
    - vérifie `gs://...`,
    - ajoute `/monitoring` si nécessaire.
    """
    raw = os.environ.get("GCS_MONITORING_PREFIX", "")
    mon = raw.strip().strip("'\"").rstrip("/")
    if not mon.startswith("gs://"):
        print(f"[monitoring.intro] GCS_MONITORING_PREFIX invalide. Lu='{raw}' nettoyé='{mon}' (backend={STORAGE_BACKEND})")
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX manquant ou invalide")
    if not mon.endswith("/monitoring"):
        mon = mon + "/monitoring"
    return mon


# Timestamp "version" attendu pour le time-travel: 2025-10-23T14-30-00Z
_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$")


def _sanitize_at(at: Optional[str]) -> str:
    """Normalise le paramètre `at` (time-travel) en slug de dossier."""
    if not at or at.strip().lower() in ("", "latest"):
        return "latest"
    s = at.strip()
    return s if _AT_RE.match(s) else "latest"


def _json_sanitize(obj):
    """Remplace NaN/±Inf par None pour garantir un JSON valide."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _proxy_json(gs_uri: str, request: Request, ttl: int = 60) -> JSONResponse:
    """Proxy générique JSON (local ou GCS) → HTTP JSON avec headers de cache."""
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


# ───────────────────────── Endpoints ─────────────────────────

@router.get("/intro/available")
def intro_available():
    """Expose la liste des documents *intro* disponibles et le mode time-travel."""
    return {
        "docs": ["intro"],
        "time_travel": "utiliser ?at=YYYY-MM-DDTHH-MM-SSZ ou sans param pour latest",
    }


@router.get("/intro")
def intro_doc(request: Request, at: Optional[str] = None):
    """
    Sert le document consolidé produit par `build_monitoring_intro.py`.

    Chemin logique :
        gs://.../monitoring/intro/<latest|YYYY-MM-DDTHH-MM-SSZ>/intro.json

    En mode local :
        data/monitoring/intro/<folder>/intro.json
    """
    base = _mon_prefix_or_500()
    folder = _sanitize_at(at)
    gs_uri = f"{base}/intro/{folder}/intro.json"
    print(f"[monitoring.intro] reading {gs_uri} (backend={STORAGE_BACKEND})")
    return _proxy_json(gs_uri, request, ttl=60)
