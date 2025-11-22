# apps/api/routes/monitoring/data_health.py

"""Monitoring — Data Health endpoints.

Ce module expose les endpoints `/monitoring/data/health/...` qui servent les
artefacts JSON produits par le job `build_data_health.py` sur GCS.

Arborescence cible (LATEST only)
--------------------------------
<GCS_MONITORING_PREFIX>/monitoring/data/health/latest/
    kpis.json
    station_health.json
    coverage_by_hour.json
    alerts.json
    anomalies/
        manifest.json
        flat.json
        duplicates.json
        missing.json

Principes :
- le backend **ne calcule rien** : il ne fait que proxifier des JSON
  pré-calculés et uploadés par les jobs batch,
- `GCS_MONITORING_PREFIX` doit être un `gs://...` valide (sinon 500),
- les floats non finis (NaN / ±Inf) sont remplacés par `null` pour produire
  un JSON valide côté client,
- certains endpoints (ex. `/anomalies`) fusionnent plusieurs shards pour
  assurer une compat ascendante avec d’anciens schémas.
"""

from __future__ import annotations
import os, json, math
from typing import Literal, Tuple, Any, Optional, List, Dict

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    # Ici GCS est requis : sans client, impossible de servir les artefacts.
    raise RuntimeError("google-cloud-storage requis côté API") from e


# ───────────────────────────── Router ─────────────────────────────
# Prefix "/monitoring/data", puis sous-chemin "health/..."
router = APIRouter(prefix="/monitoring/data")


# ───────────────────────────── GCS Helpers ─────────────────────────────

def _split_gs(uri: str) -> Tuple[str, str]:
    """Découpe une URI `gs://bucket/key` en `(bucket, key)`.

    Parameters
    ----------
    uri : str
        URI GCS commençant par `gs://`.

    Returns
    -------
    tuple[str, str]
        Nom du bucket et clé de l’objet.

    Raises
    ------
    ValueError
        Si l’URI ne commence pas par `gs://`.
    """
    if not uri.startswith("gs://"):
        raise ValueError(f"Bad GCS URI: {uri}")
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def _gcs_read_json(gs_uri: str) -> Any:
    """Lit un blob JSON GCS et retourne son contenu décodé.

    Parameters
    ----------
    gs_uri : str
        URI GCS du document JSON.

    Returns
    -------
    Any
        Dict ou list issu du JSON (selon la structure).
        Lève une exception si :
        - le blob n’existe pas,
        - le contenu n’est pas un JSON valide.
    """
    bkt, key = _split_gs(gs_uri)
    client = storage.Client()
    bucket = client.bucket(bkt)
    blob = bucket.blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"GCS blob not found: {gs_uri}")
    data = blob.download_as_bytes()
    return json.loads(data.decode("utf-8"))


def _mon_prefix_or_500() -> str:
    """Lit `GCS_MONITORING_PREFIX` et valide qu'il s'agit d'un `gs://...`.

    Nettoyage appliqué :
    - strip des espaces,
    - strip des guillemets simples / doubles,
    - suppression du slash final.

    Returns
    -------
    str
        Préfixe GCS normalisé.

    Raises
    ------
    HTTPException(500)
        Si la variable est manquante ou ne commence pas par `gs://`.
    """
    raw = os.environ.get("GCS_MONITORING_PREFIX", "")
    mon = raw.strip().strip("'\"").rstrip("/")
    if not mon.startswith("gs://"):
        print(f"[data.health.router] GCS_MONITORING_PREFIX invalide: '{raw}' → '{mon}'")
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX manquant ou invalide")
    return mon


def _mon_base_or_500() -> str:
    """Retourne le préfixe `.../monitoring` en s'appuyant sur GCS_MONITORING_PREFIX.

    Si `GCS_MONITORING_PREFIX` se termine déjà par `/monitoring`, on le renvoie
    tel quel, sinon on ajoute le suffixe `/monitoring`.

    Returns
    -------
    str
        Préfixe GCS du répertoire `monitoring`.
    """
    mon = _mon_prefix_or_500()
    return f"{mon}/monitoring" if not mon.endswith("/monitoring") else mon


def _sanitize(obj):
    """Remplace les floats non finis (NaN / ±Inf) par `None` pour le JSON.

    Fonction récursive appliquée à tout l’objet :

    - dict  → application sur toutes les valeurs,
    - list  → application sur tous les éléments,
    - float → si non fini (NaN / ±Inf) → None,
    - autres types → renvoyés tels quels.
    """
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _proxy_json(gs_uri: str, request: Request, ttl: int = 120) -> JSONResponse:
    """Lit un JSON sur GCS et le proxifie avec des headers de cache.

    Comportement :
    - si le blob n’existe pas → statut 204 + body `{}` (pas d’erreur),
    - si une autre erreur survient → 502 (erreur de lecture / parsing),
    - sinon → JSONResponse avec contenu "sanitisé" et `Cache-Control`.

    Parameters
    ----------
    gs_uri : str
        URI GCS du document JSON.
    request : Request
        Requête FastAPI (actuellement non utilisée, gardée pour extension).
    ttl : int, default 120
        Durée de vie du cache HTTP (max-age, en secondes).

    Returns
    -------
    fastapi.responses.JSONResponse
        Réponse HTTP contenant le JSON nettoyé.
    """
    try:
        payload = _gcs_read_json(gs_uri)
    except FileNotFoundError:
        resp = JSONResponse({}, status_code=204)
        resp.headers["Cache-Control"] = f"public, max-age={ttl}"
        resp.headers.setdefault("Access-Control-Allow-Origin", "*")
        return resp
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur lecture GCS: {e}")

    safe = _sanitize(payload)
    resp = JSONResponse(safe)
    resp.headers["Cache-Control"] = f"public, max-age={ttl}"
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    return resp


def _proxy_file(rel: str, request: Request, ttl: int) -> JSONResponse:
    """Proxy utilitaire sur un chemin relatif depuis `data/health/latest/`.

    Parameters
    ----------
    rel : str
        Chemin relatif à partir de `data/health/latest/`, par ex.
        `"kpis.json"` ou `"anomalies/flat.json"`.
    request : Request
        Requête FastAPI.
    ttl : int
        TTL (max-age) utilisé dans les headers `Cache-Control`.

    Returns
    -------
    fastapi.responses.JSONResponse
        Réponse produite par `_proxy_json`.
    """
    base = _mon_base_or_500()
    gs_uri = f"{base}/data/health/latest/{rel.lstrip('/')}"
    return _proxy_json(gs_uri, request, ttl=ttl)


# ───────────────────────────── ROUTES ─────────────────────────────

# Ensemble des documents supportés par ce router.
DocName = Literal[
    "kpis",
    "station_health",
    "coverage_by_hour",
    "anomalies",            # fusion des shards (compat ascendante)
    "alerts",
    "anomalies_manifest",   # nouveau: retourne manifest.json
    "anomalies_flat",       # nouveau: flat.json
    "anomalies_duplicates", # nouveau: duplicates.json
    "anomalies_missing",    # nouveau: missing.json
]

# TTL spécifiques par document (secondes)
_TTL: Dict[str, int] = {
    "kpis": 60,
    "station_health": 300,
    "coverage_by_hour": 300,
    "anomalies": 120,
    "alerts": 60,
    "anomalies_manifest": 120,
    "anomalies_flat": 120,
    "anomalies_duplicates": 120,
    "anomalies_missing": 120,
}


@router.get("/health/available")
def available_docs():
    """Petit index des documents `data health` disponibles (latest only).

    Response
    --------
    {
      "docs": [...],         # noms logiques des documents
      "time_travel": "non — latest-only"
    }
    """
    return {
        "docs": list(_TTL.keys()),
        "time_travel": "non — latest-only",
    }


@router.get("/health/anomalies_manifest")
def anomalies_manifest(request: Request):
    """Proxy direct de `anomalies/manifest.json` (latest)."""
    return _proxy_file("anomalies/manifest.json", request, _TTL["anomalies_manifest"])


@router.get("/health/anomalies_flat")
def anomalies_flat(request: Request):
    """Proxy direct de `anomalies/flat.json` (latest)."""
    return _proxy_file("anomalies/flat.json", request, _TTL["anomalies_flat"])


@router.get("/health/anomalies_duplicates")
def anomalies_duplicates(request: Request):
    """Proxy direct de `anomalies/duplicates.json` (latest)."""
    return _proxy_file("anomalies/duplicates.json", request, _TTL["anomalies_duplicates"])


@router.get("/health/anomalies_missing")
def anomalies_missing(request: Request):
    """Proxy direct de `anomalies/missing.json` (latest)."""
    return _proxy_file("anomalies/missing.json", request, _TTL["anomalies_missing"])


@router.get("/health/{doc}")
def get_doc(doc: DocName, request: Request):
    """
    Proxy JSON pour les artefacts *data health*.

    Endpoints :
      /monitoring/data/health/kpis
      /monitoring/data/health/station_health
      /monitoring/data/health/coverage_by_hour
      /monitoring/data/health/alerts
      /monitoring/data/health/anomalies                → fusionne flat+duplicates+missing (compat)
      /monitoring/data/health/anomalies_manifest       → manifest.json
      /monitoring/data/health/anomalies_flat           → flat.json
      /monitoring/data/health/anomalies_duplicates     → duplicates.json
      /monitoring/data/health/anomalies_missing        → missing.json

    Notes
    -----
    - L’alias `/anomalies` est conservé pour compat ascendante et renvoie une
      liste fusionnée de tous les shards (flat + duplicates + missing).
    - Pour plus de granularité, les nouveaux endpoints permettent d’accéder
      directement à chaque shard individuel.
    """

    # Compat ascendante : fusionner les shards si /anomalies est demandé
    if doc == "anomalies":
        base = _mon_base_or_500()
        parts: List[Any] = []
        for name in ("flat", "duplicates", "missing"):
            try:
                parts.append(_gcs_read_json(f"{base}/data/health/latest/anomalies/{name}.json"))
            except FileNotFoundError:
                continue

        # On concatène toutes les listes rencontrées en une seule.
        merged: List[Any] = []
        for p in parts:
            if isinstance(p, list):
                merged.extend(p)

        resp = JSONResponse(_sanitize(merged))
        resp.headers["Cache-Control"] = f"public, max-age={_TTL['anomalies']}"
        resp.headers.setdefault("Access-Control-Allow-Origin", "*")
        return resp

    # Cas standard (kpis, station_health, coverage_by_hour, alerts)
    if doc in ("kpis", "station_health", "coverage_by_hour", "alerts"):
        return _proxy_file(f"{doc}.json", request, _TTL.get(doc, 120))

    # Nouveaux docs (si atteints via {doc})
    if doc == "anomalies_manifest":
        return anomalies_manifest(request)
    if doc == "anomalies_flat":
        return anomalies_flat(request)
    if doc == "anomalies_duplicates":
        return anomalies_duplicates(request)
    if doc == "anomalies_missing":
        return anomalies_missing(request)

    # Défense ultime : DocName devrait empêcher d’arriver ici en pratique.
    raise HTTPException(status_code=404, detail=f"Unknown doc '{doc}'")