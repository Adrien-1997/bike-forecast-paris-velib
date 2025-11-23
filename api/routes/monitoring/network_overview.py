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

Principes :
- Le backend **ne calcule rien** ici : il ne fait que proxifier les JSON déjà
  préparés par les jobs batch.
- Tous les chemins GCS sont dérivés de `GCS_MONITORING_PREFIX` (env),
  normalisé via `_mon_prefix_or_500()`.
- Le paramètre `at` permet un time-travel simple :
    * `?at=latest` ou omis → snapshot courant,
    * `?at=YYYY-MM-DDTHH-MM-SSZ` → snapshot daté,
    * toute valeur invalide → repli sur `latest` (et évite le path traversal).
- Les nombres flottants non finis (NaN / ±Inf) sont convertis en `null`
  dans la réponse via `_json_sanitize`, pour garantir un JSON valide.

Ces endpoints sont utilisés par la page "Network Overview" du monitoring
(courbes d’occupation, KPIs globaux, distribution des snapshots, carte, etc.).
"""

from __future__ import annotations
import os
import json
import re
from typing import Optional, Literal, Tuple
import math

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    # Pour les routes de monitoring, GCS est requis côté API :
    # sans client storage, impossible de servir les artefacts.
    raise RuntimeError("google-cloud-storage requis côté API") from e


router = APIRouter(prefix="/monitoring/network")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers GCS
# ──────────────────────────────────────────────────────────────────────────────

def _split_gs(uri: str) -> Tuple[str, str]:
    """Découpe une URI `gs://bucket/key` en `(bucket, key)`.

    Parameters
    ----------
    uri : str
        URI GCS devant commencer par `gs://`.

    Returns
    -------
    tuple[str, str]
        `(bucket, key)`.

    Raises
    ------
    ValueError
        Si l’URI ne commence pas par `gs://`.
    """
    if not uri.startswith("gs://"):
        raise ValueError(f"Bad GCS URI: {uri}")
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def _gcs_read_json(gs_uri: str) -> dict:
    """Télécharge un blob JSON depuis GCS et le parse en dictionnaire Python.

    Parameters
    ----------
    gs_uri : str
        URI GCS du document JSON à lire.

    Returns
    -------
    dict
        Contenu JSON décodé.

    Raises
    ------
    FileNotFoundError
        Si le blob n'existe pas sur GCS.
    ValueError
        Si le contenu n'est pas un JSON valide.
    """
    bkt, key = _split_gs(gs_uri)
    client = storage.Client()
    bucket = client.bucket(bkt)
    blob = bucket.blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"GCS blob not found: {gs_uri}")
    data = blob.download_as_bytes()
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Invalid JSON in {gs_uri}: {e}") from e


def _mon_prefix_or_500() -> str:
    """Retourne `GCS_MONITORING_PREFIX` normalisé ou lève une 500.

    Étapes :
    - lit la variable d'environnement `GCS_MONITORING_PREFIX`,
    - strip espaces + guillemets superflus,
    - retire le slash final,
    - vérifie que le résultat commence par `gs://`.

    Returns
    -------
    str
        Préfixe GCS normalisé.

    Raises
    ------
    HTTPException(500)
        Si la variable est manquante ou invalide.
    """
    mon_raw = os.environ.get("GCS_MONITORING_PREFIX", "")
    mon = mon_raw.strip().strip("'\"").rstrip("/")
    if not mon.startswith("gs://"):
        print(f"[network_overview] GCS_MONITORING_PREFIX invalide. Lu='{mon_raw}' nettoyé='{mon}'")
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX manquant ou invalide")
    print(f"[network_overview] GCS_MONITORING_PREFIX='{mon}'")
    return mon


# Timestamp "version" attendu pour le time-travel: 2025-10-23T14-30-00Z
_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$")

def _sanitize_at(at: Optional[str]) -> str:
    """Normalise le paramètre `at` (time-travel) en nom de dossier.

    Règles :
    - `None`, vide ou "latest" (insensible à la casse) → `"latest"`,
    - chaîne qui matche `_AT_RE` → renvoyée telle quelle,
    - tout le reste → `"latest"` (évite les valeurs exotiques / path traversal).
    """
    if not at or at.strip().lower() in ("", "latest"):
        return "latest"
    s = at.strip()
    if not _AT_RE.match(s):
        # force 'latest' si format non reconnu (évite path traversal)
        return "latest"
    return s


def _json_sanitize(obj):
    """Remplace NaN/±Inf par None pour produire un JSON valide.

    Fonction récursive appliquée à toute la structure :
    - dict  → application sur les valeurs,
    - list  → application sur les éléments,
    - float → si non fini (NaN / ±Inf) → None,
    - autres types → renvoyés tels quels.
    """
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _proxy_json(gs_uri: str, request: Request, ttl: int = 120) -> JSONResponse:
    """Lit un JSON GCS et le proxifie avec des headers de cache adaptés.

    Parameters
    ----------
    gs_uri : str
        URI GCS du document à servir.
    request : Request
        Requête FastAPI (actuellement non utilisée, gardée pour extension).
    ttl : int, default 120
        Durée de vie du cache HTTP (`max-age`) en secondes.

    Returns
    -------
    fastapi.responses.JSONResponse
        Réponse contenant le JSON "sanitisé" (`_json_sanitize`).

    Raises
    ------
    HTTPException
        - 404 si le document est introuvable,
        - 502 en cas d'erreur de lecture / parsing.
    """
    try:
        payload = _gcs_read_json(gs_uri)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document non trouvé")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur lecture GCS: {e}")

    # Sanitize (évite ValueError: Out of range float values are not JSON compliant)
    safe = _json_sanitize(payload)

    resp = JSONResponse(safe)
    # Cache côté client / CDN
    resp.headers["Cache-Control"] = f"public, max-age={max(0, int(ttl))}"
    # CORS permissif si derrière un proxy (optionnel)
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

# TTL (secondes) par type de document, ajusté selon la volatilité attendue
_TTL_BY_DOC = {
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
    """Petit index des documents *network overview* disponibles.

    Response
    -------
    {
      "docs": [...],
      "time_travel": "utiliser ?at=YYYY-MM-DDTHH-MM-SSZ ou sans param pour latest"
    }

    Utile pour introspection rapide (via /docs) et pour le frontend,
    afin de découvrir dynamiquement les documents servis par ce router.
    """
    docs = [
        "kpis",
        "snapshot_distribution",
        "today_curve",
        "ref_median_curve",
        "kpis_today_vs_lags",
        "snapshot_map",
        "stations_tension",
    ]
    return {
        "docs": docs,
        "time_travel": "utiliser ?at=YYYY-MM-DDTHH-MM-SSZ ou sans param pour latest"
    }


@router.get("/overview/{doc}")
def network_overview_doc(doc: DocNameOverview, request: Request, at: Optional[str] = None):
    """
    Proxy JSON pour les documents générés par le job Overview.

    Exemples
    --------
    - /monitoring/network/overview/kpis
    - /monitoring/network/overview/today_curve?at=2025-10-13T09-12-00Z

    Paramètres
    ----------
    doc : DocNameOverview, path
        Nom logique du document à servir (voir `DocNameOverview`).
    at : str | None, query
        Slug de time-travel :
        - None / latest (ou invalide) → snapshot courant,
        - 'YYYY-MM-DDTHH-MM-SSZ' → version datée.

    Returns
    -------
    fastapi.responses.JSONResponse
        Contenu JSON "sanitisé" du document demandé.
    """
    mon = _mon_prefix_or_500()  # ex: gs://velib-forecast-472820_cloudbuild/velib
    folder = _sanitize_at(at)   # 'latest' ou timestamp normalisé

    # si l'ENV finit déjà par /monitoring, on ne le redouble pas
    base = f"{mon}/monitoring" if not mon.endswith("/monitoring") else mon
    gs_uri = f"{base}/network/overview/{folder}/{doc}.json"

    ttl = _TTL_BY_DOC.get(doc, 120)
    return _proxy_json(gs_uri, request, ttl=ttl)
