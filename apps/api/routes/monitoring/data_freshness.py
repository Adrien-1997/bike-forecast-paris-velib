# apps/api/routes/monitoring/data_freshness.py

"""Monitoring — Data Freshness endpoint.

Ce module expose l’endpoint `/monitoring/data/freshness` qui sert le JSON
produit par le job `build_data_health.py` / pipeline de fraîcheur sur GCS :

Arborescence cible (LATEST only)
--------------------------------
<MONITORING_BASE>/monitoring/data/freshness/latest.json

Contenu typique :
- P95 / P50 de fraîcheur (en minutes),
- indicateurs globaux (OK / WARN / ERROR),
- éventuellement des infos météo associées.

Principes :
- Le backend ne calcule rien : il ne fait que proxy un JSON déjà construit
  et uploadé par les jobs batch.
- L’URI est dérivée de `GCS_MONITORING_PREFIX` (env), normalisée via
  `_mon_prefix_or_500()`.
- Les JSON sont "sanitisés" avant renvoi :
  - `NaN` / `±Inf` → `null` via `_json_sanitize`.
- Les headers HTTP sont posés pour l’UI :
  - `Cache-Control: public, max-age=30`,
  - `Access-Control-Allow-Origin: *`.
"""

from __future__ import annotations
import os, json, math
from typing import Tuple

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    # Ici, GCS est requis : sans client storage on ne peut pas servir les JSON.
    raise RuntimeError("google-cloud-storage requis côté API") from e


# Préfixe global de ce module
router = APIRouter(prefix="/monitoring/data")


# ──────────────────────────────────────────────────────────────
# Helpers GCS — identiques à ceux des autres routes monitoring
# ──────────────────────────────────────────────────────────────

def _split_gs(uri: str) -> Tuple[str, str]:
    """Découpe une URI `gs://bucket/key` en `(bucket, key)`.

    Parameters
    ----------
    uri : str
        URI GCS commençant par `gs://`.

    Returns
    -------
    tuple[str, str]
        Nom du bucket et clé d’objet.

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
    """Télécharge un blob JSON sur GCS et le parse en dict Python.

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
        Si le blob n’existe pas.
    ValueError
        Si le contenu n’est pas un JSON valide.
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
    - lit `GCS_MONITORING_PREFIX` dans les variables d’environnement,
    - supprime espaces et guillemets superflus,
    - retire le `/` final,
    - vérifie que le résultat commence par `gs://`.

    Raises
    ------
    HTTPException(500)
        Si le préfixe est manquant ou invalide.
    """
    mon_raw = os.environ.get("GCS_MONITORING_PREFIX", "")
    mon = mon_raw.strip().strip("'\"").rstrip("/")
    if not mon.startswith("gs://"):
        print(f"[data_freshness] GCS_MONITORING_PREFIX invalide. Lu='{mon_raw}' nettoyé='{mon}'")
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX manquant ou invalide")
    print(f"[data_freshness] GCS_MONITORING_PREFIX='{mon}'")
    return mon


def _json_sanitize(obj):
    """Remplace NaN/±Inf par None pour produire un JSON valide.

    Fonction récursive appliquée sur l’objet avant renvoi au client :

    - dict  → application sur toutes les valeurs,
    - list  → application sur tous les éléments,
    - float → si non fini (NaN / ±Inf) → None,
    - autres types → inchangés.
    """
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _proxy_json(gs_uri: str, request: Request, ttl: int = 60) -> JSONResponse:
    """Lit un JSON sur GCS et le renvoie avec en-têtes HTTP adaptés.

    Parameters
    ----------
    gs_uri : str
        URI GCS du document à servir.
    request : Request
        Requête FastAPI (non utilisée pour l’instant, gardée pour extension).
    ttl : int, default 60
        Durée de vie du cache HTTP en secondes (max-age).

    Returns
    -------
    fastapi.responses.JSONResponse
        Réponse contenant le JSON "sanitisé" (`_json_sanitize`).

    Raises
    ------
    HTTPException
        - 404 si le document est introuvable,
        - 502 en cas d’erreur GCS / parsing JSON.
    """
    try:
        payload = _gcs_read_json(gs_uri)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document non trouvé")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur lecture GCS: {e}")

    safe = _json_sanitize(payload)
    resp = JSONResponse(safe)
    resp.headers["Cache-Control"] = f"public, max-age={max(0, int(ttl))}"
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    return resp


# ──────────────────────────────────────────────────────────────
# ROUTE UNIQUE /monitoring/data/freshness
# ──────────────────────────────────────────────────────────────

@router.get("/freshness")
def get_data_freshness_latest(request: Request):
    """Renvoie le `latest.json` de fraîcheur des données.

    GCS cible :
    ----------
    <GCS_MONITORING_PREFIX>/monitoring/data/freshness/latest.json

    Utilisation côté UI :
    ---------------------
    - cartes / badges de fraîcheur globale (P95, P50, statut),
    - éventuelles métriques météo alignées,
    - bloc "Data Freshness" dans la page Monitoring / Data Health.

    Notes
    -----
    - Le TTL HTTP est volontairement court (30 s) pour refléter au mieux
      la mise à jour des jobs de monitoring.
    """
    prefix = _mon_prefix_or_500()
    base = f"{prefix}/monitoring" if not prefix.endswith("/monitoring") else prefix
    uri = f"{base}/data/freshness/latest.json"
    print(f"[data_freshness] reading {uri}")
    return _proxy_json(uri, request, ttl=30)
