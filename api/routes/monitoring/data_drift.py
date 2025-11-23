# api/routes/monitoring_data_drift.py

"""Monitoring — Data Drift endpoints.

Ce module expose les endpoints `/monitoring/data/drift` qui servent les artefacts
JSON produits par le job `build_data_drift.py` sur GCS :

Arborescence cible (LATEST only)
--------------------------------
<MONITORING_BASE>/monitoring/data/drift/latest/
    summary.json
    psi_by_feature.json
    ks_by_feature.json
    deltas_by_feature.json
    psi_global_daily_ema.json
    alerts.json
    bounds.json
    zones.json
    features_detected.json

Principes :
- Le backend **ne recalcule rien** : il ne fait que proxy des JSON déjà
  construits et uploadés par les jobs batch.
- Les URIs GCS sont dérivées de `GCS_MONITORING_PREFIX` (env ou settings),
  normalisée via `_mon_prefix_or_500()`.
- Tous les JSON sont "sanitisés" pour le transport :
  - les `NaN` / `+/-Inf` sont convertis en `null` via `_json_sanitize`,
  - l'API retourne toujours du JSON valide côté client.
- Les headers HTTP suivants sont posés :
  - `Cache-Control: public, max-age=120` (par défaut),
  - `Access-Control-Allow-Origin: *` (CORS permissif pour les widgets UI).
"""

from __future__ import annotations
import os
import json
import math
from typing import Tuple

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    # Contrairement à d'autres modules, ici GCS est requis :
    # sans client storage, on ne peut rien servir → erreur explicite au démarrage.
    raise RuntimeError("google-cloud-storage requis côté API") from e


router = APIRouter(prefix="/monitoring/data")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers GCS — calqués sur network_overview.py
# ──────────────────────────────────────────────────────────────────────────────

def _split_gs(uri: str) -> Tuple[str, str]:
    """Découpe une URI `gs://bucket/key` en `(bucket, key)`.

    Parameters
    ----------
    uri : str
        URI GCS commençant par `gs://`.

    Returns
    -------
    tuple[str, str]
        Nom du bucket et chemin de l'objet.

    Raises
    ------
    ValueError
        Si l'URI ne commence pas par `gs://`.
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
        URI GCS du document JSON.

    Returns
    -------
    dict
        Contenu JSON parsé.

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
    """Retourne le préfixe GCS_MONITORING_PREFIX normalisé ou lève une 500.

    Lit la variable d'environnement `GCS_MONITORING_PREFIX`, applique :
    - strip des espaces,
    - strip des guillemets simples / doubles,
    - suppression du `/` final.

    Le résultat doit commencer par `gs://`, sinon une `HTTPException(500)`
    est levée (la stack de monitoring n'est pas correctement configurée).
    """
    mon_raw = os.environ.get("GCS_MONITORING_PREFIX", "")
    mon = mon_raw.strip().strip("'\"").rstrip("/")
    if not mon.startswith("gs://"):
        print(f"[data_drift] GCS_MONITORING_PREFIX invalide. Lu='{mon_raw}' nettoyé='{mon}'")
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX manquant ou invalide")
    print(f"[data_drift] GCS_MONITORING_PREFIX='{mon}'")
    return mon


def _json_sanitize(obj):
    """Remplace NaN/±Inf par None pour produire un JSON valide.

    Fonction récursive appliquée avant renvoi au client pour éviter
    les JSON non standards (NaN, Inf...).

    Règles :
    - dict → on applique la fonction à toutes les valeurs,
    - list → on applique la fonction à tous les éléments,
    - float → si non fini (NaN / ±Inf) → None,
    - le reste est renvoyé inchangé.
    """
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _proxy_json(gs_uri: str, request: Request, ttl: int = 120) -> JSONResponse:
    """Proxy générique GCS JSON → HTTP JSON avec en-têtes de cache.

    Parameters
    ----------
    gs_uri : str
        URI GCS du fichier JSON à servir.
    request : Request
        Requête FastAPI (actuellement non utilisée, gardée pour extension).
    ttl : int, default 120
        Durée de vie du cache HTTP (secondes), utilisée dans `Cache-Control`.

    Returns
    -------
    fastapi.responses.JSONResponse
        Réponse HTTP contenant le JSON nettoyé (`_json_sanitize`).

    Raises
    ------
    HTTPException
        - 404 si le document est introuvable,
        - 502 en cas d'erreur GCS / parsing JSON.
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


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES /monitoring/data/drift
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/drift")
def get_data_drift(request: Request):
    """Renvoie le `summary.json` du dossier `latest` de data drift.

    GCS cible :
    ----------
    <GCS_MONITORING_PREFIX>/monitoring/data/drift/latest/summary.json

    Utilisation côté UI :
    ---------------------
    - bloc "state global" du data drift,
    - PSI global, statut, éventuels indicateurs de stabilité,
    - sert de point d'entrée pour décider d'afficher des alertes (ou non).
    """
    prefix = _mon_prefix_or_500()
    base = f"{prefix}/monitoring" if not prefix.endswith("/monitoring") else prefix
    uri = f"{base}/data/drift/latest/summary.json"
    print(f"[data_drift] reading {uri}")
    return _proxy_json(uri, request)


@router.get("/drift/{name}")
def get_data_drift_file(name: str, request: Request):
    """Accès direct à un JSON spécifique de data drift (psi, ks, deltas…).

    Parameters
    ----------
    name : str
        Nom logique du fichier, sans extension. Doit appartenir à l'ensemble :
        {
          "psi_by_feature", "ks_by_feature", "deltas_by_feature",
          "psi_global_daily_ema", "summary", "alerts", "bounds", "zones",
          "features_detected",
        }
    request : Request
        Requête FastAPI.

    Returns
    -------
    fastapi.responses.JSONResponse
        Contenu JSON du fichier, via `_proxy_json`.

    Raises
    ------
    HTTPException
        - 404 si `name` n'est pas dans la liste des fichiers connus.
    """
    valid = {
        "psi_by_feature", "ks_by_feature", "deltas_by_feature",
        "psi_global_daily_ema", "summary", "alerts", "bounds", "zones",
        "features_detected",
    }
    if name not in valid:
        raise HTTPException(status_code=404, detail=f"Fichier inconnu '{name}'")

    prefix = _mon_prefix_or_500()
    base = f"{prefix}/monitoring" if not prefix.endswith("/monitoring") else prefix
    uri = f"{base}/data/drift/latest/{name}.json"
    print(f"[data_drift] reading {uri}")
    return _proxy_json(uri, request)


@router.get("/drift/available")
def list_available():
    """Liste statique des fichiers disponibles sous `data/drift/latest`.

    Utile pour introspection ou debug rapide depuis le navigateur / API docs.
    """
    return {
        "path": "monitoring/data/drift/latest/",
        "files": [
            "psi_by_feature.json",
            "ks_by_feature.json",
            "deltas_by_feature.json",
            "psi_global_daily_ema.json",
            "summary.json",
            "alerts.json",
            "bounds.json",
            "zones.json",
            "features_detected.json",
        ],
    }
