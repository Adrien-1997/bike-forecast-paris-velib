# api/routes/model_performance.py

"""Monitoring — Model Performance endpoints.

Ce module expose les endpoints :

    /monitoring/model/performance/...

Ils servent les artefacts JSON produits par le job
`build_model_performance.py` sur GCS, typiquement organisés comme :

    <GCS_MONITORING_PREFIX>/monitoring/model/performance/<latest|TS>/h{H}/
        kpis.json
        daily_metrics.json
        by_hour.json
        by_dow.json
        by_station.json
        by_cluster.json           (optionnel, si clusters fournis)
        lift_curve.json
        hist_residuals.json
        station_timeseries.json   (nouveau)

Principes :
- Le backend **ne calcule rien** : il ne fait que proxifier des JSON déjà
  préparés par les jobs batch.
- Tous les chemins GCS sont dérivés de `GCS_MONITORING_PREFIX` (env),
  normalisé via `_mon_prefix_or_500()`.
- Le paramètre `h` représente l’horizon en minutes (ex: 15, 60).
- Le paramètre `at` est prévu pour le time-travel, mais aujourd’hui
  on reste en `latest-only` (valeurs invalides → `latest`).
- Les nombres flottants non finis (NaN / ±Inf) sont convertis en `null`
  dans la réponse via `_json_sanitize`, pour garantir un JSON valide.
"""

from __future__ import annotations
import os, json, re, math
from typing import Optional, Literal, Tuple

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import JSONResponse

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    # Pour les routes de monitoring, GCS est requis côté API : sans client
    # storage, impossible de servir les artefacts → erreur explicite au boot.
    raise RuntimeError("google-cloud-storage requis côté API") from e

# ── Préfixe aligné sur le pattern des pages network/* ─────────────────────────
# /monitoring/model/performance/*
router = APIRouter(prefix="/monitoring/model/performance")


# ── Helpers GCS ───────────────────────────────────────────────────────────────
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
    """Télécharge un blob JSON depuis GCS et le parse en dict Python.

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
    blob = client.bucket(bkt).blob(key)
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
        print(f"[model_performance] GCS_MONITORING_PREFIX invalide. Lu='{mon_raw}' nettoyé='{mon}'")
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX manquant ou invalide")
    return mon


# Timestamp "version" attendu si on activait le time-travel : 2025-10-23T14-30-00Z
_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$")


def _sanitize_at(at: Optional[str]) -> str:
    """Normalise le paramètre `at` (time-travel) en slug de dossier.

    Comportement actuel (latest-only) :
    - `None`, vide ou "latest" (insensible à la casse) → `"latest"`,
    - chaîne non vide respectant le pattern `_AT_RE` → renvoyée telle quelle,
    - toute autre valeur → `"latest"` (fallback sécurisé).
    """
    # latest-only pour l’instant ; on garde 'at' future-proof
    if not at or at.strip().lower() in ("", "latest"):
        return "latest"
    s = at.strip()
    if not _AT_RE.match(s):
        return "latest"
    return s


def _json_sanitize(obj):
    """Remplace NaN/±Inf par None pour garantir un JSON valide.

    Fonction récursive appliquée sur toute la structure :
    - dict  → application sur toutes les valeurs,
    - list  → application sur tous les éléments,
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
    """Lit un JSON sur GCS et le proxifie avec des headers HTTP adaptés.

    Parameters
    ----------
    gs_uri : str
        URI GCS du document à servir.
    request : Request
        Requête FastAPI (non utilisée aujourd'hui, gardée pour extension).
    ttl : int, default 120
        Durée de vie du cache HTTP (`max-age`), en secondes.

    Returns
    -------
    fastapi.responses.JSONResponse
        Réponse contenant le JSON "sanitisé" (`_json_sanitize`).

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


# ── Endpoints PERF (latest only) ──────────────────────────────────────────────
DocNamePerf = Literal[
    "kpis",
    "daily_metrics",
    "by_hour",
    "by_dow",
    "by_station",
    "by_cluster",       # présent si tu fournis le CSV de clusters
    "lift_curve",
    "hist_residuals",
    "station_timeseries",  # ⬅️ NEW
]

# TTL par type de document (secondes)
_TTL_BY_DOC = {
    "kpis": 60,
    "daily_metrics": 120,
    "by_hour": 300,
    "by_dow": 300,
    "by_station": 300,
    "by_cluster": 300,
    "lift_curve": 180,
    "hist_residuals": 600,
    "station_timeseries": 300,  # ⬅️ NEW
}


@router.get("/available")
def model_perf_available():
    """Expose la liste des documents de performance modèle disponibles.

    Response
    --------
    {
      "docs": [...],
      "horizons": "utiliser ?h=<minutes> (ex: 15, 60)",
      "time_travel": "latest uniquement (param ?at=… ignoré si non valide)",
      "examples": [...]
    }

    Notes
    -----
    - Purement informatif, utile pour l’UI ou pour introspection via /docs.
    """
    return {
        "docs": [
            "kpis",
            "daily_metrics",
            "by_hour",
            "by_dow",
            "by_station",
            "by_cluster",
            "lift_curve",
            "hist_residuals",
            "station_timeseries",  # ⬅️ NEW
        ],
        "horizons": "utiliser ?h=<minutes> (ex: 15, 60)",
        "time_travel": "latest uniquement (param ?at=… ignoré si non valide)",
        "examples": [
            "/monitoring/model/performance/kpis?h=15",
            "/monitoring/model/performance/by_hour?h=60",
            "/monitoring/model/performance/station_timeseries?h=15",  # ⬅️ NEW
            "/monitoring/model/performance/manifest",
        ],
    }


@router.get("/manifest")
def model_perf_manifest(request: Request, at: Optional[str] = None):
    """Expose le `manifest.json` global de la page Model Performance.

    Chemin GCS attendu :
    --------------------
    <GCS_MONITORING_PREFIX>/monitoring/model/performance/<latest|TS>/manifest.json

    Paramètres
    ----------
    at : str | None, query
        Time-travel slug (actuellement latest-only via `_sanitize_at`).

    Utilisation
    -----------
    - permet à l’UI de connaître rapidement :
        * les horizons disponibles,
        * les dates de fenêtre de performance,
        * les versions de modèles associées.
    """
    mon = _mon_prefix_or_500()
    folder = _sanitize_at(at)  # 'latest'
    base = f"{mon}/monitoring" if not mon.endswith("/monitoring") else mon
    gs_uri = f"{base}/model/performance/{folder}/manifest.json"
    return _proxy_json(gs_uri, request, ttl=60)


@router.get("/{doc}")
def model_perf_doc(
    doc: DocNamePerf,
    request: Request,
    h: int = Query(15, description="Horizon en minutes (ex: 15, 60)"),
    at: Optional[str] = None,
):
    """Proxy JSON pour un artefact de performance donné.

    Endpoints typiques
    ------------------
    - /monitoring/model/performance/kpis?h=15
    - /monitoring/model/performance/daily_metrics?h=15
    - /monitoring/model/performance/by_hour?h=60
    - /monitoring/model/performance/by_dow?h=15
    - /monitoring/model/performance/by_station?h=15
    - /monitoring/model/performance/by_cluster?h=15
    - /monitoring/model/performance/lift_curve?h=15
    - /monitoring/model/performance/hist_residuals?h=15
    - /monitoring/model/performance/station_timeseries?h=15

    Paramètres
    ----------
    doc : DocNamePerf, path
        Nom logique du document à servir (voir `DocNamePerf`).
    h : int, query
        Horizon en minutes (ex: 15, 60). Doit être > 0.
    at : str | None, query
        Time-travel slug (latest-only aujourd’hui via `_sanitize_at`).

    Returns
    -------
    fastapi.responses.JSONResponse
        Contenu JSON "sanitisé" du document demandé pour l’horizon `h`.
    """
    if h <= 0:
        raise HTTPException(status_code=400, detail="Paramètre h (minutes) doit être > 0")
    mon = _mon_prefix_or_500()
    folder = _sanitize_at(at)  # 'latest'
    base = f"{mon}/monitoring" if not mon.endswith("/monitoring") else mon
    gs_uri = f"{base}/model/performance/{folder}/h{int(h)}/{doc}.json"
    ttl = _TTL_BY_DOC.get(doc, 120)
    return _proxy_json(gs_uri, request, ttl=ttl)

