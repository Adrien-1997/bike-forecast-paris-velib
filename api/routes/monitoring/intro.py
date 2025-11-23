# api/routes/monitoring/intro.py

"""Monitoring — Intro (landing/overview) endpoint.

Ce module sert un document JSON *consolidé* produit par le job
`build_monitoring_intro.py`, utilisé pour alimenter :

- l’intro de la page Monitoring,
- et/ou des blocs de synthèse sur la landing (couverture, fraîcheur, drift…).

Caractéristiques :
- Arborescence GCS attendue (time-travel supporté) :

    <GCS_MONITORING_PREFIX>/monitoring/intro/<latest|YYYY-MM-DDTHH-MM-SSZ>/intro.json

- Paramètre de requête `at` :
    * `?at=latest` ou omis → utilise `latest`,
    * `?at=YYYY-MM-DDTHH-MM-SSZ` → time-travel sur un snapshot précis,
      validé par `_sanitize_at()`.

- Comme les autres routes de monitoring :
    * le backend ne calcule rien → simple proxy JSON,
    * GCS est obligatoire côté API,
    * les floats non finis (NaN / ±Inf) sont convertis en `null` avant renvoi.
"""

from __future__ import annotations
import os, json, re, math
from typing import Optional, Tuple

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    # Ici GCS est requis : sans storage client, l'API ne peut pas servir les artefacts.
    raise RuntimeError("google-cloud-storage requis côté API") from e


router = APIRouter(prefix="/monitoring")

# ───────────────────────── Helpers GCS ─────────────────────────

def _split_gs(uri: str) -> Tuple[str, str]:
    """Découpe une URI `gs://bucket/key` en `(bucket, key)`.

    Parameters
    ----------
    uri : str
        URI GCS commençant par `gs://`.

    Returns
    -------
    tuple[str, str]
        Nom du bucket et chemin de l’objet.

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
    """Télécharge et parse un JSON stocké sur GCS.

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
        Si le contenu ne peut pas être décodé en JSON valide.
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
    """
    Retourne le préfixe `GCS_MONITORING_PREFIX` normalisé vers `.../monitoring`.

    Comportement :
    --------------
    - Lit la variable d'environnement `GCS_MONITORING_PREFIX`,
    - applique :
        * strip espaces,
        * strip guillemets simples / doubles,
        * suppression du slash final,
    - vérifie que le résultat commence par `gs://`,
    - si le préfixe ne se termine pas par `/monitoring`, l'ajoute.

    Raises
    ------
    HTTPException(500)
        Si le préfixe est manquant ou ne commence pas par `gs://`.
    """
    raw = os.environ.get("GCS_MONITORING_PREFIX", "")
    mon = raw.strip().strip("'\"").rstrip("/")
    if not mon.startswith("gs://"):
        print(f"[monitoring.intro] GCS_MONITORING_PREFIX invalide. Lu='{raw}' nettoyé='{mon}'")
        raise HTTPException(status_code=500, detail="GCS_MONITORING_PREFIX manquant ou invalide")
    if not mon.endswith("/monitoring"):
        mon = mon + "/monitoring"
    return mon


# Timestamp "version" attendu pour le time-travel: 2025-10-23T14-30-00Z
_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$")


def _sanitize_at(at: Optional[str]) -> str:
    """Normalise le paramètre `at` (time-travel) en slug de dossier.

    Règles :
    - `None`, vide ou "latest" (insensible à la casse) → `"latest"`,
    - chaîne qui matche `_AT_RE` → renvoyée telle quelle,
    - tout le reste → `"latest"` (fallback sécurisé).
    """
    if not at or at.strip().lower() in ("", "latest"):
        return "latest"
    s = at.strip()
    return s if _AT_RE.match(s) else "latest"


def _json_sanitize(obj):
    """Remplace NaN/±Inf par None pour garantir un JSON valide.

    Fonction récursive appliquée à l'ensemble de la structure :
    - dict  → application sur les valeurs,
    - list  → application sur les éléments,
    - float → si non fini (NaN / ±Inf) → None,
    - autres types → renvoyés inchangés.
    """
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _proxy_json(gs_uri: str, request: Request, ttl: int = 60) -> JSONResponse:
    """Proxy générique GCS JSON → HTTP JSON avec headers de cache.

    Parameters
    ----------
    gs_uri : str
        URI GCS du document JSON à servir.
    request : Request
        Requête FastAPI (actuellement non utilisée, gardée pour extension).
    ttl : int, default 60
        Durée de vie du cache HTTP (`max-age`), en secondes.

    Returns
    -------
    fastapi.responses.JSONResponse
        Réponse contenant le JSON "sanitisé".

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


# ───────────────────────── Endpoints ─────────────────────────

@router.get("/intro/available")
def intro_available():
    """Expose la liste des documents *intro* disponibles et le mode time-travel.

    Aujourd'hui il n'y a qu'un document logique (`intro`), mais l'endpoint
    reste extensible si d'autres artefacts sont ajoutés dans ce dossier.

    Response example
    ----------------
    {
      "docs": ["intro"],
      "time_travel": "utiliser ?at=YYYY-MM-DDTHH-MM-SSZ ou sans param pour latest"
    }
    """
    return {
        "docs": ["intro"],  # un seul document: intro.json
        "time_travel": "utiliser ?at=YYYY-MM-DDTHH-MM-SSZ ou sans param pour latest",
    }


@router.get("/intro")
def intro_doc(request: Request, at: Optional[str] = None):
    """
    Sert le document consolidé produit par `build_monitoring_intro.py`.

    Chemin GCS attendu :
    --------------------
    gs://.../monitoring/intro/<latest|YYYY-MM-DDTHH-MM-SSZ>/intro.json

    Paramètres
    ----------
    at : str | None, query
        - None / "latest" / vide → `latest`,
        - `YYYY-MM-DDTHH-MM-SSZ` (pattern strict) → time-travel sur ce snapshot.

    Notes
    -----
    - `_sanitize_at()` garantit qu'une valeur invalide de `at` ne casse pas
      la route et retombe sur `"latest"`.
    - TTL HTTP par défaut : 60 s.
    """
    base = _mon_prefix_or_500()
    folder = _sanitize_at(at)
    gs_uri = f"{base}/intro/{folder}/intro.json"
    return _proxy_json(gs_uri, request, ttl=60)
