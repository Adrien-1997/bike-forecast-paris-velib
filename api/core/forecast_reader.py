"""API Forecast Reader.

Ce module fournit une petite couche d’accès aux prévisions batch stockées
dans GCS (outputs de `build_serving_forecast`), avec :

- résolution des chemins GCS à partir des settings,
- lecture optimisée via HTTP public si possible (sinon client GCS),
- normalisation des colonnes (types, tri, etc.),
- cache mémoire simple par horizon,
- filtrage optionnel par liste de stations.

Format principal attendu (bundle JSON unique) sous
`SERVING_FORECAST_PREFIX/latest_forecast.json` :

    {
      "generated_at": "...Z",
      "horizons": [15, 60],
      "data": {
        "15": [ {record}, ... ],
        "60": [ {record}, ... ]
      }
    }

Un mode legacy est aussi supporté par horizon : `latest_h{h}.json`.
"""

from __future__ import annotations

import io
import os
import json
import time
import urllib.request
from typing import List, Optional, Tuple, Dict

import pandas as pd
from google.cloud import storage
from google.api_core.exceptions import NotFound  # type: ignore

from .settings import settings

# Cache mémoire très simple: {horizon_min: (epoch_loaded, DataFrame)}
_CACHE: Dict[int, Tuple[float, pd.DataFrame]] = {}


# ───────────────────────────────────────────────────────────────
# Helpers config
# ───────────────────────────────────────────────────────────────
def _supported_horizons() -> set[int]:
    """Retourne l'ensemble des horizons (en minutes) supportés par l'API.

    Lecture de `settings.FORECAST_SUPPORTED` (string CSV, ex. "15,60").
    Si vide ou None, on retombe sur `{15}` par défaut.
    """
    raw = (settings.FORECAST_SUPPORTED or "15").strip()
    if not raw:
        return {15}
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def _parse_gs(uri: str) -> tuple[str, str]:
    """Parse une URI GCS `gs://bucket/key` en `(bucket, key)`.

    Raises
    ------
    AssertionError
        Si l'URI ne commence pas par `gs://`.
    """
    assert uri.startswith("gs://"), f"Invalid GCS URI: {uri}"
    bkt, key = uri[5:].split("/", 1)
    return bkt, key


def _bundle_json_uri() -> str:
    """Construit l'URI GCS du bundle JSON principal `latest_forecast.json`."""
    prefix = settings.SERVING_FORECAST_PREFIX.rstrip("/")
    return f"{prefix}/latest_forecast.json"


def _legacy_latest_json_uri(h: int) -> str:
    """Construit l'URI GCS legacy pour un horizon donné.

    Format attendu : `<SERVING_FORECAST_PREFIX>/latest_h{h}.json`.
    Utilisé uniquement en fallback si le bundle n'existe pas.
    """
    # Backward-compat support if per-horizon files still exist
    prefix = settings.SERVING_FORECAST_PREFIX.rstrip("/")
    return f"{prefix}/latest_h{h}.json"


# ───────────────────────────────────────────────────────────────
# I/O
# ───────────────────────────────────────────────────────────────
def _http_download_optional(uri: str, timeout: float = 5.0) -> Optional[bytes]:
    """
    Essaie la lecture HTTP directe sur le endpoint public GCS (si l'objet est public):
      gs://bucket/key -> https://storage.googleapis.com/bucket/key
    Renvoie None en cas d'échec (non public, 404, etc.).
    """
    try:
        bkt, key = _parse_gs(uri)
        url = f"https://storage.googleapis.com/{bkt}/{key}"
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def _download_bytes(uri: str) -> bytes:
    """Télécharge un objet GCS via le client `google-cloud-storage`.

    Parameters
    ----------
    uri : str
        URI GCS au format `gs://bucket/key`.

    Returns
    -------
    bytes
        Contenu brut de l'objet.

    Raises
    ------
    google.api_core.exceptions.NotFound
        Si l'objet n'existe pas.
    """
    bkt, key = _parse_gs(uri)
    cli = storage.Client()
    blob = cli.bucket(bkt).blob(key)
    return blob.download_as_bytes()  # peut lever NotFound


def _download_bytes_optional(uri: str) -> Optional[bytes]:
    """Version tolérante de `_download_bytes` : renvoie None en cas de NotFound."""
    try:
        return _download_bytes(uri)
    except NotFound:
        return None


def _read_json_df_records(b: bytes) -> pd.DataFrame:
    """
    Tente d'interpréter b comme une liste de records JSON → DataFrame.
    Fallback JSON Lines si nécessaire.
    """
    if not b:
        return pd.DataFrame()
    # Essai 1: parse direct liste
    try:
        obj = json.loads(b.decode("utf-8"))
        if isinstance(obj, list):
            return pd.DataFrame(obj)
        elif isinstance(obj, dict):
            # paquet type {"generated_at": "...", "horizons": [...], "data": {"15":[...], ...}}
            # Le caller gère l'extraction; ici, si dict simple -> DF d'une ligne
            return pd.DataFrame([obj])
    except Exception:
        pass
    # Essai 2: JSON Lines
    try:
        lines = [json.loads(line) for line in b.decode("utf-8").splitlines() if line.strip()]
        if lines:
            return pd.DataFrame(lines)
    except Exception:
        pass
    # Essai 3: pandas
    try:
        return pd.read_json(io.BytesIO(b), orient="records", lines=True)
    except Exception:
        return pd.DataFrame()


# ───────────────────────────────────────────────────────────────
# Normalisation
# ───────────────────────────────────────────────────────────────
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise le DataFrame de prévisions en types/cohérence UI.

    - Convertit les colonnes datetime (`tbin_latest`, `pred_ts_utc`) en tz-naive UTC.
    - Force `station_id` en string (clé front / API).
    - Cast les colonnes entières (`capacity_bin`, `horizon_min`) en Int64 nullable.
    - Cast certaines colonnes numériques en float (prédictions, ratios, lags).
    - Trie par `station_id` si présent.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame brut issu du JSON.

    Returns
    -------
    pd.DataFrame
        DataFrame normalisé et trié.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Datetimes -> tz-naive UTC
    for c in ("tbin_latest", "pred_ts_utc"):
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce", utc=True)
            df[c] = s.dt.tz_localize(None)

    # station_id toujours string (clé UI)
    if "station_id" in df.columns:
        df["station_id"] = df["station_id"].astype("string")

    # Numériques stables pour le reste
    for c in ("capacity_bin", "horizon_min"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    for c in ("bikes_pred", "occ_ratio_bin", "lag_nb_1b", "roll_nb_12b"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "station_id" in df.columns:
        df = df.sort_values("station_id").reset_index(drop=True)

    return df


# ───────────────────────────────────────────────────────────────
# Cache
# ───────────────────────────────────────────────────────────────
def _cache_get(h: int) -> Optional[pd.DataFrame]:
    """Retourne le DataFrame en cache pour un horizon donné si encore valide.

    La durée de vie (TTL) est contrôlée par `settings.FORECAST_CACHE_TTL_SECONDS`.
    Si le TTL est <= 0, le cache est désactivé.

    Parameters
    ----------
    h : int
        Horizon (en minutes).

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame mis en cache, ou None si expiré / absent / cache désactivé.
    """
    ttl = int(getattr(settings, "FORECAST_CACHE_TTL_SECONDS", 0) or 0)
    if ttl <= 0:
        return None
    item = _CACHE.get(h)
    if not item:
        return None
    ts, df = item
    if (time.time() - ts) <= ttl:
        return df
    return None


def _cache_put(h: int, df: pd.DataFrame) -> None:
    """Enregistre un DataFrame en cache pour un horizon donné.

    Ne fait rien si le TTL est <= 0 (cache désactivé).

    Parameters
    ----------
    h : int
        Horizon (en minutes).
    df : pd.DataFrame
        DataFrame à mettre en cache.
    """
    ttl = int(getattr(settings, "FORECAST_CACHE_TTL_SECONDS", 0) or 0)
    if ttl <= 0:
        return
    _CACHE[h] = (time.time(), df)


# ───────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────
def load_latest_forecast(h: int, station_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Charge les dernières prévisions pour un horizon donné depuis GCS.

    Format principal (bundle):
      SERVING_FORECAST_PREFIX/latest_forecast.json

        {
          "generated_at": "...Z",
          "horizons": [15, 60],
          "data": {
            "15": [ {record}, ... ],
            "60": [ {record}, ... ]
          }
        }

    Fallback legacy:
      - latest_h{h}.json si présent sous SERVING_FORECAST_PREFIX.

    Le résultat est normalisé via `_normalize` et peut être filtré sur un
    sous-ensemble de stations.

    Parameters
    ----------
    h : int
        Horizon de prévision en minutes (doit appartenir à _supported_horizons).
    station_ids : list[str] | None, optional
        Liste optionnelle d'identifiants de station à conserver.
        Les identifiants sont castés en string et normalisés avec `strip()`.

    Returns
    -------
    pd.DataFrame
        DataFrame des prévisions pour l'horizon demandé, éventuellement filtré.

    Raises
    ------
    ValueError
        Si l'horizon n'est pas listé dans les horizons supportés.
    google.api_core.exceptions.NotFound
        Si aucun fichier de prévision n'est trouvé pour cet horizon.
    """
    if h not in _supported_horizons():
        raise ValueError(f"horizon {h} not supported. Supported={sorted(_supported_horizons())}")

    # 1) Cache
    cached = _cache_get(h)
    if cached is not None:
        df = cached
    else:
        # 2) Try bundle JSON
        uri_bundle = _bundle_json_uri()
        use_http = str(getattr(settings, "FORECAST_HTTP_FIRST", os.getenv("FORECAST_HTTP_FIRST", "0"))).lower() in (
            "1", "true", "yes", "y"
        )

        raw: Optional[bytes] = None
        if use_http:
            raw = _http_download_optional(uri_bundle)
        if raw is None:
            raw = _download_bytes_optional(uri_bundle)

        if raw:
            try:
                obj = json.loads(raw.decode("utf-8"))
                data = obj.get("data", {})
                rows = data.get(str(h), []) if isinstance(data, dict) else []
                df = pd.DataFrame(rows)
                df = _normalize(df)
                _cache_put(h, df)
            except Exception:
                df = pd.DataFrame()
        else:
            # 3) Fallback legacy per-horizon JSON if it still exists
            uri_legacy = _legacy_latest_json_uri(h)
            raw_legacy = _download_bytes_optional(uri_legacy) if not use_http else (
                _http_download_optional(uri_legacy) or _download_bytes_optional(uri_legacy)
            )
            if raw_legacy:
                df = _read_json_df_records(raw_legacy)
                df = _normalize(df)
                _cache_put(h, df)
            else:
                # Nothing found
                raise NotFound(f"No forecast found for horizon {h}")

    # ───────────────────────────────────────────────
    # Filter station_ids if requested (string-safe)
    # ───────────────────────────────────────────────
    if station_ids:
        sset = {str(x).strip() for x in station_ids if x is not None}
        if "station_id" in df.columns and sset:
            df = df[df["station_id"].astype(str).isin(sset)].reset_index(drop=True)

    return df