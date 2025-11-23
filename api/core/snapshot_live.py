"""Live Vélib' snapshot (GBFS → pandas DataFrame).

Ce module fournit une petite brique autonome pour récupérer **l'état live**
du réseau Vélib' via les endpoints GBFS Smovengo, et le transformer en un
DataFrame propre, cohérent avec le schéma utilisé côté backend :

    ts_utc, tbin_utc, station_id,
    bikes, capacity, mechanical, ebike,
    status, lat, lon, name

Points clés :
- URLs surchageables via les variables d'env `VELIB_STATUS_URL` et
  `VELIB_INFO_URL`,
- session HTTP `requests` avec retries robustes (exponential backoff),
- merge `station_status` + `station_information` sur `station_id`,
- arrondi des timestamps à la minute 5 la plus proche (`tbin_utc`),
- typage strict pour l'output (int, float, string).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Any

import pandas as pd
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# ---------- Config (surchargable via env) ----------
# Endpoints GBFS; peuvent être overridés pour les tests ou environnements spécifiques.
URL_STATUS = os.getenv(
    "VELIB_STATUS_URL",
    "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_status.json",
)
URL_INFO = os.getenv(
    "VELIB_INFO_URL",
    "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json",
)

# Schéma de sortie standard (ordre de colonnes).
COLS_ORDER = [
    "ts_utc", "tbin_utc", "station_id",
    "bikes", "capacity", "mechanical", "ebike",
    "status", "lat", "lon", "name"
]

# ---------- HTTP session with retries ----------
_session: requests.Session | None = None


def _session_get() -> requests.Session:
    """Return a shared `requests.Session` configured with retries.

    - utilise urllib3 Retry pour gérer :
        * erreurs de connexion,
        * timeouts de lecture,
        * statuts 429 / 5xx,
    - applique un backoff exponentiel (backoff_factor=0.4),
    - fixe un User-Agent explicite pour l'API Vélib'.

    La session est créée une fois puis réutilisée (pattern singleton).
    """
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=0.4,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
            raise_on_status=False,
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.headers.update({"User-Agent": "velib-api-snapshot-live/1.0"})
        _session = s
    return _session


def _http_get_json(url: str, timeout: int = 20) -> dict:
    """GET + JSON avec vérification du statut HTTP.

    Parameters
    ----------
    url : str
        Endpoint à appeler.
    timeout : int, default 20
        Timeout en secondes passé à `requests`.

    Returns
    -------
    dict
        JSON décodé (typiquement `{"data": {...}}` pour GBFS).

    Raises
    ------
    requests.HTTPError
        Si le code de retour n'est pas 2xx.
    """
    r = _session_get().get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------- GBFS fetch ----------
def _fetch_gbfs_velib_df() -> pd.DataFrame:
    """Télécharge et fusionne les flux GBFS (status + info) en DataFrame brut.

    Steps:
    - récupère `station_status.json` et `station_information.json`,
    - extrait les listes `data.stations` des deux payloads,
    - construit un DataFrame `df_st` (status) avec :
        * station_id, ts_utc, bikes, mechanical, ebike, status,
    - construit un DataFrame `df_in` (info) avec :
        * station_id, name, lat, lon, capacity,
    - merge inner sur `station_id`,
    - calcule `tbin_utc` = ts_utc arrondi à 5 minutes (UTC naïf).

    Returns
    -------
    pd.DataFrame
        DataFrame intermédiaire status+info, avec `tbin_utc`.
        Si un des payloads est vide ou si le merge échoue, renvoie un
        DataFrame vide avec le bon jeu de colonnes.
    """
    js_status = _http_get_json(URL_STATUS)
    js_info = _http_get_json(URL_INFO)

    status_list = (js_status.get("data") or {}).get("stations") or []
    info_list = (js_info.get("data") or {}).get("stations") or []
    if not status_list or not info_list:
        print("[gbfs] empty payloads — status:", len(status_list), "info:", len(info_list))
        return pd.DataFrame(columns=COLS_ORDER)

    # ---- status ----
    st_rows = []
    for s in status_list:
        sid = s.get("station_id")
        if not sid:
            continue

        # num_bikes_available_types peut être une liste ou un dict selon la version
        v = s.get("num_bikes_available_types")
        types = (v[0] if isinstance(v, list) and v else (v if isinstance(v, dict) else {})) or {}
        mech = types.get("mechanical", 0) or 0
        ebi = types.get("ebike", 0) or 0

        ts = pd.to_datetime(s.get("last_reported"), unit="s", utc=True, errors="coerce")
        st_rows.append(
            {
                "station_id": sid,
                "ts_utc": (ts.tz_convert(None) if ts is not pd.NaT else pd.NaT),
                "bikes": s.get("num_bikes_available"),
                "mechanical": mech,
                "ebike": ebi,
                "status": s.get("station_status")
                # fallback : station "OK" si renting + returning actifs
                or ("OK" if (s.get("is_renting", 1) and s.get("is_returning", 1)) else "CLOSED"),
            }
        )
    df_st = pd.DataFrame(st_rows)

    # ---- info ----
    in_rows = []
    for i in info_list:
        sid = i.get("station_id")
        if not sid:
            continue
        in_rows.append(
            {
                "station_id": sid,
                "name": i.get("name"),
                "lat": i.get("lat"),
                "lon": i.get("lon"),
                "capacity": i.get("capacity"),
            }
        )
    df_in = pd.DataFrame(in_rows)

    # ---- merge ----
    df = df_st.merge(df_in, on="station_id", how="inner")
    if df.empty:
        print("[gbfs] merge empty — no matching station_id")
        return df

    # Arrondi des timestamps à la fenêtre de 5 minutes (UTC, puis tz-naive)
    df["tbin_utc"] = (
        pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
        .dt.floor("5min")
        .dt.tz_convert(None)
    )
    return df


# ---------- Public API ----------
def fetch_live_snapshot() -> pd.DataFrame:
    """
    Retourne un DataFrame "snapshot live" (sans météo), schéma :
      ts_utc, tbin_utc, station_id, bikes, capacity, mechanical, ebike,
      status, lat, lon, name

    - Appelle `_fetch_gbfs_velib_df()` pour obtenir le merge status+info,
    - force les types (int pour vélos/capacité, float pour lat/lon, str pour id/nom),
    - renvoie un DataFrame vide avec `COLS_ORDER` si la récupération échoue.

    Returns
    -------
    pd.DataFrame
        Vue instantanée du réseau Vélib' prête à être consommée
        par l'API ou les features (sans composante météo).
    """
    df_v = _fetch_gbfs_velib_df()
    if df_v.empty:
        return pd.DataFrame(columns=COLS_ORDER)

    out = pd.DataFrame(
        {
            "ts_utc": pd.to_datetime(df_v["ts_utc"], utc=True, errors="coerce").dt.tz_convert(None),
            "tbin_utc": pd.to_datetime(df_v["tbin_utc"], utc=True, errors="coerce").dt.tz_convert(None),
            "station_id": df_v["station_id"].astype(str),
            "bikes": pd.to_numeric(df_v["bikes"], errors="coerce").fillna(0).astype(int),
            "capacity": pd.to_numeric(df_v["capacity"], errors="coerce").fillna(0).astype(int),
            "mechanical": pd.to_numeric(df_v["mechanical"], errors="coerce").fillna(0).astype(int),
            "ebike": pd.to_numeric(df_v["ebike"], errors="coerce").fillna(0).astype(int),
            "status": df_v["status"].astype(str),
            "lat": pd.to_numeric(df_v["lat"], errors="coerce"),
            "lon": pd.to_numeric(df_v["lon"], errors="coerce"),
            "name": df_v["name"].astype(str),
        }
    )[COLS_ORDER]

    return out
