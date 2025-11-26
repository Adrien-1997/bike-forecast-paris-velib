# service/jobs/ingest.py

"""
Job d’ingestion 5 minutes pour le pipeline Vélib’ Forecast.

Rôle
----
Ce job :
- récupère le statut des stations et les informations de stations depuis
  les endpoints officiels GBFS de Vélib ;
- force les timestamps des stations au *moment d’exécution du job*
  (ts_utc = now_utc) tout en conservant optionnellement le `last_reported`
  GBFS original en mémoire dans une colonne `src_ts_utc` ;
- calcule un bin temporel de 5 minutes (tbin_utc) aligné sur l’heure
  d’exécution du job ;
- récupère optionnellement la météo horaire auprès de l’API Open-Meteo
  autour de la fenêtre temporelle courante ;
- fusionne données vélos et météo dans un DataFrame de snapshot unique ;
- écrit ce snapshot en Parquet local et, optionnellement, sur GCS
  (couche "bronze") ;
- calcule des métriques de "fraîcheur" pour les stations et la météo
  et les publie en JSON en local et, optionnellement, sur GCS pour le
  stack de monitoring.

Schéma (strict, timestamps UTC naïfs)
-------------------------------------
ts_utc, tbin_utc, station_id, bikes, capacity, mechanical, ebike, status,
lat, lon, name, temp_C, precip_mm, wind_mps

Layout des fichiers (UTC)
-------------------------
Snapshots locaux :
  data_local/raw/date=YYYY-MM-DD/hour=HH/YYYY-MM-DDT%H-%M.parquet

Snapshots GCS (quand INGEST_TO_GCS=1) :
  gs://.../bronze/date=YYYY-MM-DD/hour=HH/YYYY-MM-DDT%H-%M.parquet

Variables d’environnement
-------------------------
INGEST_SAVE_PARQUET : "1" | "0"  (défaut "1")
    Contrôle l’écriture locale en Parquet du snapshot brut 5 minutes.

LOCAL_RAW_DIR : str (défaut "data_local/raw")
    Racine locale pour les snapshots bruts.

INGEST_TO_GCS : "1" | "0"  (défaut "0")
    Quand "1", uploade le snapshot Parquet sur GCS sous GCS_RAW_PREFIX.

GCS_RAW_PREFIX : str
    Préfixe GCS (gs://bucket/path) pour les snapshots bronze
    lorsque INGEST_TO_GCS=1.

OPENMETEO_LAT, OPENMETEO_LON : float (en tant que chaînes)
    Coordonnées utilisées pour la requête météo horaire Open-Meteo.

METEO_DISABLE : "1" | "0"  (défaut "0")
    Quand "1", saute complètement l’appel à l’API météo.

DIAG : "1" | "true" | "True" (défaut "0")
    Active un logging diagnostique verbeux (lignes d’exemple, chemins, etc.).

GCS_MONITORING_PREFIX : str (optionnel)
    Préfixe GCS (gs://bucket/path) utilisé pour uploader les JSON de fraîcheur
    pour le monitoring.

Exécution
---------
À lancer une fois depuis la racine du dépôt :

    python -m jobs.ingest
"""

from __future__ import annotations
import os
from io import BytesIO
from datetime import datetime, timezone, timedelta
from typing import Tuple
import json
import math
import numpy as np
import pandas as pd
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

try:
    from google.cloud import storage  # type: ignore
except Exception:
    storage = None  # type: ignore

# ───────────────────────── Config ─────────────────────────

URL_STATUS = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_status.json"
URL_INFO   = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

LAT = float(os.environ.get("OPENMETEO_LAT", "48.8566"))
LON = float(os.environ.get("OPENMETEO_LON", "2.3522"))
METEO_DISABLE = os.environ.get("METEO_DISABLE", "0") == "1"
DIAG = os.environ.get("DIAG", "0") in ("1", "true", "True")

COLS_ORDER = [
    "ts_utc","tbin_utc","station_id","bikes","capacity","mechanical","ebike",
    "status","lat","lon","name","temp_C","precip_mm","wind_mps"
]

# ───────────────────── Session HTTP (retries) ─────────────────────

_session: requests.Session | None = None


def _session_get() -> requests.Session:
    """
    Retourne une session HTTP partagée configurée avec des retries.

    La session :
    - retente sur les erreurs HTTP transitoires classiques (5xx, 429) ;
    - utilise un backoff exponentiel ;
    - définit un User-Agent spécifique pour tracer plus facilement côté provider.
    """
    global _session
    if _session is None:
        s = requests.Session()
        retry = Retry(
            total=5, connect=5, read=5,
            backoff_factor=0.4,
            status_forcelist=[429,500,502,503,504],
            allowed_methods=["GET","HEAD"],
            raise_on_status=False,
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.headers.update({"User-Agent": "velib-pipeline/1.1"})
        _session = s
    return _session


def _http_get_json(url: str, timeout: int = 20) -> dict:
    """
    Effectue une requête HTTP GET avec la session partagée et décode le JSON.

    Paramètres
    ----------
    url : str
        URL absolue à interroger.
    timeout : int, défaut 20
        Timeout de la requête en secondes.

    Retour
    ------
    dict
        Payload JSON parsé.

    Lève
    ----
    requests.HTTPError
        Si le code de statut HTTP n’est pas un succès.
    """
    r = _session_get().get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ───────────────────────── GBFS ─────────────────────────


def fetch_velib_df() -> pd.DataFrame:
    """
    Récupère le statut et les informations des stations Vélib, fusionne et normalise.

    Cette fonction :
    - appelle les endpoints GBFS officiels `station_status.json` et
      `station_information.json` ;
    - construit une ligne par station ;
    - force `ts_utc` au timestamp d’exécution du job d’ingestion (now UTC naïf) ;
    - ajoute un bin de 5 minutes (`tbin_utc`) aligné sur l’heure du job ;
    - conserve optionnellement le timestamp source `last_reported` dans la
      colonne en mémoire `src_ts_utc` (non persistée dans le Parquet).

    Retour
    ------
    pandas.DataFrame
        DataFrame contenant (au minimum) les colonnes de COLS_ORDER ainsi
        qu’une colonne en mémoire `src_ts_utc` utilisée ensuite pour la
        fraîcheur. Si les payloads sont vides, renvoie un DataFrame vide
        avec COLS_ORDER comme colonnes.
    """
    js_status = _http_get_json(URL_STATUS)
    js_info   = _http_get_json(URL_INFO)

    status_list = (js_status.get("data") or {}).get("stations") or []
    info_list   = (js_info.get("data") or {}).get("stations") or []
    if not status_list or not info_list:
        print("[ingest][gbfs] empty payloads — status:", len(status_list), "info:", len(info_list))
        return pd.DataFrame(columns=COLS_ORDER)

    # "Heure du job" en UTC (naïf), utilisée comme ts_utc et pour le bin 5 minutes.
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    tbin_now = pd.Timestamp(now_utc).floor("5min").to_pydatetime()

    # STATUS
    st_rows = []
    for s in status_list:
        sid = s.get("station_id")
        if not sid:
            continue

        # Extraction des vélos mécaniques / électriques quand exposée par le provider.
        types = {}
        v = s.get("num_bikes_available_types")
        if isinstance(v, list) and v:
            for d in v:
                if isinstance(d, dict):
                    types.update(d)
        elif isinstance(v, dict):
            types = v

        # Timestamp source depuis GBFS (last_reported), utilisé uniquement pour la fraîcheur.
        last_reported = s.get("last_reported")
        try:
            src_ts = datetime.utcfromtimestamp(int(last_reported)) if last_reported else now_utc
        except Exception:
            src_ts = now_utc

        st_rows.append({
            "station_id": sid,
            "ts_utc": now_utc,  # timestamp du job d’ingestion
            "bikes": s.get("num_bikes_available"),
            "mechanical": types.get("mechanical", 0),
            "ebike": types.get("ebike", 0),
            "status": s.get("station_status") or (
                "OK" if (s.get("is_renting",1) and s.get("is_returning",1)) else "CLOSED"
            ),
            # Timestamp source utilisé uniquement en mémoire pour la fraîcheur ; non écrit dans le parquet.
            "src_ts_utc": src_ts,
        })

    df_st = pd.DataFrame(st_rows)

    # INFO
    in_rows = []
    for i in info_list:
        sid = i.get("station_id")
        if not sid:
            continue
        in_rows.append({
            "station_id": sid,
            "name": i.get("name"),
            "lat":  i.get("lat"),
            "lon":  i.get("lon"),
            "capacity": i.get("capacity"),
        })
    df_in = pd.DataFrame(in_rows)

    # Fusion statut + info sur station_id
    df = df_st.merge(df_in, on="station_id", how="inner")
    df["ts_utc"]   = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce").dt.tz_localize(None)
    df["tbin_utc"] = pd.to_datetime(tbin_now)

    # Normalisation des champs numériques
    for c in ("bikes","mechanical","ebike","capacity","lat","lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"[ingest][gbfs] rows={len(df):,} tbin_utc={tbin_now} (UTC)")
    if DIAG:
        print(df[["station_id","ts_utc","tbin_utc","bikes","capacity"]].head(5).to_string(index=False))
    return df

# ───────────────────────── Météo ─────────────────────────


def _weather_window(df_velib: pd.DataFrame) -> Tuple[datetime, datetime]:
    """
    Construit une fenêtre temporelle ±3h en UTC autour du bin courant.

    La fenêtre est dérivée du min/max de `tbin_utc` dans `df_velib`,
    arrondis à l’heure inférieure. Si les timestamps ne peuvent pas être
    parsés, on retombe sur une fenêtre ±3h autour de "now UTC".

    Paramètres
    ----------
    df_velib : pandas.DataFrame
        DataFrame Vélib tel que retourné par `fetch_velib_df`.

    Retour
    ------
    (datetime, datetime)
        Bornes inférieure et supérieure de la fenêtre en UTC (datetimes naïfs).
    """
    hours = pd.to_datetime(df_velib["tbin_utc"], errors="coerce").dt.floor("h")
    lo, hi = hours.min(), hours.max()
    if pd.isna(lo) or pd.isna(hi):
        now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        return now - timedelta(hours=3), now + timedelta(hours=3)
    return lo - timedelta(hours=3), hi + timedelta(hours=3)


def fetch_weather_df(df_velib: pd.DataFrame) -> pd.DataFrame:
    """
    Récupère la météo horaire autour de la fenêtre Vélib auprès d’Open-Meteo.

    La fonction :
    - dérive une fenêtre ±3h autour des valeurs `tbin_utc` de Vélib ;
    - interroge l’API Open-Meteo pour les 2 jours passés et 2 jours futurs en UTC ;
    - filtre la série horaire pour ne garder que les heures dans la fenêtre ;
    - renvoie température, précipitations et vitesse du vent à 10 m.

    Paramètres
    ----------
    df_velib : pandas.DataFrame
        Snapshot Vélib utilisé pour calculer la fenêtre temporelle.

    Retour
    ------
    pandas.DataFrame
        DataFrame avec colonnes ["hour_utc", "temp_C", "precip_mm", "wind_mps"].
        Si METEO_DISABLE=1 ou en cas d’erreur, renvoie un DataFrame vide avec
        ces colonnes.
    """
    if METEO_DISABLE:
        print("[ingest][weather] disabled by METEO_DISABLE=1")
        return pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])
    try:
        lo, hi = _weather_window(df_velib)
        params = {
            "latitude": LAT,
            "longitude": LON,
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "windspeed_unit": "ms",
            "past_days": 2, "forecast_days": 2,
            "timezone": "UTC",
        }
        j = _session_get().get(OPEN_METEO, params=params, timeout=20)
        j.raise_for_status()
        j = j.json()

        hours = pd.to_datetime(j["hourly"]["time"], utc=True, errors="coerce").tz_convert(None)
        dfw = pd.DataFrame({
            "hour_utc":  hours,
            "temp_C":    j["hourly"]["temperature_2m"],
            "precip_mm": j["hourly"]["precipitation"],
            "wind_mps":  j["hourly"]["wind_speed_10m"],
        })
        dfw = dfw[(dfw["hour_utc"] >= lo) & (dfw["hour_utc"] <= hi)].copy()
        print(f"[ingest][weather] got={len(j['hourly']['time']):,} kept={len(dfw):,} window={lo}..{hi} (UTC)")
        return dfw
    except Exception as e:
        print(f"[ingest][weather] error: {e}")
        return pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])

# ─────────────────────── Helpers GCS ───────────────────────


def _split_gcs(gcs_url: str) -> tuple[str,str]:
    """
    Découper une URL GCS en bucket et object key.

    Paramètres
    ----------
    gcs_url : str
        URL commençant par "gs://".

    Retour
    ------
    (str, str)
        Tuple (bucket_name, object_key).

    Lève
    ----
    AssertionError
        Si l’URL ne commence pas par "gs://".
    """
    assert gcs_url.startswith("gs://")
    b, p = gcs_url[5:].split("/", 1)
    return b, p


def _upload_parquet_gcs(df: pd.DataFrame, gcs_url: str):
    """
    Uploader un DataFrame au format parquet vers une localisation GCS.

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame à sérialiser.
    gcs_url : str
        URL GCS de destination (gs://bucket/path/to/file.parquet).

    Lève
    ----
    RuntimeError
        Si `google-cloud-storage` n’est pas installé.
    """
    if storage is None:
        raise RuntimeError("google-cloud-storage not installed")
    bkt, key = _split_gcs(gcs_url)
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    storage.Client().bucket(bkt).blob(key).upload_from_file(buf, content_type="application/octet-stream")


def _upload_json_gcs(payload: dict, gcs_url: str):
    """
    Uploader un payload JSON-sérialisable au format JSON sur GCS.

    Paramètres
    ----------
    payload : dict
        Dictionnaire Python à encoder en JSON.
    gcs_url : str
        URL GCS de destination (gs://bucket/path/to/file.json).

    Lève
    ----
    RuntimeError
        Si `google-cloud-storage` n’est pas installé.
    """
    if storage is None:
        raise RuntimeError("google-cloud-storage not installed")
    bkt, key = _split_gcs(gcs_url)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    storage.Client().bucket(bkt).blob(key).upload_from_string(data, content_type="application/json")


def _write_json_local(payload: dict, path: str):
    """
    Écrire un payload JSON en local, en créant les dossiers parents si besoin.

    Paramètres
    ----------
    payload : dict
        Dictionnaire Python à encoder en JSON.
    path : str
        Chemin de fichier local où écrire le JSON.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _compute_freshness_payload(df_velib: pd.DataFrame, df_weather: pd.DataFrame) -> dict:
    """
    Calculer des métriques de fraîcheur stations et météo pour le monitoring.

    Fraîcheur des stations
    ----------------------
    - Timestamp de base :
        - si `src_ts_utc` existe (GBFS `last_reported`), on l’utilise ;
        - sinon, on retombe sur `ts_utc` (timestamp du job).
    - Les valeurs de fraîcheur (en minutes) sont calculées comme :
        `now_utc - base_timestamp`.

    Fraîcheur météo
    ---------------
    - Si `df_weather` n’est pas vide :
        - on prend le max de `hour_utc` et on calcule `now_utc - last_hour`
          en minutes.

    La fonction construit également un petit payload de diagnostic avec les
    k stations les plus "anciennes" (freshness la plus élevée).

    Paramètres
    ----------
    df_velib : pandas.DataFrame
        DataFrame Vélib en mémoire, y compris la colonne optionnelle
        `src_ts_utc`.
    df_weather : pandas.DataFrame
        DataFrame météo retourné par `fetch_weather_df`.

    Retour
    ------
    dict
        Payload JSON prêt à être sérialisé, de la forme :

        {
          "now_utc": <ISO datetime>,
          "stations": {
            "count": <int>,
            "freshness": {
              "p50_min": <float or null>,
              "p95_min": <float or null>,
              "max_min": <float or null>
            },
            "top_oldest": [
              {"station_id": ..., "freshness_min": ...},
              ...
            ]
          },
          "weather": {
            "freshness_min": <float or null>
          },
          "meta": {
            "bin_t_utc": <ISO datetime or null>,
            "schema": "v1"
          }
        }
    """
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

    # 1) Fraîcheur des stations
    if "src_ts_utc" in df_velib.columns:
        ts_base = pd.to_datetime(df_velib["src_ts_utc"], errors="coerce")
    else:
        ts_base = pd.to_datetime(df_velib["ts_utc"], errors="coerce")

    freshness_min_st = (pd.to_datetime(now_utc) - pd.to_datetime(ts_base)).dt.total_seconds() / 60.0
    freshness_min_st = pd.to_numeric(freshness_min_st, errors="coerce")
    f50 = float(np.nanquantile(freshness_min_st, 0.50)) if len(freshness_min_st) else float("nan")
    f95 = float(np.nanquantile(freshness_min_st, 0.95)) if len(freshness_min_st) else float("nan")
    fmax = float(np.nanmax(freshness_min_st)) if len(freshness_min_st) else float("nan")

    # Top-k stations les plus "anciennes" (freshness max) — utile en diagnostic.
    top_k = 50
    top_idx = np.argsort(freshness_min_st.values)[-top_k:] if len(freshness_min_st) else []
    top_payload = []
    if len(top_idx):
        sub = df_velib.iloc[top_idx]
        for sid, val in zip(sub["station_id"], freshness_min_st.iloc[top_idx]):
            if pd.notna(val):
                try:
                    top_payload.append({"station_id": int(sid), "freshness_min": round(float(val), 2)})
                except Exception:
                    top_payload.append({"station_id": sid, "freshness_min": round(float(val), 2)})
        top_payload.sort(key=lambda d: d["freshness_min"], reverse=True)

    # 2) Fraîcheur météo (si dispo)
    weather_fresh_min = None
    if not df_weather.empty and "hour_utc" in df_weather.columns:
        last_hour = pd.to_datetime(df_weather["hour_utc"], errors="coerce").max()
        if pd.notna(last_hour):
            weather_fresh_min = round(
                (pd.to_datetime(now_utc) - pd.to_datetime(last_hour)).total_seconds() / 60.0, 2
            )

    payload = {
        "now_utc": pd.to_datetime(now_utc).isoformat(),
        "stations": {
            "count": int(df_velib["station_id"].nunique()) if "station_id" in df_velib.columns else int(len(df_velib)),
            "freshness": {
                "p50_min": None if math.isnan(f50) else round(f50, 2),
                "p95_min": None if math.isnan(f95) else round(f95, 2),
                "max_min": None if math.isnan(fmax) else round(fmax, 2),
            },
            # Top 50 stations les plus anciennes, utile pour le monitoring/debug interne.
            "top_oldest": top_payload,
        },
        "weather": {
            # Peut rester None si la météo est désactivée ou indisponible.
            "freshness_min": weather_fresh_min,
        },
        "meta": {
            "bin_t_utc": pd.to_datetime(df_velib["tbin_utc"].iloc[0]).isoformat() if len(df_velib) else None,
            "schema": "v1",
        },
    }
    return payload

# ───────────────────────── Ingest principal ─────────────────────────


def ingest_once(save: bool = True) -> tuple[int, str | None]:
    """
    Exécute un cycle d’ingestion unique (5 minutes).

    Étapes
    ------
    1. Récupérer statut+info Vélib et construire le snapshot stations.
    2. Récupérer optionnellement la météo et fusionner sur l’heure.
    3. Réordonner les colonnes selon COLS_ORDER.
    4. Déduire la date/heure UTC et le nom du fichier Parquet à partir de `tbin_utc`.
    5. Écrire optionnellement le snapshot en Parquet local (contrôlé par `save`
       ou `INGEST_SAVE_PARQUET`).
    6. Uploader optionnellement le snapshot sur GCS quand `INGEST_TO_GCS=1`.
    7. Calculer les métriques de fraîcheur et les écrire en local et sur GCS
       (si configuré).

    Paramètres
    ----------
    save : bool, défaut True
        Si True, force l’écriture en Parquet local, indépendamment de la
        variable d’environnement `INGEST_SAVE_PARQUET`. Si False, on se
        repose uniquement sur la variable d’environnement.

    Retour
    ------
    (int, str ou None)
        Tuple `(n_rows, gcs_url)` où :
        - `n_rows` est le nombre de lignes du snapshot ;
        - `gcs_url` est l’URL GCS du fichier Parquet si uploadé, sinon None.
    """
    df_v = fetch_velib_df()
    if df_v.empty:
        print("[ingest] GBFS empty — no output")
        return 0, None

    df_w = fetch_weather_df(df_v)
    df_v["hour_utc"] = pd.to_datetime(df_v["tbin_utc"], errors="coerce").dt.floor("h")
    if not df_w.empty:
        df = df_v.merge(df_w, on="hour_utc", how="left")
        print(f"[ingest][merge] velib={len(df_v):,} weather={len(df_w):,} → merged={len(df):,}")
    else:
        df = df_v.assign(temp_C=None, precip_mm=None, wind_mps=None)
        print(f"[ingest][merge] weather empty → filled NaN (rows={len(df):,})")
    df = df.drop(columns=["hour_utc"])

    df_out = df[COLS_ORDER].copy()
    latest_bin_utc = pd.to_datetime(df_out["tbin_utc"].iloc[0])
    day  = latest_bin_utc.strftime("%Y-%m-%d")
    hour = latest_bin_utc.strftime("%H")
    fname = latest_bin_utc.strftime("%Y-%m-%dT%H-%M.parquet")

    if DIAG:
        print(f"[ingest][diag] rows={len(df_out)} stations={df_out['station_id'].nunique()} bin={latest_bin_utc} → {day}/{hour}/{fname}")

    wrote_gcs: str | None = None
    if save or os.environ.get("INGEST_SAVE_PARQUET","1") == "1":
        local_root = os.environ.get("LOCAL_RAW_DIR", "data_local/raw")
        local_path = os.path.join(local_root, f"date={day}", f"hour={hour}", fname)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        df_out.to_parquet(local_path, index=False)
        print(f"[ingest][snapshot] wrote {len(df_out):,} rows → {local_path}")

    if os.environ.get("INGEST_TO_GCS","0") == "1":
        gcs_prefix = os.environ.get("GCS_RAW_PREFIX")
        if not gcs_prefix or not gcs_prefix.startswith("gs://"):
            raise RuntimeError("INGEST_TO_GCS=1 mais GCS_RAW_PREFIX absent ou invalide")
        gcs_url = f"{gcs_prefix}/date={day}/hour={hour}/{fname}"
        _upload_parquet_gcs(df_out, gcs_url)
        print(f"[ingest][snapshot] uploaded {len(df_out):,} rows → {gcs_url}")
        wrote_gcs = gcs_url

    # ───────────── JSON de fraîcheur (monitoring/data/freshness) ─────────────
    freshness = _compute_freshness_payload(df, df_w)

    # Copies locales "latest" et datées.
    local_monitor_root = os.environ.get("LOCAL_MONITOR_DIR", "data_local/monitoring")
    isots = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    local_latest = os.path.join(local_monitor_root, "data/freshness/latest.json")
    local_dated  = os.path.join(local_monitor_root, f"data/freshness/{isots}.json")
    _write_json_local(freshness, local_latest)
    _write_json_local(freshness, local_dated)
    print(f"[ingest][freshness] wrote {local_latest} & {local_dated}")

    # Uploads GCS optionnels quand un préfixe de monitoring est configuré.
    gcs_mon = os.environ.get("GCS_MONITORING_PREFIX", "").rstrip("/")
    if gcs_mon.startswith("gs://"):
        g_latest = f"{gcs_mon}/monitoring/data/freshness/latest.json"
        g_dated  = f"{gcs_mon}/monitoring/data/freshness/{isots}.json"
        _upload_json_gcs(freshness, g_latest)
        _upload_json_gcs(freshness, g_dated)
        print(f"[ingest][freshness] uploaded {g_latest} & {g_dated}")

    return len(df_out), wrote_gcs


def main() -> int:
    """
    Point d’entrée CLI pour le job d’ingestion.

    Retour
    ------
    int
        Code de sortie du process (0 en cas de succès).
    """
    print("[ingest] start")
    n, out = ingest_once(save=os.environ.get("INGEST_SAVE_PARQUET","1") == "1")
    print(f"[ingest] done rows={n} out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())