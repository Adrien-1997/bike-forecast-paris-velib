# service/jobs/build_serving_forecast.py

"""
Vélib’ Forecast — Job de Serving Forecast (horizons courts, fenêtre glissante de 4 h).

Rôle
----
Ce job construit un snapshot de features sur 4 heures (bins de 5 minutes) et,
optionnellement, produit des prévisions à court horizon (ex. 15 min, 60 min)
en utilisant des modèles pré-entraînés stockés sur GCS.

Entrées
-------
GCS (ingestion brute parquet, pas de 5 minutes)
    GCS_RAW_PREFIX/date=YYYY-MM-DD/hour=HH/*.parquet

Le schéma brut est aligné avec le job d’ingestion et doit contenir au minimum
les colonnes définies dans `service.core.time_features.BASE_COLUMNS`.

Modèles (par horizon, sur GCS)
    GCS_MODEL_URI_T15   = .../models/h15/latest.joblib
    GCS_MODEL_URI_T60   = .../models/h60/latest.joblib
    (des horizons futurs peuvent être ajoutés de la même façon)

Fenêtre de features
-------------------
- Résolution temporelle : 5 minutes (BIN_MINUTES = 5)
- Taille de fenêtre     : WINDOW_HOURS (par défaut 4 heures)
- Interne               : WINDOW_BINS = max(fenêtre en bins, LAG_MAX_BINS+1)
                          où LAG_MAX_BINS = 48 (plus grand lag utilisé
                          dans les features)

À l’exécution :
    now_utc                : horloge système ou valeur fixée via NOW_UTC_ISO
    end_tbin_aware (UTC)   : now_utc arrondi par défaut au multiple de 5 minutes
    start_tbin_aware (UTC) : end_tbin_aware - (WINDOW_BINS-1)*5min

Prévisions
----------
Pour chaque horizon configuré (ex. 15, 60 minutes) :
- Construire les features par station à `tbin_latest` (dernier bin de la fenêtre).
- Appeler `predict_from_features_df` (core d’entraînement) pour obtenir les prédictions.
- Ré-aligner station_id / métadonnées et garantir une structure JSON propre.
- Uploader :

    {SERVING_FORECAST_PREFIX}/h{H}/latest.json

Exemple :
    gs://.../serving/forecast/h15/latest.json

    {
      "generated_at": "2025-11-01T17:55:00Z",
      "horizon_min": 15,
      "data": [ {...}, {...} ]
    }

Environnement
-------------
Requis :
    GCS_RAW_PREFIX
    SERVING_FORECAST_PREFIX
    FORECAST_HORIZONS        ex. "15,60"
    GCS_MODEL_URI_T15        (pour l’horizon 15 min)
    GCS_MODEL_URI_T60        (pour l’horizon 60 min)

Optionnels :
    WINDOW_HOURS             (par défaut 4)
    WITH_FORECAST            (par défaut "1" ; "0"/"false" désactive l’inférence)
    NOW_UTC_ISO              (horloge fixe en ISO pour les tests)

Notes
-----
- Ce job est stateless : pas de cache local ni d’état persistant.
- Tous les timestamps sont manipulés en UTC, convertis en datetime naïfs
  côté pandas pour le feature building, puis reconvertis en chaînes ISO
  au moment de l’export JSON.
"""

from __future__ import annotations
import os, sys, shutil, json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict
import importlib.util as _iu

import numpy as np
import pandas as pd
from google.cloud import storage

# ─────────────────────────────────────────────
#  Résolution automatique de la racine du dépôt (service/ ou train/)
# ─────────────────────────────────────────────
def _ensure_repo_root():
    """
    S’assurer que la racine du dépôt (contenant `service/` ou `train/`) est sur sys.path.

    Cela rend le module robuste à différents contextes d’exécution :
    - exécution locale : `python -m service.jobs.build_serving_forecast`
    - job Docker / Cloud Run avec `WORKDIR /app`
    - environnements CI / build où le répertoire de travail n’est pas la racine du repo.

    Stratégie
    ---------
    1. Si `service` ou `train` est déjà importable → ne rien faire.
    2. Sinon, remonter depuis le fichier courant et :
         - si un répertoire contenant `service/` ou `train/` est trouvé,
           le préfixer à `sys.path`.
    3. En dernier recours, si `/app` existe, préfixer `/app` à `sys.path`.
    """
    if _iu.find_spec("service") is not None or _iu.find_spec("train") is not None:
        return
    here = Path(__file__).resolve()
    for c in [here] + list(here.parents) + [Path("/app"), Path.cwd()]:
        if (c / "service").exists() or (c / "train").exists():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            print(f"[serving_forecast] repo_root={c}")
            return
    if Path("/app").exists() and "/app" not in sys.path:
        sys.path.insert(0, "/app")
        print("[serving_forecast] fallback /app")

_ensure_repo_root()

# ─────────────────────────────────────────────
#  Imports training/forecast (compatibles multi-layout)
# ─────────────────────────────────────────────
try:
    from service.core.cal_features import add_time_features
    from service.core.time_features import BASE_COLUMNS as TRAIN_BASE_COLUMNS
    from service.core.forecast import predict_from_features_df
except ModuleNotFoundError:
    # Imports de secours conservés pour d’anciens layouts ou problèmes de packaging.
    try:
        from service.core.cal_features import add_time_features
        from service.core.time_features import BASE_COLUMNS as TRAIN_BASE_COLUMNS
        from service.core.forecast import predict_from_features_df
    except ModuleNotFoundError:
        from service.core.cal_features import add_time_features
        from service.core.time_features import BASE_COLUMNS as TRAIN_BASE_COLUMNS
        from service.core.forecast import predict_from_features_df

# ─────────────────────────────────────────────
#  Configuration ENV
# ─────────────────────────────────────────────
# Données brutes (bronze, ingestion toutes les 5 minutes)
RAW_PREFIX = os.environ["GCS_RAW_PREFIX"]

# Résolution temporelle & fenêtre de features
BIN_MINUTES = 5
WINDOW_HOURS = int(os.environ.get("WINDOW_HOURS", "4"))
LAG_MAX_BINS = 48  # plus grand lag utilisé dans les features
# Garantir qu’on a toujours assez d’historique pour tous les lags :
WINDOW_BINS = max(WINDOW_HOURS * 60 // BIN_MINUTES, LAG_MAX_BINS + 1)

# Bascule de forecasting & sorties
WITH_FORECAST = str(os.environ.get("WITH_FORECAST", "1")).lower() in ("1", "true", "yes")
SERVING_FORECAST_PREFIX = os.environ.get("SERVING_FORECAST_PREFIX")
FORECAST_HORIZONS = os.environ.get("FORECAST_HORIZONS", "15,60")

# URIs des modèles par horizon (modèles pré-entraînés sur GCS)
GCS_MODEL_URI_T15 = os.environ.get("GCS_MODEL_URI_T15")
GCS_MODEL_URI_T60 = os.environ.get("GCS_MODEL_URI_T60")

# ─────────────────────────────────────────────
#  Utilitaires temporels
# ─────────────────────────────────────────────
def _floor_5min(dt: datetime) -> datetime:
    """
    Arrondir une datetime aware à la borne inférieure de 5 minutes.

    Exemple
    -------
    12:07:23 → 12:05:00
    12:10:00 → 12:10:00
    """
    m = (dt.minute // 5) * 5
    return dt.replace(minute=m, second=0, microsecond=0)

def _iter_hours(start: datetime, end: datetime):
    """
    Itérer sur les heures pleines entre deux datetimes (inclus).

    Chaque datetime émise est tronquée à l’heure (minutes/secondes/microsecondes = 0).

    Utilisé pour construire le chemin de partition horaire :
      bronze/date=YYYY-MM-DD/hour=HH/
    """
    cur = start.replace(minute=0, second=0, microsecond=0)
    last = end.replace(minute=0, second=0, microsecond=0)
    while cur <= last:
        yield cur
        cur += timedelta(hours=1)

# ─────────────────────────────────────────────
#  Utilitaires GCS
# ─────────────────────────────────────────────
def _parse_gs(uri: str) -> Tuple[str, str]:
    """
    Découper une URI GCS de la forme `gs://bucket/path` en (bucket, key).

    Lève
    ----
    AssertionError
        Si l’URI ne commence pas par `gs://`.
    """
    assert uri.startswith("gs://"), f"bad GCS uri: {uri}"
    bkt, key = uri[5:].split("/", 1)
    return bkt, key

def _upload_bytes(cli: storage.Client, data: bytes, dest_uri: str, content_type: str = "application/json") -> None:
    """
    Uploader un payload de bytes vers une URI GCS à l’aide du client fourni.

    Paramètres
    ----------
    cli : google.cloud.storage.Client
        Instance de client GCS.
    data : bytes
        Données brutes à uploader.
    dest_uri : str
        URI GCS de destination (gs://bucket/path).
    content_type : str
        Type MIME (par défaut "application/json").
    """
    bkt, key = _parse_gs(dest_uri)
    cli.bucket(bkt).blob(key).upload_from_string(data, content_type=content_type)

def _list_raw_files_for_window(cli: storage.Client, start: datetime, end: datetime) -> List[str]:
    """
    Lister tous les fichiers parquet bruts dans le bucket d’ingestion (bronze)
    pour une fenêtre temporelle donnée.

    Le layout d’ingestion est :

        {RAW_PREFIX}/date=YYYY-MM-DD/hour=HH/...

    Pour chaque heure entre `start` et `end` (inclus), cette fonction :
      - dérive le préfixe de partition date/heure correspondant,
      - liste les blobs sous ce préfixe,
      - ne garde que les fichiers se terminant par `.parquet`.

    Retour
    ------
    list[str]
        Liste triée d’URIs `gs://bucket/path`.
    """
    bkt, pfx_root = _parse_gs(RAW_PREFIX)
    out: List[str] = []
    for h in _iter_hours(start, end):
        day = h.strftime("%Y-%m-%d"); hh = h.strftime("%H")
        prefix = f"{pfx_root}/date={day}/hour={hh}/"
        for b in cli.list_blobs(bkt, prefix=prefix):
            if b.name.endswith(".parquet"):
                out.append(f"gs://{bkt}/{b.name}")
    return sorted(out)

def _download_gs_files(cli: storage.Client, uris: List[str], dest_dir: Path) -> List[Path]:
    """
    Télécharger une liste d’URIs GCS localement dans un répertoire de travail.

    Paramètres
    ----------
    cli : google.cloud.storage.Client
        Instance de client GCS.
    uris : list[str]
        URIs GCS à télécharger.
    dest_dir : pathlib.Path
        Répertoire local où les fichiers seront écrits.

    Retour
    ------
    list[pathlib.Path]
        Liste des chemins locaux correspondant aux fichiers téléchargés.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for uri in uris:
        bkt, key = _parse_gs(uri)
        local = dest_dir / Path(key).name
        cli.bucket(bkt).blob(key).download_to_filename(str(local))
        paths.append(local)
    return paths

# ─────────────────────────────────────────────
#  IO / normalisation
# ─────────────────────────────────────────────
BASE_COLS = list(TRAIN_BASE_COLUMNS)  # aligné avec l’entraînement

def _to_naive_utc(series: pd.Series) -> pd.Series:
    """
    Convertir une série de timestamps en datetimes UTC naïves.

    Étapes
    ------
    1. Parser en datetime aware UTC (erreurs → NaT).
    2. Convertir explicitement en timezone UTC.
    3. Retirer l’info de timezone pour obtenir un datetime64[ns] naïf.
    """
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    return dt.dt.tz_convert("UTC").dt.tz_localize(None)

def _read_concat_parquets(files: List[Path]) -> pd.DataFrame:
    """
    Lire plusieurs fichiers parquet et retourner un DataFrame unique normalisé.

    Responsabilités
    ---------------
    - Lire tous les fichiers (en ignorant ceux corrompus avec un warning).
    - Normaliser les timestamps (`ts_utc`, `tbin_utc`) en UTC naïf.
    - Forcer le typage des colonnes numériques principales
      (bikes/capacity/mechanical/ebike, météo).
    - Normaliser les colonnes texte (`status`, `name`) en dtype string pandas.
    - Construire une colonne `station_id` robuste à partir de :
        station_id ou stationcode (si station_id manquant/vide).
    - Dédupliquer les lignes par (station_id, tbin_utc) en utilisant ts_utc
      comme tie-breaker (dernier échantillon conservé).
    - Garantir que toutes les colonnes de base d’entraînement (`BASE_COLS`)
      existent dans le DataFrame final, remplies par NA si manquantes.

    Retour
    ------
    pandas.DataFrame
        DataFrame restreint aux colonnes `BASE_COLS`.
    """
    dfs = []
    for p in files:
        try:
            df = pd.read_parquet(p)
            dfs.append(df)
        except Exception as e:
            print(f"[read] skip {p.name}: {e}")
    if not dfs:
        return pd.DataFrame(columns=BASE_COLS)

    df = pd.concat(dfs, ignore_index=True)

    # normalisation des colonnes temporelles
    if "ts_utc" in df.columns:
        df["ts_utc"] = _to_naive_utc(df["ts_utc"])
    if "tbin_utc" in df.columns:
        df["tbin_utc"] = _to_naive_utc(df["tbin_utc"])

    # numériques (ne pas toucher à station_id)
    for c in ["bikes", "capacity", "mechanical", "ebike"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["lat", "lon", "temp_C", "precip_mm", "wind_mps"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # texte
    if "status" in df.columns:
        df["status"] = df["status"].astype("string")
    if "name" in df.columns:
        df["name"] = df["name"].astype("string")

    # station_id robuste
    if "station_id" not in df.columns:
        if "stationcode" in df.columns:
            df["station_id"] = df["stationcode"].astype("string")
        else:
            df["station_id"] = pd.NA
    df["station_id"] = df["station_id"].astype("string")
    if "stationcode" in df.columns:
        sc = df["stationcode"].astype("string")
        m_empty = df["station_id"].isna() | (df["station_id"].str.strip() == "")
        df.loc[m_empty, "station_id"] = sc

    # déduplication
    if set(["station_id", "tbin_utc", "ts_utc"]).issubset(df.columns):
        df = df.sort_values(["station_id", "tbin_utc", "ts_utc"])
        df = df.groupby(["station_id", "tbin_utc"], as_index=False, dropna=True).tail(1).reset_index(drop=True)

    # colonnes attendues
    for c in BASE_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    return df[BASE_COLS]

# ─────────────────────────────────────────────
#  Helpers de construction de features (par station)
# ─────────────────────────────────────────────
def _build_one_station_full(st_df: pd.DataFrame) -> pd.Series:
    """
    Construire un vecteur de features compact, au format entraînement,
    pour une station donnée.

    Hypothèses
    ----------
    - `st_df` contient la fenêtre historique complète (≥ 48 bins) pour
      une seule station.
    - `st_df` est trié par `tbin_utc`.

    La série retournée contient :
      - identification / bin courant :
          station_id, tbin_latest, capacity_bin, occ_ratio_bin
      - composition des vélos :
          mechanical, ebike
      - statique / géo :
          lat, lon
      - météo :
          temp_C, precip_mm, wind_mps, et leurs valeurs à lag 1
      - signaux historiques :
          lag_bikes_{1,2,3,6,12,24,48}
          roll_mean_{3,6,12}, roll_std_{3,6,12}
          trend_nb_12b, trend_occ_12b (pente par pas de 5 min
          sur les 12 derniers bins)
      - features calendaires (UTC et Paris) :
          hour, minute, dow, month, is_weekend,
          hod_sin/hod_cos, dow_sin/dow_cos,
          paris_hour, paris_dow, paris_is_we

    Notes
    -----
    - Toutes les tendances utilisent une fenêtre sans fuite :
      calculées sur des données strictement antérieures au dernier bin.
    - Les features météo en lag 1 sont basées sur la série décalée d’un bin.
    """
    st_df = st_df.sort_values("tbin_utc").copy()

    cur = st_df.iloc[-1]
    st_id = cur.get("station_id", np.nan)
    tbin_latest = pd.to_datetime(cur["tbin_utc"], errors="coerce")

    bikes = pd.to_numeric(st_df["bikes"], errors="coerce")
    capacity = pd.to_numeric(st_df["capacity"], errors="coerce")

    # lags
    def lag_last(s: pd.Series, n: int):
        """
        Retourner la valeur en position -1 de `s` décalé de n pas,
        si disponible. Sinon retourne NaN.
        """
        return s.shift(n).iloc[-1] if len(s) >= (n + 1) else np.nan
    lag_set = (1, 2, 3, 6, 12, 24, 48)
    lag_vals = {f"lag_bikes_{L}": lag_last(bikes, L) for L in lag_set}

    # moyennes mobiles (sans fuite → décalage de 1)
    roll_vals: Dict[str, float] = {}
    for W in (3, 6, 12):
        s_shift = bikes.shift(1)
        roll_vals[f"roll_mean_{W}"] = s_shift.rolling(W, min_periods=max(1, W // 2)).mean().iloc[-1]
        roll_vals[f"roll_std_{W}"]  = s_shift.rolling(W, min_periods=max(1, W // 2)).std().iloc[-1]

    # tendances (12 bins)
    def _slope_per_5m(ts: pd.Series, y: pd.Series) -> float:
        """
        Calculer la pente linéaire de y en fonction du temps exprimé
        en pas de 5 minutes.

        Paramètres
        ----------
        ts : pandas.Series
            Série temporelle (type datetime).
        y : pandas.Series
            Série numérique.

        Retour
        ------
        float
            Pente par pas de 5 minutes. NaN si pas assez de points
            ou variance nulle.
        """
        t = pd.to_datetime(ts, errors="coerce")
        m = t.notna() & y.notna()
        if m.sum() < 2:
            return np.nan
        x = t[m].astype("datetime64[s]").astype("int64").astype(np.float64) / (BIN_MINUTES * 60.0)
        yy = y[m].astype(float).to_numpy()
        vx = np.var(x)
        if vx == 0:
            return np.nan
        return float(np.cov(x, yy, ddof=0)[0, 1] / vx)

    win_nb  = st_df.tail(13).iloc[:-1]
    win_occ = st_df.tail(13).iloc[:-1]
    trend_nb_12b  = _slope_per_5m(win_nb["tbin_utc"],  win_nb["bikes"]) if "bikes" in win_nb.columns else np.nan
    trend_occ_12b = _slope_per_5m(
        win_occ["tbin_utc"],
        (win_occ["bikes"] / win_occ["capacity"].where(win_occ["capacity"] > 0))
    ) if {"bikes", "capacity"}.issubset(win_occ.columns) else np.nan

    # ratios & météo (lag 1)
    occ_ratio_bin = (
        cur.get("bikes", np.nan) / cur.get("capacity", np.nan)
        if pd.notna(cur.get("capacity", np.nan)) and cur.get("capacity", 0) > 0
        else np.nan
    )
    temp_C    = pd.to_numeric(st_df["temp_C"], errors="coerce")
    precip_mm = pd.to_numeric(st_df["precip_mm"], errors="coerce")
    wind_mps  = pd.to_numeric(st_df["wind_mps"], errors="coerce")
    temp_C_lag1    = lag_last(temp_C, 1)
    precip_mm_lag1 = lag_last(precip_mm, 1)
    wind_mps_lag1  = lag_last(wind_mps, 1)

    # features calendaires (UTC + Paris)
    cal = pd.DataFrame({"tbin_utc": [tbin_latest]})
    cal = add_time_features(cal, ts_col="tbin_utc", add_paris_derived=True).iloc[0].to_dict()

    status = cur.get("status", None)

    out = {
        "station_id":      st_id,
        "tbin_latest":     tbin_latest,
        "capacity_bin":    cur.get("capacity", np.nan),
        "occ_ratio_bin":   occ_ratio_bin,
        "occ_ratio":       occ_ratio_bin,
        "mechanical":      cur.get("mechanical", np.nan),
        "ebike":           cur.get("ebike", np.nan),
        "lat":             cur.get("lat", np.nan),
        "lon":             cur.get("lon", np.nan),
        "temp_C":          cur.get("temp_C", np.nan),
        "precip_mm":       cur.get("precip_mm", np.nan),
        "wind_mps":        cur.get("wind_mps", np.nan),
        "temp_C_lag1":     temp_C_lag1,
        "precip_mm_lag1":  precip_mm_lag1,
        "wind_mps_lag1":   wind_mps_lag1,
        "trend_nb_12b":    trend_nb_12b,
        "trend_occ_12b":   trend_occ_12b,
        "status":          status,     # encodé en status_code plus loin
        # calendaires
        "hour":        cal.get("hour"),
        "minute":      cal.get("minute"),
        "dow":         cal.get("dow"),
        "month":       cal.get("month"),
        "is_weekend":  cal.get("is_weekend"),
        "hod_sin":     cal.get("hod_sin"),
        "hod_cos":     cal.get("hod_cos"),
        "dow_sin":     cal.get("dow_sin"),
        "dow_cos":     cal.get("dow_cos"),
        "paris_hour":  cal.get("paris_hour"),
        "paris_dow":   cal.get("paris_dow"),
        "paris_is_we": cal.get("paris_is_we"),
    }
    out.update(lag_vals)
    out.update(roll_vals)
    return pd.Series(out)

def _build_features(df: pd.DataFrame, start_tbin: datetime, end_tbin: datetime) -> pd.DataFrame:
    """
    Construire une ligne de features par station pour la fenêtre [start_tbin, end_tbin].

    Étapes
    ------
    1. Filtrer le DataFrame brut sur la fenêtre temporelle et ne garder
       que les colonnes nécessaires.
    2. Grouper par `station_id` et appeler `_build_one_station_full` sur
       chaque groupe.
    3. Fixer un typage robuste pour station_id (string) et les datetimes.
    4. Encoder `status_code` en entier (encodage catégoriel de `status`).

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame brut normalisé (bronze).
    start_tbin : datetime
        Borne inférieure (UTC naïf) de la fenêtre historique.
    end_tbin : datetime
        Borne supérieure (UTC naïf) de la fenêtre historique.

    Retour
    ------
    pandas.DataFrame
        DataFrame de features avec une ligne par station.
    """
    # sélection de la fenêtre historique pour calculer les features
    m = (df["tbin_utc"] >= start_tbin) & (df["tbin_utc"] <= end_tbin)
    cols_needed = [
        "tbin_utc", "station_id", "bikes", "capacity", "mechanical", "ebike", "status",
        "lat", "lon", "temp_C", "precip_mm", "wind_mps"
    ]
    dfw = df.loc[m, cols_needed].dropna(subset=["station_id", "tbin_utc"]).copy()
    if dfw.empty:
        return pd.DataFrame()

    # Important : définir station_id à partir de la clé du groupby
    # (et non des valeurs de colonne éventuellement modifiées)
    def _build_with_key(g: pd.DataFrame) -> pd.Series:
        out = _build_one_station_full(g)
        out["station_id"] = str(g.name) if g.name is not None else pd.NA
        return out

    try:
        feats = (
            dfw.groupby("station_id", dropna=True, group_keys=False)
               .apply(_build_with_key, include_groups=False)  # pandas ≥ 2.2
               .reset_index(drop=True)
        )
    except TypeError:
        # Chemin de compatibilité pour les versions de pandas plus anciennes.
        feats = (
            dfw.groupby("station_id", dropna=True, group_keys=False)
               .apply(_build_with_key)
               .reset_index(drop=True)
        )

    # typage final
    feats["station_id"]  = feats["station_id"].astype("string")
    feats["tbin_latest"] = pd.to_datetime(feats["tbin_latest"], errors="coerce")

    # encodage de status_code
    if "status" in feats.columns:
        cats = sorted([s for s in feats["status"].dropna().unique()])
        status_map = {s: i for i, s in enumerate(cats)}
        feats["status_code"] = feats["status"].map(status_map).astype("Int64")
        feats = feats.drop(columns=["status"])

    try:
        print("[features] sample station_id:", feats["station_id"].head(3).tolist())
    except Exception:
        pass

    return feats

# ─────────────────────────────────────────────
#  Sanitation JSON
# ─────────────────────────────────────────────
def _to_jsonable(v):
    """
    Convertir une valeur scalaire en représentation compatible JSON.

    Règles
    ------
    - None → None
    - NaN / NA → None
    - numpy integer → int
    - numpy float   → float
    - pandas.Timestamp / datetime →
         chaîne ISO8601 en UTC (suffixe Z)
    - sinon → valeur inchangée
    """
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        dt = v
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    return v

def _records_jsonable(df: pd.DataFrame) -> list[dict]:
    """
    Convertir un DataFrame en liste d’enregistrements JSON-compatibles.

    - Les colonnes datetime sont converties en chaînes ISO en UTC avec suffixe "Z".
    - Toutes les valeurs scalaires passent par `_to_jsonable`.

    Paramètres
    ----------
    df : pandas.DataFrame

    Retour
    ------
    list[dict]
        Liste de dictionnaires sérialisables en JSON.
    """
    if df.empty:
        return []
    df2 = df.copy()
    for c in df2.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        s = pd.to_datetime(df2[c], utc=True, errors="coerce")
        df2[c] = s.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    recs = df2.to_dict(orient="records")
    return [{k: _to_jsonable(v) for k, v in r.items()} for r in recs]

# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main() -> int:
    """
    Point d’entrée CLI pour le job de Serving Forecast.

    Pipeline
    --------
    1. Résoudre `now_utc` :
         - si NOW_UTC_ISO est présent → parser en datetime UTC fixe,
         - sinon utiliser l’heure UTC courante.
    2. Calculer la fenêtre de features :
         - end_tbin_aware = _floor_5min(now_utc)
         - start_tbin_aware = end_tbin_aware - (WINDOW_BINS - 1)*5min
    3. Lister tous les fichiers parquet bruts sous GCS_RAW_PREFIX pour la fenêtre.
         - S’il n’y en a aucun → exit(0).
    4. Télécharger ces parquets dans /tmp, les lire et les normaliser.
    5. Construire une ligne de features par station avec `_build_features`.
         - S’il n’y a pas de features → exit(0).
    6. Si WITH_FORECAST est désactivé → s’arrêter après le calcul des features.
    7. Pour chaque horizon dans FORECAST_HORIZONS :
         - résoudre l’URI du modèle correspondant (GCS_MODEL_URI_T{H}),
         - appeler `predict_from_features_df` (core d’entraînement),
         - réaligner station_id et métadonnées, ajouter les métadonnées
           d’horizon et de modèle,
         - dériver bikes_pred_int à partir de bikes_pred si nécessaire,
         - convertir en enregistrements JSON-compatibles.
    8. Uploader les bundles JSON par horizon vers :
         SERVING_FORECAST_PREFIX/h{H}/latest.json

    Retour
    ------
    int
        0 en cas de succès (même si certains horizons sont vides ou ignorés).
    """
    # 0) now / fenêtre de features
    if "NOW_UTC_ISO" in os.environ:
        now_utc = datetime.fromisoformat(os.environ["NOW_UTC_ISO"].replace("Z", "+00:00")).astimezone(timezone.utc)
    else:
        now_utc = datetime.now(timezone.utc)

    end_tbin_aware = _floor_5min(now_utc)
    start_tbin_aware = end_tbin_aware - timedelta(minutes=BIN_MINUTES * (WINDOW_BINS - 1))
    end_naive   = end_tbin_aware.replace(tzinfo=None)
    start_naive = start_tbin_aware.replace(tzinfo=None)

    print(f"[features_4h][cfg] RAW={RAW_PREFIX} WIN_H={WINDOW_HOURS} BINS={WINDOW_BINS}", flush=True)
    print(f"[features_4h] window UTC: {start_tbin_aware.isoformat()} → {end_tbin_aware.isoformat()} (inclusive)", flush=True)

    cli = storage.Client()

    # 1) listage + 2) téléchargement
    gcs_files = _list_raw_files_for_window(cli, start_tbin_aware, end_tbin_aware)
    print(f"[features_4h] gcs files found = {len(gcs_files)}", flush=True)
    if not gcs_files:
        print("[features_4h] no raw files in window — exit 0", flush=True)
        return 0

    work = Path("/tmp/features_4h_raw"); shutil.rmtree(work, ignore_errors=True)
    local_files = _download_gs_files(cli, gcs_files, work)
    print(f"[features_4h] local files = {len(local_files)}", flush=True)

    # 3) lecture + features
    df = _read_concat_parquets(local_files)
    feats = _build_features(df, start_naive, end_naive)
    print(f"[features_4h] features rows={len(feats):,}", flush=True)
    if feats.empty:
        print("[features_4h] no features → no forecast", flush=True)
        return 0

    # 4) inférence → JSONs par horizon
    if not WITH_FORECAST:
        print("[forecast] WITH_FORECAST disabled — nothing to do", flush=True)
        return 0
    if not SERVING_FORECAST_PREFIX:
        raise RuntimeError("SERVING_FORECAST_PREFIX is required to write latest.json")

    def _model_uri_for(hmin: int) -> str | None:
        """
        Retourner l’URI de modèle configurée pour un horizon donné (en minutes).

        Actuellement pris en charge :
          - 15 → GCS_MODEL_URI_T15
          - 60 → GCS_MODEL_URI_T60

        De futurs horizons peuvent être ajoutés en étendant ce mapping.
        """
        if hmin == 15 and GCS_MODEL_URI_T15:
            return GCS_MODEL_URI_T15
        if hmin == 60 and GCS_MODEL_URI_T60:
            return GCS_MODEL_URI_T60
        # futurs horizons à ajouter ici (ex. 120 → GCS_MODEL_URI_T120)
        return None  # horizon non configuré → ignoré

    horizons_min = [int(x.strip()) for x in FORECAST_HORIZONS.split(",") if x.strip()]
    consolidated: Dict[str, list] = {}
    generated_at = end_tbin_aware.isoformat().replace("+00:00", "Z")

    for hmin in horizons_min:
        uri = _model_uri_for(hmin)
        if not uri:
            print(f"[forecast][skip] no model configured for h={hmin} — skipping")
            consolidated[str(hmin)] = []
            continue

        # Inférence principale : appel aux utilitaires du pipeline d’entraînement.
        preds = predict_from_features_df(
            feats_df=feats,
            model_uri=uri,
            horizon_bins=max(1, hmin // 5),  # 15→3 bins, 60→12 bins
            model_alias=None,
        )

        if preds.empty:
            print(f"[forecast] empty preds for h={hmin}")
            consolidated[str(hmin)] = []
            continue

        # réalignement positionnel (sécurité)
        preds = preds.reset_index(drop=True).copy()
        feats_idx = feats.reset_index(drop=True).copy()

        # s’assurer que station_id vient des features (string)
        preds["station_id"] = feats_idx["station_id"].astype("string")

        # champs utiles pour l’UI
        if "tbin_latest" not in preds.columns:
            preds["tbin_latest"] = feats_idx["tbin_latest"].values
        if "capacity_bin" not in preds.columns and "capacity_bin" in feats_idx.columns:
            preds["capacity_bin"] = feats_idx["capacity_bin"].values

        # champs méta
        preds["horizon_min"] = hmin
        preds["pred_ts_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if "model_version" not in preds.columns:
            try:
                preds["model_version"] = uri.rsplit("/", 1)[-1].replace(".joblib", "")
            except Exception:
                preds["model_version"] = f"model_h{hmin}"

        # prédiction entière de vélos si seule une version float est présente
        if "bikes_pred_int" not in preds.columns and "bikes_pred" in preds.columns:
            preds["bikes_pred_int"] = (
                np.rint(pd.to_numeric(preds["bikes_pred"], errors="coerce"))
                .clip(lower=0)
                .astype("Int64")
            )

        try:
            print(
                "[forecast][sample]",
                preds[["station_id", "bikes_pred", "bikes_pred_int"]].head(3).to_dict("records"),
            )
        except Exception:
            pass

        consolidated[str(hmin)] = _records_jsonable(preds)

    # 5) upload par horizon : serving/h15/latest.json, serving/h60/latest.json
    for hmin, recs in consolidated.items():
        sub_bundle = {
            "generated_at": generated_at,
            "horizon_min": int(hmin),
            "data": recs,
        }
        dest_uri = f"{SERVING_FORECAST_PREFIX.rstrip('/')}/h{hmin}/latest.json"
        _upload_bytes(cli, json.dumps(sub_bundle, ensure_ascii=False).encode("utf-8"), dest_uri)
        print(f"[forecast] uploaded → {dest_uri}", flush=True)

    print(f"[forecast] done: {len(consolidated)} horizons uploaded", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())