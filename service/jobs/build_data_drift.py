# service/jobs/build_data_drift.py

"""
Vélib’ Forecast — Data Drift builder (Schéma v1.4)

Ce job :
- lit les exports d’événements (events_YYYY-MM-DD.parquet) depuis le bucket exports ;
- normalise le schéma des événements (timestamps, station_id, numériques) ;
- découpe les données en fenêtre **référence** et fenêtre **courante** ;
- calcule, pour chaque feature, des métriques de dérive :
    - Population Stability Index (PSI)
    - Statistique de Kolmogorov–Smirnov (KS)
    - Deltas standardisés de moyenne et de variance
- calcule un proxy de PSI global et une EMA journalière sur `occ_ratio` ;
- produit des métriques de dérive par zone (PSI par zone spatiale) pour les cartes ;
- écrit un bundle JSON LATEST-only pour l’UI de monitoring :

  {GCS_MONITORING_PREFIX}/monitoring/data/drift/latest/
    - psi_by_feature.json
    - ks_by_feature.json
    - deltas_by_feature.json
    - psi_global_daily_ema.json
    - summary.json
    - alerts.json
    - bounds.json
    - zones.json
    - features_detected.json

Environnement
-------------
Noms de variables d’environnement unifiés (single source of truth) :

- GCS_EXPORTS_PREFIX
    gs://.../velib/exports
    Requis. Préfixe racine contenant les events_YYYY-MM-DD.parquet.

- GCS_MONITORING_PREFIX
    gs://.../velib[/monitoring]
    Requis. Préfixe racine pour les artefacts de monitoring ; '/monitoring' est
    ajouté automatiquement s’il manque.

- MON_TZ
    Fuseau horaire du monitoring (ex. "Europe/Paris").
    Défaut : "Europe/Paris".

- MON_LAST_DAYS
    Taille de la fenêtre courante de dérive (en jours). Exemple : 14.

- MON_REF_DAYS
    Taille de la fenêtre de référence de dérive (en jours). Exemple : 28.

- FORECAST_HORIZONS
    Liste d’horizons de prévision séparés par des virgules ("15,60").
    Non utilisé dans ce job, conservé pour homogénéité d’interface.

Principes de design de la dérive
--------------------------------
- Seules les **vraies features** sont utilisées pour la dérive (pas de timestamp
  brut, pas de lat/lon, pas d’encodages sin/cos ou d’indicateurs temporels dérivés).
- PSI / KS sont calculés sur des moyennes journalières par station.
- Les features numériques doivent avoir au moins 5 valeurs valides dans les
  fenêtres ref & courante, sinon la feature est ignorée avec un log.
- Le PSI global est :
    - le PSI sur `occ_ratio` si présent,
    - sinon la médiane des PSI de toutes les features.
- Des alertes sont levées selon des seuils de PSI global :
    - >= 0.25 → high
    - >= 0.10 → medium
"""

from __future__ import annotations
import os, re, json, sys
from io import BytesIO
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception as e:
    raise RuntimeError("pyarrow is required") from e

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage is required") from e

SCHEMA_VERSION = "1.4"

# ──────────────────────────────────────────────────────────────────────────────
# ENV — noms unifiés
# ──────────────────────────────────────────────────────────────────────────────

def _env(name: str, default=None):
    """
    Lire une variable d’environnement, avec valeur par défaut si absente ou vide.

    Paramètres
    ----------
    name : str
        Nom de la variable d’environnement.
    default :
        Valeur par défaut à renvoyer si la variable est absente ou vide.

    Retour
    ------
    Any
        Valeur de la variable (string) ou la valeur par défaut.
    """
    v = os.environ.get(name)
    return v if (v is not None and v != "") else default


def _env_int(name: str, default: int) -> int:
    """
    Lire une variable d’environnement entière avec une valeur par défaut.

    Paramètres
    ----------
    name : str
        Nom de la variable d’environnement.
    default : int
        Valeur entière par défaut si le parsing échoue ou si la variable manque.

    Retour
    ------
    int
        Valeur entière parsée.
    """
    try:
        return int(_env(name, default))
    except Exception:
        return default


def _env_list_int(name: str, default_csv: str) -> list[int]:
    """
    Lire une liste d’entiers séparés par des virgules depuis une variable d’env.

    Paramètres
    ----------
    name : str
        Nom de la variable d’environnement.
    default_csv : str
        CSV par défaut à utiliser si la variable est absente.

    Retour
    ------
    list[int]
        Liste d’entiers parsés ; les tokens invalides sont ignorés.
    """
    raw = str(_env(name, default_csv))
    out: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except Exception:
            pass
    return out

GCS_EXPORTS_PREFIX    = _env("GCS_EXPORTS_PREFIX")
GCS_MONITORING_PREFIX = _env("GCS_MONITORING_PREFIX")
MON_TZ                = _env("MON_TZ", "Europe/Paris")
MON_LAST_DAYS         = _env_int("MON_LAST_DAYS", 14)
MON_REF_DAYS          = _env_int("MON_REF_DAYS", 28)
FORECAST_HORIZONS     = _env_list_int("FORECAST_HORIZONS", "15,60")  # (non utilisé ici)

# Vérifications de base GCS
if not (GCS_EXPORTS_PREFIX and GCS_EXPORTS_PREFIX.startswith("gs://")):
    raise RuntimeError("GCS_EXPORTS_PREFIX missing or invalid")
if not (GCS_MONITORING_PREFIX and GCS_MONITORING_PREFIX.startswith("gs://")):
    raise RuntimeError("GCS_MONITORING_PREFIX missing or invalid")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers GCS
# ──────────────────────────────────────────────────────────────────────────────

def _split(gs: str) -> Tuple[str, str]:
    """
    Découper une URI GCS `gs://bucket/path` en (bucket, key).

    Paramètres
    ----------
    gs : str
        URI GCS commençant par "gs://".

    Retour
    ------
    (str, str)
        Tuple (nom_du_bucket, clé_objet).
    """
    assert gs.startswith("gs://"), f"bad GCS uri: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k.rstrip("/")


def _read_parquet_blob_to_df(blob: "storage.Blob") -> pd.DataFrame:
    """
    Télécharger un blob Parquet depuis GCS et le charger en DataFrame pandas.

    Paramètres
    ----------
    blob : google.cloud.storage.Blob
        Blob du fichier Parquet.

    Retour
    ------
    pandas.DataFrame
        DataFrame chargé depuis le contenu Parquet.
    """
    buf = BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    tbl = pq.read_table(buf)
    return tbl.to_pandas()


def _list_event_blobs(exports_prefix: str, start_date: datetime, end_date: datetime) -> List["storage.Blob"]:
    """
    Lister les blobs d’événements (events_YYYY-MM-DD.parquet) sur une fenêtre donnée.

    Paramètres
    ----------
    exports_prefix : str
        Préfixe GCS où sont stockés les events_*.parquet (GCS_EXPORTS_PREFIX).
    start_date : datetime
        Date UTC de début (incluse).
    end_date : datetime
        Date UTC de fin (incluse).

    Retour
    ------
    list[google.cloud.storage.Blob]
        Liste triée de blobs correspondant aux contraintes de date.
    """
    bkt, key_prefix = _split(exports_prefix)
    client = storage.Client()
    bucket = client.bucket(bkt)
    blobs = list(client.list_blobs(bucket, prefix=key_prefix.strip("/") + "/"))
    pat = re.compile(r"events_(\d{4}-\d{2}-\d{2})\.parquet$")
    out: List["storage.Blob"] = []
    for bl in blobs:
        m = pat.search(bl.name)
        if not m:
            continue
        d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        if start_date.date() <= d <= end_date.date():
            out.append(bl)
    out.sort(key=lambda b: b.name)
    return out


def _upload_json_gs(obj: dict | list, gs_uri: str):
    """
    Uploader un objet JSON-sérialisable en fichier JSON sur GCS.

    Notes
    -----
    - Tous les floats sont nettoyés pour éviter NaN/Inf (remplacés par null).
    - Le JSON est encodé en UTF-8 avec des séparateurs compacts.

    Paramètres
    ----------
    obj : dict ou list
        Objet JSON-sérialisable à uploader.
    gs_uri : str
        URI GCS de destination (gs://bucket/path/to/file.json).
    """
    def _san(o):
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_san(v) for v in o]
        if isinstance(o, float):
            return float(o) if np.isfinite(o) else None
        return o
    safe = _san(obj)
    bkt, key = _split(gs_uri)
    data = json.dumps(safe, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    storage.Client().bucket(bkt).blob(key).upload_from_string(data, content_type="application/json")
    print(f"[data.drift] wrote → {gs_uri} ({len(data):,} bytes)")

# ──────────────────────────────────────────────────────────────────────────────
# Normalisation des events (schéma build_datasets.py)
# ──────────────────────────────────────────────────────────────────────────────

KEY_STR = {"station_id", "status", "name"}
KEY_TIME = {"tbin_utc", "ts", "timestamp"}

# Exclusions explicites pour les features de dérive
EXCLUDE_EXACT = {
    # identifiants & time-like
    "station_id", "date_local", "dow", "hour", "h", "min", "ts_local",
    # géo
    "lat", "lon",
    # on n’utilise jamais les timestamps bruts comme features
    "tbin_utc", "ts", "timestamp",
}
EXCLUDE_PATTERNS = [
    r".*_(sin|cos)$",
    r"^(hour|minute|dow)(_|$)",
]

def _is_time_or_coord(col: str) -> bool:
    """
    Retourne True si une colonne est time-like ou coordonnée (à exclure).

    Règles
    ------
    - Exclure les noms explicites (station_id, lat, lon, tbin_utc, etc.).
    - Exclure les colonnes qui matchent EXCLUDE_PATTERNS (ex. *_sin, *_cos, hour_*).
    """
    if col in EXCLUDE_EXACT:
        return True
    return any(re.match(p, col) for p in EXCLUDE_PATTERNS)


def _ensure_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliser le DataFrame des événements et préserver les types critiques.

    Pré-requis
    ----------
    - Doit contenir au minimum 'tbin_utc' et 'station_id'.

    Transformations
    ---------------
    - 'station_id' → string.
    - 'tbin_utc'   → datetime64[ns] naïf (UTC).
    - les colonnes datetime sont laissées en l’état ;
    - les colonnes numériques sont laissées en l’état ;
    - les autres colonnes object/bool sont converties en numérique si possible.

    Paramètres
    ----------
    df : pandas.DataFrame
        Données brutes des événements.

    Retour
    ------
    pandas.DataFrame
        DataFrame d’événements normalisé.
    """
    if not {"tbin_utc", "station_id"}.issubset(df.columns):
        raise RuntimeError("Missing minimal columns: tbin_utc/station_id")

    out = df.copy()

    # station_id en string
    out["station_id"] = out["station_id"].astype("string")

    # tbin_utc en datetime UTC naïf
    out["tbin_utc"] = pd.to_datetime(out["tbin_utc"], errors="coerce", utc=True).dt.tz_convert(None)

    # Pour chaque colonne : coercer object/bool en numérique, laisser datetime/numérique
    for c in out.columns:
        if c in KEY_STR or c in KEY_TIME:
            continue  # station_id/status/name et timestamps intouchés
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            continue
        try:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        except Exception:
            pass

    return out


def _to_local(df: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    """
    Ajouter des colonnes de temps local à partir de tbin_utc (UTC).

    Colonnes ajoutées
    -----------------
    - date_local : date calendaire locale
    - dow        : jour de semaine (0=lundi)
    - hour       : heure locale
    - ts_local   : timestamp localisé

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame d’entrée avec la colonne 'tbin_utc'.
    tz : str ou None
        Nom du fuseau horaire (ex. 'Europe/Paris'). Si None, reste en UTC.

    Retour
    ------
    pandas.DataFrame
        DataFrame avec colonnes de temps local supplémentaires.
    """
    dt = pd.to_datetime(df["tbin_utc"], errors="coerce", utc=True)
    dt_local = dt.dt.tz_convert(tz) if tz else dt
    return df.assign(date_local=dt_local.dt.date, dow=dt_local.dt.dayofweek, hour=dt_local.dt.hour, ts_local=dt_local)

# ──────────────────────────────────────────────────────────────────────────────
# Métriques de dérive
# ──────────────────────────────────────────────────────────────────────────────

def _valid_series(x) -> pd.Series:
    """
    Convertir une entrée en Series 1D numérique sans NaN.

    Paramètres
    ----------
    x : array-like
        Valeurs en entrée.

    Retour
    ------
    pandas.Series
        Série numérique nettoyée ; série vide si la coercition échoue.
    """
    try:
        s = pd.to_numeric(pd.Series(x, copy=False), errors="coerce").dropna()
        return s if isinstance(s, pd.Series) else pd.Series([], dtype=float)
    except Exception:
        return pd.Series([], dtype=float)


def _psi_continuous(ref: pd.Series, cur: pd.Series, bins: int = 20, eps: float = 1e-9) -> float:
    """
    Calculer un PSI (Population Stability Index) binned entre deux échantillons.

    Étapes
    ------
    - Nettoyer les deux échantillons avec _valid_series.
    - Construire des bins basés sur les quantiles de l’échantillon de référence.
    - Calculer les distributions (p_ref, p_cur) sur ces bins.
    - PSI = somme((p_ref - p_cur) * log(p_ref / p_cur)).

    Paramètres
    ----------
    ref : pandas.Series
        Échantillon de référence.
    cur : pandas.Series
        Échantillon courant.
    bins : int, défaut 20
        Nombre cible de bins de quantiles.
    eps : float, défaut 1e-9
        Petite valeur ajoutée aux probabilités pour la stabilité numérique.

    Retour
    ------
    float
        Valeur de PSI ; NaN si données insuffisantes ou dégénérées.
    """
    a = _valid_series(ref)
    b = _valid_series(cur)
    if len(a) < 5 or len(b) < 5:
        return np.nan
    if a.min() == a.max() or b.min() == b.max():
        return np.nan
    try:
        q = np.unique(np.nanquantile(a, np.linspace(0, 1, bins + 1)))
    except Exception:
        return np.nan
    if q.size < 3:
        return np.nan
    ca, _ = np.histogram(a, bins=q)
    cb, _ = np.histogram(b, bins=q)
    if ca.sum() == 0 or cb.sum() == 0:
        return np.nan
    pa = (ca / ca.sum()).astype(float) + eps
    pb = (cb / cb.sum()).astype(float) + eps
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def _ks_stat(ref: pd.Series, cur: pd.Series) -> float:
    """
    Calculer une statistique de Kolmogorov–Smirnov discrétisée entre deux échantillons.

    Détails d’implémentation
    ------------------------
    - Utilise les quantiles combinés des deux échantillons comme grille.
    - Approxime les CDF via des histogrammes.
    - Retourne la différence absolue maximale entre les deux CDF.

    Paramètres
    ----------
    ref : pandas.Series
        Échantillon de référence.
    cur : pandas.Series
        Échantillon courant.

    Retour
    ------
    float
        Statistique KS ; NaN si données insuffisantes ou dégénérées.
    """
    a = _valid_series(ref)
    b = _valid_series(cur)
    if len(a) < 5 or len(b) < 5:
        return np.nan
    both = pd.concat([a, b], ignore_index=True)
    if both.empty or both.min() == both.max():
        return np.nan
    try:
        q = np.unique(np.nanquantile(both, np.linspace(0, 1, 201)))
    except Exception:
        return np.nan
    if q.size < 3:
        return np.nan
    ca, _ = np.histogram(a, bins=q)
    cb, _ = np.histogram(b, bins=q)
    if ca.sum() == 0 or cb.sum() == 0:
        return np.nan
    cdfa = np.cumsum(ca) / ca.sum()
    cdfb = np.cumsum(cb) / cb.sum()
    return float(np.max(np.abs(cdfa - cdfb)))


def _delta_mean_var(ref: pd.Series, cur: pd.Series) -> tuple[float, float]:
    """
    Calculer les deltas standardisés de moyenne et de variance entre deux échantillons.

    Définitions
    -----------
    - dm = (mean_cur - mean_ref) / std_ref
    - dv = (var_cur - var_ref) / var_ref

    Paramètres
    ----------
    ref : pandas.Series
        Échantillon de référence.
    cur : pandas.Series
        Échantillon courant.

    Retour
    ------
    (float, float)
        Tuple (delta_mean, delta_var), possiblement NaN si données insuffisantes.
    """
    a = _valid_series(ref)
    b = _valid_series(cur)
    if len(a) < 5 or len(b) < 5:
        return (np.nan, np.nan)
    if a.std(ddof=1) == 0:
        return (np.nan, np.nan)
    dm = (b.mean() - a.mean()) / (a.std(ddof=1) + 1e-9)
    avar = a.var(ddof=1)
    dv = (b.var(ddof=1) - avar) / (avar + 1e-9)
    return (float(dm), float(dv))


def _split_windows(df: pd.DataFrame, current_days: int, reference_days: int) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Timestamp]]:
    """
    Découper les événements en fenêtres temporelles de référence et courante.

    Fenêtres
    --------
    - tmax = max(tbin_utc)
    - fenêtre courante   : [tmax - current_days, tmax]
    - fenêtre référence  : [tmax - current_days - reference_days, tmax - current_days)

    Paramètres
    ----------
    df : pandas.DataFrame
        Événements normalisés avec 'tbin_utc'.
    current_days : int
        Taille de la fenêtre courante (en jours).
    reference_days : int
        Taille de la fenêtre de référence (en jours).

    Retour
    ------
    (DataFrame, DataFrame, dict)
        Tuple (ref_df, cur_df, bounds) où bounds contient :
        - tmax
        - t_cur_start
        - t_ref_start
        - t_ref_end
    """
    if df.empty:
        return df.iloc[0:0].copy(), df.iloc[0:0].copy(), {}
    tmax = df["tbin_utc"].max()
    t_cur_start = tmax - pd.Timedelta(days=current_days)
    t_ref_end = t_cur_start
    t_ref_start = t_ref_end - pd.Timedelta(days=reference_days)
    ref = df[(df["tbin_utc"] >= t_ref_start) & (df["tbin_utc"] < t_ref_end)].copy()
    cur = df[(df["tbin_utc"] >= t_cur_start) & (df["tbin_utc"] <= tmax)].copy()
    bounds = {"tmax": tmax, "t_cur_start": t_cur_start, "t_ref_start": t_ref_start, "t_ref_end": t_ref_end}
    return ref, cur, bounds


def _assign_zone(df: pd.DataFrame) -> pd.Series:
    """
    Calculer un identifiant de "zone" générique par station pour la dérive spatiale.

    Stratégie
    ---------
    1. On privilégie un champ de zone sémantique s’il existe :
        - 'arrondissement', 'arr', 'zone', 'district'
    2. Sinon, on dérive une pseudo-zone depuis lat/lon par arrondi :
        - lat_rounded = round(lat * 100) / 100
        - lon_rounded = round(lon * 100) / 100
        - zone = "lat_rounded,lon_rounded"

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame avec éventuellement les colonnes 'lat' et 'lon'.

    Retour
    ------
    pandas.Series
        Série d’identifiants de zone (dtype object).
    """
    for c in ("arrondissement", "arr", "zone", "district"):
        if c in df.columns:
            return df[c].astype(str)
    lat = pd.to_numeric(df.get("lat"), errors="coerce")
    lon = pd.to_numeric(df.get("lon"), errors="coerce")
    mask = lat.notna() & lon.notna()
    z = pd.Series(index=df.index, dtype=object, name="zone")
    lat_r = (lat[mask] * 100).round() / 100.0
    lon_r = (lon[mask] * 100).round() / 100.0
    z.loc[mask] = lat_r.astype(str) + "," + lon_r.astype(str)
    return z


def _compute_drift(events: pd.DataFrame, current_days: int, reference_days: int, tz: Optional[str]) -> dict:
    """
    Calculer les métriques de dérive pour le dataset des événements.

    Étapes
    ------
    1. Ajouter les colonnes de temps local (date_local, dow, hour, ts_local).
    2. Découper en fenêtres référence et courante selon MON_REF_DAYS / MON_LAST_DAYS.
    3. Agréger les événements en moyennes journalières par station.
    4. Découvrir automatiquement les features numériques candidates
       (coord/time exclues).
    5. Pour chaque feature :
        - PSI (continu)
        - Statistique KS
        - Deltas standardisés de moyenne et de variance
    6. Calculer le PSI global (sur occ_ratio si dispo, sinon médiane des PSI).
    7. Calculer une EMA journalière sur occ_ratio pour la visualisation de tendance.
    8. Construire des métriques de PSI par zone (sur occ_ratio si dispo, sinon bikes).
    9. Construire les payloads de résumé et d’alertes.

    Paramètres
    ----------
    events : pandas.DataFrame
        DataFrame d’événements normalisé, tel que retourné par `_ensure_events`.
    current_days : int
        Nombre de jours de la fenêtre courante.
    reference_days : int
        Nombre de jours de la fenêtre de référence.
    tz : str ou None
        Fuseau horaire pour l’agrégation calendaire locale.

    Retour
    ------
    dict
        Dictionnaire contenant :
        - psi_df, ks_df, deltas_df : métriques par feature (DataFrames)
        - psi_daily_ema            : DataFrame d’EMA journalière
        - summary                  : dict de résumé global
        - alerts                   : liste de dicts d’alertes
        - bounds                   : dict avec les bornes de fenêtres UTC
        - zones                    : dict avec les lignes PSI par zone
        - feature_list             : liste des features numériques utilisées
    """
    df = _to_local(events, tz)
    ref, cur, bounds = _split_windows(df, current_days=current_days, reference_days=reference_days)

    # Agrégation par (date_local, station) — moyennes journalières
    def agg(d: pd.DataFrame):
        # Colonnes numériques candidates (coord/time exclues ici aussi)
        num_cols = [
            c for c in d.columns
            if c not in {"station_id","tbin_utc","status","name","date_local","dow","hour","ts_local"}
            and pd.api.types.is_numeric_dtype(d[c])
            and not _is_time_or_coord(c)
        ]
        # Garder lat/lon à part pour la carte des zones
        keep = ["date_local","station_id"] + num_cols + [c for c in ("lat","lon") if c in d.columns]
        return d[keep].groupby(["date_local","station_id"], dropna=True).mean(numeric_only=True).reset_index()

    ref = agg(ref); cur = agg(cur)

    # Auto features : intersection numérique, coord/time exclues
    common_num = [
        c for c in ref.columns
        if c not in {"date_local","station_id"} and c in cur.columns
        and pd.api.types.is_numeric_dtype(ref[c]) and not _is_time_or_coord(c)
    ]

    rows_psi, rows_ks, rows_delta = [], [], []
    for f in sorted(common_num):
        try:
            rf = _valid_series(ref[f]); cf = _valid_series(cur[f])
            if len(rf) < 5 or len(cf) < 5:
                print(f"[data.drift][info] skip feature '{f}' (insufficient data)")
                continue
            psi = _psi_continuous(rf, cf)
            ks  = _ks_stat(rf, cf)
            dm, dv = _delta_mean_var(rf, cf)
            rows_psi.append({"feature": f, "psi": float(psi) if np.isfinite(psi) else None})
            rows_ks.append({"feature": f, "ks":  float(ks)  if np.isfinite(ks)  else None})
            rows_delta.append({"feature": f,
                               "delta_mean": float(dm) if np.isfinite(dm) else None,
                               "delta_var":  float(dv) if np.isfinite(dv) else None})
        except Exception as e:
            print(f"[data.drift][warn] metrics failed for feature '{f}': {e}")

    psi_df = pd.DataFrame(rows_psi)
    ks_df = pd.DataFrame(rows_ks)
    d_df = pd.DataFrame(rows_delta)

    # EMA journalière sur occ_ratio (si dispo)
    ema_df = pd.DataFrame(columns=["date_local","psi_ema"])
    if "occ_ratio" in events.columns:
        by_day = df.groupby("date_local")["occ_ratio"].apply(
            lambda s: float(np.nanmean(pd.to_numeric(s, errors='coerce')))
        ).reset_index()
        by_day = by_day.sort_values("date_local")
        alpha = 2 / (7 + 1.0)  # EMA 7 jours
        ema, last = [], None
        for _, r in by_day.iterrows():
            x = r["occ_ratio"]
            if pd.isna(x):
                ema.append(np.nan); continue
            last = x if last is None else (alpha * x + (1 - alpha) * last)
            ema.append(last)
        ema_df = pd.DataFrame({"date_local": by_day["date_local"].astype(str), "psi_ema": ema})

    # PSI global
    psi_global = None
    if not psi_df.empty:
        if "occ_ratio" in list(psi_df["feature"]):
            v = psi_df.loc[psi_df["feature"]=="occ_ratio","psi"].values[0]
            psi_global = float(v) if v is not None and np.isfinite(v) else None
        else:
            v = np.nanmedian([p for p in psi_df["psi"] if p is not None])
            psi_global = float(v) if np.isfinite(v) else None

    alerts = []
    if psi_global is not None:
        if psi_global >= 0.25:
            alerts.append({"level": "high", "code": "psi_global_high", "text": f"High global PSI ({psi_global:.3f})"})
        elif psi_global >= 0.10:
            alerts.append({"level": "medium", "code": "psi_global_medium", "text": f"Moderate global PSI ({psi_global:.3f})"})

    summary = {
        "psi_global": psi_global,
        "top_feature": (psi_df.sort_values("psi", ascending=False).iloc[0]["feature"] if not psi_df.empty else None),
        "top_feature_psi": (float(psi_df.sort_values("psi", ascending=False).iloc[0]["psi"]) if not psi_df.empty and psi_df.sort_values("psi", ascending=False).iloc[0]["psi"] is not None else None),
    }

    # PSI par zone (occ_ratio si dispo, sinon bikes) — robuste
    zones_doc = {"rows": []}
    try:
        rows = []
        if {"lat", "lon"}.issubset(df.columns):
            ref_z = ref.assign(zone=_assign_zone(ref))
            cur_z = cur.assign(zone=_assign_zone(cur))
            for z, rsub in ref_z.groupby("zone", dropna=True):
                csub = cur_z[cur_z["zone"] == z]
                if csub.empty:
                    continue
                metric = None
                for m in ("occ_ratio", "bikes"):
                    if m in rsub.columns and m in csub.columns:
                        metric = m
                        break
                if not metric:
                    continue

                rvals = _valid_series(rsub[metric])
                cvals = _valid_series(csub[metric])
                if len(rvals) < 5 or len(cvals) < 5:
                    continue

                psi = _psi_continuous(rvals, cvals)
                lat = float(rsub["lat"].median()) if "lat" in rsub.columns and rsub["lat"].notna().any() else None
                lon = float(rsub["lon"].median()) if "lon" in rsub.columns and rsub["lon"].notna().any() else None
                zname = None if (z is None or (isinstance(z, float) and np.isnan(z))) else str(z)
                rows.append({"zone": zname, "psi": float(psi) if np.isfinite(psi) else None, "lat": lat, "lon": lon})
        zones_doc = {"rows": rows}
    except Exception as e:
        print(f"[data.drift] zones PSI failed: {e}")

    return {
        "psi_df": psi_df, "ks_df": ks_df, "deltas_df": d_df,
        "psi_daily_ema": ema_df,
        "summary": summary, "alerts": alerts,
        "bounds": {
            "tmax_utc": pd.Timestamp(bounds.get("tmax")).tz_localize("UTC").isoformat().replace("+00:00","Z") if bounds else None,
            "cur_start_utc": pd.Timestamp(bounds.get("t_cur_start")).tz_localize("UTC").isoformat().replace("+00:00","Z") if bounds else None,
            "ref_start_utc": pd.Timestamp(bounds.get("t_ref_start")).tz_localize("UTC").isoformat().replace("+00:00","Z") if bounds else None,
            "ref_end_utc": pd.Timestamp(bounds.get("t_ref_end")).tz_localize("UTC").isoformat().replace("+00:00","Z") if bounds else None,
        },
        "zones": zones_doc,
        "feature_list": sorted(common_num),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    """
    Entrypoint CLI pour le job de monitoring de data drift.

    Étapes
    ------
    1. Déterminer la fenêtre de lecture UTC :
       days = max(MON_LAST_DAYS, MON_REF_DAYS, 7).
    2. Lister et lire tous les events_YYYY-MM-DD.parquet dans cette fenêtre.
    3. Normaliser les events avec `_ensure_events`.
    4. Calculer les métriques de dérive via `_compute_drift`.
    5. Normaliser le préfixe de monitoring (s’assurer d’un segment `/monitoring`).
    6. Écrire un bundle JSON LATEST-only sous :

       {GCS_MONITORING_PREFIX}/monitoring/data/drift/latest

    Retour
    ------
    int
        Code de sortie (0 en cas de succès, 2 en cas d’erreur fatale).
    """
    # Fenêtre de lecture sur GCS = >= max(MON_LAST_DAYS, MON_REF_DAYS, 7)
    now = datetime.now(timezone.utc)
    window_days = max(int(MON_LAST_DAYS), int(MON_REF_DAYS), 7)
    start = (now - timedelta(days=window_days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"[data.drift] window UTC: {start.date()} → {now.date()} (days={window_days})")

    blobs = _list_event_blobs(GCS_EXPORTS_PREFIX, start, now)
    if not blobs:
        print("[data.drift] no event blobs in window — nothing to do")
        return 0

    frames: List[pd.DataFrame] = []
    for bl in blobs:
        print(f"[read] {bl.name}")
        try:
            df = _read_parquet_blob_to_df(bl)
            frames.append(df)
        except Exception as e:
            print(f"[warn] failed to read {bl.name}: {e}")

    if not frames:
        print("[data.drift] no readable data — nothing to do")
        return 0

    ev = pd.concat(frames, ignore_index=True)
    if ev.empty:
        print("[data.drift] events empty — nothing to do")
        return 0

    ev_norm = _ensure_events(ev)
    res = _compute_drift(ev_norm, current_days=int(MON_LAST_DAYS), reference_days=int(MON_REF_DAYS), tz=MON_TZ)

    # Normaliser le préfixe monitoring (ajout de /monitoring si absent)
    mon_base = GCS_MONITORING_PREFIX.rstrip("/")
    if not mon_base.endswith("/monitoring"):
        mon_base = mon_base + "/monitoring"
    out_prefix = f"{mon_base}/data/drift/latest"

    files = {
        "psi_by_feature.json":       res["psi_df"].to_dict(orient="records"),
        "ks_by_feature.json":        res["ks_df"].to_dict(orient="records"),
        "deltas_by_feature.json":    res["deltas_df"].to_dict(orient="records"),
        "psi_global_daily_ema.json": res["psi_daily_ema"].to_dict(orient="records"),
        "summary.json":              {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"), **res["summary"]},
        "alerts.json":               res["alerts"],
        "bounds.json":               res["bounds"],
        "zones.json":                res["zones"],
        "features_detected.json":    res["feature_list"],
    }
    for fname, payload in files.items():
        _upload_json_gs(payload, f"{out_prefix}/{fname}")

    print(f"[data.drift] done → {out_prefix}/ (LATEST only)")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[data.drift][fatal] {e}", file=sys.stderr)
        sys.exit(2)
