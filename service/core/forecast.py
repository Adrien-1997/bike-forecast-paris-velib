# service/core/forecast.py

# =============================================================================
# Training + inference utilities for Vélib' forecasting.
#
# ✅ Tout-en-un : cal_features, time_features et spatial_features sont intégrés ici
# ✅ Spatial TOUJOURS activé (aucune option/ENV nécessaire)
# ✅ API:
#    - train_model(src, horizon_bins=3, out_path='model.joblib', lookback_days=None, start_date=None, end_date=None)
#    - predict_from_features_df(feats_df, model_uri, horizon_bins)
#    - predict_latest_offline(src, model_path_or_prefix)
#
# CLI:
#    python -m service.core.forecast train --src <glob|dir|parquet> --horizon 3 [--start YYYY-MM-DD --end YYYY-MM-DD --lookback-days N]
#    python -m service.core.forecast predict --src <glob|dir|parquet> --model <joblib|prefix> [--horizon N]
# =============================================================================

"""
Vélib' Forecast — core training & inference module.

Ce module regroupe **tout** ce qui concerne :
    - la préparation des features temporelles + météo,
    - les features spatiales (statique + dynamiques KNN),
    - l'entraînement XGBoost avec versioning,
    - l'inférence en production (via `predict_from_features_df`),
    - la prédiction offline (via `predict_latest_offline`),
    - une petite CLI pour entraîner / tester en local.

Les blocs sont structurés en 3 couches :

(A) cal_features
    - `add_time_features`: encodage calendaire (hour, dow, sin/cos, proxies Paris).

(B) time_features
    - lecture/normalisation des parquets 5 min,
    - construction de la cible `y_nb`,
    - lags / rollings / météo / encodage status.

(C) spatial_features
    - features statiques par station (distance centre, grille, KMeans),
    - carte KNN entre stations,
    - lags dynamiques des voisins.

L'API publique utilisée par le reste du projet :

    - `train_model(...)`
    - `predict_from_features_df(...)`
    - `predict_latest_offline(...)`
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor  # type: ignore

# =============================================================================
# Shared utils
# =============================================================================


def _utc_now_naive() -> datetime:
    """
    Get the current UTC time as a **naive** datetime.

    Returns
    -------
    datetime
        `datetime.now(timezone.utc)` avec tz-info supprimée (naïf, UTC implicite).
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _is_gcs_uri(uri: str) -> bool:
    """
    Check if a string looks like a GCS URI (`gs://...`).

    Parameters
    ----------
    uri : str

    Returns
    -------
    bool
        True si la chaîne commence par 'gs://'.
    """
    return isinstance(uri, str) and uri.startswith("gs://")


def _split_gs(uri: str) -> Tuple[str, str]:
    """
    Split a GCS URI `gs://bucket/key` into `(bucket, key)`.

    Parameters
    ----------
    uri : str
        GCS URI.

    Returns
    -------
    (str, str)
        Bucket name et chemin interne (key).
    """
    assert _is_gcs_uri(uri), f"Not a GCS URI: {uri}"
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def _gcs_blob_exists(bucket: str, key: str) -> bool:
    """
    Teste l'existence d'un blob dans GCS.

    Paramètres
    ----------
    bucket : str
        Nom du bucket.
    key : str
        Clé/chemin de l'objet.

    Returns
    -------
    bool
        True si le blob existe.
    """
    from google.cloud import storage
    cli = storage.Client()
    return cli.bucket(bucket).blob(key).exists()


def _resolve_model_uri(model_or_prefix: str) -> str:
    """
    Résoudre un chemin de modèle qui peut être :
      - un joblib direct,
      - un dossier contenant `latest.joblib` ou `model.joblib`,
      - un préfixe GCS (`gs://bucket/path`).

    La résolution se fait dans l'ordre :
      1. si c'est déjà un `.joblib` → retour direct,
      2. si c'est un dossier ou un préfixe → on cherche `latest.joblib`,
         puis `model.joblib`,
      3. sinon `FileNotFoundError`.

    Parameters
    ----------
    model_or_prefix : str

    Returns
    -------
    str
        URI ou chemin résolu vers le fichier `.joblib`.
    """
    if _is_gcs_uri(model_or_prefix):
        bkt, key = _split_gs(model_or_prefix.rstrip("/"))
        if key.endswith(".joblib"):
            return model_or_prefix
        latest_key = f"{key}/latest.joblib"
        if _gcs_blob_exists(bkt, latest_key):
            return f"gs://{bkt}/{latest_key}"
        model_key = f"{key}/model.joblib"
        if _gcs_blob_exists(bkt, model_key):
            return f"gs://{bkt}/{model_key}"
        raise FileNotFoundError(f"No latest.joblib or model.joblib under gs://{bkt}/{key}")
    else:
        p = Path(model_or_prefix)
        if p.suffix == ".joblib":
            return str(p)
        if p.is_dir():
            latest = p / "latest.joblib"
            if latest.exists():
                return str(latest)
            model = p / "model.joblib"
            if model.exists():
                return str(model)
            raise FileNotFoundError(f"No latest.joblib or model.joblib under {p}")
        if p.exists() and p.is_file():
            return str(p)
        latest = p / "latest.joblib"
        if latest.exists():
            return str(latest)
        raise FileNotFoundError(f"Cannot resolve model uri from: {model_or_prefix}")


def _load_bytes_from_gcs(uri: str) -> bytes:
    """
    Télécharger un blob GCS en mémoire (bytes).

    Parameters
    ----------
    uri : str
        URI GCS (`gs://bucket/key`).

    Returns
    -------
    bytes
        Contenu brut du blob.
    """
    from google.cloud import storage
    bucket, key = _split_gs(uri)
    cli = storage.Client()
    return cli.bucket(bucket).blob(key).download_as_bytes()


def _load_joblib_from_uri(uri_or_prefix: str):
    """
    Charger un objet joblib depuis :
      - un chemin local,
      - un préfixe local,
      - une URI GCS.

    Parameters
    ----------
    uri_or_prefix : str

    Returns
    -------
    Any
        Objet Python désérialisé via joblib.load.
    """
    uri = _resolve_model_uri(uri_or_prefix)
    if _is_gcs_uri(uri):
        data = _load_bytes_from_gcs(uri)
        with io.BytesIO(data) as buf:
            return joblib.load(buf)
    return joblib.load(uri)


def _ensure_columns(df: pd.DataFrame, cols: Iterable[str], fill: float = 0.0) -> pd.DataFrame:
    """
    S'assurer qu'un DataFrame contient au moins les colonnes `cols`.

    Toute colonne absente est créée avec la valeur `fill`.

    Parameters
    ----------
    df : pandas.DataFrame
    cols : Iterable[str]
        Noms attendus.
    fill : float, default 0.0

    Returns
    -------
    pandas.DataFrame
        Le même DataFrame (modifié en place), renvoyé pour chaînage.
    """
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df


def _select_in_order(df: pd.DataFrame, ordered_cols: Iterable[str]) -> pd.DataFrame:
    """
    Sélectionner les colonnes dans un ordre précis (sans ajout ni suppression).

    Parameters
    ----------
    df : pandas.DataFrame
    ordered_cols : Iterable[str]

    Returns
    -------
    pandas.DataFrame
        DataFrame restreint aux colonnes `ordered_cols` dans le bon ordre.
    """
    return df[[c for c in ordered_cols]]


def _clip_and_round(y: np.ndarray, cap: np.ndarray | float | int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clip puis arrondir les prédictions en respectant la capacité.

    Étapes :
      1. clip y dans [0, capacity],
      2. arrondi au plus proche entier (np.rint),
      3. cast en int16.

    Parameters
    ----------
    y : array-like
        Prédictions flottantes.
    cap : array-like | float | int
        Capacité (peut être scalaire ou vectoriel).

    Returns
    -------
    (y_clipped, y_int)
        y_clipped : float32
        y_int     : int16
    """
    y = np.asarray(y, dtype=np.float32)
    cap_arr = np.asarray(cap) if not np.isscalar(cap) else np.full_like(y, float(cap))
    y_clipped = np.clip(y, 0.0, cap_arr.astype(np.float32))
    y_int = np.rint(y_clipped).astype(np.int16)
    return y_clipped, y_int


def _derive_model_version_from_uri(uri_or_prefix: str) -> str:
    """
    Déduire un identifiant de version à partir de l'URI du modèle.

    Exemple :
      - '.../latest.joblib' → 'latest'
      - '.../model_2.1.3.joblib' → 'model_2.1.3'

    Parameters
    ----------
    uri_or_prefix : str

    Returns
    -------
    str
        Nom de fichier sans extension.
    """
    uri = _resolve_model_uri(uri_or_prefix)
    base = uri.rstrip("/").split("/")[-1]
    return os.path.splitext(base)[0]  # 'latest' le cas échéant


def _ts_utc_slug() -> str:
    """
    Timestamp UTC compact pour nommer les répertoires d'artefacts.

    Exemple : '20251121-145501Z'
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")


def _safe_hash_feat_cols(cols: Iterable[str]) -> str:
    """
    Calculer un hash court (8 caractères) des noms de features.

    Utilisé pour décider si un changement de features doit bump la minor.

    Parameters
    ----------
    cols : Iterable[str]

    Returns
    -------
    str
        Préfixe de hash SHA-256 (8 premiers caractères).
    """
    m = hashlib.sha256()
    for c in cols:
        m.update(str(c).encode("utf-8"))
    return m.hexdigest()[:8]


# =============================================================================
# Versioning helpers (latest.json → bump)
# =============================================================================


def _parse_semver(s: str) -> Tuple[int, int, int]:
    """
    Parser une chaîne de type 'M.m.p' en tuple (M, m, p).

    Si la chaîne n'est pas au bon format, on retourne (1,0,0).
    """
    try:
        parts = [int(x) for x in s.strip().split(".")]
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts[:3])
    except Exception:
        return (1, 0, 0)


def _fmt_semver(M: int, m: int, p: int) -> str:
    """
    Formatter un semver (M,m,p) en string 'M.m.p'.
    """
    return f"{M}.{m}.{p}"


def _bump_patch(v: str) -> str:
    """
    Incrémenter le patch d'une version 'M.m.p' → 'M.m.(p+1)'.
    """
    M, m, p = _parse_semver(v)
    return _fmt_semver(M, m, p + 1)


def _bump_minor(v: str) -> str:
    """
    Incrémenter la minor d'une version 'M.m.p' → 'M.(m+1).0'.
    """
    M, m, _ = _parse_semver(v)
    return _fmt_semver(M, m + 1, 0)


def _read_latest_manifest_gcs(bucket: str, prefix_h15: str) -> dict | None:
    """
    Lire `latest.json` dans GCS pour un certain horizon (h15, h60, ...).

    Parameters
    ----------
    bucket : str
    prefix_h15 : str
        Préfixe de l'horizon (ex: 'velib/models/h15').

    Returns
    -------
    dict | None
        Manifest JSON ou None si absent / erreur.
    """
    try:
        from google.cloud import storage
        cli = storage.Client()
        blob = cli.bucket(bucket).blob(f"{prefix_h15.rstrip('/')}/latest.json")
        if not blob.exists():
            return None
        data = blob.download_as_bytes()
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def _read_latest_manifest_local(arte_base: str, horizon_slug: str) -> dict | None:
    """
    Lire `latest.json` en local pour un horizon donné.

    Parameters
    ----------
    arte_base : str
        Dossier racine des artefacts (ex: 'artifacts').
    horizon_slug : str
        Sous-dossier d'horizon (ex: 'h15').

    Returns
    -------
    dict | None
        Manifest ou None si absent/erreur.
    """
    try:
        p = Path(arte_base) / horizon_slug / "latest.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _compute_next_version(
    *,
    base_version: str,
    feat_hash8: str,
    horizon_bins: int,
    gcs_bucket: Optional[str],
    gcs_prefix: Optional[str],
    arte_base: str,
    horizon_slug: str,
) -> str:
    """
    Décider de la prochaine version du modèle à partir de l'historique.

    Règle :
      - si (features_hash8 OU horizon_bins) changent → bump minor,
      - sinon → bump patch.

    Parameters
    ----------
    base_version : str
        Version de base (ex: '2.0').
    feat_hash8 : str
        Hash court des features.
    horizon_bins : int
        Horizon en bins de 5 min.
    gcs_bucket : str | None
        Bucket GCS où lire le manifest latest.json.
    gcs_prefix : str | None
        Préfixe GCS pour l'horizon.
    arte_base : str
        Dossier local d'artefacts.
    horizon_slug : str
        Slug d'horizon (ex: 'h15').

    Returns
    -------
    str
        Version suivante (semver).
    """
    latest = None
    if gcs_bucket and gcs_prefix:
        latest = _read_latest_manifest_gcs(gcs_bucket, gcs_prefix)
    if latest is None:
        latest = _read_latest_manifest_local(arte_base, horizon_slug)

    if base_version.count(".") == 1:
        base_version += ".0"
    if not latest:
        return base_version

    prev_v = latest.get("version") or latest.get("semver") or base_version
    prev_hash = str(latest.get("features_hash8", ""))
    prev_hz = int(latest.get("horizon_bins", horizon_bins))
    return _bump_minor(prev_v) if (prev_hash != feat_hash8 or prev_hz != horizon_bins) else _bump_patch(prev_v)


# =============================================================================
# JSON / GCS publish
# =============================================================================


def _write_version_json(
    out_json_path: str,
    *,
    model_type: str,
    horizon_bins: int,
    feat_cols: List[str],
    metrics: dict | None = None,
    extras: dict | None = None,
    include_features: bool = False,
) -> None:
    """
    Écrire le fichier `version.json` associé à un modèle entraîné.

    Parameters
    ----------
    out_json_path : str
        Chemin local de sortie.
    model_type : str
        Type de modèle (ex: 'xgb').
    horizon_bins : int
        Horizon en bins (5 minutes).
    feat_cols : list[str]
        Liste des noms de features utilisées.
    metrics : dict | None
        Dictionnaire de métriques (MAE, RMSE...).
    extras : dict | None
        Champs supplémentaires à inclure (version, n_train, etc.).
    include_features : bool
        Si True, inclut la liste détaillée des features (verbeux mais explicite).
    """
    payload = {
        "model_type": model_type,
        "horizon_bins": int(horizon_bins),
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": os.environ.get("GIT_SHA"),
        "feature_count": len(feat_cols),
        "features_hash8": _safe_hash_feat_cols(feat_cols),
        "metrics": metrics or {},
    }
    if include_features:
        payload["features"] = list(feat_cols)
    if extras:
        payload.update(extras)
    Path(out_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[publish] wrote version json → {out_json_path}")


def _maybe_publish_to_gcs(local_model: str, local_json: str, *, bucket: str, prefix_h15: str) -> None:
    """
    Publier un modèle + version.json dans GCS + mettre à jour les alias latest.*.

    Layout GCS :
      - {prefix_h15}/{stamp}/model.joblib
      - {prefix_h15}/{stamp}/version.json
      - {prefix_h15}/latest.joblib
      - {prefix_h15}/latest.json

    Parameters
    ----------
    local_model : str
        Chemin local du `model.joblib` daté.
    local_json : str
        Chemin local de `version.json`.
    bucket : str
        Nom du bucket GCS.
    prefix_h15 : str
        Préfixe d'horizon (ex: 'velib/models/h15').
    """
    from google.cloud import storage
    client = storage.Client()
    bkt = client.bucket(bucket)

    stamp = Path(local_model).parent.name
    base_dir = f"{prefix_h15.rstrip('/')}/{stamp}"
    latest_dir = f"{prefix_h15.rstrip('/')}"

    src_model_blob = bkt.blob(f"{base_dir}/model.joblib")
    src_json_blob  = bkt.blob(f"{base_dir}/version.json")
    src_model_blob.upload_from_filename(local_model, content_type="application/octet-stream")
    src_json_blob.upload_from_filename(local_json, content_type="application/json")
    print(f"[publish] uploaded → gs://{bucket}/{base_dir}/model.joblib")
    print(f"[publish] uploaded → gs://{bucket}/{base_dir}/version.json")

    bkt.blob(f"{latest_dir}/latest.joblib").rewrite(src_model_blob)
    bkt.blob(f"{latest_dir}/latest.json").rewrite(src_json_blob)
    print(f"[publish] updated aliases → gs://{bucket}/{latest_dir}/latest.*")


# =============================================================================
# (A) cal_features — add_time_features
# =============================================================================


def add_time_features(d: pd.DataFrame, ts_col: str = "tbin_utc", add_paris_derived: bool = True) -> pd.DataFrame:
    """
    Ajouter des features calendaires simples (UTC naïf, proxies Paris).

    Features générées
    -----------------
    - hour, minute
    - dow (0=lundi), month
    - is_weekend (1 si samedi/dimanche)
    - hod_sin, hod_cos (cycle 24h)
    - dow_sin, dow_cos (cycle 7j)
    - paris_hour, paris_dow, paris_is_we (si `add_paris_derived=True`)

    Notes
    -----
    Ici on considère la colonne `ts_col` déjà en naïf UTC (cohérent avec les
    parquets du projet). Les features 'paris_*' sont simplement copiées de
    l'UTC, ce sont donc des proxies (pas une vraie conversion timezone).
    """
    dts = pd.to_datetime(d[ts_col], errors="coerce", utc=False)
    d["hour"]   = dts.dt.hour.astype("Int16")
    d["minute"] = dts.dt.minute.astype("Int16")
    d["dow"]    = dts.dt.dayofweek.astype("Int16")
    d["month"]  = dts.dt.month.astype("Int16")
    d["is_weekend"] = (d["dow"].isin([5, 6])).astype("Int8")

    mins = d["hour"] * 60 + d["minute"]
    d["hod_sin"] = np.sin(2 * np.pi * mins / 1440.0).astype("float32")
    d["hod_cos"] = np.cos(2 * np.pi * mins / 1440.0).astype("float32")
    d["dow_sin"] = np.sin(2 * np.pi * d["dow"] / 7.0).astype("float32")
    d["dow_cos"] = np.cos(2 * np.pi * d["dow"] / 7.0).astype("float32")

    if add_paris_derived:
        # Proxies simples : mêmes valeurs qu'en UTC
        d["paris_hour"] = d["hour"].astype("Int16")
        d["paris_dow"]  = d["dow"].astype("Int16")
        d["paris_is_we"] = d["is_weekend"].astype("Int8")
    return d


# =============================================================================
# (B) time_features — base I/O + lags/rollings/météo + target
# =============================================================================

BASE_COLUMNS = [
    "ts_utc", "tbin_utc", "station_id", "bikes", "capacity", "mechanical", "ebike",
    "status", "lat", "lon", "name", "temp_C", "precip_mm", "wind_mps"
]


def _read_many_parquets(path_or_glob: str) -> pd.DataFrame:
    """
    Lire un ensemble de parquets (fichier, dossier, glob) et normaliser le schéma.

    Parameters
    ----------
    path_or_glob : str
        - dossier contenant des .parquet,
        - pattern glob (`.../*.parquet`),
        - fichier unique.

    Returns
    -------
    pandas.DataFrame
        DataFrame concaténé avec au moins les colonnes `BASE_COLUMNS`.
        Les colonnes manquantes sont créées avec des NA.
    """
    paths: List[str] = []
    if os.path.isdir(path_or_glob):
        paths = sorted(glob.glob(os.path.join(path_or_glob, "*.parquet")))
    else:
        if "*" in path_or_glob or "?" in path_or_glob:
            paths = sorted(glob.glob(path_or_glob))
        else:
            paths = [path_or_glob]
    if not paths:
        return pd.DataFrame(columns=BASE_COLUMNS)

    dfs = []
    for p in paths:
        try:
            dfs.append(pd.read_parquet(p))
        except Exception as e:
            print(f"[features][warn] failed to read parquet: {p} → {e}")
    if not dfs:
        return pd.DataFrame(columns=BASE_COLUMNS)

    out = pd.concat(dfs, ignore_index=True, sort=False)
    for c in BASE_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    return out[BASE_COLUMNS]


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliser les types des colonnes de base.

    - ts_utc, tbin_utc : datetime naïf UTC
    - station_id, status, name : string pandas
    - bikes, capacity, mechanical, ebike, lat, lon, météo : numeric
    """
    df["ts_utc"]   = pd.to_datetime(df["ts_utc"],   utc=True, errors="coerce").dt.tz_convert(None)
    df["tbin_utc"] = pd.to_datetime(df["tbin_utc"], utc=True, errors="coerce").dt.tz_convert(None)
    df["station_id"] = df["station_id"].astype("string")
    for c in ["bikes", "capacity", "mechanical", "ebike"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["lat", "lon", "temp_C", "precip_mm", "wind_mps"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["status"] = df["status"].astype("string")
    df["name"]   = df["name"].astype("string")
    return df


def _dedupe_per_bin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprimer les doublons par (station_id, tbin_utc), en gardant **le dernier** échantillon.

    Le tri est fait sur (station_id, tbin_utc, ts_utc), puis `tail(1)` par groupe.
    """
    df = df.sort_values(["station_id", "tbin_utc", "ts_utc"])
    dedup = df.groupby(["station_id", "tbin_utc"], as_index=False).tail(1)
    return dedup.reset_index(drop=True)


def _add_target_and_lags(
    df: pd.DataFrame,
    horizon_bins: int = 3,
    lag_bins: Iterable[int] = (1, 2, 3, 6, 12, 24, 36, 48),
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Ajouter la cible `y_nb` et toutes les features temporelles classiques.

    Features créées
    ---------------
    - y_nb : bikes décalé de -horizon_bins (cible à horizon H)
    - lag_bikes_L : lags sur bikes pour L ∈ `lag_bins`
    - roll_mean_W / roll_std_W : statistiques glissantes sur bikes (avec shift(1) anti-leak)
    - occ_ratio : bikes / capacity
    - temp_C_lag1, precip_mm_lag1, wind_mps_lag1 : météo décalée de 1 bin
    - features calendaires via `add_time_features`
    - status_code : encodage entier de `status`

    Parameters
    ----------
    df : pandas.DataFrame
        Données brutes normalisées.
    horizon_bins : int, default 3
        Horizon en bins de 5 minutes (ex: 3 → 15 minutes).
    lag_bins : Iterable[int]
        Lags sur bikes à générer.

    Returns
    -------
    (df, feat_cols)
        df : DataFrame enrichi
        feat_cols : liste ordonnée des noms de features utilisables pour XGBoost.
    """
    df = df.sort_values(["station_id", "tbin_utc"]).copy()

    # Target
    df["y_nb"] = df.groupby("station_id", group_keys=False)["bikes"].shift(-horizon_bins)

    # Bike lags
    for L in lag_bins:
        df[f"lag_bikes_{L}"] = df.groupby("station_id", group_keys=False)["bikes"].shift(L)

    # Rolling windows avec shift(1) pour éviter les fuites de cible
    rolling_windows = (3, 6, 12, 24, 36, 48)
    for W in rolling_windows:
        df[f"roll_mean_{W}"] = (
            df.groupby("station_id", group_keys=False)["bikes"]
              .apply(lambda s: s.shift(1).rolling(W, min_periods=max(1, W//2)).mean())
        )
        df[f"roll_std_{W}"] = (
            df.groupby("station_id", group_keys=False)["bikes"]
              .apply(lambda s: s.shift(1).rolling(W, min_periods=max(1, W//2)).std())
        )

    # Occupancy ratio
    df["occ_ratio"] = df["bikes"] / df["capacity"].where(df["capacity"] > 0)

    # Weather lags (L=1)
    for c in ("temp_C", "precip_mm", "wind_mps"):
        df[f"{c}_lag1"] = df.groupby("station_id", group_keys=False)[c].shift(1)

    # Calendar/time features
    add_time_features(df, ts_col="tbin_utc", add_paris_derived=True)

    # Status code
    status_map = {s: i for i, s in enumerate(sorted(df["status"].dropna().unique()))}
    df["status_code"] = df["status"].map(status_map).astype("Int64")

    feat_cols = [
        "capacity", "mechanical", "ebike",
        "lat", "lon",
        "temp_C", "precip_mm", "wind_mps",
        "occ_ratio",
        *(f"lag_bikes_{L}" for L in lag_bins),
        *(f"roll_mean_{W}" for W in rolling_windows),
        *(f"roll_std_{W}" for W in rolling_windows),
        "temp_C_lag1", "precip_mm_lag1", "wind_mps_lag1",
        "hour", "minute", "dow", "month", "is_weekend",
        "hod_sin", "hod_cos", "dow_sin", "dow_cos",
        "paris_hour", "paris_dow", "paris_is_we",
        "status_code",
    ]
    return df, feat_cols


# =============================================================================
# (C) spatial_features — statics & KNN dynamics (s’additionnent à time_features)
# =============================================================================

from sklearn.cluster import KMeans


def _haversine_km(lat1, lon1, lat2, lon2):
    """
    Distance haversine en kilomètres entre deux ensembles de points.

    Paramètres
    ----------
    lat1, lon1, lat2, lon2 : array-like
        Coordonnées en degrés.

    Returns
    -------
    numpy.ndarray
        Distances en km (broadcast compatible).
    """
    R = 6371.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))


def build_station_static(df: pd.DataFrame, center=(48.853, 2.349), grid_m=300, k_clusters=30, random_state=42) -> pd.DataFrame:
    """
    Construire des features **statiques** par station (lat/lon, capacité, dist centre, grille, KMeans, target encoding).

    Parameters
    ----------
    df : pandas.DataFrame
        Données contenant au moins `station_id`, `lat`, `lon`, `capacity`.
    center : tuple(float, float)
        Coordonnées (lat, lon) du centre de référence (Paris centre par défaut).
    grid_m : int
        Taille de la grille spatiale en mètres (~300m).
    k_clusters : int
        Nombre de clusters KMeans.
    random_state : int

    Returns
    -------
    pandas.DataFrame
        Une ligne par station, avec :
          - station_id
          - lat, lon
          - dist_center_km
          - grid_x, grid_y
          - kmeans_cluster
          - te_cap_mean (target encoding de la capacité).
    """
    cols_needed = {"station_id", "lat", "lon", "capacity"}
    missing = cols_needed - set(df.columns)
    if missing:
        raise KeyError(f"build_station_static: colonnes manquantes: {missing}")

    if "tbin_utc" in df.columns:
        st = (df.sort_values(["station_id", "tbin_utc"])
                .groupby("station_id", as_index=False).tail(1))[["station_id", "lat", "lon", "capacity"]].drop_duplicates()
    else:
        st = df[["station_id", "lat", "lon", "capacity"]].drop_duplicates()

    st["station_id"] = st["station_id"].astype("string")

    # distance au centre
    st["dist_center_km"] = _haversine_km(st["lat"], st["lon"], center[0], center[1]).astype("float32")

    # grille ~300 m (approx à Paris)
    lat_deg_per_m = 1.0 / 111_000.0
    lon_deg_per_m = 1.0 / 75_000.0
    st["grid_x"] = np.floor(st["lon"] / (grid_m * lon_deg_per_m)).astype("int32")
    st["grid_y"] = np.floor(st["lat"] / (grid_m * lat_deg_per_m)).astype("int32")

    # KMeans spatial (+ capacité)
    km = KMeans(n_clusters=k_clusters, n_init=10, random_state=random_state)
    st["kmeans_cluster"] = km.fit_predict(st[["lat", "lon", "capacity"]]).astype("int32")

    # target encoding statique (capacités moyennes par cluster)
    te = st.groupby("kmeans_cluster", as_index=False)["capacity"].mean().rename(columns={"capacity": "te_cap_mean"})
    st = st.merge(te, on="kmeans_cluster", how="left")
    st["te_cap_mean"] = st["te_cap_mean"].astype("float32")

    return st[["station_id", "lat", "lon", "dist_center_km", "grid_x", "grid_y", "kmeans_cluster", "te_cap_mean"]]


def compute_knn_map(stations_df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Construire une carte KNN (station → k voisins les plus proches).

    Parameters
    ----------
    stations_df : pandas.DataFrame
        Doit contenir 'station_id', 'lat', 'lon'.
    k : int, default 5
        Nombre de voisins à conserver.

    Returns
    -------
    pandas.DataFrame
        Colonnes :
          - station_id
          - rank (1..k)
          - neighbor_id
          - dist_km
    """
    required = {"station_id", "lat", "lon"}
    missing = required - set(stations_df.columns)
    if missing:
        raise KeyError(f"compute_knn_map: stations_df missing columns: {missing}")
    stations = stations_df[["station_id", "lat", "lon"]].drop_duplicates().copy()
    stations["station_id"] = stations["station_id"].astype("string")
    S = stations.shape[0]
    lat = stations["lat"].to_numpy(); lon = stations["lon"].to_numpy()
    lat_mat1 = np.repeat(lat[:, None], S, axis=1)
    lon_mat1 = np.repeat(lon[:, None], S, axis=1)
    lat_mat2 = lat_mat1.T
    lon_mat2 = lon_mat1.T
    D = _haversine_km(lat_mat1, lon_mat1, lat_mat2, lon_mat2)
    np.fill_diagonal(D, np.inf)
    kk = min(k, max(1, S-1))
    idx = np.argpartition(D, kth=kk-1, axis=1)[:, :kk]
    rows = []
    for i in range(S):
        sid = stations.iloc[i]["station_id"]
        neigh_idx = idx[i]
        for rank, j in enumerate(neigh_idx, 1):
            rows.append((sid, int(rank), stations.iloc[j]["station_id"], float(D[i, j])))
    knn = pd.DataFrame(rows, columns=["station_id", "rank", "neighbor_id", "dist_km"])
    return knn


def _add_features_spatial(df: pd.DataFrame, horizon_bins: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    Ajouter la **pile complète** de features : temporelles + spatiales.

    Contenu
    -------
    - Temps :
        cible y_nb, lags bikes, rollings, météo, calendrier, status_code.
    - Spatial statique :
        dist_center_km, grid_x/y, kmeans_cluster, te_cap_mean.
    - Spatial dynamique KNN :
        knn_mean_bikes_lag1 / 3 / 12 sur les voisins (k=5).

    Parameters
    ----------
    df : pandas.DataFrame
        Données brutes normalisées (5-min bins).
    horizon_bins : int
        Horizon de prédiction en bins.

    Returns
    -------
    (enhanced, feat_cols)
        enhanced : DataFrame enrichi pour le training / inference.
        feat_cols : liste des colonnes de features.
    """
    base_df, base_cols = _add_target_and_lags(df, horizon_bins=horizon_bins)

    # === statiques ===
    st_static_src = base_df[["station_id", "lat", "lon", "capacity"]].drop_duplicates()
    st_static = build_station_static(st_static_src, center=(48.853, 2.349), grid_m=300, k_clusters=30, random_state=42)
    # on merge sans dupliquer lat/lon
    enhanced = base_df.merge(st_static.drop(columns=["lat", "lon"]), on="station_id", how="left")

    # === dynamiques KNN (k=5) ===
    knn = compute_knn_map(stations_df=st_static[["station_id", "lat", "lon"]], k=5)
    base = enhanced[["tbin_utc", "station_id", "bikes"]].copy()
    base["station_id"] = base["station_id"].astype("string")

    for L in (1, 3, 12):  # bins (5 min/bin)
        neigh = knn.merge(base.rename(columns={"station_id": "neighbor_id"}), on="neighbor_id", how="left")
        neigh["tbin_join"] = pd.to_datetime(neigh["tbin_utc"], errors="coerce") + pd.Timedelta(minutes=5 * L)
        tmp = (neigh[["station_id", "tbin_join", "bikes"]]
               .rename(columns={"tbin_join": "tbin_utc", "bikes": f"knn_mean_bikes_lag{L}_tmp"}))
        tmp = tmp.groupby(["station_id", "tbin_utc"], as_index=False)[f"knn_mean_bikes_lag{L}_tmp"].mean()
        enhanced = enhanced.merge(tmp, on=["station_id", "tbin_utc"], how="left")
        enhanced.rename(columns={f"knn_mean_bikes_lag{L}_tmp": f"knn_mean_bikes_lag{L}"}, inplace=True)

    feat_cols = (
        base_cols
        + ["dist_center_km", "grid_x", "grid_y", "kmeans_cluster", "te_cap_mean"]
        + [f"knn_mean_bikes_lag{L}" for L in (1, 3, 12)]
    )

    # types cohérents
    for c in ["dist_center_km", "te_cap_mean", "grid_x", "grid_y", "kmeans_cluster",
              "knn_mean_bikes_lag1", "knn_mean_bikes_lag3", "knn_mean_bikes_lag12"]:
        enhanced[c] = pd.to_numeric(enhanced[c], errors="coerce")

    return enhanced, feat_cols


# =============================================================================
# In-pipeline inference API
# =============================================================================


def predict_from_features_df(
    feats_df: pd.DataFrame,
    model_uri: str,
    horizon_bins: Optional[int] = None,
    model_alias: Optional[str] = None,
) -> pd.DataFrame:
    """
    Appliquer un modèle entraîné sur un DataFrame de features déjà construites.

    Cette fonction est utilisée dans la **pipeline serving** (Cloud Run job
    `build_serving_forecast`) et dans des scripts offline.

    Contract
    --------
    - `feats_df` doit contenir :
        * station_id
        * tbin_latest
        * capacity_bin
        * toutes les colonnes de `pack["feat_cols"]` (les manquantes sont
          injectées avec 0.0 par `_ensure_columns`).

    Parameters
    ----------
    feats_df : pandas.DataFrame
        Features prêtes à être ingérées par le modèle.
    model_uri : str
        Chemin / URI vers un modèle entraîné (ou préfixe).
    horizon_bins : int | None
        Horizon en bins (5 min). Si None, utilise celui enregistré dans le pack.
    model_alias : str | None
        Alias de version à injecter dans `model_version` (sinon dérivé de l'URI).

    Returns
    -------
    pandas.DataFrame
        Colonnes :
          - station_id (Int64)
          - tbin_latest (datetime)
          - horizon_min
          - bikes_pred (float32)
          - bikes_pred_int (int16)
          - capacity_bin (Int64)
          - pred_ts_utc (datetime naïf UTC)
          - target_ts_utc (datetime naïf UTC)
          - model_version (str)
    """
    if feats_df is None or len(feats_df) == 0:
        return pd.DataFrame(
            columns=[
                "station_id", "tbin_latest", "horizon_min", "bikes_pred", "bikes_pred_int",
                "capacity_bin", "pred_ts_utc", "target_ts_utc", "model_version",
            ]
        )

    pack = _load_joblib_from_uri(model_uri)
    if not isinstance(pack, dict) or "model" not in pack or "feat_cols" not in pack:
        raise ValueError("Invalid model pack. Expect dict with keys: 'model', 'feat_cols', 'horizon_bins'.")

    model = pack["model"]
    feat_cols: List[str] = list(pack["feat_cols"])
    baked_hz = int(pack.get("horizon_bins", 3))
    hz_bins = int(horizon_bins) if horizon_bins is not None else baked_hz
    hz_min = int(hz_bins * 5)

    for col in ["station_id", "tbin_latest", "capacity_bin"]:
        if col not in feats_df.columns:
            raise KeyError(f"Missing required column in features: '{col}'")

    X_df = feats_df.copy()
    _ensure_columns(X_df, feat_cols, fill=0.0)
    X = _select_in_order(X_df, feat_cols).astype(np.float32)

    y_hat = model.predict(X)

    cap = feats_df["capacity_bin"].to_numpy()
    y_clip, y_int = _clip_and_round(y_hat, cap)

    tbin_latest = pd.to_datetime(feats_df["tbin_latest"], errors="coerce")
    target_ts = tbin_latest + pd.to_timedelta(hz_min, unit="m")

    out = pd.DataFrame(
        {
            "station_id": feats_df["station_id"].astype("Int64"),
            "tbin_latest": tbin_latest,
            "horizon_min": np.int16(hz_min),
            "bikes_pred": y_clip.astype(np.float32),
            "bikes_pred_int": y_int.astype(np.int16),
            "capacity_bin": feats_df["capacity_bin"].astype("Int64"),
            "pred_ts_utc": _utc_now_naive(),
            "target_ts_utc": target_ts,
            "model_version": (model_alias or _derive_model_version_from_uri(model_uri)),
        }
    )

    return out[
        [
            "station_id", "tbin_latest", "horizon_min", "bikes_pred", "bikes_pred_int",
            "capacity_bin", "pred_ts_utc", "target_ts_utc", "model_version",
        ]
    ]


# =============================================================================
# Training (XGBoost-only) & offline predict
# =============================================================================


def _build_training_frame_always_spatial(
    src: str,
    start_date: Optional[str] = None,
    end_date: Optional[str]   = None,
    horizon_bins: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """
    Construire le DataFrame d'entraînement avec la **pile SPATIALE forcée**.

    Étapes
    ------
    1. Lire les parquets 5min (`_read_many_parquets`).
    2. Normaliser types + dédoublonner par bin.
    3. Filtrer la fenêtre temporelle [start_date, end_date] (si fournie).
    4. Ajouter toutes les features temporelles + spatiales via `_add_features_spatial`.
    5. Extraire :
          - `df2` : DataFrame enrichi complet,
          - `X`   : features uniquement,
          - `y`   : cible y_nb,
          - `feat_cols` : noms des features.

    Returns
    -------
    (df2, X, y, feat_cols)
        df2 : DataFrame enrichi
        X   : features float32
        y   : série float32
        feat_cols : liste des colonnes features.
    """
    df = _read_many_parquets(src)
    if df.empty:
        return df, pd.DataFrame(), pd.Series(dtype="float64"), []
    df = _coerce_types(df)
    df = _dedupe_per_bin(df)
    if start_date:
        df = df[df["tbin_utc"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["tbin_utc"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)]
    df2, feat_cols = _add_features_spatial(df, horizon_bins=horizon_bins)
    used_cols = ["station_id", "tbin_utc", "bikes", "y_nb"] + feat_cols
    df_model = df2[used_cols].dropna(subset=["y_nb"])
    X = df_model[feat_cols].astype("float32").copy()
    y = df_model["y_nb"].astype("float32").copy()
    return df2, X, y, feat_cols


def train_model(
    src: str,
    horizon_bins: int = 3,
    out_path: str = "model.joblib",
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    lookback_days: Optional[int] = None,
):
    """
    Entraîner un XGBoost (hyperparamètres Optuna) avec **features spatiales forcées**.

    Paramètres principaux
    ---------------------
    src : str
        Chemin/glob/dossier de parquets 5min (ou daily) alignés sur le schéma.
    horizon_bins : int, default 3
        Horizon en bins (5 min). Ex: 3 → 15 minutes.
    out_path : str
        Chemin de compatibilité pour enregistrer le modèle.
        Le vrai artefact versionné est dans MODEL_ARTEFACTS_DIR.

    Fenêtre temporelle
    ------------------
    - start_date / end_date : dates UTC "YYYY-MM-DD".
    - lookback_days : si fourni **et** start/end non fournis, on calcule
      automatiquement [tmax - N, tmax] à partir de max(tbin_utc).

    Versioning
    ----------
    - Les artefacts sont écrits sous :
        {MODEL_ARTEFACTS_DIR}/h{horizon_min}/{stamp}/
      avec `model.joblib` et `version.json`.
    - Si MODEL_GCS_BUCKET + MODEL_GCS_PREFIX sont présents, les fichiers sont
      également publiés sur GCS + alias latest.* mis à jour.
    """
    # Résolution lookback → fenêtre [tmax-N, tmax] si demandée
    if lookback_days is not None and (start_date is None and end_date is None):
        df_tmp = _read_many_parquets(src)
        if df_tmp.empty:
            raise RuntimeError("No data found in source for lookback window resolution.")
        df_tmp = _coerce_types(df_tmp)
        df_tmp = _dedupe_per_bin(df_tmp)
        tmax = pd.to_datetime(df_tmp["tbin_utc"], errors="coerce").max()
        if pd.isna(tmax):
            raise RuntimeError("Could not infer max tbin_utc for lookback.")
        tmin = (tmax - timedelta(days=int(lookback_days))).normalize()
        start_date = tmin.strftime("%Y-%m-%d"); end_date = tmax.strftime("%Y-%m-%d")

    # Build SPATIAL frame (toujours)
    df, X, y, feat_cols = _build_training_frame_always_spatial(
        src, start_date=start_date, end_date=end_date, horizon_bins=horizon_bins
    )
    if X.empty:
        raise RuntimeError("No training data (X is empty). Check your parquet source / date filters.")

    # 🔍 Show all feature columns before training
    print("\n────────────────────────────── FEATURES ──────────────────────────────")
    print(f"[train] Stack: spatial  |  Total: {len(feat_cols)} features")
    for i, c in enumerate(feat_cols, 1):
        print(f"{i:3d}. {c}")
    print("──────────────────────────────────────────────────────────────────────\n")

    # === Hyperparams (Optuna best trial fournis) ===
    model = XGBRegressor(
        max_depth=9,
        min_child_weight=9.207818833112405,
        subsample=0.8047635514805875,
        colsample_bytree=0.8291598175121464,
        reg_lambda=3.8462732799788806,
        reg_alpha=7.378469990313361,
        learning_rate=0.0324253002728162,
        max_bin=256,
        n_estimators=1000,
        tree_method="hist",
        n_jobs=int(os.environ.get("N_JOBS", "-1")),
        random_state=int(os.environ.get("SEED", "42")),
        objective="reg:squarederror",
        verbosity=1,
        eval_metric="rmse",
    )

    # Temporal split aligné avec y.index
    t_order_base = pd.to_datetime(df.loc[y.index, "tbin_utc"], errors="coerce")
    valid_mask = t_order_base.notna()
    if not valid_mask.all():
        X = X.loc[valid_mask]; y = y.loc[valid_mask]; t_order_base = t_order_base.loc[valid_mask]

    order = np.argsort(t_order_base.values)
    n = len(order)
    if n < 2:
        raise RuntimeError("Not enough samples after filtering to split train/valid.")
    split = max(1, int(0.9 * n))
    if split >= n: split = n - 1

    X_train = X.iloc[order[:split]].astype("float32")
    y_train = y.iloc[order[:split]].astype("float32")
    X_valid = X.iloc[order[split:]].astype("float32")
    y_valid = y.iloc[order[split:]].astype("float32")

    # --- Fit avec compat ultra-large versions XGBoost ---
    try:
        # 1) versions récentes: supportent early_stopping_rounds
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=200,
            verbose=True
        )
    except TypeError:
        try:
            # 2) versions un peu anciennes: acceptent eval_set mais PAS early_stopping_rounds
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=True
            )
        except TypeError:
            # 3) versions très anciennes: pas d'arguments supplémentaires
            model.fit(X_train, y_train)

    yhat = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, yhat)
    rmse = mean_squared_error(y_valid, yhat, squared=False)
    print(f"[train] xgb  MAE={mae:.3f}  RMSE={rmse:.3f}  (n_valid={len(y_valid)})")

    # Versioning + save
    arte_base = os.environ.get("MODEL_ARTEFACTS_DIR", "artifacts")
    horizon_slug = f"h{int(horizon_bins)*5:02d}"  # e.g., h15
    stamp = _ts_utc_slug()
    local_dir = Path(arte_base) / horizon_slug / stamp
    local_dir.mkdir(parents=True, exist_ok=True)

    gcs_bucket = os.environ.get("MODEL_GCS_BUCKET")
    gcs_prefix = os.environ.get("MODEL_GCS_PREFIX")  # e.g., "velib/models/h15"
    feat_hash8 = _safe_hash_feat_cols(feat_cols)
    base_version = os.environ.get("MODEL_BASE_VERSION", "2.0")

    version_str = _compute_next_version(
        base_version=base_version,
        feat_hash8=feat_hash8,
        horizon_bins=horizon_bins,
        gcs_bucket=gcs_bucket,
        gcs_prefix=gcs_prefix,
        arte_base=arte_base,
        horizon_slug=horizon_slug,
    )

    local_model_path = str(local_dir / "model.joblib")
    joblib.dump({"model": model, "feat_cols": list(feat_cols), "horizon_bins": int(horizon_bins)}, local_model_path)
    print(f"[train] model saved → {local_model_path}")

    metrics_payload = {"mae": float(mae), "rmse": float(rmse)}
    extras = {
        "version": version_str,
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
        "seed": int(os.environ.get("SEED", "42")),
        "features_stack": "spatial",
        "start_date": start_date,
        "end_date": end_date,
    }
    local_json_path = str(local_dir / "version.json")
    _write_version_json(
        local_json_path,
        model_type="xgb",
        horizon_bins=horizon_bins,
        feat_cols=list(feat_cols),
        metrics=metrics_payload,
        extras=extras,
        include_features=False,
    )

    # garder une copie locale latest.json (utile pour next-version en local)
    try:
        latest_local = Path(arte_base) / horizon_slug / "latest.json"
        latest_local.write_text(Path(local_json_path).read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    # Publication GCS (répertoire daté + alias latest.*)
    if gcs_bucket and gcs_prefix:
        try:
            _maybe_publish_to_gcs(local_model_path, local_json_path, bucket=gcs_bucket, prefix_h15=gcs_prefix)
        except Exception as e:
            print(f"[publish][warn] GCS publish failed: {e}")

    # Compat out_path si appelé par un orchestrateur
    if out_path and out_path != "model.joblib":
        try:
            joblib.dump({"model": model, "feat_cols": list(feat_cols), "horizon_bins": int(horizon_bins)}, out_path)
            print(f"[train] (compat) model saved → {out_path}")
        except Exception as e:
            print(f"[train][warn] could not save compat path '{out_path}': {e}")


def _latest_feature_frame_for_predict(src: str, horizon_bins: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construire les features spatiales pour la **dernière observation par station**.

    Utilisé pour `predict_latest_offline`.

    Étapes
    ------
    1. Lire / normaliser / dédoublonner les parquets.
    2. Identifier, pour chaque station, le dernier `tbin_utc`.
    3. Construire un historique court (60 minutes) avant ce dernier bin.
    4. Appliquer `_add_features_spatial` sur cet historique.
    5. Garder la dernière ligne par station dans ce frame enrichi.

    Returns
    -------
    (Xstar, feat_cols)
        Xstar : features pour la dernière observation par station.
        feat_cols : liste de toutes les colonnes utilisables.
    """
    df = _read_many_parquets(src)
    if df.empty:
        return pd.DataFrame(), []
    df = _coerce_types(df)
    df = _dedupe_per_bin(df)

    # dernière observation par station
    last = (
        df.sort_values(["station_id", "tbin_utc"])
        .groupby("station_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    # historique court (60 min) pour lissage/voisins
    hist = df.merge(last[["station_id", "tbin_utc"]], on="station_id", how="inner", suffixes=("", "_last"))
    hist = hist[hist["tbin_utc"] <= hist["tbin_utc_last"]]
    hist = hist[hist["tbin_utc"] >= hist["tbin_utc_last"] - pd.Timedelta(minutes=60)]

    hist2, feat_cols = _add_features_spatial(hist, horizon_bins=horizon_bins)

    Xstar = (
        hist2.sort_values(["station_id", "tbin_utc"])
        .groupby("station_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return Xstar, list(feat_cols)


def predict_latest_offline(src: str, model_path_or_prefix: str) -> pd.DataFrame:
    """
    Prédire offline sur la **dernière observation** de chaque station.

    Utilise la même pile SPATIALE que le training pour garantir la cohérence
    des features.

    Parameters
    ----------
    src : str
        Fichier/dossier/glob de parquets 5min.
    model_path_or_prefix : str
        Modèle entraîné (joblib direct ou préfixe local/GCS).

    Returns
    -------
    pandas.DataFrame
        Même format que `predict_from_features_df`, mais en lisant les données
        brutes directement.
    """
    pack = _load_joblib_from_uri(model_path_or_prefix)
    model = pack["model"]
    feat_cols = list(pack["feat_cols"])
    horizon_bins = int(pack.get("horizon_bins", 3))

    Xstar, _ = _latest_feature_frame_for_predict(src, horizon_bins)
    if Xstar.empty:
        print("[predict] no base to predict.")
        return pd.DataFrame()

    if "station_id" not in Xstar.columns:
        raise KeyError("Offline predict: missing 'station_id' in Xstar.")
    if "tbin_latest" not in Xstar.columns:
        Xstar["tbin_latest"] = pd.to_datetime(Xstar.get("tbin_utc"), errors="coerce")

    if "capacity_bin" not in Xstar.columns:
        try:
            Xstar["capacity_bin"] = Xstar["capacity"].round().astype("Int64")
        except Exception:
            Xstar["capacity_bin"] = pd.Series([pd.NA] * len(Xstar), dtype="Int64")

    for c in feat_cols:
        if c not in Xstar.columns:
            Xstar[c] = 0.0
    X = Xstar[feat_cols].astype(np.float32)

    yhat = model.predict(X)
    cap = (Xstar["capacity_bin"] if "capacity_bin" in Xstar.columns else Xstar.get("capacity", 0)).to_numpy()
    y_clip, y_int = _clip_and_round(yhat, cap)

    tbin_latest = pd.to_datetime(Xstar["tbin_latest"], errors="coerce")
    hz_min = int(horizon_bins * 5)
    target_ts = tbin_latest + pd.to_timedelta(hz_min, unit="m")

    out = pd.DataFrame(
        {
            "station_id": Xstar["station_id"].astype("Int64"),
            "tbin_latest": tbin_latest,
            "horizon_min": np.int16(hz_min),
            "bikes_pred": y_clip.astype(np.float32),
            "bikes_pred_int": y_int.astype(np.int16),
            "capacity_bin": Xstar["capacity_bin"].astype("Int64"),
            "pred_ts_utc": _utc_now_naive(),
            "target_ts_utc": target_ts,
            "model_version": _derive_model_version_from_uri(model_path_or_prefix),
        }
    )
    return out


# =============================================================================
# CLI
# =============================================================================


def _cli():
    """
    Petit CLI pour lancer un training ou une prédiction offline depuis le shell.

    Commandes
    ---------
    - train :
        python -m service.core.forecast train --src <glob|dir|parquet> --horizon 3 \
            [--start YYYY-MM-DD --end YYYY-MM-DD --lookback-days N]

    - predict :
        python -m service.core.forecast predict --src <glob|dir|parquet> \
            --model <joblib|prefix> [--horizon N]

    Notes
    -----
    - En mode `predict`, si `--horizon` est fourni, on met à jour `horizon_bins`
      dans le pack chargé (en local).
    """
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_tr = sub.add_parser("train", help="Train the XGBoost model (Optuna-best params, SPATIAL forced)")
    ap_tr.add_argument("--src", required=True, help="file/glob/folder of 5-min parquet snapshots (or daily)")
    ap_tr.add_argument("--horizon", type=int, default=3, help="forecast horizon in 5-min bins")
    ap_tr.add_argument("--out", default="model.joblib", help="optional extra output joblib path")
    ap_tr.add_argument("--start", default=None, help="UTC start date YYYY-MM-DD")
    ap_tr.add_argument("--end",   default=None, help="UTC end date YYYY-MM-DD")
    ap_tr.add_argument("--lookback-days", type=int, default=None, help="limit training to the last N days")

    ap_pr = sub.add_parser("predict", help="Predict on the latest observed bin per station (offline, reads sources)")
    ap_pr.add_argument("--src", required=True, help="file/glob/folder of 5-min parquet snapshots")
    ap_pr.add_argument("--horizon", type=int, default=None, help="(optional) override horizon (will update pack)")
    ap_pr.add_argument("--model", required=True, help="trained model joblib OR prefix (GCS/local)")

    args = ap.parse_args()

    if args.cmd == "train":
        train_model(
            args.src, horizon_bins=args.horizon, out_path=args.out,
            start_date=args.start, end_date=args.end, lookback_days=args.lookback_days
        )
    elif args.cmd == "predict":
        if args.horizon is not None:
            pack = _load_joblib_from_uri(args.model)
            pack["horizon_bins"] = int(args.horizon)
            try:
                if not _is_gcs_uri(args.model):
                    resolved = _resolve_model_uri(args.model)
                    joblib.dump(pack, resolved)
            except Exception:
                pass
        preds = predict_latest_offline(args.src, args.model)
        if not preds.empty:
            print(preds.head(12).to_string(index=False))


if __name__ == "__main__":
    _cli()
