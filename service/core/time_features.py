# service/core/time_features.py
# =============================================================================
# Construction de la frame d'entraînement à partir de snapshots Parquet 5 minutes
# (météo déjà jointe) :
# - schéma strict
# - déduplication par bin
# - cible y_nb à +horizon_bins (5 minutes par bin)
# - lags sur les vélos, stats glissantes, lags météo simples
# - features calendaires + sinusoïdales (UTC + Paris)
#
# Aligné avec le notebook (velib-lgbm.ipynb) :
#   - lag_bins : (1, 2, 3, 6, 12, 24, 36, 48)
#   - fenêtres rolling : (3, 6, 12, 24, 36, 48) sur bikes avec shift(1)
#   - lags météo : 1 pour (temp_C, precip_mm, wind_mps)
#   - occ_ratio = bikes / capacity (>0)
# =============================================================================

"""
Feature engineering temporel pour les snapshots Vélib' en pas de 5 minutes.

Ce module construit la frame d'entraînement **purement temporelle** (non spatiale)
utilisée par les modèles de prévision. Il suppose que :

- les données brutes viennent de snapshots Parquet 5 minutes avec la météo
  **déjà jointe** (colonnes temp_C, precip_mm, wind_mps),
- le schéma est (ou sera forcé à) `BASE_COLUMNS`,
- station_id est traité comme une chaîne au moment de la construction des features.

Pipeline haut niveau
--------------------
1. I/O
   - `_read_many_parquets` lit un ou plusieurs fichiers Parquet, normalise les
     colonnes et garantit la présence de toutes les `BASE_COLUMNS`.
   - `_coerce_types` impose un régime de types stable (timestamps, numériques, chaînes).
   - `_dedupe_per_bin` conserve le **dernier** enregistrement par `(station_id, tbin_utc)`.

2. Feature engineering
   - `_add_target_and_lags` ajoute :
       * la cible y_nb à +horizon_bins (5 minutes x horizon_bins),
       * des lags sur le nombre de vélos,
       * des moyennes/écarts-types glissants avec shift(1) pour éviter les fuites,
       * des lags météo simples (L=1),
       * le taux d’occupation,
       * des features calendaires + sinusoïdales via `add_time_features`,
       * un encodage ordinal du status → status_code.
   - `build_training_frame` orchestre l’ensemble et retourne :
       (full_df, X, y, feat_cols).

API publique
------------
- `build_training_frame(src, start_date=None, end_date=None, horizon_bins=3)`
    → (full_df, X, y, feat_cols)

Contrat avec le code d'entraînement :
- X est un DataFrame float32 contenant **uniquement** les colonnes listées
  dans `feat_cols`,
- y est la Series float32 `y_nb`,
- full_df est le DataFrame enrichi (cible, features et colonnes brutes),
- la déduplication par bin et le filtrage par dates ont déjà été appliqués.
"""

from __future__ import annotations
import os, glob
from typing import Iterable, Tuple, List, Optional
import pandas as pd

# Import relatif préféré ; fallback vers service.train.* pour anciens layouts
try:
    from .cal_features import add_time_features
except Exception:
    from service.train.cal_features import add_time_features  # type: ignore

# Colonnes attendues après ingestion + jointure météo.
BASE_COLUMNS = [
    "ts_utc", "tbin_utc", "station_id", "bikes", "capacity", "mechanical", "ebike",
    "status", "lat", "lon", "name", "temp_C", "precip_mm", "wind_mps"
]

# ───────────────────────── I/O ─────────────────────────

def _read_many_parquets(path_or_glob: str) -> pd.DataFrame:
    """
    Lire un ou plusieurs fichiers Parquet et normaliser sur `BASE_COLUMNS`.

    Paramètres
    ----------
    path_or_glob : str
        - Chemin vers un fichier `.parquet`,
        - Chemin vers un répertoire contenant des `.parquet`,
        - Motif glob du type `"path/to/*.parquet"`.

    Retour
    ------
    pandas.DataFrame
        DataFrame concaténé avec au minimum `BASE_COLUMNS` :
        toute colonne de base manquante est créée remplie avec NA,
        et l’ordre final des colonnes est exactement `BASE_COLUMNS`.

    Notes
    -----
    - Tout fichier qui échoue au chargement est ignoré avec un warning.
    - Si aucun fichier ne peut être lu, on retourne un DataFrame vide
      avec `BASE_COLUMNS`.
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

    # assurer la présence du schéma
    for c in BASE_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    return out[BASE_COLUMNS]

# ───────────────────────── Cleaning ─────────────────────────

def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forcer les types des colonnes de base (aligné avec le notebook).

    Transformations
    ---------------
    - `ts_utc`, `tbin_utc` :
        * convertis en datetimes UTC aware,
        * puis convertis en datetimes UTC **naïfs** via `tz_convert(None)`.
    - `station_id` :
        * typé en `string` pandas (plus robuste que object).
    - `bikes`, `capacity`, `mechanical`, `ebike` :
        * cast en numérique via `pd.to_numeric(errors="coerce")`.
    - `lat`, `lon`, `temp_C`, `precip_mm`, `wind_mps` :
        * même coercition numérique.
    - `status`, `name` :
        * typés en `string`.

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame brut en entrée.

    Retour
    ------
    pandas.DataFrame
        Même DataFrame avec des dtypes normalisés.
    """
    df["ts_utc"]   = pd.to_datetime(df["ts_utc"],   utc=True, errors="coerce").dt.tz_convert(None)
    df["tbin_utc"] = pd.to_datetime(df["tbin_utc"], utc=True, errors="coerce").dt.tz_convert(None)

    # Conserver station_id en string pour la frame d'entraînement (comme dans le notebook)
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
    Conserver le **dernier** enregistrement par (station_id, tbin_utc) via ts_utc.

    Stratégie
    ---------
    - Tri sur (station_id, tbin_utc, ts_utc) ascendant.
    - Pour chaque groupe (station_id, tbin_utc), on garde le `tail(1)`
      (i.e. l’enregistrement avec le ts_utc le plus récent).
    - Reset de l’index pour faciliter les traitements downstream.

    Paramètres
    ----------
    df : pandas.DataFrame
        Frame d’entrée où plusieurs snapshots peuvent exister par bin.

    Retour
    ------
    pandas.DataFrame
        Frame dédupliquée, avec au plus une ligne par (station_id, tbin_utc).
    """
    df = df.sort_values(["station_id", "tbin_utc", "ts_utc"], ascending=[True, True, True])
    dedup = df.groupby(["station_id", "tbin_utc"], as_index=False).tail(1)
    return dedup.reset_index(drop=True)

# ───────────────────────── Feature engineering ─────────────────────────

def _add_target_and_lags(
    df: pd.DataFrame,
    horizon_bins: int = 3,
    lag_bins: Iterable[int] = (1, 2, 3, 6, 12, 24, 36, 48),
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Enrichir la frame avec la **cible** et les features temporelles.

    Ajouts
    ------
    - `y_nb` :
        Cible = bikes à +`horizon_bins` bins de 5 minutes
        (soit horizon_bins * 5 minutes dans le futur).
    - `lag_bikes_L` :
        Pour chaque L dans `lag_bins`, bikes décalé de L bins.
    - `roll_mean_W`, `roll_std_W` :
        Pour chaque fenêtre W dans (3, 6, 12, 24, 36, 48) bins :
        moyenne / écart-type glissants de bikes, après un `shift(1)` pour
        éviter toute fuite d’information sur la cible.
    - `occ_ratio` :
        bikes / capacity, uniquement pour `capacity > 0` (sinon NaN).
    - lags météo simples (L=1 bin) :
        `temp_C_lag1`, `precip_mm_lag1`, `wind_mps_lag1`.
    - features calendaires :
        via `add_time_features(df, ts_col="tbin_utc", add_paris_derived=True)`.
    - `status_code` :
        encodage ordinal de la variable catégorielle `status`
        (valeurs uniques triées).

    Paramètres
    ----------
    df : pandas.DataFrame
        Frame propre et dédupliquée (types déjà forcés).
    horizon_bins : int, par défaut 3
        Horizon de prévision en bins de 5 minutes (3 → 15 minutes, 12 → 1 heure, etc.).
    lag_bins : Iterable[int]
        Collection de tailles de lags (en bins) à utiliser pour les vélos.

    Retour
    ------
    (df, feat_cols)
        df : pandas.DataFrame
            Même DataFrame, enrichi de la cible et de toutes les features.
        feat_cols : list[str]
            Liste ordonnée des noms de colonnes de features (hors cible).
    """
    df = df.sort_values(["station_id", "tbin_utc"])

    # Cible
    df["y_nb"] = df.groupby("station_id", group_keys=False)["bikes"].shift(-horizon_bins)

    # Lags sur bikes
    for L in lag_bins:
        df[f"lag_bikes_{L}"] = df.groupby("station_id", group_keys=False)["bikes"].shift(L)

    # Fenêtres rolling avec shift(1) pour éviter les fuites
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

    # Taux d'occupation
    df["occ_ratio"] = df["bikes"] / df["capacity"].where(df["capacity"] > 0)

    # Lags météo (L=1)
    for c in ("temp_C", "precip_mm", "wind_mps"):
        df[f"{c}_lag1"] = df.groupby("station_id", group_keys=False)[c].shift(1)

    # Features calendaires / temporelles
    add_time_features(df, ts_col="tbin_utc", add_paris_derived=True)

    # Categorical status → code ordinal (stable)
    status_map = {s: i for i, s in enumerate(sorted(df["status"].dropna().unique()))}
    df["status_code"] = df["status"].map(status_map).astype("Int64")

    # Colonnes de features finales (exactement ce qu’attend le notebook)
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

# ───────────────────────── API publique ─────────────────────────

def build_training_frame(
    src: str,
    start_date: Optional[str] = None,
    end_date: Optional[str]   = None,
    horizon_bins: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """
    Construire la frame d'entraînement **temporelle** à partir de snapshots Parquet 5 minutes.

    Étapes
    ------
    1. Chargement des données brutes
       - `_read_many_parquets(src)` collecte tous les snapshots et les normalise
         sur `BASE_COLUMNS`.
    2. Nettoyage / normalisation
       - `_coerce_types` assure des dtypes stables.
       - `_dedupe_per_bin` garde le dernier enregistrement par `(station_id, tbin_utc)`.
    3. Filtrage par dates (optionnel, UTC)
       - Si `start_date` est fourni, on garde les lignes où `tbin_utc >= start_date`.
       - Si `end_date` est fourni, on garde les lignes où
         `tbin_utc <= end_date + 1 jour - 5 minutes`.
    4. Feature engineering
       - `_add_target_and_lags` ajoute la cible y_nb et toutes les features temporelles.
    5. Construction de la frame modèle
       - `df_model` = sous-ensemble de df avec :
           ["station_id", "tbin_utc", "bikes", "y_nb"] + feat_cols,
         après suppression des lignes où `y_nb` est NA.
       - X = df_model[feat_cols].astype("float32").
       - y = df_model["y_nb"].astype("float32").

    Paramètres
    ----------
    src : str
        Chemin fichier, répertoire ou motif glob pointant vers les snapshots Parquet 5 minutes.
    start_date : str | None, défaut None
        Borne inférieure (incluse) sur `tbin_utc`, au format "YYYY-MM-DD".
    end_date : str | None, défaut None
        Borne supérieure (incluse) sur `tbin_utc`, au format "YYYY-MM-DD".
        Convertie en interne en `end_date + 1 jour - 5 minutes`.
    horizon_bins : int, défaut 3
        Horizon de prévision en bins de 5 minutes (passé à `_add_target_and_lags`).

    Retour
    ------
    (full_df, X, y, feat_cols)
        full_df : pandas.DataFrame
            Frame enrichie (colonnes brutes + y_nb + toutes les features).
        X : pandas.DataFrame
            Matrice de features prête pour l'entraînement (float32).
        y : pandas.Series
            Série cible y_nb (float32).
        feat_cols : list[str]
            Liste ordonnée des colonnes de features correspondant aux colonnes de X.

    Notes
    -----
    - Si aucune donnée ne peut être lue, retourne :
        (empty_df, empty_df, empty_series, []).
    - Cette fonction ne réalise **aucun** split train/validation ; elle est
      volontairement générique pour laisser différentes stratégies
      d'entraînement se brancher au-dessus.
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

    df, feat_cols = _add_target_and_lags(df, horizon_bins=horizon_bins)

    # Suppression des NA sur la cible
    used_cols = ["station_id", "tbin_utc", "bikes", "y_nb"] + feat_cols
    df_model = df[used_cols].dropna(subset=["y_nb"])

    X = df_model[feat_cols].astype("float32").copy()
    y = df_model["y_nb"].astype("float32").copy()

    return df, X, y, feat_cols