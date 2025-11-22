# cal_features.py

# ============================================================================
# Petites features calendaires (UTC + Paris) et encodages sin/cos.
# Entrée attendue: une colonne temporelle (par défaut 'tbin_utc') au format
# pandas.Timestamp naive UTC.
# ============================================================================

from __future__ import annotations
import pandas as pd


def _safe_to_datetime_utc(s: pd.Series) -> pd.Series:
    """
    Convert a Series to UTC datetimes and return a **naive UTC** Series.

    Parameters
    ----------
    s : pandas.Series
        Series contenant des timestamps (ou des chaînes) à convertir.

    Returns
    -------
    pandas.Series
        Datetime64[ns] **naïf** (sans tz-info), mais interprété comme UTC.

    Notes
    -----
    - On parse avec `utc=True` pour normaliser en UTC.
    - Puis on enlève la timezone pour rester cohérent avec les parquets
      (toutes les colonnes temporelles du projet sont stockées en naïf UTC).
    - Les valeurs non convertibles deviennent NaT.
    """
    t = pd.to_datetime(s, utc=True, errors="coerce")
    # on garde naïf UTC (pas de tz-info) pour rester cohérent avec les parquets
    return t.dt.tz_convert(None)


def add_time_features(
    df: pd.DataFrame,
    ts_col: str = "tbin_utc",
    add_paris_derived: bool = True,
) -> pd.DataFrame:
    """
    Ajouter des features calendaires (UTC + Paris) + encodages sin/cos.

    Features ajoutées (UTC)
    -----------------------
    À partir de la colonne `ts_col` (naïf UTC, ou convertissable en datetime) :

    - hour        : heure (0–23) en UTC
    - minute      : minute (0–59)
    - dow         : day-of-week, 0 = lundi, 6 = dimanche (UTC)
    - month       : mois (1–12)
    - is_weekend  : 1 si dow ∈ {5,6}, sinon 0

    Encodages cycliques (sin/cos)
    ------------------------------
    - hod_sin / hod_cos : encodage horaire sur 24h
          angle = (hour + minute/60) / 24 * 2π
    - dow_sin / dow_cos : encodage du jour de semaine sur 7 jours

    Dérivés Paris (optionnels)
    --------------------------
    Si `add_paris_derived=True` :

    - paris_hour  : heure locale Europe/Paris (Int64)
    - paris_dow   : day-of-week en Europe/Paris (0=lundi)
    - paris_is_we : 1 si samedi/dimanche (en Europe/Paris), sinon 0

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame dans lequel les colonnes seront ajoutées (modifié **en place**).
    ts_col : str, default "tbin_utc"
        Nom de la colonne temporelle d'entrée.
    add_paris_derived : bool, default True
        Si True, calcule aussi les features dérivées dans le fuseau Europe/Paris.

    Returns
    -------
    pandas.DataFrame
        Le même DataFrame `df` (modifié en place), renvoyé pour chaînage éventuel.

    Comportement en cas d'absence de colonne
    ----------------------------------------
    Si `ts_col` n'existe pas dans `df`, la fonction renvoie `df` sans modification.
    """
    if ts_col not in df.columns:
        # rien à faire si la colonne temporelle n'existe pas
        return df

    # Normalisation en datetime naïf UTC
    t = _safe_to_datetime_utc(df[ts_col])

    # Features UTC de base
    df["hour"]   = t.dt.hour
    df["minute"] = t.dt.minute
    df["dow"]    = t.dt.dayofweek  # 0=lundi
    df["month"]  = t.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")

    # Encodages sin/cos (cercle 24h et 7j)
    import numpy as np
    df["hod_sin"] = np.sin(2 * np.pi * (df["hour"] + df["minute"]/60.0) / 24.0)
    df["hod_cos"] = np.cos(2 * np.pi * (df["hour"] + df["minute"]/60.0) / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * (df["dow"]) / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * (df["dow"]) / 7.0)

    if add_paris_derived:
        # Reparse avec timezone et conversion en Europe/Paris
        t_paris = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dt.tz_convert("Europe/Paris")
        df["paris_hour"] = t_paris.dt.hour.astype("Int64")
        df["paris_dow"]  = t_paris.dt.dayofweek.astype("Int64")
        df["paris_is_we"] = (df["paris_dow"] >= 5).astype("Int64")

    return df
