# service/jobs/build_network_overview.py

"""
Vélib’ Forecast — Job de monitoring "Network Overview" (LATEST ONLY).

Rôle
----
Ce job lit les fichiers parquet *events* :

    gs://.../velib/exports/events_YYYY-MM-DD.parquet

sur une fenêtre glissante de plusieurs jours et construit tous les artefacts JSON
nécessaires à la page **Network Overview** de l’UI de monitoring :

- kpis.json
    * KPIs globaux (stations actives, disponibilité du snapshot,
      couverture 7 jours, volatilité aujourd’hui, etc.)
- snapshot_distribution.json
    * décomposition en comptes / pourcentages (vélos disponibles, bornes dispo,
      pénurie, saturation) pour le dernier snapshot réseau
- snapshot_map.json
    * info par station pour la carte (bikes, docks_avail, flags pénurie/saturation)
- today_curve.json
    * série temporelle du "% de stations avec ≥1 vélo" sur le "today" choisi
      (day_for_today_utc)
- ref_median_curve.json
    * courbe médiane de référence sur les jours passés ayant le même weekday
      (UTC, bins 5 minutes)
- kpis_today_vs_lags.json
    * KPIs journaliers pour today vs J-7 / J-14 / J-21 (disponibilité & pénurie/saturation)
- stations_tension.json
    * taux de pénurie/saturation par station (pour le ranking de "tension")

Toutes les sorties sont publiées sous :

    <GCS_MONITORING_PREFIX>/monitoring/network/overview/latest/

avec un `manifest.json` top-level qui décrit la fenêtre, la timezone, les sources
et les artefacts produits.

Environnement
-------------
Obligatoire :
    GCS_EXPORTS_PREFIX    = gs://bucket/velib/exports
    GCS_MONITORING_PREFIX = gs://bucket/velib      (ou .../monitoring)

Optionnel (ENV unifié MON_* avec compat legacy) :
    MON_TZ         = Europe/Paris
    MON_LAST_DAYS  = 7    (utilisé pour coverage & tension)
    MON_REF_DAYS   = 28   (historique pour la courbe médiane de référence)

Alias legacy encore pris en compte :
    OVERVIEW_TZ, OVERVIEW_LAST_DAYS, OVERVIEW_REF_DAYS

Notes
-----
- Le JSON est nettoyé : NaN/±Inf → null.
- Sorties en mode *LATEST ONLY* : pas de dossiers datés ; le front lit toujours dans `latest/`.
- `day_for_today_utc` est dérivé du nom du dernier fichier `events_YYYY-MM-DD.parquet`;
  tous les calculs "today vs ref" sont ancrés sur ce jour UTC.
"""

from __future__ import annotations
import os, re, json, sys
from io import BytesIO
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception as e:
    raise RuntimeError("pyarrow requis") from e

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage requis") from e

SCHEMA_VERSION = "1.3"  # 1.2 → 1.3 (ENV unifié MON_*, manifest, LATEST only)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers ENV (unifiés)
# ──────────────────────────────────────────────────────────────────────────────

def _env(name: str, default=None):
    """
    Lit une variable d’environnement avec une valeur par défaut.

    Paramètres
    ----------
    name : str
        Nom de la variable d’environnement.
    default : Any
        Valeur de repli si la variable est absente ou vide.

    Retourne
    --------
    Any
        Valeur brute (str) lue dans l’ENV, ou la valeur par défaut.
    """
    v = os.environ.get(name)
    return v if (v is not None and v != "") else default


def _env_int(name: str, default: int) -> int:
    """
    Lit une variable d’environnement de type entier.

    En cas d’échec de parsing, retourne la valeur par défaut.

    Paramètres
    ----------
    name : str
        Nom de la variable d’environnement.
    default : int
        Valeur entière de repli.

    Retourne
    --------
    int
        Entier parsé ou valeur par défaut.
    """
    try:
        return int(_env(name, default))
    except Exception:
        return default


# ──────────────────────────────────────────────────────────────────────────────
# Helpers GCS
# ──────────────────────────────────────────────────────────────────────────────

def _split(gs: str) -> Tuple[str, str]:
    """
    Découpe une URI GCS `gs://bucket/path` en (bucket, key).

    Paramètres
    ----------
    gs : str
        URI GCS.

    Retourne
    --------
    (str, str)
        Nom de bucket et clé d’objet (sans slash final).

    Lève
    ----
    AssertionError
        Si l’URI ne commence pas par `gs://`.
    """
    assert gs.startswith("gs://"), f"bad GCS uri: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k.rstrip("/")


def _read_parquet_blob_to_df(blob: "storage.Blob") -> pd.DataFrame:
    """
    Lit un blob parquet GCS dans un DataFrame pandas.

    Paramètres
    ----------
    blob : google.cloud.storage.Blob
        Blob GCS pointant vers un fichier parquet.

    Retourne
    --------
    pandas.DataFrame
        DataFrame chargé depuis le parquet.
    """
    buf = BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    tbl = pq.read_table(buf)
    return tbl.to_pandas()


def _list_event_blobs(exports_prefix: str, start_date: datetime, end_date: datetime) -> List["storage.Blob"]:
    """
    Liste les blobs `events_YYYY-MM-DD.parquet` sur une fenêtre de dates UTC.

    Le filtrage se fait sur la date encodée dans le nom de fichier.

    Paramètres
    ----------
    exports_prefix : str
        Préfixe GCS de base pour les exports (GCS_EXPORTS_PREFIX).
    start_date : datetime
        Début de fenêtre (inclus, UTC).
    end_date : datetime
        Fin de fenêtre (inclus, UTC).

    Retourne
    --------
    list[google.cloud.storage.Blob]
        Liste triée de blobs correspondant à `events_YYYY-MM-DD.parquet`
        dans la fenêtre demandée.
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


def _upload_json_gs(obj: dict, gs_uri: str, log_prefix: str = "network.overview"):
    """
    Envoie un document JSON vers GCS, en remplaçant NaN/±Inf par null.

    Notes
    -----
    - Parcourt récursivement dicts/listes et remplace les floats non finis par null.
    - Encode le JSON en UTF-8, séparateurs compacts.
    - Écrit sur l’URI GCS cible avec content-type `application/json`.

    Paramètres
    ----------
    obj : dict
        Payload JSON-sérialisable (sera nettoyé).
    gs_uri : str
        URI GCS cible.
    log_prefix : str, défaut "network.overview"
        Préfixe utilisé dans les logs.
    """
    # Nettoyage JSON : remplace NaN/±Inf par null
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
    print(f"[{log_prefix}] wrote → {gs_uri} ({len(data):,} bytes)")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers cœur
# ──────────────────────────────────────────────────────────────────────────────

def _detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Auto-détecte les colonnes clés du DataFrame d’évènements.

    La fonction est tolérante à différents schémas et renvoie un mapping
    avec les clés (valeurs éventuellement None) :

        ts, station, bikes, capacity, docks, name, lat, lon

    Les colonnes minimales requises sont : timestamp, station_id, bikes.

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame brut d’évènements.

    Retourne
    --------
    dict
        Mapping des noms logiques vers les noms effectifs.

    Lève
    ----
    KeyError
        Si les colonnes minimales sont absentes.
    """
    lower = {c.lower(): c for c in df.columns}

    def any_of(*cands):
        for c in cands:
            if c in lower:
                return lower[c]
        return None

    ts = any_of("ts", "tbin_utc", "timestamp", "datetime")
    st = any_of("station_id", "stationcode", "id", "station")
    bikes = any_of("bikes", "num_bikes_available", "velos", "velos_disponibles")
    cap = any_of("capacity", "num_docks_total", "dock_count", "cap")
    docks = any_of("docks_avail", "num_docks_available", "places_disponibles", "free_docks")
    name = any_of("name", "station_name", "nom")
    lat = any_of("lat", "latitude")
    lon = any_of("lon", "lng", "longitude")
    if not ts or not st or not bikes:
        raise KeyError(f"[overview] Colonnes minimales absentes (ts={ts}, station={st}, bikes={bikes})")
    return dict(ts=ts, station=st, bikes=bikes, capacity=cap, docks=docks, name=name, lat=lat, lon=lon)


def _to_local(s_utc_like: pd.Series, tzname: str) -> pd.Series:
    """
    Convertit une série de timestamps en datetimes localisés (tz-aware).

    Paramètres
    ----------
    s_utc_like : pandas.Series
        Série convertible en datetime (supposée en UTC).
    tzname : str
        Nom de la timezone cible (ex. "Europe/Paris").

    Retourne
    --------
    pandas.Series
        Série de datetimes localisés dans la timezone cible.
    """
    s = pd.to_datetime(s_utc_like, utc=True, errors="coerce")
    return s.dt.tz_convert(tzname)


def _today_bounds_local(series_local: pd.Series) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Donne les bornes [start, end) du jour local du dernier timestamp de la série.

    Paramètres
    ----------
    series_local : pandas.Series
        Série de datetimes localisés.

    Retourne
    --------
    (pandas.Timestamp, pandas.Timestamp)
        Début (00:00) et fin (00:00 du jour suivant) de ce jour local.
    """
    tmax = series_local.max()
    start = tmax.normalize()
    end = start + pd.Timedelta(days=1)
    return start, end


def _safe_ratio(n: float, d: float) -> float:
    """
    Calcule en sécurité (n / d * 100), arrondi à 2 décimales.

    Retourne
    --------
    float
        Pourcentage n/d*100, ou NaN si d == 0.
    """
    return float("nan") if d == 0 else round(100.0 * n / d, 2)


def _part_bool(x: pd.Series) -> float:
    """
    Pourcentage (0–100) de True dans une série booléenne.

    Retourne NaN pour une série vide.

    Paramètres
    ----------
    x : pandas.Series
        Série booléenne.

    Retourne
    --------
    float
        Pourcentage de valeurs True.
    """
    if x.size == 0:
        return float("nan")
    return float((x.mean() * 100.0).round(2))


def _safe_num(v: object) -> Optional[float]:
    """
    Conversion robuste vers float, renvoyant None pour NaN/Inf ou erreur.

    Paramètres
    ----------
    v : Any
        Valeur d’entrée.

    Retourne
    --------
    float | None
        Float fini ou None.
    """
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except Exception:
        return None


def _compute_snapshot(
    df: pd.DataFrame,
    cols: Dict[str, Optional[str]],
    tzname: str,
    station_universe: List[str],
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Construit le snapshot global réseau et les structures dérivées.

    Définition du snapshot
    ----------------------
    - On prend le dernier timestamp présent dans la fenêtre.
    - On sélectionne toutes les lignes exactement à ce timestamp.
    - Par station, on dérive :
        * has_bike, has_dock
        * penury (bikes == 0)
        * saturation (docks == 0)
    - KPIs globaux :
        * stations_active vs univers
        * availability_bike_pct, availability_dock_pct
        * penury_pct, saturation_pct

    La fonction renvoie aussi :
    - `dist` : petit DataFrame de distribution pour le snapshot
    - `map_df` : index pour la carte (une ligne par station).

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame évènementiel déjà filtré sur la fenêtre temporelle.
    cols : dict
        Mapping de colonnes tel que `_detect_columns`.
    tzname : str
        Timezone pour snapshot_ts_local.
    station_universe : list[str]
        Liste des stations faisant partie de l’univers.

    Retourne
    --------
    (dict, pandas.DataFrame, pandas.DataFrame)
        - kpis : KPIs globaux (dict JSON-ready)
        - dist : distribution instantanée
        - map_df : DataFrame pour la carte snapshot.
    """
    df = df.copy()
    last_ts = pd.to_datetime(df[cols["ts"]], utc=True, errors="coerce").max()
    snap = df[df[cols["ts"]] == last_ts].copy()

    # dérive docks_avail si absent
    docks_col = cols["docks"]
    if docks_col is None and cols["capacity"]:
        docks = (
            pd.to_numeric(snap[cols["capacity"]], errors="coerce")
            - pd.to_numeric(snap[cols["bikes"]], errors="coerce")
        ).clip(lower=0)
        snap["__docks_avail"] = docks
        docks_col = "__docks_avail"

    bikes = pd.to_numeric(snap[cols["bikes"]], errors="coerce")
    has_bike = bikes > 0
    has_dock = None
    sat = pen = None
    if docks_col and docks_col in snap.columns:
        docks = pd.to_numeric(snap[docks_col], errors="coerce")
        has_dock = docks > 0
        sat = docks == 0
    pen = bikes == 0

    active = snap[cols["station"]].astype(str).nunique()
    universe = len(station_universe)
    offline = max(universe - active, 0)

    kpis = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "snapshot_ts_utc": pd.Timestamp(last_ts).tz_convert("UTC").isoformat().replace("+00:00", "Z"),
        "snapshot_ts_local": str(_to_local(pd.Series([last_ts]), tzname).iloc[0]),
        "stations_universe": universe,
        "stations_active": int(active),
        "stations_offline": int(offline),
        "availability_bike_pct": _part_bool(has_bike) if has_bike is not None else float("nan"),
        "availability_dock_pct": _part_bool(has_dock) if has_dock is not None else float("nan"),
        "penury_pct": _part_bool(pen) if pen is not None else float("nan"),
        "saturation_pct": _part_bool(sat) if sat is not None else float("nan"),
    }

    # Distribution instantanée (counts + pct)
    dist = pd.DataFrame(
        {
            "metric": ["bike_avail", "dock_avail", "penury", "saturation"],
            "count": [
                int(has_bike.sum()) if has_bike is not None else 0,
                int(has_dock.sum()) if has_dock is not None else 0,
                int(pen.sum()) if pen is not None else 0,
                int(sat.sum()) if sat is not None else 0,
            ],
        }
    )
    dist["total_active"] = active
    dist["pct"] = dist.apply(lambda r: _safe_ratio(r["count"], r["total_active"]), axis=1)

    # Index carte snapshot
    map_rows = []
    latc = cols["lat"]
    lonc = cols["lon"]
    namec = cols["name"]
    for _, row in snap.iterrows():
        sid = str(row[cols["station"]])
        lat = float(row[latc]) if latc and pd.notna(row[latc]) else None
        lon = float(row[lonc]) if lonc and pd.notna(row[lonc]) else None
        name = str(row[namec]) if namec and pd.notna(row[namec]) else sid
        b = int(pd.to_numeric(row[cols["bikes"]], errors="coerce")) if pd.notna(row[cols["bikes"]]) else None
        d = None
        if docks_col and docks_col in snap.columns:
            d = int(pd.to_numeric(row[docks_col], errors="coerce")) if pd.notna(row[docks_col]) else None
        map_rows.append(
            {
                "station_id": sid,
                "name": name,
                "lat": lat,
                "lon": lon,
                "bikes": b,
                "docks_avail": d,
                "is_penury": (1 if (b == 0 if b is not None else False) else 0),
                "is_saturation": (1 if (d == 0 if d is not None else False) else 0),
            }
        )
    map_df = pd.DataFrame(map_rows)

    return kpis, dist, map_df


def _coverage_volatility(
    df: pd.DataFrame,
    cols: Dict[str, Optional[str]],
    tzname: str,
    last_days: int,
) -> Tuple[float, float]:
    """
    Calcule la couverture 7j et la KPI "volatility_today".

    Coverage (par station)
    ----------------------
    - Fenêtre : `last_days` derniers jours (temps local).
    - Pour chaque station :
        coverage = (# timestamps vus pour la station) / (# timestamps distincts globaux)
    - coverage_7d_pct global = moyenne coverage stations × 100.

    Volatility today
    ----------------
    - Fenêtre : jour local courant (00:00–24:00).
    - Pour chaque station : écart-type des bikes sur la journée.
    - volatility_today = médiane de ces écart-types.

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame évènementiel sur la fenêtre.
    cols : dict
        Mapping de colonnes.
    tzname : str
        Timezone.
    last_days : int
        Nombre de jours pour la couverture.

    Retourne
    --------
    (float, float)
        coverage_pct, volatility_today.
    """
    if last_days <= 0:
        return float("nan"), float("nan")
    ts_local = _to_local(df[cols["ts"]], tzname)
    tmax = ts_local.max()
    start = tmax - pd.Timedelta(days=last_days)
    win = df.loc[(ts_local >= start) & (ts_local <= tmax)].copy()
    if win.empty:
        return float("nan"), float("nan")
    total_ts = win[cols["ts"]].nunique()
    per_station = (
        win.groupby(cols["station"])[cols["ts"]].nunique() / max(total_ts, 1)
    ).reindex(win[cols["station"]].unique()).fillna(0.0)
    coverage_pct = float((per_station.mean() * 100.0).round(2))

    start_day, end_day = _today_bounds_local(ts_local)
    today = df[(ts_local >= start_day) & (ts_local < end_day)].copy()
    vol = (today.groupby(cols["station"])[cols["bikes"]].std(ddof=0)).median()
    return coverage_pct, float(0.0 if pd.isna(vol) else round(vol, 2))


def _today_vs_median_utc(
    df: pd.DataFrame,
    cols: Dict[str, Optional[str]],
    tzname: str,
    ref_days: int,
    day_for_today_utc: datetime,
) -> Tuple[dict, dict]:
    """
    Construit :
      - today_curve : % de stations avec ≥1 vélo, par bin de 5 minutes,
        pour le jour UTC choisi (day_for_today_utc)
      - ref_median_curve : courbe médiane sur les `ref_days` jours précédents
        ayant le même weekday, par bin UTC 5 minutes.

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame évènementiel sur la fenêtre.
    cols : dict
        Mapping de colonnes.
    tzname : str
        Nom de timezone (gardé pour symétrie, peu utilisé ici).
    ref_days : int
        Nombre de jours pour construire la référence.
    day_for_today_utc : datetime
        Jour UTC de référence utilisé comme "today".

    Retourne
    --------
    (dict, dict)
        today_curve_doc, ref_median_doc (dicts JSON-ready).
    """
    df = df.copy()
    ts_utc = pd.to_datetime(df[cols["ts"]], utc=True, errors="coerce")
    df["_ts_utc"] = ts_utc
    df["_hhmm"] = df["_ts_utc"].dt.strftime("%H:%M")
    df["_weekday"] = df["_ts_utc"].dt.weekday

    bikes_all = pd.to_numeric(df[cols["bikes"]], errors="coerce")
    df["__has_bike"] = bikes_all > 0

    start_utc = pd.Timestamp(day_for_today_utc)
    end_utc = start_utc + pd.Timedelta(days=1)

    # sous-ensemble du jour "today" en UTC
    today = df[(ts_utc >= start_utc) & (ts_utc < end_utc)].copy()

    # fenêtre de référence : ref_days, même weekday
    ref_start = start_utc - pd.Timedelta(days=ref_days)
    ref = df[
        (ts_utc >= ref_start)
        & (ts_utc < start_utc)
        & (df["_weekday"] == start_utc.weekday())
    ].copy()

    def agg_part(d: pd.DataFrame) -> pd.DataFrame:
        """
        Agrège une sous-fenêtre en % de stations avec ≥1 vélo, par ts & hhmm.
        """
        if d.empty:
            return pd.DataFrame(columns=["_hhmm", "pct"])
        out = (
            d.groupby(["_ts_utc", "_hhmm"])[["__has_bike"]]
            .agg(has_bike=("__has_bike", "mean"), n=("__has_bike", "size"))
            .reset_index()
        )
        out["pct"] = out["has_bike"] * 100.0
        return out

    cur = agg_part(today)
    ref_curve = agg_part(ref)

    # médiane par hh:mm sur la référence (UTC)
    bins = pd.Index(pd.date_range("00:00", "23:55", freq="5min").strftime("%H:%M"))
    ref_med = ref_curve.groupby("_hhmm")["pct"].median().reindex(bins)

    today_doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "day_for_today_utc": start_utc.strftime("%Y-%m-%d"),
        "points": [
            {"hhmm": str(h), "pct": _safe_num(v)}
            for h, v in (
                cur[["_hhmm", "pct"]].itertuples(index=False, name=None)
                if not cur.empty
                else []
            )
        ],
    }
    ref_doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "day_for_today_utc": start_utc.strftime("%Y-%m-%d"),
        "median": [
            {"hhmm": str(h), "pct_median": _safe_num(v)}
            for h, v in ref_med.items()
        ],
    }
    return today_doc, ref_doc


def _kpis_today_vs_lags_utc(
    df: pd.DataFrame,
    cols: Dict[str, Optional[str]],
    day_for_today_utc: datetime,
) -> dict:
    """
    Calcule les KPIs journalières pour today vs J-7 / J-14 / J-21 (UTC).

    Pour chaque jour on calcule :
      - avail_bike :  moyenne sur la journée du "% de stations avec ≥1 vélo"
      - avail_dock :  moyenne sur la journée du "% de stations avec ≥1 borne libre"
      - pen :         moyenne du "% de stations en pénurie (bikes == 0)"
      - sat :         moyenne du "% de stations en saturation (docks == 0)"

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame évènementiel.
    cols : dict
        Mapping de colonnes.
    day_for_today_utc : datetime
        Jour UTC de référence pour "today".

    Retourne
    --------
    dict
        Document JSON avec champs "today" et "lags".
    """
    ts_utc_all = pd.to_datetime(df[cols["ts"]], utc=True, errors="coerce")

    def day_kpis(day_start_utc: pd.Timestamp) -> dict:
        """
        Calcule les KPIs disponibilité / pénurie / saturation pour un jour UTC.
        """
        day_end = day_start_utc + pd.Timedelta(days=1)
        d = df[(ts_utc_all >= day_start_utc) & (ts_utc_all < day_end)].copy()
        if d.empty:
            return {"avail_bike": np.nan, "avail_dock": np.nan, "pen": np.nan, "sat": np.nan}

        b = pd.to_numeric(d[cols["bikes"]], errors="coerce")
        has_bike = b > 0

        docks = None
        if cols["docks"] and cols["docks"] in d.columns:
            docks = pd.to_numeric(d[cols["docks"]], errors="coerce")
        elif cols["capacity"] and cols["capacity"] in d.columns:
            docks = (pd.to_numeric(d[cols["capacity"]], errors="coerce") - b).clip(lower=0)

        has_dock = (docks > 0) if docks is not None else None
        pen = b == 0
        sat = (docks == 0) if docks is not None else None

        return {
            "avail_bike": float((has_bike.mean() * 100.0).round(2)),
            "avail_dock": float((has_dock.mean() * 100.0).round(2)) if has_dock is not None else np.nan,
            "pen": float((pen.mean() * 100.0).round(2)),
            "sat": float((sat.mean() * 100.0).round(2)) if sat is not None else np.nan,
        }

    start_today = pd.Timestamp(day_for_today_utc)
    k_today = day_kpis(start_today)
    k_lags = {
        "J-7": day_kpis(start_today - pd.Timedelta(days=7)),
        "J-14": day_kpis(start_today - pd.Timedelta(days=14)),
        "J-21": day_kpis(start_today - pd.Timedelta(days=21)),
    }

    def pack(d: dict) -> dict:
        """
        Convertit toutes les valeurs numériques en floats "safe" (None pour NaN/Inf).
        """
        return {k: (_safe_num(v) if v is not None else None) for k, v in d.items()}

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "day_for_today_utc": start_today.strftime("%Y-%m-%d"),
        "today": pack(k_today),
        "lags": {k: pack(v) for k, v in k_lags.items()},
    }


def _stations_tension(
    df: pd.DataFrame,
    cols: Dict[str, Optional[str]],
    tzname: str,
    last_days: int,
) -> dict:
    """
    Calcule les taux de pénurie / saturation par station sur les `last_days` derniers jours (temps local).

    Pour chaque station :
      - penury_rate     = moyenne sur la fenêtre de (bikes == 0)
      - saturation_rate = moyenne sur la fenêtre de (docks == 0
                          ou capacity - bikes <= 0)

    Utilisé par l’UI Monitoring pour afficher un ranking de "tension".

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame évènementiel sur la fenêtre.
    cols : dict
        Mapping de colonnes.
    tzname : str
        Nom de timezone.
    last_days : int
        Nombre de jours de fenêtre.

    Retourne
    --------
    dict
        Document JSON avec "rows": [{station_id, penury_rate, saturation_rate}, ...].
    """
    ts_loc = _to_local(df[cols["ts"]], tzname)
    tmax = ts_loc.max()
    start_ld = tmax - pd.Timedelta(days=last_days)
    win = df[(ts_loc >= start_ld) & (ts_loc <= tmax)].copy()
    if win.empty:
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "rows": [],
        }

    # taux de pénurie
    pen_rate = win.groupby(cols["station"])[cols["bikes"]].apply(
        lambda s: (pd.to_numeric(s, errors="coerce") == 0).mean()
    )
    # taux de saturation
    if cols["docks"] and cols["docks"] in win.columns:
        sat_rate = win.groupby(cols["station"])[cols["docks"]].apply(
            lambda s: (pd.to_numeric(s, errors="coerce") == 0).mean()
        )
    elif cols["capacity"] and cols["capacity"] in win.columns:
        cap = pd.to_numeric(win[cols["capacity"]], errors="coerce")
        bks = pd.to_numeric(win[cols["bikes"]], errors="coerce")
        sat_rate = ((cap - bks) <= 0).groupby(win[cols["station"]]).mean()
    else:
        sat_rate = pd.Series(np.nan, index=pen_rate.index)

    out = pd.DataFrame(
        {
            "station_id": pen_rate.index.astype(str),
            "penury_rate": pen_rate.values,
            "saturation_rate": sat_rate.values,
        }
    ).sort_values(["penury_rate", "saturation_rate"], ascending=False)
    rows = out.to_dict(orient="records")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "rows": rows,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    """
    Point d’entrée CLI pour le job Network Overview (LATEST ONLY).

    Pipeline
    --------
    1. Lire `GCS_EXPORTS_PREFIX`, `GCS_MONITORING_PREFIX`.
    2. Résoudre la configuration effective :
         - TZNAME (MON_TZ / OVERVIEW_TZ)
         - LAST_DAYS (MON_LAST_DAYS / OVERVIEW_LAST_DAYS)
         - REF_DAYS  (MON_REF_DAYS / OVERVIEW_REF_DAYS)
    3. Définir une fenêtre de lecture UTC `WINDOW_DAYS = max(LAST_DAYS, REF_DAYS, 7)` :
         start = 00:00 de (now - (WINDOW_DAYS - 1))
         end   = now (UTC)
    4. Lister et lire tous les `events_YYYY-MM-DD.parquet` dans la fenêtre.
       Si aucun, publier un manifest minimal et retourner.
    5. Détecter les colonnes, normaliser les types, drop les lignes invalides,
       puis appliquer la fenêtre stricte UTC.
    6. Construire l’univers de stations sur les `WINDOW_DAYS`.
    7. Dériver `day_for_today_utc` du dernier fichier events_* (jour opérationnel).
    8. Calculer tous les artefacts :
         - KPIs snapshot + distribution + index carte,
         - couverture & volatilité,
         - courbes today vs médiane (UTC),
         - KPIs journalières today vs J-7 / J-14 / J-21,
         - tension par station (pénurie/saturation).
    9. Uploader tous les JSON sous :
         <GCS_MONITORING_PREFIX>/monitoring/network/overview/latest/
    10. Construire et uploader `manifest.json` décrivant :
         - version de schéma, window_days, tz,
         - sources (exports prefix),
         - artefacts produits,
         - day_for_today_utc.

    Retourne
    --------
    int
        Code de sortie (0 = succès).
    """
    EXPORTS_PREFIX = _env("GCS_EXPORTS_PREFIX")    # gs://bucket/velib/exports
    MON_PREFIX     = _env("GCS_MONITORING_PREFIX") # gs://bucket/velib (ou .../monitoring)
    if not (EXPORTS_PREFIX and str(EXPORTS_PREFIX).startswith("gs://")):
        raise RuntimeError("GCS_EXPORTS_PREFIX manquant ou invalide")
    if not (MON_PREFIX and str(MON_PREFIX).startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX manquant ou invalide")

    TZNAME    = _env("MON_TZ", _env("OVERVIEW_TZ", "Europe/Paris"))
    LAST_DAYS = _env_int("MON_LAST_DAYS", _env_int("OVERVIEW_LAST_DAYS", 7))
    REF_DAYS  = _env_int("MON_REF_DAYS", _env_int("OVERVIEW_REF_DAYS", 28))

    now = datetime.now(timezone.utc)
    # Fenêtre de lecture : max(LAST_DAYS, REF_DAYS, 7) pour couvrir tous les calculs
    WINDOW_DAYS = max(LAST_DAYS, REF_DAYS, 7)
    start = (now - timedelta(days=WINDOW_DAYS - 1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    print(f"[network.overview] window UTC: {start.date()} → {now.date()} (days={WINDOW_DAYS})")

    # 1) Lire les parquets évènementiels dans la fenêtre
    blobs = _list_event_blobs(EXPORTS_PREFIX, start, now)
    mon_base = MON_PREFIX.rstrip("/")
    if not mon_base.endswith("/monitoring"):
        mon_base = mon_base + "/monitoring"
    base_latest = f"{mon_base}/network/overview/latest"

    if not blobs:
        print("[network.overview] no event blobs in window — nothing to do")
        # publier un manifest minimal pour latest
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": now.isoformat().replace("+00:00", "Z"),
            "latest_prefix": base_latest,
            "window_days": int(LAST_DAYS),
            "tz": TZNAME,
            "sources": {"exports_prefix": EXPORTS_PREFIX},
            "artifacts": [],
        }
        _upload_json_gs(manifest, f"{base_latest}/manifest.json")
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
        print("[network.overview] no readable data — nothing to do")
        return 0

    ev = pd.concat(frames, ignore_index=True)
    if ev.empty:
        print("[network.overview] events empty — nothing to do")
        return 0

    # 2) Colonnes & typage
    cols = _detect_columns(ev)
    for c in [cols["ts"], cols["station"]]:
        if c is None:
            raise RuntimeError("colonnes minimales manquantes")

    # Parsing UTC tz-aware
    ev[cols["ts"]] = pd.to_datetime(ev[cols["ts"]], utc=True, errors="coerce")
    ev[cols["station"]] = ev[cols["station"]].astype("string")

    # colonnes numériques
    for c in [cols["bikes"], cols["capacity"], cols["docks"], cols["lat"], cols["lon"]]:
        if c and c in ev.columns:
            ev[c] = pd.to_numeric(ev[c], errors="coerce")
    if cols["name"] and cols["name"] in ev.columns:
        ev[cols["name"]] = ev[cols["name"]].astype("string")

    ev = ev.dropna(subset=[cols["ts"], cols["station"]]).copy()

    # fenêtre stricte (UTC aware)
    ev = ev[
        (ev[cols["ts"]] >= pd.Timestamp(start))
        & (ev[cols["ts"]] <= pd.Timestamp(now))
    ].copy()

    # 3) Univers de stations (sur WINDOW_DAYS)
    ts_loc_all = _to_local(ev[cols["ts"]], TZNAME)
    tmax = ts_loc_all.max()
    uni_start = tmax - pd.Timedelta(days=WINDOW_DAYS)
    station_universe = (
        ev.loc[(ts_loc_all >= uni_start) & (ts_loc_all <= tmax), cols["station"]]
        .astype(str)
        .dropna()
        .unique()
        .tolist()
    )

    # 4) Jour opérationnel basé sur le DERNIER fichier events_* disponible
    m = re.search(r"events_(\d{4}-\d{2}-\d{2})\.parquet$", blobs[-1].name)
    if not m:
        raise RuntimeError("Impossible de déterminer la date du dernier fichier events")
    day_for_today_utc = datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    print(
        f"[network.overview] day_for_today_utc auto-detected → {day_for_today_utc.date()} (from last events_*.parquet)"
    )

    # 5) Calculs principaux
    kpis, dist, map_df = _compute_snapshot(ev, cols, TZNAME, station_universe)
    cov, vol = _coverage_volatility(ev, cols, TZNAME, LAST_DAYS)
    kpis["coverage_pct"] = cov
    kpis["volatility_today"] = vol
    kpis["last_days"] = LAST_DAYS
    kpis["ref_days"] = REF_DAYS
    kpis["day_for_today_utc"] = day_for_today_utc.strftime("%Y-%m-%d")

    # Courbes & comparatifs en UTC (jour choisi)
    today_curve_doc, ref_median_doc = _today_vs_median_utc(
        ev, cols, TZNAME, REF_DAYS, day_for_today_utc
    )
    kpi_bars_doc = _kpis_today_vs_lags_utc(ev, cols, day_for_today_utc)
    tension_doc = _stations_tension(ev, cols, TZNAME, LAST_DAYS)

    # 6) Uploads (LATEST only + manifest)
    _upload_json_gs(kpis, f"{base_latest}/kpis.json")
    _upload_json_gs(dist.to_dict(orient="records"), f"{base_latest}/snapshot_distribution.json")
    _upload_json_gs(today_curve_doc, f"{base_latest}/today_curve.json")
    _upload_json_gs(ref_median_doc, f"{base_latest}/ref_median_curve.json")
    _upload_json_gs(kpi_bars_doc, f"{base_latest}/kpis_today_vs_lags.json")
    _upload_json_gs(
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "rows": map_df.to_dict(orient="records"),
        },
        f"{base_latest}/snapshot_map.json",
    )
    _upload_json_gs(tension_doc, f"{base_latest}/stations_tension.json")

    # Manifest top-level (pour la page)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "latest_prefix": base_latest,
        "window_days": int(LAST_DAYS),
        "tz": TZNAME,
        "sources": {"exports_prefix": EXPORTS_PREFIX},
        "artifacts": [
            "kpis.json",
            "snapshot_distribution.json",
            "today_curve.json",
            "ref_median_curve.json",
            "kpis_today_vs_lags.json",
            "snapshot_map.json",
            "stations_tension.json",
        ],
        "day_for_today_utc": day_for_today_utc.strftime("%Y-%m-%d"),
    }
    _upload_json_gs(manifest, f"{base_latest}/manifest.json")

    print("[network.overview] done (latest only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
