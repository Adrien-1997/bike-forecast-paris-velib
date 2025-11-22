# service/jobs/build_data_health.py

# Vélib’ Forecast — Data Health (v1.6 “yesterday-window”)
#
# ▶ ENV HEADER UNIFIÉ (v1) — mêmes noms que les autres jobs
#   - GCS_EXPORTS_PREFIX      (gs://.../velib/exports)            [requis]
#   - GCS_MONITORING_PREFIX   (gs://.../velib[/monitoring])        [requis]
#   - MON_TZ                  (ex: Europe/Paris)                   [def=Europe/Paris]
#   - MON_LAST_DAYS           (fenêtre courante, int)              [def=14]
#   - (MON_REF_DAYS ignoré ici)
#
#   - DATA_HEALTH_BIN_MIN           (def=5)    taille de pas (min)
#   - DATA_HEALTH_FRESH_SLO_MIN     (def=5)    SLO fraicheur p95 (min)
#   - DATA_HEALTH_DUP_ALERT_PCT     (def=1.0)  seuil alerte duplication (% des lignes)
#   - DATA_HEALTH_FLAT_STEPS        (def=6)    min steps séquence plate (en steps)
#   - DATA_HEALTH_COMPL_ALERT_PCT   (def=98.0) seuil OK couverture globale (%)
#   - DATA_HEALTH_LAT_MAX_MIN       (def=4320) cap latence (min, 72h) pour robustesse
#   - DATA_HEALTH_ACTIVE_MIN_FRAC   (optionnel 0..1) filtre actives presence-aware
#   - DATA_HEALTH_WEIGHTED          (0/1) moyenne PA pondérée par expected_i
#   - DATA_HEALTH_ANOM_TOPK         (def=200) 0=illimité
#   - DATA_HEALTH_ANOM_SAMPLE       (def=0)   0..1 sous-échantillonnage anomalies
#   - DATA_HEALTH_ROUND             (def=3)   arrondi flottants JSON
#   - DATA_HEALTH_ANOM_INCLUDE_NAMES (def=true) inclure "name" quand dispo
#
# v1.6 — Changement clé
# ---------------------
# ✅ La fenêtre se termine au **dernier bin de J-1** (ex: 23:55 pour 5 min),
#    et la fraîcheur utilise la **même référence "now"** pour éviter les décalages.
#    On obtient bien: [J-14, ..., J-1], donc “14 jours collectés”.
# ✅ Coverage par heure corrigée: attendu = nb de jours où la station a été
#    vue **sur CETTE heure** × bins/heure (presence-aware par heure).
# =============================================================================

"""
Data Health builder for the Vélib’ Forecast monitoring suite (schema v1.6).

This job:
- Lit les exports d’événements (events_YYYY-MM-DD.parquet) sur GCS.
- Normalise les colonnes clés (timestamps, station_id, bikes, etc.).
- Délimite une fenêtre de monitoring sur J-1: [J-N .. J-1].
- Calcule les KPIs globaux de qualité de données :
    • Fraîcheur (p50 / p95 d’âge des données, SLO).
    • Complétude (coverage files-aware + presence-aware).
    • Latence ingestion → event.
    • Duplications et séquences plates (bikes constants).
- Produit des artefacts JSON pour le monitoring UI :
    • kpis.json
    • station_health.json
    • coverage_by_hour.json
    • alerts.json
    • anomalies/flat.json, anomalies/duplicates.json, anomalies/missing.json
    • anomalies/manifest.json
"""

from __future__ import annotations
import os, re, json, sys, math
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

SCHEMA_VERSION = "1.6"  # yesterday-window + freshness alignée + hourly PA fix

# ─────────────────────────── ENV unifié ───────────────────────────

def _env(name: str, default=None):
    """
    Lire une variable d'environnement avec valeur par défaut.

    Paramètres
    ----------
    name : str
        Nom de la variable d’environnement.
    default :
        Valeur de fallback si la variable est absente ou vide.

    Retour
    ------
    Any
        Chaîne ou valeur par défaut.
    """
    v = os.environ.get(name)
    return v if (v is not None and v != "") else default


def _env_int(name: str, default: int) -> int:
    """
    Lire une variable d'environnement entière avec valeur par défaut.

    Si la conversion échoue, on renvoie simplement `default`.

    Paramètres
    ----------
    name : str
        Nom de la variable.
    default : int
        Valeur par défaut.

    Retour
    ------
    int
        Valeur entière parsée ou valeur par défaut.
    """
    try:
        return int(_env(name, default))
    except Exception:
        return default


GCS_EXPORTS_PREFIX    = _env("GCS_EXPORTS_PREFIX")
GCS_MONITORING_PREFIX = _env("GCS_MONITORING_PREFIX")
MON_TZ                = _env("MON_TZ", "Europe/Paris")
MON_LAST_DAYS         = _env_int("MON_LAST_DAYS", 14)

if not (GCS_EXPORTS_PREFIX and str(GCS_EXPORTS_PREFIX).startswith("gs://")):
    raise RuntimeError("GCS_EXPORTS_PREFIX manquant ou invalide")
if not (GCS_MONITORING_PREFIX and str(GCS_MONITORING_PREFIX).startswith("gs://")):
    raise RuntimeError("GCS_MONITORING_PREFIX manquant ou invalide")

# ─────────────────────────── Helpers GCS ───────────────────────────

def _split(gs: str) -> Tuple[str, str]:
    """
    Découper une URI GCS `gs://bucket/path` en (bucket, key).

    Paramètres
    ----------
    gs : str
        URI GCS commençant par 'gs://'.

    Retour
    ------
    (str, str)
        Tuple (nom_bucket, chemin_objet).
    """
    assert gs.startswith("gs://"), f"bad GCS uri: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k.rstrip("/")


def _list_event_blobs(exports_prefix: str, start_date: datetime, end_date: datetime) -> List["storage.Blob"]:
    """
    Lister les blobs events_YYYY-MM-DD.parquet dans une fenêtre de dates.

    Paramètres
    ----------
    exports_prefix : str
        Préfixe GCS contenant les événements (GCS_EXPORTS_PREFIX).
    start_date : datetime
        Date UTC de début (inclus).
    end_date : datetime
        Date UTC de fin (inclus).

    Retour
    ------
    list[google.cloud.storage.Blob]
        Liste triée de blobs d’événements.
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


def _read_parquet_blob_to_df(blob: "storage.Blob") -> pd.DataFrame:
    """
    Télécharger un blob Parquet GCS et le charger en DataFrame pandas.

    Paramètres
    ----------
    blob : google.cloud.storage.Blob
        Blob Parquet à lire.

    Retour
    ------
    pandas.DataFrame
        Données du fichier Parquet.
    """
    buf = BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    tbl = pq.read_table(buf)
    return tbl.to_pandas()


def _upload_json_gs(obj: object, gs_uri: str) -> None:
    """
    Sérialiser un objet Python en JSON et l’uploader sur GCS.

    Règles
    ------
    - Conversion propre des types numpy (float/int/bool).
    - NaN / infinities → None pour du JSON valide.

    Paramètres
    ----------
    obj : object
        Objet JSON-sérialisable (dict, list, ...).
    gs_uri : str
        Destination GCS (gs://bucket/path/file.json).
    """
    def san(o):
        if isinstance(o, dict):  return {k: san(v) for k, v in o.items()}
        if isinstance(o, list):  return [san(v) for v in o]
        if isinstance(o, (np.floating,)): return float(o) if np.isfinite(o) else None
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.bool_,)):    return bool(o)
        if o is None: return None
        try:
            if pd.isna(o): return None
        except Exception:
            pass
        return o
    safe = san(obj)
    bkt, key = _split(gs_uri)
    data = json.dumps(safe, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    storage.Client().bucket(bkt).blob(key).upload_from_string(data, content_type="application/json")
    print(f"[data.health] wrote → {gs_uri} ({len(data):,} bytes)")

# ─────────────────────────── Small helpers ───────────────────────────

def _r(x: float, nd: int) -> Optional[float]:
    """
    Arrondir un float à `nd` décimales, en retournant None si non fini.

    Paramètres
    ----------
    x : float
        Valeur à arrondir.
    nd : int
        Nombre de décimales.

    Retour
    ------
    float or None
        Valeur arrondie ou None si NaN/Inf ou erreur.
    """
    try:
        if not np.isfinite(x): return None
        return float(np.round(float(x), nd))
    except Exception:
        return None


def _detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Détecter les colonnes clés (ts, station_id, bikes, docks, etc.) de façon robuste.

    La détection est basée sur des variantes de noms en minuscules, pour
    tolérer différents schémas d’export.

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame source.

    Retour
    ------
    dict
        Mapping avec les clés:
        - ts, station, bikes, docks, capacity, ingested_at, name

    Lève
    ----
    KeyError
        Si les colonnes minimales (ts, station, bikes) sont introuvables.
    """
    lower = {c.lower(): c for c in df.columns}
    def any_of(*cands):
        for c in cands:
            if c in lower:
                return lower[c]
        return None
    ts  = any_of("ts","tbin_utc","timestamp","datetime")
    sid = any_of("station_id","stationcode","station","id")
    bikes = any_of("bikes","nb_velos_bin","num_bikes_available","velos","velos_disponibles")
    docks = any_of("docks_avail","nb_docks_bin","num_docks_available","free_docks","places_disponibles")
    cap   = any_of("capacity","num_docks_total","dock_count","cap")
    ing   = any_of("ingested_at","ingest_ts","ingest_time","received_at","etl_ts","load_ts","created_at")
    name  = any_of("name","station_name","nom")
    if not ts or not sid or not bikes:
        raise KeyError(f"[data.health] Colonnes minimales absentes (ts={ts}, station={sid}, bikes={bikes})")
    return dict(ts=ts, station=sid, bikes=bikes, docks=docks, capacity=cap, ingested_at=ing, name=name)


def _yesterday_end_floor(bin_min: int) -> pd.Timestamp:
    """
    Dernier timestamp aligné sur `bin_min` pour J-1 en UTC (naïf).

    Exemples
    --------
    - bin_min=5  → J-1 23:55:00
    - bin_min=15 → J-1 23:45:00
    """
    now_utc = datetime.now(timezone.utc)
    y_end = (now_utc - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
    ts = pd.Timestamp(y_end)                 # tz-aware (UTC)
    ts = ts.floor(f"{int(bin_min)}min")     # alignement
    return ts.tz_localize(None)             # naïf UTC


def _window_start_for_days(end_ts: pd.Timestamp, days: int) -> pd.Timestamp:
    """
    Début de fenêtre à 00:00:00 UTC pour couvrir EXACTEMENT `days` jours.

    La fenêtre inclut le jour de end_ts comme J-1, donc days=7 → J-7..J-1.

    Paramètres
    ----------
    end_ts : pandas.Timestamp
        Fin de fenêtre (J-1 aligné).
    days : int
        Nombre de jours à couvrir (>= 0).

    Retour
    ------
    pandas.Timestamp
        Début de fenêtre à 00:00 UTC.
    """
    d = max(int(days), 0)
    start_day = end_ts.normalize() - pd.Timedelta(days=max(d - 1, 0))
    return start_day  # 00:00 UTC


def _bins_between(a: pd.Timestamp, b: pd.Timestamp, bin_min: int) -> int:
    """
    Nombre de bins (strict à gauche, inclusif à droite) de taille `bin_min`.

    Intervalle considéré : (a, b]

    Paramètres
    ----------
    a : pandas.Timestamp
        Début (exclu).
    b : pandas.Timestamp
        Fin (inclus).
    bin_min : int
        Taille du pas en minutes.

    Retour
    ------
    int
        Nombre de bins.
    """
    if pd.isna(a) or pd.isna(b) or b <= a:
        return 1
    return int(math.floor((b - a).total_seconds() / 60.0 / max(1, int(bin_min))) + 1)

# ─────────────────────────── KPI builders ───────────────────────────

def kpi_freshness(
    df: pd.DataFrame,
    sid_col: str,
    ts_col: str,
    slo_min: int,
    bin_min: int,
    now_ref: pd.Timestamp,
) -> dict:
    """
    Calculer la fraîcheur des données (âge des derniers points par station).

    Métriques retournées
    --------------------
    - freshness_age_p50_min : médiane de l’âge en minutes.
    - freshness_age_p95_min : p95 de l’âge en minutes.
    - freshness_slo_min     : SLO cible (env).
    - freshness_p95_ok      : bool (p95 <= slo_min) ou None si non calculable.

    Paramètres
    ----------
    df : pandas.DataFrame
        Données filtrées dans la fenêtre.
    sid_col : str
        Nom de la colonne station_id.
    ts_col : str
        Nom de la colonne timestamp.
    slo_min : int
        SLO de fraîcheur (minutes).
    bin_min : int
        Pas en minutes (non utilisé directement mais gardé pour symétrie).
    now_ref : pandas.Timestamp
        Référence temporelle globale (fin J-1 alignée).

    Retour
    ------
    dict
        Dictionnaire de KPIs de fraîcheur.
    """
    now = now_ref  # déjà aligné J-1 23:55 etc.
    last = df.groupby(sid_col)[ts_col].max().reset_index(name="last_ts")
    last["age_min"] = (now - last["last_ts"]).dt.total_seconds() / 60.0
    p50 = float(np.nanpercentile(last["age_min"], 50)) if len(last) else np.nan
    p95 = float(np.nanpercentile(last["age_min"], 95)) if len(last) else np.nan
    return {
        "now_utc": now.isoformat(),
        "ts_global_max": (df[ts_col].max().isoformat() if len(df) else None),
        "freshness_age_p50_min": _r(p50, 2),
        "freshness_age_p95_min": _r(p95, 2),
        "freshness_slo_min": float(slo_min),
        "freshness_p95_ok": (p95 <= slo_min) if np.isfinite(p95) else None,
    }


def kpi_completeness_files_aware(
    df: pd.DataFrame,
    sid_col: str,
    ts_col: str,
    current_days: int,
    bin_min: int,
    tz: Optional[str],
    *,
    active_min_frac: Optional[float] = None,
    weighted: bool = False,
    end_ref: Optional[pd.Timestamp] = None,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    """
    Calculer les KPIs de complétude (files-aware & presence-aware).

    Définitions
    -----------
    - Fenêtre : (start_ref, end_ref] avec end_ref = fin J-1 alignée.
    - days_present_all :
        # de jours (fichiers) réellement présents dans la fenêtre.
    - expected_global :
        days_present_all × bins_par_jour.
    - coverage_global_pct :
        moyenne des coverage_pct_global par station (files-aware).
    - by_station_pa :
        coverage presence-aware par station (expected_i individuel).
    - cov_by_hour :
        coverage moyen par heure locale (presence-aware par heure).

    Paramètres
    ----------
    df : pandas.DataFrame
        Événements filtrés.
    sid_col : str
        Colonne station.
    ts_col : str
        Colonne timestamp.
    current_days : int
        Taille de la fenêtre actuelle (en jours).
    bin_min : int
        Taille du pas en minutes.
    tz : str or None
        Timezone pour les heures locales (coverage_by_hour).
    active_min_frac : float, optionnel
        Filtre des stations "actives" (ratio obs/expected_i minimal).
    weighted : bool
        Si True, moyenne pondérée par expected_i pour coverage_active.
    end_ref : pandas.Timestamp, optionnel
        Fin de fenêtre (J-1 aligné). Si None, recalcul via _yesterday_end_floor.

    Retour
    ------
    tuple
        (kpis, by_station_pa, cov_by_hour, by_station_global, days_present_all, bins_per_day)
    """
    if current_days <= 0 or df.empty:
        empty = pd.DataFrame()
        return {"coverage_global_pct": np.nan, "coverage_active_pct": np.nan}, empty, empty, empty, 0, (60 // max(1,int(bin_min))) * 24

    if end_ref is None:
        end_ref = _yesterday_end_floor(bin_min)

    start_ref = _window_start_for_days(end_ref, current_days)
    win = df[(df[ts_col] > start_ref) & (df[ts_col] <= end_ref)].copy()
    if win.empty:
        empty = pd.DataFrame()
        return {"coverage_global_pct": np.nan, "coverage_active_pct": np.nan}, empty, empty, empty, 0, (60 // max(1,int(bin_min))) * 24

    # Jours de FICHIERS réellement présents (global)
    days_present_all = int(win[ts_col].dt.normalize().nunique())
    bins_per_hour = (60 // max(1, int(bin_min)))
    bins_per_day = bins_per_hour * 24
    expected_global = int(days_present_all * bins_per_day)

    # Observé par station (bins uniques)
    per_station_obs = win.groupby(sid_col)[ts_col].nunique().rename_axis(sid_col).reset_index(name="obs")
    per_station_obs["obs"] = per_station_obs["obs"].clip(upper=expected_global)

    # Table “global files-aware”
    by_station_global = (
        per_station_obs.assign(expected_global=expected_global)
        .assign(coverage_pct_global=lambda d: d["obs"] / d["expected_global"] * 100.0)
        .rename(columns={sid_col: "station_id"})
        [["station_id", "obs", "expected_global", "coverage_pct_global"]]
        .copy()
    )
    coverage_global = float(by_station_global["coverage_pct_global"].mean()) if len(by_station_global) else np.nan

    # Presence-aware par station (sur la fenêtre)
    pres = win.groupby(sid_col)[ts_col].agg(first="min", last="max").reset_index()
    pres["expected_i"] = pres.apply(
        lambda r: _bins_between(max(start_ref, r["first"]), min(end_ref, r["last"]), bin_min),
        axis=1,
    )
    pa = per_station_obs.merge(pres[[sid_col, "expected_i"]], on=sid_col, how="left")
    pa["expected_i"] = pa["expected_i"].fillna(1).astype(int)
    pa["obs_clamped"] = pa.apply(lambda r: min(int(r["obs"]), int(r["expected_i"])), axis=1)
    pa["coverage_pct_i"] = pa["obs_clamped"] / pa["expected_i"] * 100.0

    # Filtre actives (optionnel)
    if active_min_frac is not None and 0 < active_min_frac <= 1:
        pa_active = pa[pa["obs_clamped"] >= (active_min_frac * pa["expected_i"])].copy()
    else:
        pa_active = pa.copy()

    # Moyenne presence-aware (pondérée ou non)
    if len(pa_active):
        if weighted:
            w = pa_active["expected_i"].astype(float).replace(0, np.nan)
            coverage_active = float(np.nansum(pa_active["coverage_pct_i"] * w) / np.nansum(w))
        else:
            coverage_active = float(pa_active["coverage_pct_i"].mean())
    else:
        coverage_active = np.nan

    # ───────── Coverage par heure (presence-aware **par heure**) ─────────
    if tz:
        win["_hour"] = pd.to_datetime(win[ts_col]).dt.tz_localize("UTC").dt.tz_convert(tz).dt.hour
        win["_day"]  = pd.to_datetime(win[ts_col]).dt.tz_localize("UTC").dt.tz_convert(tz).dt.normalize()
    else:
        win["_hour"] = pd.to_datetime(win[ts_col]).dt.hour
        win["_day"]  = pd.to_datetime(win[ts_col]).dt.normalize()

    # Observé: nb de bins par (heure, station)
    per_hour_obs = (
        win.groupby(["_hour", sid_col])[ts_col]
           .nunique()
           .rename("obs")
           .reset_index()
    )

    # Présence par (heure, station): nb de jours où la station a été vue à CETTE heure
    days_by_station_hour = (
        win.groupby([sid_col, "_hour"])["_day"]
           .nunique()
           .rename("days_present_i_hour")
           .reset_index()
    )

    per_hour = per_hour_obs.merge(days_by_station_hour, on=[sid_col, "_hour"], how="left")
    per_hour["expected_i"] = (per_hour["days_present_i_hour"].fillna(0).astype(int) * int(bins_per_hour)).astype(int)
    per_hour["coverage_pct_i"] = np.where(
        per_hour["expected_i"] > 0,
        per_hour["obs"].clip(upper=per_hour["expected_i"]) / per_hour["expected_i"] * 100.0,
        np.nan,
    )

    cov_by_hour = (
        per_hour.groupby("_hour")["coverage_pct_i"]
                .mean()
                .rename("coverage_pct")
                .rename_axis("hour")
                .reset_index()
    )

    # Table PA (UI)
    by_station_pa = pa.rename(columns={
        sid_col: "station_id",
        "expected_i": "expected",
        "coverage_pct_i": "coverage_pct"
    })[["station_id","obs","expected","coverage_pct"]].sort_values("coverage_pct", ascending=True).reset_index(drop=True)

    k = {
        "coverage_global_pct": _r(coverage_global, 2),
        "coverage_active_pct": _r(coverage_active, 2),
        "active_frac": (_r(active_min_frac, 3) if isinstance(active_min_frac, (int, float)) else None),
        "weighted": bool(weighted),
    }
    return k, by_station_pa, cov_by_hour, by_station_global, days_present_all, bins_per_day


def kpi_latency(
    df: pd.DataFrame,
    ts_col: str,
    ing_col: Optional[str],
    cap_min: Optional[int] = None
) -> Tuple[dict, Optional[pd.DataFrame]]:
    """
    Calculer la latence ingestion→event (en minutes) et ses quantiles.

    Paramètres
    ----------
    df : pandas.DataFrame
        Données contenant timestamps event et ingestion.
    ts_col : str
        Nom de la colonne event timestamp.
    ing_col : str or None
        Nom de la colonne ingestion timestamp.
    cap_min : int or None
        Cap maximum de latence (minutes) pour robustesse.

    Retour
    ------
    (dict, DataFrame or None)
        - dict avec latency_p50_min, latency_p95_min.
        - DataFrame latence brute par station (optionnel) pour debug.
    """
    if not ing_col or not ( ing_col in df.columns ) or df[ing_col].isna().all():
        return {"latency_p50_min": np.nan, "latency_p95_min": np.nan}, None
    lat = df.dropna(subset=[ts_col, ing_col]).copy()
    lat["latency_min"] = (lat[ing_col] - lat[ts_col]).dt.total_seconds() / 60.0
    lat = lat[np.isfinite(lat["latency_min"]) & (lat["latency_min"] >= 0)]
    if cap_min is not None and np.isfinite(cap_min):
        lat = lat[lat["latency_min"] <= float(cap_min)]
    if lat.empty:
        return {"latency_p50_min": np.nan, "latency_p95_min": np.nan}, None
    p50 = float(np.nanpercentile(lat["latency_min"], 50))
    p95 = float(np.nanpercentile(lat["latency_min"], 95))
    cols_out = ["station_id","latency_min"] if "station_id" in lat.columns else ["latency_min"]
    return {"latency_p50_min": _r(p50, 2), "latency_p95_min": _r(p95, 2)}, lat[cols_out]


def duplication_stats(df: pd.DataFrame, sid_col: str, ts_col: str) -> pd.DataFrame:
    """
    Compter le nombre de doublons (ts, station) par station.

    Paramètres
    ----------
    df : pandas.DataFrame
        Données d’événements.
    sid_col : str
        Colonne station.
    ts_col : str
        Colonne timestamp.

    Retour
    ------
    pandas.DataFrame
        Colonnes: [sid_col, "dups"], trié par dups décroissant.
    """
    if df.empty:
        return pd.DataFrame(columns=[sid_col, "dups"])
    dups = df.duplicated(subset=[ts_col, sid_col]).groupby(df[sid_col]).sum().astype(int)
    out = dups.rename_axis(sid_col).reset_index(name="dups").sort_values("dups", ascending=False)
    out[sid_col] = out[sid_col].astype("string")
    out["dups"] = pd.to_numeric(out["dups"], errors="coerce").fillna(0).astype(int)
    return out


def flat_sequences(
    df: pd.DataFrame,
    sid_col: str,
    ts_col: str,
    bikes_col: str,
    min_steps: int,
    current_days: int,
    bin_min: int
) -> pd.DataFrame:
    """
    Détecter des séquences plates (bikes constants) par station.

    Une séquence plate est définie par une suite de pas consécutifs
    où bikes ne varie pas (delta=0), d’une longueur >= min_steps.

    Paramètres
    ----------
    df : pandas.DataFrame
        Données complètes.
    sid_col : str
        Colonne station.
    ts_col : str
        Colonne timestamp.
    bikes_col : str
        Colonne nombre de vélos.
    min_steps : int
        Longueur minimale de la séquence plate (en bins).
    current_days : int
        Fenêtre de jours considérée (J-N..J-1).
    bin_min : int
        Taille du pas en minutes (pour convertir steps → minutes).

    Retour
    ------
    pandas.DataFrame
        Colonnes: [sid_col, "steps", "start", "end", "duration_min"].
    """
    if df.empty:
        return pd.DataFrame(columns=[sid_col,"steps","start","end","duration_min"])
    # Fenêtre cohérente (J-1)
    end_ref = _yesterday_end_floor(bin_min)
    start_ref = _window_start_for_days(end_ref, current_days)
    win = df[(df[ts_col] > start_ref) & (df[ts_col] <= end_ref)].copy()
    if win.empty:
        return pd.DataFrame(columns=[sid_col,"steps","start","end","duration_min"])
    win = win.sort_values([sid_col, ts_col])
    win["delta"] = win.groupby(sid_col)[bikes_col].diff().fillna(0.0).abs()
    win["is_flat"] = (win["delta"] == 0.0)
    win["grp"] = (~win["is_flat"]).groupby(win[sid_col]).cumsum()
    agg = (win[win["is_flat"]]
           .groupby([sid_col,"grp"])
           .agg(steps=("is_flat","size"), start=(ts_col,"min"), end=(ts_col,"max"))
           .reset_index())
    out = agg[agg["steps"] >= min_steps][[sid_col,"steps","start","end"]]
    out["duration_min"] = (out["steps"] * bin_min).astype(int)
    out = out.sort_values("steps", ascending=False).reset_index(drop=True)
    out[sid_col] = out[sid_col].astype("string")
    return out

# ─────────────────────────── Main ───────────────────────────

def main() -> int:
    """
    Entrypoint CLI pour le job Data Health.

    Pipeline
    --------
    1. Calculer la fenêtre temporelle [J-N .. J-1] en UTC, alignée sur BIN_MIN.
    2. Lister et lire les fichiers events_YYYY-MM-DD.parquet dans cette fenêtre.
    3. Détecter les colonnes clés et typer les données.
    4. Calculer :
        - Fraîcheur (kpi_freshness)
        - Complétude (kpi_completeness_files_aware)
        - Latence ingestion (kpi_latency)
        - Duplications (duplication_stats)
        - Séquences plates (flat_sequences)
    5. Construire les tables station_health + coverage_by_hour.
    6. Générer les anomalies (flat / duplicates / missing).
    7. Uploader l’ensemble en JSON sous:
       {GCS_MONITORING_PREFIX}/monitoring/data/health/latest

    Retour
    ------
    int
        Code de sortie (0 = succès).
    """
    # Options spécifiques au job (seuils & granularité)
    TZNAME            = MON_TZ
    CURRENT_DAYS      = int(MON_LAST_DAYS)  # ← unifié
    BIN_MIN           = _env_int("DATA_HEALTH_BIN_MIN", 5)
    FRESH_SLO_MIN     = _env_int("DATA_HEALTH_FRESH_SLO_MIN", 5)
    DUP_ALERT_PCT     = float(_env("DATA_HEALTH_DUP_ALERT_PCT", "1.0"))
    FLAT_STEPS        = _env_int("DATA_HEALTH_FLAT_STEPS", 6)
    COMPL_ALERT_PCT   = float(_env("DATA_HEALTH_COMPL_ALERT_PCT", "98.0"))
    LAT_MAX_MIN       = _env_int("DATA_HEALTH_LAT_MAX_MIN", 72*60)

    _active_frac_env = _env("DATA_HEALTH_ACTIVE_MIN_FRAC", "").strip()
    ACTIVE_MIN_FRAC  = float(_active_frac_env) if _active_frac_env not in ("", None) else None
    WEIGHTED         = _env("DATA_HEALTH_WEIGHTED", "0").lower() in ("1","true","yes","y")

    ANOM_TOPK          = _env_int("DATA_HEALTH_ANOM_TOPK", 200)
    ANOM_SAMPLE        = float(_env("DATA_HEALTH_ANOM_SAMPLE", "0"))
    ROUND_ND           = _env_int("DATA_HEALTH_ROUND", 3)
    ANOM_INCLUDE_NAMES = _env("DATA_HEALTH_ANOM_INCLUDE_NAMES", "true").lower() in ("1","true","yes","y")

    # ✅ Référence temporelle = fin de J-1 alignée au pas BIN_MIN
    end_ref = _yesterday_end_floor(BIN_MIN)
    start_ref = _window_start_for_days(end_ref, max(CURRENT_DAYS, 7))

    print(f"[data.health] window UTC: {start_ref.date()} → {end_ref.date()} (days≥{CURRENT_DAYS})")

    blobs = _list_event_blobs(GCS_EXPORTS_PREFIX, start_ref, end_ref)
    if not blobs:
        print("[data.health] no event blobs in window — nothing to do")
        return 0

    frames: List[pd.DataFrame] = []
    for bl in blobs:
        print(f"[read] {bl.name}")
        try:
            frames.append(_read_parquet_blob_to_df(bl))
        except Exception as e:
            print(f"[warn] failed to read {bl.name}: {e}")

    if not frames:
        print("[data.health] no readable data — nothing to do")
        return 0

    df = pd.concat(frames, ignore_index=True)

    # Détection colonnes + typage
    cols = _detect_columns(df)
    ts_col, sid_col, bikes_col = cols["ts"], cols["station"], cols["bikes"]
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dt.tz_convert(None)
    df[sid_col] = df[sid_col].astype("string")
    df[bikes_col] = pd.to_numeric(df[bikes_col], errors="coerce")
    if cols["docks"] and cols["docks"] in df.columns:
        df[cols["docks"]] = pd.to_numeric(df[cols["docks"]], errors="coerce")
    if cols["capacity"] and cols["capacity"] in df.columns:
        df[cols["capacity"]] = pd.to_numeric(df[cols["capacity"]], errors="coerce")
    if cols["ingested_at"] and cols["ingested_at"] in df.columns:
        df[cols["ingested_at"]] = pd.to_datetime(df[ing_col := cols["ingested_at"]], utc=True, errors="coerce").dt.tz_convert(None)

    df = df.dropna(subset=[ts_col, sid_col]).copy()

    # Tronquer immédiatement à la fenêtre J-1 (sécurité)
    df = df[(df[ts_col] > start_ref) & (df[ts_col] <= end_ref)].copy()

    # KPIs principaux — files-aware & presence-aware
    kfresh = kpi_freshness(df, sid_col, ts_col, FRESH_SLO_MIN, BIN_MIN, now_ref=end_ref)
    kcov, by_station_pa, cov_by_hour, by_station_global, DAYS_PRESENT_ALL, BINS_PER_DAY = kpi_completeness_files_aware(
        df, sid_col, ts_col, CURRENT_DAYS, BIN_MIN, TZNAME,
        active_min_frac=ACTIVE_MIN_FRAC, weighted=WEIGHTED, end_ref=end_ref
    )
    klat, lat_df = kpi_latency(df, ts_col, cols["ingested_at"], cap_min=LAT_MAX_MIN)

    # Duplications / séquences plates
    flats = flat_sequences(df, sid_col, ts_col, bikes_col, FLAT_STEPS, CURRENT_DAYS, BIN_MIN)
    dups = duplication_stats(df, sid_col, ts_col)
    dup_total = int(dups["dups"].sum()) if len(dups) else 0
    dups_pct = (dup_total / max(1, len(df))) * 100.0 if len(df) else 0.0

    # Missing bins (global files-aware): expected_global - obs
    miss = by_station_global.rename(columns={"expected_global": "expected"})[["station_id","obs","expected"]].copy()
    miss["missing"] = (miss["expected"] - miss["obs"]).clip(lower=0).astype(int)
    missing_stations = int((miss["missing"] > 0).sum()) if len(miss) else 0

    # Table station_health
    station_health_rows = (
        by_station_global
        .rename(columns={
            "expected_global": "expected",
            "coverage_pct_global": "coverage_pct"
        })
        [["station_id","obs","expected","coverage_pct"]]
        .merge(miss[["station_id","missing"]], on="station_id", how="left")
        .sort_values("coverage_pct", ascending=True)
        .reset_index(drop=True)
    )
    station_health_rows["expected"] = station_health_rows["expected"].astype(int)
    station_health_rows["missing"] = station_health_rows["missing"].fillna(
        (station_health_rows["expected"] - station_health_rows["obs"]).clip(lower=0)
    ).astype(int)

    # Ajout des noms défensif
    if ANOM_INCLUDE_NAMES and cols.get("name") and cols["name"] in df.columns:
        name_col = cols["name"]
        name_map_df = (
            df.dropna(subset=[name_col])
              .sort_values([sid_col, ts_col])
              .groupby(sid_col, as_index=False)
              .tail(1)[[sid_col, name_col]]
              .drop_duplicates(subset=[sid_col])
              .rename(columns={sid_col: "station_id", name_col: "name"})
        )
        if not name_map_df.empty:
            name_map_df["station_id"] = name_map_df["station_id"].astype("string")
            station_health_rows = station_health_rows.merge(
                name_map_df[["station_id", "name"]],
                on="station_id",
                how="left",
            )
    else:
        station_health_rows["name"] = None

    # Colonnes finales & ordre
    cols_out = ["station_id", "name", "obs", "expected", "coverage_pct", "missing"]
    for c in cols_out:
        if c not in station_health_rows.columns:
            station_health_rows[c] = None
    station_health_rows = station_health_rows[cols_out]

    # Coverage by hour → records
    cov_by_hour_rows = cov_by_hour.to_dict(orient="records")

    # ── Anomalies shardées ────────────────────────────────────────────────
    name_lookup: Optional[Dict[str, str]] = None
    if ANOM_INCLUDE_NAMES and cols.get("name") and cols["name"] in df.columns:
        name_col = cols["name"]
        name_lookup_df = (
            df.dropna(subset=[name_col])
              .sort_values([sid_col, ts_col])
              .groupby(sid_col, as_index=False)
              .tail(1)[[sid_col, name_col]]
              .drop_duplicates(subset=[sid_col])
              .rename(columns={sid_col: "station_id", name_col: "name"})
        )
        name_lookup = dict(zip(name_lookup_df["station_id"].astype(str), name_lookup_df["name"].astype(str)))

    def _nm(sid: str) -> Optional[str]:
        """
        Chercher le nom de station correspondant à un station_id.

        Retourne None si `name_lookup` n’est pas initialisé ou si absent.
        """
        if not name_lookup: return None
        return name_lookup.get(str(sid))

    # 3.a flat sequences
    an_flat: List[Dict[str, object]] = []
    if len(flats):
        for _, r in flats.iterrows():
            sidv = str(r["station_id"] if "station_id" in r else r[sid_col])
            an_flat.append({
                "type": "flat_sequence",
                "station_id": sidv,
                **({"name": _nm(sidv)} if name_lookup else {}),
                "start": r["start"].isoformat(),
                "end": r["end"].isoformat(),
                "steps": int(r["steps"]),
                "duration_min": int(r["duration_min"]),
            })

    # 3.b duplicates
    an_dups: List[Dict[str, object]] = []
    if len(dups):
        dups_sorted = dups.sort_values("dups", ascending=False)
        for _, r in dups_sorted.iterrows():
            val = int(r["dups"]) if pd.notna(r["dups"]) else 0
            if val <= 0: continue
            sidv = str(r[sid_col])
            an_dups.append({
                "type": "duplicates",
                "station_id": sidv,
                **({"name": _nm(sidv)} if name_lookup else {}),
                "dups": val,
            })

    # 3.c missing (top manquants)
    an_missing: List[Dict[str, object]] = []
    if len(miss):
        worst_miss = miss.sort_values("missing", ascending=False)
        for _, r in worst_miss.iterrows():
            if int(r["missing"]) > 0:
                sidv = str(r["station_id"])
                an_missing.append({
                    "type": "missing_bins",
                    "station_id": sidv,
                    **({"name": _nm(sidv)} if name_lookup else {}),
                    "missing": int(r["missing"]),
                    "expected": int(r["expected"]),
                })

    # Post-traitement anomalies
    def _postproc(an_rows: List[Dict[str, object]], sort_key: Optional[str] = None) -> List[Dict[str, object]]:
        """
        Trier / tronquer / échantillonner les anomalies avant JSON.

        - sort_key : clé de tri décroissant.
        - ANOM_TOPK : garde les top-k.
        - ANOM_SAMPLE : sous-échantillonnage systématique (pas).
        - ROUND_ND : arrondi des floats.
        """
        rows = list(an_rows)
        if not rows: return rows
        if sort_key and all((sort_key in r) for r in rows):
            rows = sorted(rows, key=lambda r: r.get(sort_key, 0), reverse=True)
        if ANOM_TOPK > 0 and len(rows) > ANOM_TOPK:
            rows = rows[:ANOM_TOPK]
        if 0.0 < ANOM_SAMPLE < 1.0 and len(rows) > 0:
            step = max(1, int(round(1.0 / ANOM_SAMPLE)))
            rows = rows[::step]
        for r in rows:
            for k, v in list(r.items()):
                if isinstance(v, (float, np.floating)):
                    r[k] = _r(float(v), ROUND_ND)
        return rows

    an_flat     = _postproc(an_flat, sort_key="duration_min")
    an_dups     = _postproc(an_dups, sort_key="dups")
    an_missing  = _postproc(an_missing, sort_key="missing")

    # Alertes globales
    alerts = {
        "freshness_p95_ok": (None if kfresh.get("freshness_p95_ok") is None else bool(kfresh["freshness_p95_ok"])),
        "coverage_ok": bool(float(kcov.get("coverage_global_pct", 0.0)) >= float(COMPL_ALERT_PCT)),
        "duplication_alert": bool(dups_pct >= float(DUP_ALERT_PCT)),
        "flat_sequences_found": bool(len(an_flat)),
    }

    # En-tête paramètres (audit)
    params_header = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "tz": TZNAME or "UTC",
        "bin_min": int(BIN_MIN),
        "current_days": int(CURRENT_DAYS),
        "files_aware_days": int(DAYS_PRESENT_ALL),
        "bins_per_day": int(BINS_PER_DAY),
        "thresholds": {
            "fresh_slo_min": FRESH_SLO_MIN,
            "compl_alert_pct": COMPL_ALERT_PCT,
            "dup_alert_pct": DUP_ALERT_PCT,
            "flat_steps": FLAT_STEPS,
            "lat_max_min": LAT_MAX_MIN,
        },
        "completeness_options": {
            "active_min_frac": ACTIVE_MIN_FRAC,
            "weighted": WEIGHTED,
        },
        "anomaly_params": {
            "topk": ANOM_TOPK,
            "sample_frac": ANOM_SAMPLE,
            "round_nd": ROUND_ND,
            "include_names": ANOM_INCLUDE_NAMES,
        },
        "sources": {
            "exports_prefix": GCS_EXPORTS_PREFIX,
        },
        "window": {
            "start_utc": start_ref.isoformat(),
            "end_utc":   end_ref.isoformat(),
        },
    }

    data_health = {
        **params_header,
        "rows": int(len(df)),
        "stations": int(df[sid_col].nunique()),
        "span": [
            df[ts_col].min().isoformat() if len(df) else None,
            df[ts_col].max().isoformat() if len(df) else None
        ],
        **kfresh, **kcov, **klat,
        "dups_pct": _r(dups_pct, 3),
        "missing_stations": missing_stations,
        "alerts": alerts,
    }

    # ─────────────────────────── Uploads → latest ───────────────────────────
    mon_base = str(GCS_MONITORING_PREFIX).rstrip("/")
    if not mon_base.endswith("/monitoring"):
        mon_base = mon_base + "/monitoring"
    base_alias = f"{mon_base}/data/health/latest"

    _upload_json_gs(data_health,                               f"{base_alias}/kpis.json")
    _upload_json_gs(station_health_rows.to_dict("records"),    f"{base_alias}/station_health.json")
    _upload_json_gs(cov_by_hour_rows,                          f"{base_alias}/coverage_by_hour.json")
    _upload_json_gs(alerts,                                    f"{base_alias}/alerts.json")

    anomalies_prefix = f"{base_alias}/anomalies"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "params": params_header["anomaly_params"],
        "files": [],
        "counts": {
            "flat": len(an_flat),
            "duplicates": len(an_dups),
            "missing": len(an_missing),
        }
    }

    def _upload_anom(name: str, rows: List[Dict[str, object]]):
        """
        Uploader une liste d’anomalies (rows) sous anomalies/<name>.json
        et l’ajouter au manifest.
        """
        uri = f"{anomalies_prefix}/{name}.json"
        # correction: pas d'accolade en trop
        uri = f"{anomalies_prefix}/{name}.json"
        _upload_json_gs(rows, uri)
        manifest["files"].append({"name": name, "uri": uri, "count": len(rows)})

    _upload_anom("flat", an_flat)
    _upload_anom("duplicates", an_dups)
    _upload_anom("missing", an_missing)
    _upload_json_gs(manifest, f"{anomalies_prefix}/manifest.json")

    print("[data.health] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
