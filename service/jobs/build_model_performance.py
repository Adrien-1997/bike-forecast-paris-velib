# service/jobs/build_model_performance.py

# -----------------------------------------------------------------------------
# Build "Model Performance" monitoring artifacts from perf_YYYY-MM-DD.parquet
# ➜ VERSION "LATEST ONLY" (no timestamped versioned folders)
#
# Time window aligned with Data Health:
#   - End  = last aligned bin of Y-1 (UTC, aligned on BIN_MIN)
#   - Start= 00:00:00 UTC of Y-(MON_LAST_DAYS-1)
#
# Inputs  (GCS):
#   GCS_EXPORTS_PREFIX/perf_YYYY-MM-DD.parquet  (one per day)
#
# Outputs (GCS JSON — LATEST ONLY):
#   <MONITORING_BASE>/model/performance/latest/manifest.json
#   <MONITORING_BASE>/model/performance/latest/h{H}/kpis.json
#   <MONITORING_BASE>/model/performance/latest/h{H}/daily_metrics.json
#   <MONITORING_BASE>/model/performance/latest/h{H}/by_hour.json
#   <MONITORING_BASE>/model/performance/latest/h{H}/by_dow.json
#   <MONITORING_BASE>/model/performance/latest/h{H}/by_station.json
#   <MONITORING_BASE>/model/performance/latest/h{H}/by_cluster.json  (optional)
#   <MONITORING_BASE>/model/performance/latest/h{H}/lift_curve.json
#   <MONITORING_BASE>/model/performance/latest/h{H}/hist_residuals.json
#   <MONITORING_BASE>/model/performance/latest/h{H}/station_timeseries.json   # 24h
#
# ENV (required):
#   GCS_EXPORTS_PREFIX     = gs://<bucket>/velib/exports
#   GCS_MONITORING_PREFIX  = gs://<bucket>/velib
#
# Unified ENV (like other jobs):
#   MON_TZ            = Europe/Paris      (fallback PERF_TZ, default "Europe/Paris")
#   MON_LAST_DAYS     = 14                (fallback PERF_LAST_DAYS, default 14)
#   MON_HORIZONS      = "15"              (CSV; fallback PERF_HORIZONS / FORECAST_HORIZONS)
#
# Optional:
#   PERF_RESID_BINS      = 40
#   PERF_TOP_STATIONS    = 10
#   PERF_CLUSTERS_CSV    = gs://.../station_clusters.csv
#   PERF_TS_MIN_POINTS   = 24
#
# Run:
#   python -m service.jobs.build_model_performance
# -----------------------------------------------------------------------------

"""
Vélib’ Forecast — Model Performance monitoring job (LATEST ONLY).

Rôle du job
-----------
Ce job construit tous les artefacts de **performance modèle** consommés par
l’UI de monitoring, à partir des fichiers journaliers `perf_YYYY-MM-DD.parquet`
produits par `build_datasets.py`.

Pour chaque horizon H (15, 60, …), il génère (en "latest only") :

- `kpis.json` : métriques globales (MAE, RMSE, lift vs baseline, coverage…)
- `daily_metrics.json` : MAE / lift par jour (courbe de lift)
- `by_hour.json` : performance par heure locale
- `by_dow.json` : performance par jour de la semaine
- `by_station.json` : performance par station
- `by_cluster.json` : performance par cluster (optionnel, via CSV)
- `lift_curve.json` : lift vs baseline dans le temps
- `hist_residuals.json` : histogramme des résidus (y_true - y_pred)
- `station_timeseries.json` : série temporelle 24h d’une station choisie

La fenêtre temporelle est **alignée avec Data Health** :

- fenêtre CALENDRIER LOCAL (MON_TZ) sur J-(MON_LAST_DAYS-1) → J-1,
- transformée en UTC via `_local_window_yesterday`,
- les données sont ensuite **troncées strictement** dans cette fenêtre.

En sortie, tout est publié sous :

    <GCS_MONITORING_PREFIX>/monitoring/model/performance/latest/...

Sans versionnement daté : l’UI ne lit que la dernière version disponible.
"""

from __future__ import annotations
import os, re, json, sys, random
from io import BytesIO
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception as e:
    raise RuntimeError("pyarrow required") from e

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage required") from e


SCHEMA_VERSION = "1.2"  # 1.1→1.2 : aligned Y-1 window + strict truncation + effective window in manifest
BIN_MIN = 5             # 5-minute cadence (pipeline invariant)


# ──────────────────────────────────────────────────────────────────────────────
# ENV helpers (unified)
# ──────────────────────────────────────────────────────────────────────────────

def _env(name: str, default=None):
    """
    Lire une variable d’environnement avec valeur par défaut.

    Paramètres
    ----------
    name : str
        Nom de la variable d’environnement.
    default : Any
        Valeur de repli si la variable n’est pas définie ou vide.

    Retour
    ------
    Any
        Valeur de la variable ou valeur de repli.
    """
    v = os.environ.get(name)
    return v if (v is not None and v != "") else default

def _env_int(name: str, default: int) -> int:
    """
    Lire une variable d’environnement et la convertir en entier.

    Si la conversion échoue, renvoie la valeur de repli.

    Paramètres
    ----------
    name : str
        Nom de la variable.
    default : int
        Valeur par défaut si non définie ou conversion impossible.

    Retour
    ------
    int
        Valeur entière ou valeur de repli.
    """
    try:
        return int(_env(name, default))
    except Exception:
        return default

def _parse_horizons() -> List[int]:
    """
    Résoudre la liste des horizons en minutes à monitorer.

    Priorité :
    1. MON_HORIZONS
    2. PERF_HORIZONS
    3. FORECAST_HORIZONS
    4. "15" par défaut

    Retour
    ------
    list[int]
        Horizons triés, uniques, > 0.
    """
    # Priority: MON_HORIZONS, then PERF_HORIZONS / FORECAST_HORIZONS
    raw = _env("MON_HORIZONS") or _env("PERF_HORIZONS") or _env("FORECAST_HORIZONS") or "15"
    hs: List[int] = []
    for x in str(raw).split(","):
        x = x.strip()
        if not x:
            continue
        try:
            v = int(x)
            if v > 0:
                hs.append(v)
        except Exception:
            pass
    return sorted(list(set(hs))) or [15]


# ──────────────────────────────────────────────────────────────────────────────
# GCS helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split(gs: str) -> Tuple[str, str]:
    """
    Découper une URI GCS `gs://bucket/path` en (bucket, key).

    Paramètres
    ----------
    gs : str
        URI GCS.

    Retour
    ------
    (str, str)
        Nom du bucket et chemin d’objet.
    """
    assert gs.startswith("gs://"), f"bad GCS uri: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k.rstrip("/")

def _read_parquet_blob_to_df(blob: "storage.Blob") -> pd.DataFrame:
    """
    Lire un blob Parquet GCS dans un DataFrame pandas.

    Paramètres
    ----------
    blob : google.cloud.storage.Blob
        Blob à lire.

    Retour
    ------
    pandas.DataFrame
        Contenu du Parquet.
    """
    buf = BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    tbl = pq.read_table(buf)
    return tbl.to_pandas()

def _list_perf_blobs(exports_prefix: str, start_date: datetime, end_date: datetime) -> List["storage.Blob"]:
    """List perf_YYYY-MM-DD.parquet blobs between start_date and end_date (UTC-day inclusive)."""
    bkt, key_prefix = _split(exports_prefix)
    client = storage.Client()
    bucket = client.bucket(bkt)
    blobs = list(client.list_blobs(bucket, prefix=key_prefix.strip("/") + "/"))
    pat = re.compile(r"perf_(\d{4}-\d{2}-\d{2})\.parquet$")
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

def _upload_json_gs(obj: dict, gs_uri: str):
    """Upload JSON with NaN/Inf sanitized to null."""
    def _san(o):
        if isinstance(o, dict):  return {k: _san(v) for k, v in o.items()}
        if isinstance(o, list):  return [_san(v) for v in o]
        if isinstance(o, float): return float(o) if np.isfinite(o) else None
        return o

    safe = _san(obj)
    bkt, key = _split(gs_uri)
    data = json.dumps(safe, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    storage.Client().bucket(bkt).blob(key).upload_from_string(data, content_type="application/json")
    print(f"[model.performance] wrote → {gs_uri} ({len(data):,} bytes)")


# ──────────────────────────────────────────────────────────────────────────────
# Time & base path helpers (aligned with Data Health)
# ──────────────────────────────────────────────────────────────────────────────

# ── remplace _yesterday_end_floor / _window_start_for_days par :

def _local_window_yesterday(tzname: str, days: int, bin_min: int = BIN_MIN) -> tuple[datetime, datetime]:
    """
    Fenêtre alignée CALENDRIER LOCAL pour la perf (cohérente avec Data Health).

    Définition (en timezone locale MON_TZ) :
      - end_local   = 23:55:00 de J-1 (floor sur BIN_MIN)
      - start_local = 00:00:00 de J-(days-1)

    La fonction renvoie la fenêtre en UTC (tz-aware) : (start_utc, end_utc).

    Paramètres
    ----------
    tzname : str
        Nom de timezone (ex: "Europe/Paris").
    days : int
        Nombre de jours dans la fenêtre (incluant J-1).
    bin_min : int
        Taille d’un bin en minutes (BIN_MIN=5).

    Retour
    ------
    (datetime, datetime)
        (start_utc, end_utc) tz-aware.
    """
    now_local = pd.Timestamp.now(tz=tzname)
    end_local = (now_local - pd.Timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
    end_local = end_local.floor(f"{int(bin_min)}min")
    start_local = end_local.normalize() - pd.Timedelta(days=max(int(days) - 1, 0))
    return start_local.tz_convert("UTC").to_pydatetime(), end_local.tz_convert("UTC").to_pydatetime()

def _ensure_mon_base(mon_prefix: str) -> str:
    """Ensure path ends with '/monitoring' once."""
    base = mon_prefix.rstrip("/")
    if not base.endswith("/monitoring"):
        base = base + "/monitoring"
    return base

def _to_local(utc_series: pd.Series, tzname: str) -> pd.Series:
    """
    Convertir une série UTC en timezone locale.

    Paramètres
    ----------
    utc_series : pandas.Series
        Série de timestamps (tz-aware ou naïfs).
    tzname : str
        Timezone cible (ex: "Europe/Paris").

    Retour
    ------
    pandas.Series
        Série de timestamps tz-aware dans la timezone cible.
    """
    s = pd.to_datetime(utc_series, utc=True, errors="coerce")
    return s.dt.tz_convert(tzname)


# ──────────────────────────────────────────────────────────────────────────────
# Schema detection & normalization
# ──────────────────────────────────────────────────────────────────────────────

def _detect_perf_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Auto-detect key columns with robust fallbacks."""
    lower = {c.lower(): c for c in df.columns}

    def any_of(*cands):
        for c in cands:
            if c.lower() in lower:
                return lower[c.lower()]
        return None

    tbin = any_of("tbin_utc", "ts", "timestamp", "datetime")
    sid = any_of("station_id", "stationcode", "id", "station")
    ytrue = any_of("y_true", "y", "target", "bikes_true", "nb_velos_bin")
    ypred_int = any_of("y_pred_int", "bikes_pred_int")
    ypred_float = any_of("y_pred", "bikes_pred", "yhat", "prediction", "pred")
    ybase = any_of("y_baseline_persist", "y_pred_baseline", "baseline", "persist")
    cap = any_of("capacity", "cap", "num_docks_total", "dock_count")
    hb = any_of("horizon_bins",)
    hmin = any_of("horizon_min",)

    if not tbin or not sid or not ytrue or not ybase:
        raise KeyError(f"[model.performance] Missing minimal columns (tbin={tbin}, station={sid}, y_true={ytrue}, baseline={ybase})")

    return dict(
        tbin=tbin, station=sid, y_true=ytrue, y_pred_int=ypred_int,
        y_pred_float=ypred_float, y_baseline=ybase, capacity=cap,
        horizon_bins=hb, horizon_min=hmin
    )

def _normalize_types(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    """
    Normaliser les types des colonnes structurantes de perf.

    - tbin     : datetime UTC tz-aware
    - station  : string
    - y_true   : float
    - y_baseline, y_pred_* : float
    - capacity : float
    - horizons : float (avant conversion en Int64)

    Paramètres
    ----------
    df : pandas.DataFrame
        Dataset perf brut.
    cols : dict
        Mapping des noms de colonnes.

    Retour
    ------
    pandas.DataFrame
        DataFrame copié/typé.
    """
    out = df.copy()
    out[cols["tbin"]] = pd.to_datetime(out[cols["tbin"]], utc=True, errors="coerce")  # tz-aware UTC
    out[cols["station"]] = out[cols["station"]].astype("string")
    out[cols["y_true"]] = pd.to_numeric(out[cols["y_true"]], errors="coerce")
    out[cols["y_baseline"]] = pd.to_numeric(out[cols["y_baseline"]], errors="coerce")
    if cols["y_pred_int"] and cols["y_pred_int"] in out.columns:
        out[cols["y_pred_int"]] = pd.to_numeric(out[cols["y_pred_int"]], errors="coerce")
    if cols["y_pred_float"] and cols["y_pred_float"] in out.columns:
        out[cols["y_pred_float"]] = pd.to_numeric(out[cols["y_pred_float"]], errors="coerce")
    if cols["capacity"] and cols["capacity"] in out.columns:
        out[cols["capacity"]] = pd.to_numeric(out[cols["capacity"]], errors="coerce")
    # horizons
    if cols["horizon_bins"] and cols["horizon_bins"] in out.columns:
        out[cols["horizon_bins"]] = pd.to_numeric(out[cols["horizon_bins"]], errors="coerce")
    if cols["horizon_min"] and cols["horizon_min"] in out.columns:
        out[cols["horizon_min"]] = pd.to_numeric(out[cols["horizon_min"]], errors="coerce")
    return out

def _select_pred_series(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.Series:
    """Return the best available prediction series (Int preferred, else float)."""
    if cols["y_pred_int"] and cols["y_pred_int"] in df.columns:
        return pd.to_numeric(df[cols["y_pred_int"]], errors="coerce")
    if cols["y_pred_float"] and cols["y_pred_float"] in df.columns:
        return pd.to_numeric(df[cols["y_pred_float"]], errors="coerce")
    return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")

def _normalize_horizon_bin(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.Series:
    """Return a Series 'hbin' of integer horizon bins."""
    if cols["horizon_bins"] and cols["horizon_bins"] in df.columns:
        s = pd.to_numeric(df[cols["horizon_bins"]], errors="coerce")
        return s.round().astype("Int64")
    if cols["horizon_min"] and cols["horizon_min"] in df.columns:
        m = pd.to_numeric(df[cols["horizon_min"]], errors="coerce")
        return (m / BIN_MIN).round().astype("Int64")
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def _metrics(y_true: pd.Series, y_hat: pd.Series) -> Dict[str, float]:
    """
    Calculer les métriques de base : MAE, RMSE, ME (mean error).

    Paramètres
    ----------
    y_true : pandas.Series
        Observations.
    y_hat : pandas.Series
        Prédictions.

    Retour
    ------
    dict
        {"mae": float, "rmse": float, "me": float}
    """
    err = y_true.astype("float64") - y_hat.astype("float64")
    mae = float(np.nanmean(np.abs(err))) if len(err) else float("nan")
    rmse = float(np.sqrt(np.nanmean(np.square(err)))) if len(err) else float("nan")
    me = float(np.nanmean(err)) if len(err) else float("nan")
    return {"mae": mae, "rmse": rmse, "me": me}

def _lift(mae_base: float, mae_model: float) -> float:
    """
    Calculer le lift relatif vs baseline : (MAE_base - MAE_model) / MAE_base.

    Gestion de sécurité si MAE_base est nul ou NaN.

    Paramètres
    ----------
    mae_base : float
        MAE de la baseline.
    mae_model : float
        MAE du modèle.

    Retour
    ------
    float
        Lift, ou NaN si non défini.
    """
    if mae_base is None or np.isnan(mae_base) or mae_base == 0 or mae_model is None or np.isnan(mae_model):
        return float("nan")
    return float((mae_base - mae_model) / mae_base)

def _coverage_pct(pred: pd.Series) -> float:
    """
    Pourcentage de lignes pour lesquelles une prédiction est disponible.

    Paramètres
    ----------
    pred : pandas.Series
        Série de prédictions (peut contenir des NaN).

    Retour
    ------
    float
        Pourcentage (0–100) de non-NaN, arrondi à 2 décimales.
    """
    return float((pred.notna().mean() * 100.0).round(2)) if len(pred) else float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Computations (global/daily/segments/hist)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_global(df: pd.DataFrame, cols: Dict[str, Optional[str]], tzname: str, window_days: int, horizon_min: Optional[int]) -> dict:
    """
    Calculer les KPIs globaux pour un horizon.

    Paramètres
    ----------
    df : pandas.DataFrame
        Sous-ensemble perf (une horizon bin, fenêtre tronquée).
    cols : dict
        Mapping de colonnes (tbin, station, y_true, y_baseline, y_pred_*).
    tzname : str
        Timezone (non utilisée ici mais homogène avec autres helpers).
    window_days : int
        Fenêtre théorique en jours (pour info).
    horizon_min : int | None
        Horizon en minutes, ou None pour global.

    Retour
    ------
    dict
        KPIs globaux (mae, rmse, me, coverage, lift…).
    """
    y_true = pd.to_numeric(df[cols["y_true"]], errors="coerce")
    y_base = pd.to_numeric(df[cols["y_baseline"]], errors="coerce")
    y_pred = _select_pred_series(df, cols)

    m_model = _metrics(y_true, y_pred)
    m_base = _metrics(y_true, y_base)
    cov = _coverage_pct(y_pred)

    g = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "window_days": int(window_days),
        "horizon_min": int(horizon_min) if horizon_min is not None else None,
        "coverage_pred_pct": cov,
        "mae_model": m_model["mae"], "rmse_model": m_model["rmse"], "me_model": m_model["me"],
        "mae_baseline": m_base["mae"], "rmse_baseline": m_base["rmse"], "me_baseline": m_base["me"],
        "lift_vs_baseline": _lift(m_base["mae"], m_model["mae"]),
        "n_rows": int(len(df)),
        "n_stations": int(df[cols["station"]].nunique()),
        "ts_min_utc": pd.to_datetime(df[cols["tbin"]], utc=True, errors="coerce").min().isoformat().replace("+00:00","Z") if len(df) else None,
        "ts_max_utc": pd.to_datetime(df[cols["tbin"]], utc=True, errors="coerce").max().isoformat().replace("+00:00","Z") if len(df) else None,
    }
    return g

def _compute_daily(df: pd.DataFrame, cols: Dict[str, Optional[str]], tzname: str, last_days: int) -> List[dict]:
    """
    Calculer les métriques quotidiennes (MAE, RMSE, lift, coverage) par jour.

    Paramètres
    ----------
    df : pandas.DataFrame
        Sous-ensemble perf.
    cols : dict
        Mapping des colonnes (tbin, station, y_true, y_baseline, y_pred_*).
    tzname : str
        Timezone pour le regroupement par date locale.
    last_days : int
        Nombre de jours max à conserver (taille de la courbe), ou 0 pour tout.

    Retour
    ------
    list[dict]
        Une ligne par date locale.
    """
    tloc = _to_local(df[cols["tbin"]], tzname)
    y_true = pd.to_numeric(df[cols["y_true"]], errors="coerce")
    y_base = pd.to_numeric(df[cols["y_baseline"]], errors="coerce")
    y_pred = _select_pred_series(df, cols)

    tmp = pd.DataFrame({
        "_date": tloc.dt.date,
        "_y": y_true,
        "_b": y_base,
        "_p": y_pred
    })
    grp = tmp.groupby("_date", dropna=False)
    rows = []
    for d, g in grp:
        mm = _metrics(g["_y"], g["_p"])
        mb = _metrics(g["_y"], g["_b"])
        cov = _coverage_pct(g["_p"])
        rows.append({
            "date": str(d),
            "mae_model": mm["mae"],
            "mae_baseline": mb["mae"],
            "rmse_model": mm["rmse"],
            "rmse_baseline": mb["rmse"],
            "coverage_pred_pct": cov,
            "lift_vs_baseline": _lift(mb["mae"], mm["mae"]),
            "n": int(len(g)),
        })
    rows = sorted(rows, key=lambda r: r["date"])
    if last_days and last_days > 0:
        rows = rows[-last_days:]
    return rows

def _compute_by_hour(df: pd.DataFrame, cols: Dict[str, Optional[str]], tzname: str) -> List[dict]:
    """
    Calculer les métriques agrégées par heure locale (0–23).

    Paramètres
    ----------
    df : pandas.DataFrame
        Sous-ensemble perf.
    cols : dict
        Mapping des colonnes (tbin, station, y_true, y_baseline, y_pred_*).
    tzname : str
        Timezone pour l’heure locale.

    Retour
    ------
    list[dict]
        Une ligne par heure locale.
    """
    tloc = _to_local(df[cols["tbin"]], tzname)
    y_true = pd.to_numeric(df[cols["y_true"]], errors="coerce")
    y_base = pd.to_numeric(df[cols["y_baseline"]], errors="coerce")
    y_pred = _select_pred_series(df, cols)

    tmp = pd.DataFrame({
        "_hour": tloc.dt.hour,
        "_y": y_true,
        "_b": y_base,
        "_p": y_pred
    })
    grp = tmp.groupby("_hour", dropna=False)
    rows = []
    for h, g in grp:
        mm = _metrics(g["_y"], g["_p"])
        mb = _metrics(g["_y"], g["_b"])
        cov = _coverage_pct(g["_p"])
        rows.append({
            "hour": int(h),
            "mae_model": mm["mae"],
            "mae_baseline": mb["mae"],
            "coverage_pred_pct": cov,
            "n": int(len(g)),
        })
    rows = sorted(rows, key=lambda r: r["hour"])
    return rows

def _compute_by_dow(df: pd.DataFrame, cols: Dict[str, Optional[str]], tzname: str) -> List[dict]:
    """
    Calculer les métriques agrégées par jour de semaine (0=lundi,…,6=dimanche).

    Paramètres
    ----------
    df : pandas.DataFrame
        Sous-ensemble perf.
    cols : dict
        Mapping des colonnes (tbin, station, y_true, y_baseline, y_pred_*).
    tzname : str
        Timezone pour le calcul du dayofweek.

    Retour
    ------
    list[dict]
        Une ligne par jour de la semaine.
    """
    tloc = _to_local(df[cols["tbin"]], tzname)
    y_true = pd.to_numeric(df[cols["y_true"]], errors="coerce")
    y_base = pd.to_numeric(df[cols["y_baseline"]], errors="coerce")
    y_pred = _select_pred_series(df, cols)

    tmp = pd.DataFrame({
        "_dow": tloc.dt.dayofweek,
        "_y": y_true,
        "_b": y_base,
        "_p": y_pred
    })
    grp = tmp.groupby("_dow", dropna=False)
    rows = []
    for d, g in grp:
        mm = _metrics(g["_y"], g["_p"])
        mb = _metrics(g["_y"], g["_b"])
        cov = _coverage_pct(g["_p"])
        rows.append({
            "dow": int(d),
            "mae_model": mm["mae"],
            "mae_baseline": mb["mae"],
            "coverage_pred_pct": cov,
            "n": int(len(g)),
        })
    rows = sorted(rows, key=lambda r: r["dow"])
    return rows

def _compute_by_station(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> List[dict]:
    """
    Calculer les métriques agrégées par station (station_id).

    Paramètres
    ----------
    df : pandas.DataFrame
        Sous-ensemble perf.
    cols : dict
        Mapping des colonnes (tbin, station, y_true, y_baseline, y_pred_*).

    Retour
    ------
    list[dict]
        Une ligne par station, avec lift vs baseline.
    """
    y_true = pd.to_numeric(df[cols["y_true"]], errors="coerce")
    y_base = pd.to_numeric(df[cols["y_baseline"]], errors="coerce")
    y_pred = _select_pred_series(df, cols)

    tmp = pd.DataFrame({
        "_sid": df[cols["station"]].astype("string"),
        "_y": y_true,
        "_b": y_base,
        "_p": y_pred
    })
    grp = tmp.groupby("_sid", dropna=False)
    rows = []
    for sid, g in grp:
        mm = _metrics(g["_y"], g["_p"])
        mb = _metrics(g["_y"], g["_b"])
        cov = _coverage_pct(g["_p"])
        rows.append({
            "station_id": str(sid),
            "mae_model": mm["mae"],
            "mae_baseline": mb["mae"],
            "coverage_pred_pct": cov,
            "n": int(len(g)),
            "lift_vs_baseline": _lift(mb["mae"], mm["mae"]),
        })
    rows = sorted(rows, key=lambda r: r["station_id"])
    return rows

def _compute_by_cluster(df: pd.DataFrame, cols: Dict[str, Optional[str]], clusters_df: Optional[pd.DataFrame]) -> Optional[List[dict]]:
    """
    Calculer les métriques agrégées par cluster, si un CSV de clusters est fourni.

    Paramètres
    ----------
    df : pandas.DataFrame
        Sous-ensemble perf.
    cols : dict
        Mapping des colonnes (tbin, station, y_true, y_baseline, y_pred_*).
    clusters_df : pandas.DataFrame | None
        DataFrame avec au moins `station_id` et `cluster`.

    Retour
    ------
    list[dict] | None
        Une ligne par cluster, ou None si clusters_df absent / incompatible.
    """
    if clusters_df is None or clusters_df.empty:
        return None
    m = df.copy()
    m[cols["station"]] = m[cols["station"]].astype("string")
    clusters_df = clusters_df.copy()
    if "station_id" not in clusters_df.columns or "cluster" not in clusters_df.columns:
        return None
    clusters_df["station_id"] = clusters_df["station_id"].astype("string")
    m = m.merge(clusters_df[["station_id", "cluster"]], left_on=cols["station"], right_on="station_id", how="left")
    if m["cluster"].notna().sum() == 0:
        return None

    y_true = pd.to_numeric(m[cols["y_true"]], errors="coerce")
    y_base = pd.to_numeric(m[cols["y_baseline"]], errors="coerce")
    y_pred = _select_pred_series(m, cols)

    tmp = pd.DataFrame({
        "_cluster": m["cluster"],
        "_y": y_true, "_b": y_base, "_p": y_pred
    }).dropna(subset=["_cluster"])
    grp = tmp.groupby("_cluster", dropna=False)
    rows = []
    for cl, g in grp:
        mm = _metrics(g["_y"], g["_p"])
        mb = _metrics(g["_y"], g["_b"])
        cov = _coverage_pct(g["_p"])
        rows.append({
            "cluster": cl,
            "mae_model": mm["mae"],
            "mae_baseline": mb["mae"],
            "coverage_pred_pct": cov,
            "n": int(len(g)),
            "lift_vs_baseline": _lift(mb["mae"], mm["mae"]),
        })
    rows = sorted(rows, key=lambda r: (str(r["cluster"])))
    return rows

def _build_lift_curve(daily_rows: List[dict]) -> dict:
    """
    Construire la courbe de lift à partir des daily_metrics.

    Paramètres
    ----------
    daily_rows : list[dict]
        Lignes de `daily_metrics` (une par date).

    Retour
    ------
    dict
        {"schema_version": ..., "points": [{"date": ..., "lift_vs_baseline": ...}, ...]}
    """
    pts = [{"date": r["date"], "lift_vs_baseline": r.get("lift_vs_baseline", None)} for r in daily_rows]
    return {"schema_version": SCHEMA_VERSION, "points": pts}

def _build_residual_hist(df: pd.DataFrame, cols: Dict[str, Optional[str]], bins: int) -> dict:
    """
    Construire un histogramme des résidus (y_true - y_pred).

    - Résidus tronqués au 99e percentile absolu.
    - Bins linéaires symétriques [-max_abs, max_abs].

    Paramètres
    ----------
    df : pandas.DataFrame
        Sous-ensemble perf.
    cols : dict
        Mapping des colonnes (y_true, y_pred_*).
    bins : int
        Nombre de bins de l’histogramme.

    Retour
    ------
    dict
        JSON-ready avec `bins`, `counts`, `n`.
    """
    y_true = pd.to_numeric(df[cols["y_true"]], errors="coerce")
    y_pred = _select_pred_series(df, cols)
    mask = y_pred.notna() & y_true.notna()
    if not mask.any():
        return {"schema_version": SCHEMA_VERSION, "bins": [], "counts": [], "n": 0}
    err = (y_true[mask] - y_pred[mask]).astype("float64").to_numpy()
    max_abs = float(np.nanpercentile(np.abs(err), 99)) if err.size else 1.0
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0
    hist_counts, bin_edges = np.histogram(err, bins=bins, range=(-max_abs, max_abs))
    return {
        "schema_version": SCHEMA_VERSION,
        "bins": [float(x) for x in bin_edges.tolist()],
        "counts": [int(c) for c in hist_counts.tolist()],
        "n": int(len(err))
    }

# ──────────────────────────────────────────────────────────────────────────────
# Station 24h timeseries (random eligible)
# ──────────────────────────────────────────────────────────────────────────────

def _build_station_timeseries_24h(df: pd.DataFrame, min_points: int, tzname: str, horizon_min: int) -> dict:
    """
    Construire une série temporelle 24h pour une station "représentative".

    Stratégie
    ---------
    - On identifie la dernière heure alignée (ts_max floor heure).
    - On prend la fenêtre [ts_max-24h, ts_max[ en UTC.
    - On filtre les stations ayant au moins `min_points` dans cette fenêtre.
      Si aucune n’en a assez, on prend la station la plus dense.
    - On renvoie les séries `ts`, `y_true`, `y_pred`, `y_base` pour cette station.

    Paramètres
    ----------
    df : pandas.DataFrame
        Sous-ensemble perf, avec colonnes normalisées `__tbin`, `__sid`, `__y`, `__p`, `__b`.
    min_points : int
        Nombre minimal de points pour être éligible.
    tzname : str
        Timezone (stockée dans le JSON, mais ts restent en UTC string).
    horizon_min : int
        Horizon en minutes (pour taguer la série).

    Retour
    ------
    dict
        Payload JSON-ready pour l’UI.
    """
    if df.empty:
        return {
            "schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "h": int(horizon_min), "station_id": None, "name": None, "tz": tzname,
            "ts": [], "y_true": [], "y_pred": [], "y_base": []
        }

    ts_max = pd.to_datetime(df["__tbin"], utc=True, errors="coerce").max()
    if pd.isna(ts_max):
        ts_max = datetime.now(timezone.utc)
    end_utc = ts_max.replace(minute=0, second=0, microsecond=0)
    start_utc = end_utc - timedelta(hours=24)

    w = (pd.to_datetime(df["__tbin"], utc=True, errors="coerce") >= start_utc) & \
        (pd.to_datetime(df["__tbin"], utc=True, errors="coerce") < end_utc)
    sub = df.loc[w, ["__tbin", "__sid", "__y", "__p", "__b"]].dropna(subset=["__tbin", "__sid"])
    if sub.empty:
        return {
            "schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "h": int(horizon_min), "station_id": None, "name": None, "tz": tzname,
            "ts": [], "y_true": [], "y_pred": [], "y_base": []
        }

    counts = sub.groupby("__sid")["__tbin"].count().reset_index(name="n")
    eligible = counts[counts["n"] >= int(min_points)]
    if eligible.empty:
        sid = counts.sort_values("n", ascending=False).iloc[0]["__sid"]
    else:
        sid = random.choice(eligible["__sid"].tolist())

    s = sub[sub["__sid"] == sid].sort_values("__tbin")

    ts = pd.to_datetime(s["__tbin"], utc=True, errors="coerce")

    def _series_to_num_list(x: pd.Series) -> list[Optional[float]]:
        arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype="float64")
        return [float(v) if np.isfinite(v) else None for v in arr]

    y_true = _series_to_num_list(s["__y"])
    y_pred = _series_to_num_list(s["__p"])
    y_base = _series_to_num_list(s["__b"])

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "h": int(horizon_min),
        "station_id": str(sid),
        "name": None,
        "tz": tzname,
        "ts": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in ts if pd.notna(t)],
        "y_true": y_true,
        "y_pred": y_pred,
        "y_base": y_base,
    }
    return payload

# ──────────────────────────────────────────────────────────────────────────────
# Clusters CSV (optional)
# ──────────────────────────────────────────────────────────────────────────────

def _read_clusters_csv(gs_uri: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Lire un CSV de clusters station → cluster depuis GCS (optionnel).

    Le fichier doit contenir au minimum les colonnes :
        - station_id
        - cluster

    Paramètres
    ----------
    gs_uri : str | None
        URI GCS du CSV (PERF_CLUSTERS_CSV).

    Retour
    ------
    pandas.DataFrame | None
        DataFrame (station_id, cluster) ou None si absent / invalide.
    """
    if not gs_uri or not gs_uri.startswith("gs://"):
        return None
    bkt, key = _split(gs_uri)
    cli = storage.Client()
    bl = cli.bucket(bkt).blob(key)
    if not bl.exists():
        print(f"[model.performance] clusters csv not found: {gs_uri}")
        return None
    data = bl.download_as_bytes()
    try:
        df = pd.read_csv(BytesIO(data), dtype={"station_id": "string"})
        if "station_id" in df.columns and "cluster" in df.columns:
            return df[["station_id", "cluster"]].drop_duplicates()
        print(f"[model.performance][warn] clusters csv missing required columns: {list(df.columns)}")
        return None
    except Exception as e:
        print(f"[model.performance][warn] failed to read clusters csv: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Publishing (LATEST ONLY)
# ──────────────────────────────────────────────────────────────────────────────

def _publish_latest(base_alias: str, horizon_min: int, payloads: Dict[str, dict]) -> None:
    """Publish artifacts for a given horizon under latest/h{H} only."""
    hdir = f"h{int(horizon_min)}"
    for name, obj in payloads.items():
        _upload_json_gs(obj, f"{base_alias}/{hdir}/{name}.json")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    """
    Entrypoint CLI du job Model Performance (LATEST ONLY).

    Étapes principales
    ------------------
    1. Lire les ENV unifiées (GCS_EXPORTS_PREFIX, GCS_MONITORING_PREFIX,
       MON_TZ, MON_LAST_DAYS, MON_HORIZONS…).
    2. Construire la fenêtre "théorique" alignée Y-1 via `_local_window_yesterday`.
    3. Lister et lire les fichiers `perf_YYYY-MM-DD.parquet` dans cette fenêtre.
    4. Détecter les colonnes de perf et normaliser les types.
    5. Appliquer une **troncature stricte** sur la fenêtre UTC (start_ref, end_ref).
    6. Enregistrer la fenêtre **effective** réellement couverte par les données.
    7. Normaliser les horizons en bins (`__hbin`) et filtrer sur les horizons demandés.
    8. Pour chaque horizon cible :
        - calculer les KPIs, daily_metrics, by_hour, by_dow, by_station,
          by_cluster (optionnel), lift_curve, hist_residuals, station_timeseries.
        - publier ces JSON sous `model/performance/latest/h{H}/`.
    9. Construire un `manifest.json` global pour la page Monitoring (Model Perf).

    Retour
    ------
    int
        Code de sortie (0 = succès).
    """
    EXPORTS_PREFIX = _env("GCS_EXPORTS_PREFIX")
    MON_PREFIX     = _env("GCS_MONITORING_PREFIX")
    if not (EXPORTS_PREFIX and str(EXPORTS_PREFIX).startswith("gs://")):
        raise RuntimeError("GCS_EXPORTS_PREFIX missing or invalid")
    if not (MON_PREFIX and str(MON_PREFIX).startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX missing or invalid")

    # Unified ENV + fallbacks
    TZNAME        = _env("MON_TZ", _env("PERF_TZ", "Europe/Paris"))
    LAST_DAYS     = _env_int("MON_LAST_DAYS", _env_int("PERF_LAST_DAYS", 14))
    HORIZONS      = _parse_horizons()

    # Optionals (unchanged)
    RESID_BINS    = _env_int("PERF_RESID_BINS", 40)
    CLUSTERS_URI  = _env("PERF_CLUSTERS_CSV", None)
    TS_MIN_POINTS = _env_int("PERF_TS_MIN_POINTS", 24)

    start_ref, end_ref = _local_window_yesterday(TZNAME, LAST_DAYS, BIN_MIN)
    print(f"[model.performance] window LOCAL({TZNAME}) -> UTC: "
        f"{start_ref.date()} → {end_ref.date()} | horizons={HORIZONS}")

    print(f"[env] MON_TZ={TZNAME} MON_LAST_DAYS={LAST_DAYS} MON_HORIZONS={','.join(map(str,HORIZONS))}")
    print(f"[model.performance] window (theoretical) UTC: {start_ref.date()} → {end_ref.date()} | horizons={HORIZONS}")

    # Read perf blobs in theoretical window (by file dates)
    blobs = _list_perf_blobs(EXPORTS_PREFIX, start_ref, end_ref)
    if not blobs:
        print("[model.performance] no perf_* blobs in window — nothing to do")
        mon_base = _ensure_mon_base(MON_PREFIX)
        base_alias = f"{mon_base}/model/performance/latest"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "window_days": int(LAST_DAYS),
            "window": {"start_utc": start_ref.isoformat().replace("+00:00","Z"),
                       "end_utc": end_ref.isoformat().replace("+00:00","Z")},
            "latest_prefix": base_alias,
            "horizons": [],
            "horizons_info": [],
            "tz": TZNAME,
            "sources": {"exports_prefix": EXPORTS_PREFIX},
        }
        _upload_json_gs(manifest, f"{base_alias}/manifest.json")
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
        print("[model.performance] no readable data — nothing to do")
        return 0

    perf = pd.concat(frames, ignore_index=True)
    if perf.empty:
        print("[model.performance] perf empty — nothing to do")
        return 0

    # Detect & normalize columns / types
    cols = _detect_perf_columns(perf)
    perf = _normalize_types(perf, cols)

    # STRICT truncation to the aligned Y-1 window (timestamps UTC tz-aware)
    tbin_col = cols["tbin"]
    perf = perf[(perf[tbin_col] > start_ref) & (perf[tbin_col] <= end_ref)].copy()
    if perf.empty:
        print("[model.performance] perf empty after window truncation — nothing to do")
        return 0

    # Effective window (ONLY what is present in files after truncation)
    t_min_eff = pd.to_datetime(perf[tbin_col], utc=True, errors="coerce").min()
    t_max_eff = pd.to_datetime(perf[tbin_col], utc=True, errors="coerce").max()
    print(f"[model.performance] window (effective) UTC: {t_min_eff.date()} → {t_max_eff.date()} | horizons={HORIZONS}")

    # Explicit working columns
    sid_col = cols["station"]
    y_col = cols["y_true"]; b_col = cols["y_baseline"]
    pred_series = _select_pred_series(perf, cols)
    hbin_series = _normalize_horizon_bin(perf, cols)

    # Add normalized columns for processing
    perf = perf.assign(
        __tbin=perf[tbin_col],
        __sid=perf[sid_col].astype("string"),
        __y=pd.to_numeric(perf[y_col], errors="coerce"),
        __b=pd.to_numeric(perf[b_col], errors="coerce"),
        __p=pred_series,
        __hbin=hbin_series
    ).dropna(subset=["__tbin", "__sid"])

    # Optional clusters
    clusters_df = _read_clusters_csv(CLUSTERS_URI)

    # Output base (LATEST ONLY)
    mon_base = _ensure_mon_base(MON_PREFIX)
    base_alias = f"{mon_base}/model/performance/latest"

    # Determine available horizons in data
    available_hbins = sorted([int(x) for x in pd.Series(perf["__hbin"].dropna().unique()).tolist() if pd.notna(x)])
    requested_hbins = sorted(list(set([max(1, int(round(h / BIN_MIN))) for h in HORIZONS])))
    target_hbins = [hb for hb in requested_hbins if hb in available_hbins]
    if not target_hbins:
        print(f"[model.performance] no matching horizons in data. available_hbins={available_hbins}, requested={requested_hbins}")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "window_days": int(LAST_DAYS),
            "window": {"start_utc": (t_min_eff.isoformat().replace("+00:00","Z") if pd.notna(t_min_eff) else start_ref.isoformat().replace("+00:00","Z")),
                       "end_utc":   (t_max_eff.isoformat().replace("+00:00","Z") if pd.notna(t_max_eff) else end_ref.isoformat().replace("+00:00","Z"))},
            "latest_prefix": base_alias,
            "horizons": [],
            "horizons_info": [],
            "tz": TZNAME,
            "sources": {"exports_prefix": EXPORTS_PREFIX},
        }
        _upload_json_gs(manifest, f"{base_alias}/manifest.json")
        return 0

    manifests_items = []
    for hb in target_hbins:
        hmin = hb * BIN_MIN
        sub = perf.loc[perf["__hbin"] == hb].copy()
        if sub.empty:
            print(f"[model.performance] skip h={hmin} (empty after filter)")
            continue

        # Compute artifacts (on subset already truncated to Y-1 window)
        kpis = _compute_global(sub, cols={
            "tbin": "__tbin", "station": "__sid",
            "y_true": "__y", "y_baseline": "__b",
            "y_pred_int": "__p", "y_pred_float": "__p",
            "capacity": cols.get("capacity")
        }, tzname=TZNAME, window_days=LAST_DAYS, horizon_min=hmin)

        daily_rows = _compute_daily(sub, cols={
            "tbin": "__tbin", "station": "__sid",
            "y_true": "__y", "y_baseline": "__b",
            "y_pred_int": "__p", "y_pred_float": "__p",
        }, tzname=TZNAME, last_days=LAST_DAYS)

        by_hour_rows = _compute_by_hour(sub, cols={
            "tbin": "__tbin", "station": "__sid",
            "y_true": "__y", "y_baseline": "__b",
            "y_pred_int": "__p", "y_pred_float": "__p",
        }, tzname=TZNAME)

        by_dow_rows = _compute_by_dow(sub, cols={
            "tbin": "__tbin", "station": "__sid",
            "y_true": "__y", "y_baseline": "__b",
            "y_pred_int": "__p", "y_pred_float": "__p",
        }, tzname=TZNAME)

        by_station_rows = _compute_by_station(sub, cols={
            "tbin": "__tbin", "station": "__sid",
            "y_true": "__y", "y_baseline": "__b",
            "y_pred_int": "__p", "y_pred_float": "__p",
        })

        by_cluster_rows = _compute_by_cluster(sub, cols={
            "tbin": "__tbin", "station": "__sid",
            "y_true": "__y", "y_baseline": "__b",
            "y_pred_int": "__p", "y_pred_float": "__p",
        }, clusters_df=clusters_df)

        lift_curve = _build_lift_curve(daily_rows)
        hist_res = _build_residual_hist(sub, cols={
            "tbin": "__tbin", "station": "__sid",
            "y_true": "__y", "y_baseline": "__b",
            "y_pred_int": "__p", "y_pred_float": "__p",
        }, bins=RESID_BINS)

        station_ts = _build_station_timeseries_24h(sub, min_points=int(TS_MIN_POINTS), tzname=TZNAME, horizon_min=hmin)

        # Assemble payloads per horizon
        payloads: Dict[str, dict] = {
            "kpis": kpis,
            "daily_metrics": {"schema_version": SCHEMA_VERSION, "horizon_min": int(hmin), "rows": daily_rows},
            "by_hour": {"schema_version": SCHEMA_VERSION, "horizon_min": int(hmin), "rows": by_hour_rows},
            "by_dow": {"schema_version": SCHEMA_VERSION, "horizon_min": int(hmin), "rows": by_dow_rows},
            "by_station": {"schema_version": SCHEMA_VERSION, "horizon_min": int(hmin), "rows": by_station_rows},
            "lift_curve": dict(lift_curve, **{"horizon_min": int(hmin)}),
            "hist_residuals": dict(hist_res, **{"horizon_min": int(hmin)}),
            "station_timeseries": station_ts,
        }
        if by_cluster_rows is not None:
            payloads["by_cluster"] = {"schema_version": SCHEMA_VERSION, "horizon_min": int(hmin), "rows": by_cluster_rows}

        # Publish LATEST ONLY under h{H}
        _publish_latest(base_alias, horizon_min=hmin, payloads=payloads)

        manifests_items.append({
            "horizon_min": int(hmin),
            "prefix_latest": f"{base_alias}/h{int(hmin)}",
            "artifacts": list(payloads.keys())
        })

    # Top-level manifest (LATEST ONLY) — uses the EFFECTIVE window actually present
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "window_days": int(LAST_DAYS),
        "window": {
            "start_utc": t_min_eff.isoformat().replace("+00:00","Z") if pd.notna(t_min_eff) else start_ref.isoformat().replace("+00:00","Z"),
            "end_utc":   t_max_eff.isoformat().replace("+00:00","Z") if pd.notna(t_max_eff) else end_ref.isoformat().replace("+00:00","Z"),
        },
        "latest_prefix": base_alias,
        "horizons": [int(it["horizon_min"]) for it in manifests_items],
        "horizons_info": manifests_items,
        "tz": TZNAME,
        "sources": {"exports_prefix": EXPORTS_PREFIX},
    }
    _upload_json_gs(manifest, f"{base_alias}/manifest.json")

    print("[model.performance] done (latest only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
