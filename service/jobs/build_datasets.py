# service/jobs/build_datasets.py

# Daily build (à partir d'un compact_YYYY-MM-DD.parquet) :
#  - exports/events.parquet               (jour ancre)
#  - exports/events_YYYY-MM-DD.parquet
#  - exports/perf.parquet                 (jour ancre, avec y_pred_int)
#  - exports/perf_YYYY-MM-DD.parquet
#
# Entrée : GCS_DAILY_PREFIX/compact_YYYY-MM-DD.parquet
#
# ENV requis :
#   GCS_DAILY_PREFIX    = gs://<bucket>/velib/daily
#   GCS_EXPORTS_PREFIX  = gs://<bucket>/velib/exports
#
# Modèles (un par horizon en minutes) — peuvent être:
#   - un FICHIER: gs://.../models/h15/latest.joblib (ou *.joblib)
#   - un PRÉFIXE/DOSSIER: gs://.../models/h15         → résolu en latest.joblib
#   MODEL_URI_15, MODEL_URI_60, etc.
#
# Optionnels :
#   FORECAST_HORIZONS   = "15,60" (défaut "15,60")
#   PENURY_THRESH       = 2
#   SATURATION_THRESH   = 2
#   DAY                 = "YYYY-MM-DD" (défaut today UTC)
#   HISTORY_BACK_MIN    = 240  (min d'historique J-1 à ajouter avant minuit ; défaut 4h)
#
# Exécution :
#   python -m service.jobs.build_datasets

"""
Vélib’ Forecast – Daily datasets builder.

Rôle du job
-----------
À partir d’un **compact_YYYY-MM-DD.parquet** (granularité 5 minutes) :
- produit les exports "événements" du jour ancre :
    - events.parquet (alias latest)
    - events_YYYY-MM-DD.parquet
- produit les exports "perf" (targets + baseline + prédiction modèle) :
    - perf.parquet (alias latest)
    - perf_YYYY-MM-DD.parquet

Schéma général
--------------
1. Lecture du compact du jour ancre (et éventuellement d’une "queue" du jour-1
   pour éviter la cassure à minuit sur les features).
2. Normalisation du schéma (timestamps, station_id, bikes, météo...).
3. Construction du dataset d’événements enrichi :
    - occ_ratio
    - is_penury / is_saturation
    - status_code
    - h / min
4. Construction d’une base de performance T → T+h pour chaque horizon :
    - y_true, y_baseline_persist, bikes, capacity, occ_ratio.
5. Construction des features d’inférence strictement alignées sur le training :
    - via `_add_features_spatial` importé de `service.core.forecast`.
6. Inférence des prédictions modèle pour chaque horizon :
    - chargement via `predict_from_features_df` (GCS file ou prefix).
    - post-traitement de la cible (clip 0..capacity, cast en Int64).
7. Écriture des Parquet events_YYYY-MM-DD.parquet et perf_YYYY-MM-DD.parquet
   sous `GCS_EXPORTS_PREFIX`.

Variables d’environnement
-------------------------
Obligatoires:
- GCS_DAILY_PREFIX   : gs://bucket/velib/daily
- GCS_EXPORTS_PREFIX : gs://bucket/velib/exports

Modèles:
- MODEL_URI_15, MODEL_URI_60, etc.
  → URI GCS d’un fichier joblib ou d’un **prefix** contenant latest.joblib.

Optionnelles:
- FORECAST_HORIZONS : "15,60" par défaut.
- PENURY_THRESH     : seuil pénurie (bikes ≤ N).
- SATURATION_THRESH : seuil saturation (capacity - bikes ≤ N).
- DAY               : jour ancre "YYYY-MM-DD" (défaut today UTC).
- HISTORY_BACK_MIN  : minutes d’historique J-1 à concaténer (défaut 240).
"""

from __future__ import annotations
import os, re, sys
from io import BytesIO
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    raise RuntimeError("pyarrow est requis pour build_datasets.py") from e

try:
    from google.cloud import storage
except Exception as e:
    raise RuntimeError("google-cloud-storage est requis (GCS I/O)") from e

# ─────────────────────────────────────────────
#  Imports training/forecast (STRICT ALIGNEMENT)
# ─────────────────────────────────────────────
# On réutilise EXACTEMENT la fabrique de features du training.
from service.core.forecast import _add_features_spatial, predict_from_features_df

BIN_MIN = 5
COLS = [
    "ts_utc","tbin_utc","station_id","bikes","capacity","mechanical","ebike",
    "status","lat","lon","name","temp_C","precip_mm","wind_mps"
]

# minutes d'historique du jour-1 à concaténer (évite la cassure à minuit)
HISTORY_BACK_MIN = int(os.environ.get("HISTORY_BACK_MIN", "240"))

# ───────────────────────────── GCS helpers ─────────────────────────────

def _split(gs: str) -> Tuple[str, str]:
    """
    Découper une URI GCS `gs://bucket/path` en (bucket, key).

    Paramètres
    ----------
    gs : str
        URI GCS à analyser.

    Retour
    ------
    (str, str)
        Nom du bucket et chemin de l’objet (sans slash final).

    Lève
    ----
    AssertionError
        Si la chaîne ne commence pas par 'gs://'.
    """
    assert gs.startswith("gs://"), f"bad GCS uri: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k.rstrip("/")


def _open_blob(gs: str):
    """
    Renvoyer un objet `Blob` GCS pour une URI donnée.

    Paramètres
    ----------
    gs : str
        URI GCS de la forme 'gs://bucket/path/to/object'.

    Retour
    ------
    google.cloud.storage.Blob
        Blob GCS correspondant.
    """
    b, k = _split(gs)
    return storage.Client().bucket(b).blob(k)


def _read_gcs_parquet(gs: str) -> pd.DataFrame:
    """
    Lire un fichier Parquet stocké sur GCS dans un DataFrame pandas.

    Paramètres
    ----------
    gs : str
        URI GCS du Parquet à lire.

    Retour
    ------
    pandas.DataFrame
        Contenu du fichier Parquet.

    Lève
    ----
    FileNotFoundError
        Si le blob n’existe pas.
    """
    blob = _open_blob(gs)
    if not blob.exists():
        raise FileNotFoundError(gs)
    buf = BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return pq.read_table(buf).to_pandas()


def _write_gcs_parquet(df: pd.DataFrame, gs: str):
    """
    Écrire un DataFrame pandas en Parquet sur GCS (compression snappy).

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame à sérialiser.
    gs : str
        URI GCS de destination (fichier Parquet).
    """
    b, k = _split(gs)
    buf = BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf, compression="snappy")
    buf.seek(0)
    storage.Client().bucket(b).blob(k).upload_from_file(buf, content_type="application/octet-stream")
    print(f"[build_datasets] wrote → {gs} (rows={len(df):,})")


def _copy_gcs(src_gs: str, dst_gs: str):
    """
    Copier un objet GCS vers un autre emplacement GCS.

    Paramètres
    ----------
    src_gs : str
        URI GCS source.
    dst_gs : str
        URI GCS destination.
    """
    src_b, src_k = _split(src_gs)
    dst_b, dst_k = _split(dst_gs)
    cli = storage.Client()
    src_blob = cli.bucket(src_b).blob(src_k)
    cli.bucket(dst_b).copy_blob(src_blob, cli.bucket(dst_b), dst_k)
    print(f"[build_datasets] duplicated → {dst_gs}")

# ───────────────────────────── Date helpers ─────────────────────────────

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def _anchor_day_utc() -> str:
    """
    Résoudre le "jour ancre" en date ISO (UTC).

    Stratégie
    ---------
    1. Si l’ENV `DAY` est défini :
        - Si le format est YYYY-MM-DD, le renvoyer tel quel.
        - Sinon, essayer de le parser comme datetime et renvoyer .date().
    2. Sinon, utiliser la date UTC du moment.

    Retour
    ------
    str
        Chaîne ISO "YYYY-MM-DD".
    """
    day_env = os.environ.get("DAY")
    if day_env:
        d = day_env.strip()
        if _DATE_RE.match(d):
            return d
        try:
            return pd.to_datetime(d, utc=True, errors="coerce").date().isoformat()
        except Exception:
            pass
    return datetime.now(timezone.utc).date().isoformat()


def _list_available_dailies(prefix: str) -> List[str]:
    """
    Lister toutes les dates de dailies disponibles (compact_YYYY-MM-DD.parquet).

    Paramètres
    ----------
    prefix : str
        Préfixe GCS (GCS_DAILY_PREFIX).

    Retour
    ------
    list[str]
        Liste triée de dates "YYYY-MM-DD" trouvées.
    """
    bkt, pfx = _split(prefix)
    cli = storage.Client()
    days = []
    p = re.compile(r".*/compact_(\d{4}-\d{2}-\d{2})\.parquet$")
    for b in cli.list_blobs(bkt, prefix=pfx):
        m = p.match(b.name)
        if m:
            days.append(m.group(1))
    days.sort()
    return days


def _find_best_daily(prefix: str, anchor_day: str) -> Optional[str]:
    """
    Choisir la meilleure date de daily disponible par rapport au jour ancre.

    Règles
    ------
    - Si des dailies ≤ anchor_day existent → prendre la plus récente.
    - Sinon → prendre la première date > anchor_day.
    - Sinon → None (rien disponible).

    Paramètres
    ----------
    prefix : str
        Préfixe GCS GCS_DAILY_PREFIX.
    anchor_day : str
        Jour ancre "YYYY-MM-DD".

    Retour
    ------
    str or None
        Jour retenu, ou None si aucun fichier compact n’est dispo.
    """
    avail = _list_available_dailies(prefix)
    if not avail:
        return None
    candidates = [d for d in avail if d <= anchor_day]
    if candidates:
        return candidates[-1]
    for d in avail:
        if d > anchor_day:
            return d
    return None

# ───────────────────────────── Normalisation ─────────────────────────────

def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forcer le schéma standard des events (COLS) avec types homogènes.

    Règles
    ------
    - Ajoute les colonnes manquantes initialisées à None.
    - Convertit :
        • ts_utc, tbin_utc → datetime UTC naïf.
        • station_id, status, name → string.
        • bikes, capacity, mechanical, ebike → Int64 (nullable).
        • lat, lon, météo → float.
    """
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    out = pd.DataFrame({
        "ts_utc":     pd.to_datetime(df["ts_utc"],   utc=True, errors="coerce").dt.tz_convert(None),
        "tbin_utc":   pd.to_datetime(df["tbin_utc"], utc=True, errors="coerce").dt.tz_convert(None),
        "station_id": df["station_id"].astype("string"),
        "bikes":      pd.to_numeric(df["bikes"],      errors="coerce"),
        "capacity":   pd.to_numeric(df["capacity"],   errors="coerce"),
        "mechanical": pd.to_numeric(df["mechanical"], errors="coerce"),
        "ebike":      pd.to_numeric(df["ebike"],      errors="coerce"),
        "status":     df["status"].astype("string"),
        "lat":        pd.to_numeric(df["lat"],        errors="coerce"),
        "lon":        pd.to_numeric(df["lon"],        errors="coerce"),
        "name":       df["name"].astype("string"),
        "temp_C":     pd.to_numeric(df["temp_C"],     errors="coerce"),
        "precip_mm":  pd.to_numeric(df["precip_mm"],  errors="coerce"),
        "wind_mps":   pd.to_numeric(df["wind_mps"],   errors="coerce"),
    })
    for c in ["bikes","capacity","mechanical","ebike"]:
        out[c] = out[c].astype("Int64")
    return out


def _dedup_latest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprimer les doublons sur (station_id, tbin_utc) en gardant le dernier ts_utc.

    Paramètres
    ----------
    df : pandas.DataFrame
        Données d’événements.

    Retour
    ------
    pandas.DataFrame
        DataFrame dédupliqué avec une ligne par (station_id, tbin_utc).
    """
    if not {"station_id","tbin_utc","ts_utc"}.issubset(df.columns):
        return df
    df = df.sort_values(["station_id","tbin_utc","ts_utc"])
    return df.groupby(["station_id","tbin_utc"], as_index=False).tail(1).reset_index(drop=True)

# ───────────────────────────── Prev-day tail helper ─────────────────────────────

def _maybe_read_prev_tail(prefix: str, day_iso: str, back_min: int) -> pd.DataFrame:
    """Lit compact_(jour-1).parquet et renvoie la fenêtre [J-1 24h-back, J 00:00[ (UTC)."""
    if back_min <= 0:
        return pd.DataFrame()
    day = pd.to_datetime(day_iso).tz_localize("UTC").tz_convert(None)
    start_prev_tail = day - pd.Timedelta(minutes=back_min)
    prev_day_iso = (day - pd.Timedelta(days=1)).date().isoformat()
    prev_uri = f"{prefix.rstrip('/')}/compact_{prev_day_iso}.parquet"
    try:
        df_prev = _read_gcs_parquet(prev_uri)
    except FileNotFoundError:
        return pd.DataFrame()
    df_prev = _enforce_schema(df_prev)
    df_prev = _dedup_latest(df_prev)
    m = (df_prev["tbin_utc"] >= start_prev_tail) & (df_prev["tbin_utc"] < day)
    tail = df_prev.loc[m].copy()
    return tail

# ───────────────────────────── Events build (jour unique) ─────────────────────────────

def _build_events(df_day: pd.DataFrame, penury: int, saturation: int) -> pd.DataFrame:
    """
    Construire le dataset d’événements enrichis pour un jour (avec historique).

    Enrichissements
    ---------------
    - occ_ratio       : bikes / capacity.
    - is_penury       : bikes ≤ PENURY_THRESH.
    - is_saturation   : capacity - bikes ≤ SATURATION_THRESH.
    - status_code     : encodage Int64 de la variable catégorielle `status`.
    - h / min         : heure & minute de tbin_utc.

    Le résultat est trié par (tbin_utc, station_id) et typé proprement.
    """
    df = _enforce_schema(df_day)
    df = _dedup_latest(df)

    cap = pd.to_numeric(df["capacity"], errors="coerce")
    bikes = pd.to_numeric(df["bikes"], errors="coerce")
    df["occ_ratio"] = np.where((cap > 0) & bikes.notna(), bikes / cap, np.nan)
    df["is_penury"] = ((bikes <= penury) & bikes.notna()).astype("Int8")
    df["is_saturation"] = ((cap - bikes <= saturation) & cap.notna() & bikes.notna()).astype("Int8")

    if "status" in df.columns:
        cats = sorted([s for s in df["status"].dropna().unique()])
        s_map = {s:i for i,s in enumerate(cats)}
        df["status_code"] = df["status"].map(s_map).astype("Int64")
    else:
        df["status_code"] = pd.NA

    t = pd.to_datetime(df["tbin_utc"], errors="coerce")
    df["h"]   = t.dt.hour.astype("Int8")
    df["min"] = t.dt.minute.astype("Int8")

    keep = [
        "tbin_utc","station_id","bikes","capacity","mechanical","ebike",
        "status","status_code","lat","lon","name","temp_C","precip_mm","wind_mps",
        "occ_ratio","is_penury","is_saturation",
        "h","min",
    ]
    out = df[keep].sort_values(["tbin_utc","station_id"]).reset_index(drop_index=True)
    out["station_id"] = out["station_id"].astype("string")
    out["tbin_utc"]   = pd.to_datetime(out["tbin_utc"], errors="coerce")
    return out

# ───────────────────────────── Perf base (T, T+h) ─────────────────────────────

def _build_perf_base(events: pd.DataFrame, horizons_min: List[int]) -> pd.DataFrame:
    """
    Construire la base de performance T → T+h pour tous les horizons.

    Pour chaque horizon h (en minutes) :
    - hb = h/BIN_MIN bins.
    - T = tbin_utc (temps source).
    - tbin_target = T + hb * BIN_MIN (cible).
    - y_true = bikes à tbin_target.
    - y_baseline_persist = bikes(T) (persistante).
    - On conserve également bikes(T), capacity(T), occ_ratio(T).

    Paramètres
    ----------
    events : pandas.DataFrame
        Dataset events du jour (already enriched via `_build_events`).
    horizons_min : list[int]
        Horizons en minutes (ex: [15, 60]).

    Retour
    ------
    pandas.DataFrame
        Base de performance avec une ligne par (T, station, horizon_bins).
    """
    if events.empty:
        return pd.DataFrame(columns=[
            "tbin_utc","station_id","horizon_bins","tbin_target",
            "y_true","y_baseline_persist","bikes","capacity","occ_ratio"
        ])

    events = events.sort_values(["station_id","tbin_utc"]).reset_index(drop=True)
    base_cols = ["tbin_utc","station_id","bikes","capacity","occ_ratio"]

    out_frames: List[pd.DataFrame] = []
    for hmin in horizons_min:
        hb = max(1, int(round(hmin / BIN_MIN)))
        # 1) table source (t) avec la cible temporelle
        t = events[base_cols].copy()
        t["tbin_target"] = pd.to_datetime(t["tbin_utc"], errors="coerce") + timedelta(minutes=BIN_MIN * hb)

        # 2) table cible (future) pour récupérer y_true
        tgt = events[["station_id","tbin_utc","bikes"]].rename(
            columns={"tbin_utc": "tbin_target", "bikes": "y_true"}
        )

        # 3) merge vectorisé par station + tbin_target
        merged = t.merge(tgt, on=["station_id","tbin_target"], how="left")

        merged["y_baseline_persist"] = merged["bikes"]
        merged["horizon_bins"] = hb
        out_frames.append(merged)

    perf = pd.concat(out_frames, ignore_index=True, sort=False)
    keep = ["tbin_utc","station_id","horizon_bins","tbin_target",
            "y_true","y_baseline_persist","bikes","capacity","occ_ratio"]
    perf = perf[keep].sort_values(["tbin_utc","station_id","horizon_bins"]).reset_index(drop=True)
    return perf

# ───────────────────────────── Features d'inférence (ALIGNÉES TRAINING) ─────────────────────────────

def _build_inference_features(events: pd.DataFrame, horizon_bins: int = 3) -> pd.DataFrame:
    """
    Fabrique STRICTEMENT alignée avec le training :
    - utilise _add_features_spatial (lags/rollings complets, lags météo, statiques, KNN)
    - prépare tbin_latest & capacity_bin pour predict_from_features_df

    Paramètres
    ----------
    events : pandas.DataFrame
        Dataset d’événements (incluant éventuellement la queue J-1).
    horizon_bins : int, défaut 3
        Horizon utilisé pour la fabrique (_add_features_spatial).

    Retour
    ------
    pandas.DataFrame
        DataFrame de features prêtes pour `predict_from_features_df`.
    """
    ev = _enforce_schema(events)
    ev = _dedup_latest(ev)

    enhanced, _feat_cols = _add_features_spatial(ev, horizon_bins=horizon_bins)

    enhanced = enhanced.copy()
    enhanced["tbin_latest"] = pd.to_datetime(enhanced["tbin_utc"], errors="coerce")
    try:
        enhanced["capacity_bin"] = enhanced["capacity"].round().astype("Int64")
    except Exception:
        enhanced["capacity_bin"] = pd.Series([pd.NA] * len(enhanced), dtype="Int64")

    enhanced["station_id"] = enhanced["station_id"].astype("string")
    enhanced["tbin_utc"]   = pd.to_datetime(enhanced["tbin_utc"], errors="coerce")

    return enhanced

# ───────────────────────────── Inférence (y_pred_int) ─────────────────────────────

def _infer_y_pred(perf_base: pd.DataFrame, feats_all: pd.DataFrame, horizons_min: List[int]) -> pd.DataFrame:
    """
    Inférer les prédictions `y_pred_int` pour chaque (T, station, horizon).

    Pipeline
    --------
    1. Pour chaque horizon h (en minutes) → horizon_bins hb:
       - extraire la sous-base perf_base[horizon_bins==hb].
       - construire la table d’input X via merge avec feats_all (tbin_utc, station_id).
    2. Charger le modèle pour cet horizon:
       - MODEL_URI_hmin peut être un fichier ou un préfixe (latest.joblib).
       - appel à `predict_from_features_df`.
    3. Réconcilier la prédiction avec les lignes de perf (clé station_id + tbin_target).
    4. Arrondir et clipper la prédiction en [0, capacity].

    En sortie, on renvoie la base perf avec:
        tbin_utc, station_id, horizon_bins,
        y_true, y_baseline_persist, y_pred_int, bikes, capacity, occ_ratio, h, min.
    """
    if perf_base.empty:
        merged = perf_base.assign(y_pred_int=pd.Series([pd.NA]*len(perf_base), dtype="Int64"))
        t = pd.to_datetime(merged["tbin_utc"], errors="coerce")
        merged["h"]   = t.dt.hour.astype("Int8")
        merged["min"] = t.dt.minute.astype("Int8")
        return merged[["tbin_utc","station_id","horizon_bins","y_true","y_baseline_persist","y_pred_int","bikes","capacity","occ_ratio","h","min"]]

    uri_map: Dict[int, Optional[str]] = {}
    for hmin in horizons_min:
        hb = max(1, int(round(hmin / BIN_MIN)))
        uri_map[hb] = os.environ.get(f"MODEL_URI_{hmin}")  # fichier *ou* prefix (résolu en latest)

    def _pick_pred_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """
        Heuristique pour trouver les colonnes de prédiction dans le DF de sortie.

        Renvoie:
        - int_col   : une colonne d’entiers si dispo (bikes_pred_int, y_pred_int…).
        - float_col : sinon, une colonne de floats (bikes_pred, y_pred, yhat…).

        Si aucune colonne standard n’est trouvée, tente un fallback sur les
        colonnes commençant par 'y_pred', 'yhat' ou 'bikes_pred'.
        """
        int_prefs   = ["bikes_pred_int", "y_pred_int"]
        float_prefs = ["bikes_pred", "y_pred", "yhat", "prediction", "pred"]
        int_col = next((c for c in int_prefs if c in df.columns), None)
        float_col = next((c for c in float_prefs if c in df.columns), None)
        if not int_col and not float_col:
            cand = [c for c in df.columns if c.startswith(("y_pred", "yhat", "bikes_pred"))]
            float_col = cand[0] if cand else None
        return int_col, float_col

    out_parts: List[pd.DataFrame] = []

    for hmin in horizons_min:
        hb = max(1, int(round(hmin / BIN_MIN)))
        sub = perf_base.loc[perf_base["horizon_bins"] == hb, ["tbin_utc","tbin_target","station_id"]].drop_duplicates()
        if sub.empty:
            continue

        sub = sub.copy()
        sub["station_id"]  = sub["station_id"].astype("string")
        sub["tbin_utc"]    = pd.to_datetime(sub["tbin_utc"], errors="coerce")
        sub["tbin_target"] = pd.to_datetime(sub["tbin_target"], errors="coerce")

        feats = feats_all.copy()
        feats["station_id"] = feats["station_id"].astype("string")
        feats["tbin_utc"]   = pd.to_datetime(feats["tbin_utc"], errors="coerce")

        # Features au temps source (tbin_utc)
        X = sub.merge(feats, on=["tbin_utc","station_id"], how="left")

        # Table des prédictions à merger au temps CIBLE
        pred_df = sub[["station_id","tbin_target"]].rename(columns={"tbin_target":"_key_target"}).copy()
        pred_df["y_pred_int"] = pd.Series([pd.NA]*len(pred_df), dtype="Int64")

        uri = uri_map.get(hb)
        if uri:
            try:
                preds = predict_from_features_df(
                    feats_df=X,
                    model_uri=uri,          # peut être prefix: résolu en latest.joblib
                    horizon_bins=hb,
                    model_alias=None,
                )
                if not preds.empty:
                    preds = preds.copy()
                    preds["station_id"] = preds["station_id"].astype("string")

                    time_key: Optional[str] = None
                    if "target_ts_utc" in preds.columns:
                        preds["target_ts_utc"] = pd.to_datetime(preds["target_ts_utc"], errors="coerce")
                        time_key = "target_ts_utc"
                    elif "tbin_target" in preds.columns:
                        preds["tbin_target"] = pd.to_datetime(preds["tbin_target"], errors="coerce")
                        time_key = "tbin_target"
                    elif "tbin_latest" in preds.columns:
                        preds["tbin_latest"] = pd.to_datetime(preds["tbin_latest"], errors="coerce")
                        time_key = "tbin_latest"
                    elif "tbin_utc" in preds.columns:
                        preds["tbin_utc"] = pd.to_datetime(preds["tbin_utc"], errors="coerce")
                        time_key = "tbin_utc"

                    int_col, float_col = _pick_pred_cols(preds)
                    if time_key and (int_col or float_col):
                        if int_col:
                            keep = preds[["station_id", time_key, int_col]].rename(
                                columns={time_key:"_key_target", int_col:"_y_int"}
                            )
                            pred_df = pred_df.merge(keep, on=["station_id","_key_target"], how="left")
                            pred_df["y_pred_int"] = pred_df["y_pred_int"].fillna(pred_df["_y_int"]).astype("Int64")
                            pred_df.drop(columns=["_y_int"], inplace=True)
                        else:
                            keep = preds[["station_id", time_key, float_col]].rename(
                                columns={time_key:"_key_target", float_col:"_y_f"}
                            )
                            pred_df = pred_df.merge(keep, on=["station_id","_key_target"], how="left")
                            ypf = pred_df["_y_f"].astype("float64")
                            pred_df["y_pred_int"] = pd.Series(np.rint(ypf).astype("float64"), dtype="Int64")
                            pred_df.drop(columns=["_y_f"], inplace=True)
                    else:
                        print(f"[build_datasets][warn] modèle {uri} (h={hmin}) sans colonne/clé reconnue: {list(preds.columns)}")
                else:
                    print(f"[build_datasets][warn] modèle {uri} (h={hmin}) a renvoyé un DF vide.")
            except Exception as e:
                print(f"[build_datasets][warn] inference failed for h={hmin} min: {e}")
        else:
            print(f"[build_datasets][warn] MODEL_URI_{hmin} non défini — y_pred_int=<NA>")

        # Re-mapper la prédiction *cible* sur la table perf (clé = station_id + tbin_target)
        sub_perf = perf_base.loc[perf_base["horizon_bins"] == hb].copy()
        sub_perf = sub_perf.merge(
            pred_df.rename(columns={"_key_target":"tbin_target"})[["station_id","tbin_target","y_pred_int"]],
            on=["station_id","tbin_target"], how="left"
        )

        # Bornage par capacité
        if "capacity" in sub_perf.columns:
            cap = pd.to_numeric(sub_perf["capacity"], errors="coerce")
            ypi = sub_perf["y_pred_int"].astype("Float64")
            ypi = np.clip(ypi, 0, cap)
            sub_perf["y_pred_int"] = pd.Series(np.rint(ypi).astype("float64"), dtype="Int64")

        out_parts.append(sub_perf)

    merged = pd.concat(out_parts, ignore_index=True, sort=False) if out_parts else perf_base.assign(y_pred_int=pd.Series([pd.NA]*len(perf_base), dtype="Int64"))

    # h / min depuis tbin_utc (temps source)
    t = pd.to_datetime(merged["tbin_utc"], errors="coerce")
    merged["h"]   = t.dt.hour.astype("Int8")
    merged["min"] = t.dt.minute.astype("Int8")

    cols = ["tbin_utc","station_id","horizon_bins","y_true","y_baseline_persist","y_pred_int","bikes","capacity","occ_ratio","h","min"]
    merged = merged[cols].sort_values(["tbin_utc","station_id","horizon_bins"]).reset_index(drop=True)

    merged["tbin_utc"]   = pd.to_datetime(merged["tbin_utc"], errors="coerce")
    merged["station_id"] = merged["station_id"].astype("string")
    merged["y_pred_int"] = merged["y_pred_int"].astype("Int64")
    merged["h"]          = merged["h"].astype("Int8")
    merged["min"]        = merged["min"].astype("Int8")
    return merged

# ───────────────────────────── Main (daily) ─────────────────────────────

def main() -> int:
    """
    Entrypoint CLI pour le job quotidien build_datasets.

    Étapes
    ------
    1. Résoudre le jour ancre (DAY ou today UTC) et sélectionner le meilleur
       compact_YYYY-MM-DD.parquet disponible sous GCS_DAILY_PREFIX.
    2. Définir la fenêtre [day_start, day_end[ en UTC.
    3. Lire le compact du jour + la "queue" de J-1 (HISTORY_BACK_MIN).
    4. Construire les events combinés sur (queue + jour) puis filtrer sur le
       jour ancre uniquement pour l’export.
    5. Construire la base perf à partir des events du jour.
    6. Construire les features d’inférence sur l’historique (queue+jour).
    7. Inférer les prédictions pour chaque horizon (MODEL_URI_*).
    8. Écrire :
        - exports/events_YYYY-MM-DD.parquet
        - exports/perf_YYYY-MM-DD.parquet

    Retour
    ------
    int
        Code de sortie (0 = succès).
    """
    DAILY_PREFIX   = os.environ.get("GCS_DAILY_PREFIX")
    EXPORTS_PREFIX = os.environ.get("GCS_EXPORTS_PREFIX")
    HORIZONS       = [int(x.strip()) for x in os.environ.get("FORECAST_HORIZONS","15,60").split(",") if x.strip()]
    PENURY_T       = int(os.environ.get("PENURY_THRESH","2"))
    SAT_T          = int(os.environ.get("SATURATION_THRESH","2"))

    if not (DAILY_PREFIX and DAILY_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_DAILY_PREFIX manquant ou invalide")
    if not (EXPORTS_PREFIX and EXPORTS_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_EXPORTS_PREFIX manquant ou invalide")

    anchor = _anchor_day_utc()
    best_day = _find_best_daily(DAILY_PREFIX, anchor)
    if not best_day:
        print(f"[build_datasets] no compact_* found under {DAILY_PREFIX}")
        return 0
    if best_day != anchor:
        print(f"[build_datasets] anchor={anchor} -> using available day={best_day}")

    day = best_day
    daily_uri = f"{DAILY_PREFIX.rstrip('/')}/compact_{day}.parquet"
    print(f"[build_datasets] day={day} daily_uri={daily_uri} horizons={HORIZONS} (history_back_min={HISTORY_BACK_MIN})")

    # Fenêtre du jour ancre
    day_start = pd.to_datetime(day).tz_localize("UTC").tz_convert(None)
    day_end   = day_start + timedelta(days=1)

    # Lecture jour + éventuelle "queue" du jour-1 (ex: 4h)
    df_day = _read_gcs_parquet(daily_uri)
    prev_tail = _maybe_read_prev_tail(DAILY_PREFIX, day, HISTORY_BACK_MIN)
    if len(prev_tail):
        print(f"[build_datasets] prev tail rows: {len(prev_tail):,} (from {HISTORY_BACK_MIN} min before midnight)")
        df_hist = pd.concat([prev_tail, df_day], ignore_index=True)
    else:
        print("[build_datasets] no prev tail available")
        df_hist = df_day

    # EVENTS construits sur l'historique (évite la cassure à minuit)
    events_combined = _build_events(df_hist, penury=PENURY_T, saturation=SAT_T)

    # On NE PUBLIE que le jour ancre
    events = events_combined[(events_combined["tbin_utc"] >= day_start) & (events_combined["tbin_utc"] < day_end)].copy()
    if events.empty:
        print("[build_datasets] events empty after day filter — exit 0")
        return 0
    events_dated = f"{EXPORTS_PREFIX.rstrip('/')}/events_{day}.parquet"
    _write_gcs_parquet(events, events_dated)

    # PERF + y_pred_int
    perf_base = _build_perf_base(events, horizons_min=HORIZONS)  # base à partir des events DU JOUR
    # Features d'inférence construites sur l'HISTORIQUE (prev tail + jour)
    feats_all = _build_inference_features(events_combined, horizon_bins=3)
    perf = _infer_y_pred(perf_base=perf_base, feats_all=feats_all, horizons_min=HORIZONS)

    # Filtre final (sécurité) : ne garder que le jour ancre
    if not perf.empty:
        perf = perf[(perf["tbin_utc"] >= day_start) & (perf["tbin_utc"] < day_end)].copy()

    if perf.empty:
        print("[build_datasets] perf empty — nothing written")
        return 0
    perf_dated = f"{EXPORTS_PREFIX.rstrip('/')}/perf_{day}.parquet"
    _write_gcs_parquet(perf, perf_dated)

    print("[build_datasets] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
