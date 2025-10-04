# apps/api/routes/stations.py
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

from api.core.settings import settings
from api.core.features_live import _latest_parquet  # retourne un dict {'uri','etag','size'}
from api.core.snapshot_live import fetch_live_snapshot

router = APIRouter()

# ───────────────────────── helpers ─────────────────────────

def _to_num(s, default=0):
    """Convertit série/scalaire en numérique, remplace NaN par `default`."""
    try:
        v = pd.to_numeric(s, errors="coerce")
        if hasattr(v, "fillna"):
            v = v.fillna(default)
        return v
    except Exception:
        return default

def _read_station_ref(uri: str) -> pd.DataFrame:
    """Référentiel facultatif: stationcode,name,lat,lon,capacity (csv/json/parquet)."""
    if not uri:
        return pd.DataFrame()
    try:
        if uri.endswith(".csv"):
            df = pd.read_csv(uri)
        elif uri.endswith(".json"):
            df = pd.read_json(uri)
        else:
            df = pd.read_parquet(uri)
        if "stationcode" not in df.columns and "station_id" in df.columns:
            df["stationcode"] = df["station_id"].astype(str)
        if "stationcode" in df.columns:
            df["stationcode"] = df["stationcode"].astype(str)
        return df
    except Exception as e:
        print(f"[stations] WARN: unable to read station ref '{uri}': {e}")
        return pd.DataFrame()

# --- dans apps/api/routes/stations.py ---

def _json_safe_df(df: pd.DataFrame) -> list[dict]:
    """
    DataFrame -> liste de dicts JSON-compatibles :
    - datetimes -> ISO8601 (UTC, string)
    - ±Inf -> Nonereturn _json_safe_df(out_json).to_dict(orient="records")
    - NaN/NA/NaT -> None
    - numpy scalaires -> scalaires Python, puis re-nettoyage NaN/Inf
    """
    if df is None or df.empty:
        return []

    import math

    df = df.copy()

    # Datetimes -> ISO8601 (UTC)
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            s = pd.to_datetime(df[c], errors="coerce", utc=True)
            df[c] = s.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Remplacer ±Inf par NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # On convertit en records, puis on nettoie valeur par valeur
    recs = df.to_dict(orient="records")

    for r in recs:
        for k, v in list(r.items()):
            # pandas isna couvre NaN/NA/NaT
            if pd.isna(v):
                r[k] = None
                continue

            # numpy scalaires -> Python
            if hasattr(v, "item") and callable(getattr(v, "item", None)):
                try:
                    v = v.item()
                except Exception:
                    pass

            # Re-nettoyage après conversion : float nan/inf Python natifs
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                r[k] = None
            else:
                r[k] = v

    return recs

# ───────────────────────── /stations ─────────────────────────

@router.get("/stations")
def list_stations():
    """
    Liste minimale pour la carte :
      stationcode, name, lat/lon, capacity, num_bikes_available, num_docks_available.
    Source:
      - base "features" latest.parquet pour station_id/capacity/occ_ratio (estimation des vélos),
      - enrichissement name/lat/lon via snapshot live (GBFS),
      - puis référentiel statique (si STATIONS_REF_URI défini).
    """
    try:
        # Récupère l'URI exact du latest.parquet
        serving_prefix = getattr(settings, "GCS_SERVING_PREFIX", None) or getattr(settings, "gcs_serving_prefix", None)
        meta = _latest_parquet(serving_prefix)  # dict{'uri','etag','size'}
        uri = meta.get("uri") if isinstance(meta, dict) else None

        feats = pd.read_parquet(uri) if uri else pd.DataFrame()

        # stationcode
        if "station_id" in feats.columns:
            stationcode = feats["station_id"].astype(str)
        elif "stationcode" in feats.columns:
            stationcode = feats["stationcode"].astype(str)
        else:
            # Pas de features valides → snapshot live pour la carte
            live = fetch_live_snapshot()
            if live.empty:
                return []
            out_live = live.rename(columns={"station_id": "stationcode"}).copy()
            out_live["stationcode"] = out_live["stationcode"].astype(str)
            out_live["num_bikes_available"] = out_live["bikes"]
            out_live["num_docks_available"] = (out_live["capacity"] - out_live["num_bikes_available"]).clip(lower=0)
            out_live = out_live[[
                "stationcode","name","lat","lon","capacity","num_bikes_available","num_docks_available"
            ]]
            return _json_safe_df(out_json)

        # capacity
        if "capacity" in feats.columns:
            capacity = _to_num(feats["capacity"], 0).astype(int)
        elif "capacity_bin" in feats.columns:
            capacity = _to_num(feats["capacity_bin"], 0).astype(int)
        else:
            capacity = pd.Series(0, index=feats.index, dtype=int)

        # vélos (estimation à partir d'un ratio si besoin)
        if "num_bikes_available" in feats.columns:
            bikes = _to_num(feats["num_bikes_available"], 0).astype(int)
        elif "bikes_latest" in feats.columns:
            bikes = _to_num(feats["bikes_latest"], 0).astype(int)
        elif "occ_ratio_bin" in feats.columns:
            occ = _to_num(feats["occ_ratio_bin"], 0.0)
            bikes = (capacity * occ).round().clip(lower=0).astype(int)
        else:
            bikes = pd.Series(0, index=feats.index, dtype=int)

        docks = (capacity - bikes).clip(lower=0).astype(int)

        out = pd.DataFrame({
            "stationcode": stationcode,
            "capacity": capacity,
            "num_bikes_available": bikes,
            "num_docks_available": docks,
        }).drop_duplicates(subset=["stationcode"], keep="last")

        # Enrichissement name/lat/lon via snapshot live
        try:
            live = fetch_live_snapshot()
        except Exception as e:
            print(f"[/stations] live snapshot error: {e}")
            live = pd.DataFrame()

        merged = out
        if not live.empty:
            live2 = live.rename(columns={"station_id": "stationcode"})[
                ["stationcode","name","lat","lon"]
            ].copy()
            live2["stationcode"] = live2["stationcode"].astype(str)
            merged = merged.merge(live2.drop_duplicates("stationcode"), on="stationcode", how="left")

        # Référentiel statique (si fourni)
        ref_uri = os.getenv("STATIONS_REF_URI", "")
        if ref_uri:
            ref = _read_station_ref(ref_uri)
            if not ref.empty and "stationcode" in ref.columns:
                keep = ["stationcode"]
                for c in ["name","lat","lon","capacity"]:
                    if c in ref.columns:
                        keep.append(c)
                ref2 = ref[keep].drop_duplicates("stationcode", keep="last")
                merged = merged.merge(ref2, on="stationcode", how="left", suffixes=("", "_ref"))
                # capacity du référentiel prioritaire si présent
                if "capacity_ref" in merged.columns:
                    merged["capacity"] = merged["capacity_ref"].fillna(merged["capacity"]).astype(int)
                    merged.drop(columns=["capacity_ref"], inplace=True, errors="ignore")
                # compléter name/lat/lon
                for c in ["name","lat","lon"]:
                    rc = f"{c}_ref"
                    if rc in merged.columns:
                        merged[c] = merged[c].fillna(merged[rc])
                        merged.drop(columns=[rc], inplace=True, errors="ignore")

        merged["stationcode"] = merged["stationcode"].astype(str)
        out_json = merged[
            ["stationcode","name","lat","lon","capacity","num_bikes_available","num_docks_available"]
        ]

        return _json_safe_df(out_json)

    except Exception as e:
        print(f"❌ error in /stations: {e}")
        return []

# ───────────────────────── /stations/features ─────────────────────────

_EXPECTED_FEATURE_COLS = [
    "station_id","tbin_latest","capacity_bin","occ_ratio_bin",
    "lag_nb_1b","lag_nb_24b","lag_nb_48b",
    "lag_occ_1b","lag_occ_24b","lag_occ_48b",
    "roll_nb_12b","roll_occ_12b",
    "trend_nb_12b","trend_occ_12b",
    "hour","hour_sin","hour_cos",
    # extras si présents
    "temp_C","precip_mm","wind_mps",
]

@router.get("/stations/features")
def features_latest(
    stationcodes: Optional[List[str]] = Query(default=None, description="Liste de station_id/stationcode à filtrer"),
    columns: Optional[List[str]] = Query(default=None, description="Sous-ensemble de colonnes à retourner"),
    limit: int = Query(default=0, ge=0, le=20000, description="Limiter le nb de lignes (0 = pas de limite)"),
):
    """
    Renvoie les features du `latest.parquet`, proches du tableau d’exemple.
      - `stationcodes` : filtre optionnel (accepté sur station_id/stationcode).
      - `columns`      : sous-ensemble de colonnes.
      - `limit`        : limiter le nombre de lignes renvoyées.
    """
    try:
        serving_prefix = getattr(settings, "GCS_SERVING_PREFIX", None) or getattr(settings, "gcs_serving_prefix", None)
        meta = _latest_parquet(serving_prefix)  # {'uri','etag','size'}
        uri = meta.get("uri") if isinstance(meta, dict) else None
        if not uri:
            return {"error": "cannot resolve latest parquet URI", "meta": meta}

        df = pd.read_parquet(uri)
        if df is None or df.empty:
            return []

        # Normalise la clé
        if "station_id" not in df.columns and "stationcode" in df.columns:
            df["station_id"] = df["stationcode"]

        # Filtrage stationcodes
        if stationcodes:
            codes = set(map(str, stationcodes))
            df = df[df["station_id"].astype(str).isin(codes)]

        # Colonnes à retourner
        keep = [c for c in _EXPECTED_FEATURE_COLS if c in df.columns]
        if "station_id" not in keep:
            keep = ["station_id"] + keep
        if columns:
            req = ["station_id"] + [c for c in columns if c != "station_id"]
            keep = [c for c in req if c in df.columns]

        out_df = df[keep].copy()

        # station_id en string AVANT la conversion JSON-safe
        if "station_id" in out_df.columns:
            out_df["station_id"] = out_df["station_id"].astype(str)

        # Limite de lignes
        if limit and limit > 0:
            out_df = out_df.head(limit)

        # Conversion JSON-safe (dates -> ISO, NaN/Inf -> None, dtypes nullable cassés)
        return _json_safe_df(out_df)


    except Exception as e:
        import traceback
        return {"error": repr(e), "trace": traceback.format_exc()}

# ───────────────────────── DEBUG ─────────────────────────
@router.get("/stations/_debug_features")
def stations_debug_features():
    try:
        serving_prefix = getattr(settings, "GCS_SERVING_PREFIX", None) or getattr(settings, "gcs_serving_prefix", None)
        meta = _latest_parquet(serving_prefix)
        uri = meta.get("uri") if isinstance(meta, dict) else None
        if not uri:
            return {"ok": False, "error": "cannot resolve latest parquet URI", "meta": meta}

        df = pd.read_parquet(uri)
        return {
            "ok": True,
            "uri": uri,
            "shape": None if df is None else list(df.shape),
            "cols": [] if df is None else list(df.columns),
            "sample": [] if df is None or df.empty else _json_safe_df(df.head(3)),
        }
    except Exception as e:
        import traceback
        return {
            "ok": False,
            "error": repr(e),
            "trace": traceback.format_exc(),
            "serving_prefix": getattr(settings, "GCS_SERVING_PREFIX", None) or getattr(settings, "gcs_serving_prefix", None),
        }   


@router.get("/stations/_debug_guess")
def debug_guess():
    """Aperçu minimal “liste station” à partir des features (sans lat/lon)."""
    try:
        serving_prefix = getattr(settings, "GCS_SERVING_PREFIX", None) or getattr(settings, "gcs_serving_prefix", None)
        meta = _latest_parquet(serving_prefix)
        uri = meta.get("uri") if isinstance(meta, dict) else None
        if not uri:
            return {"n": 0, "sample": []}
        df = pd.read_parquet(uri)
        if df is None or df.empty:
            return {"n": 0, "sample": []}
        cap = _to_num(df.get("capacity", df.get("capacity_bin", 0)), 0).astype(int)
        occ = _to_num(df.get("occ_ratio_bin", 0.0), 0.0)
        bikes = (cap * occ).round().clip(lower=0).astype(int)
        out = pd.DataFrame({
            "stationcode": df.get("station_id", df.get("stationcode", "")).astype(str),
            "name": df.get("station_id", "").astype(str),
            "capacity": cap,
            "num_bikes_available": bikes,
            "num_docks_available": (cap - bikes).clip(lower=0).astype(int),
            "lat": df.get("lat"),
            "lon": df.get("lon"),
        })
        sample = _json_safe_df(out.head(3))
        return {"n": int(len(out)), "sample": sample}
    except Exception as e:
        return {"n": 0, "sample": [], "error": str(e)}
