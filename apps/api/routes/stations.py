# apps/api/routes/stations.py
from __future__ import annotations

import io
import os
import json
import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter

# settings + live snapshot helper
from core.settings import settings
from core.snapshot_live import fetch_live_snapshot

try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    storage = None  # type: ignore

router = APIRouter(tags=["stations"])


# ───────────────────────── Helpers JSON ─────────────────────────
def _json_safe_df(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    df = df.copy()

    # Datetimes → ISO 8601 Z
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            s = pd.to_datetime(df[c], errors="coerce", utc=True)
            df[c] = s.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Inf/NaN → None
    df = df.replace([np.inf, -np.inf], np.nan)
    recs = df.to_dict(orient="records")

    for r in recs:
        for k, v in list(r.items()):
            if pd.isna(v):
                r[k] = None
                continue
            if hasattr(v, "item") and callable(getattr(v, "item", None)):
                try:
                    v = v.item()
                except Exception:
                    pass
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                r[k] = None
            else:
                r[k] = v
    return recs


# ───────────────────────── GCS / Serving helpers ─────────────────────────
def _serving_forecast_prefix() -> str:
    """
    Source de vérité pour le bundle latest_forecast.json.
    1) ENV GCS_SERVING_PREFIX (si présent)
    2) settings.SERVING_FORECAST_PREFIX (legacy)
    """
    env_val = (os.environ.get("GCS_SERVING_PREFIX") or "").strip()
    if env_val:
        return env_val.rstrip("/")
    return (getattr(settings, "SERVING_FORECAST_PREFIX", "") or "").rstrip("/")


def _bundle_uri() -> str:
    base = _serving_forecast_prefix()
    # si base est vide → on tentera de lire un chemin local "latest_forecast.json"
    return f"{base}/latest_forecast.json" if base else "latest_forecast.json"


def _split_gs(uri: str) -> Tuple[str, str]:
    assert uri.startswith("gs://"), f"Invalid GCS URI: {uri}"
    b, k = uri[5:].split("/", 1)
    return b, k


def _read_forecast_bundle_latest() -> dict:
    """
    Lit le bundle latest_forecast.json et le retourne en dict.
    Forme attendue: {"generated_at": "...", "horizons":[...], "data":{"15":[...], ...}}
    """
    uri = _bundle_uri()

    # Lecture locale si pas de GCS disponible ou si l'URI n'est pas gs://
    if not uri.startswith("gs://") or storage is None:
        try:
            with open(uri, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    # Lecture GCS
    try:
        bkt, key = _split_gs(uri)
        cli = storage.Client()
        blob = cli.bucket(bkt).get_blob(key)
        if not blob:
            return {}
        data = blob.download_as_bytes()
        return json.loads(data.decode("utf-8"))
    except Exception:
        return {}


def _to_int_series(s: pd.Series, default: int = 0) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").fillna(default)
    return v.astype(int)


# ───────────────────────── Route: /stations ─────────────────────────
@router.get("/stations")
def list_stations():
    """
    Liste minimale pour la carte :
      station_id (string), name, lat, lon, capacity,
      num_bikes_available, num_docks_available.

    Priorité:
      1) snapshot live (GBFS) → plus frais
      2) fallback bundle forecast JSON → minimal (id + capacité) si live indisponible
    """
    # 1) Source principale : live
    try:
        live = fetch_live_snapshot()
    except Exception as e:
        print(f"[/stations] live snapshot error: {e}")
        live = pd.DataFrame()

    if live is not None and not live.empty:
        df = live.copy()

        # Clé canonique station_id (string)
        if "station_id" in df.columns:
            station_id = df["station_id"].astype("string")
        elif "stationcode" in df.columns:
            station_id = df["stationcode"].astype("string")
        elif "id" in df.columns:
            station_id = df["id"].astype("string")
        else:
            # pas de clé exploitable → liste vide
            return []

        capacity = _to_int_series(df.get("capacity", pd.Series([0] * len(df))))
        bikes = _to_int_series(df.get("bikes", df.get("num_bikes_available", pd.Series([0] * len(df)))))

        out = pd.DataFrame(
            {
                "station_id": station_id.astype(str),
                "name": df.get("name"),
                "lat": df.get("lat"),
                "lon": df.get("lon"),
                "capacity": capacity,
                "num_bikes_available": bikes,
                "num_docks_available": (capacity - bikes).clip(lower=0),
            }
        ).drop_duplicates(subset=["station_id"], keep="last")
        return _json_safe_df(out)

    # 2) Fallback : bundle forecast JSON
    try:
        bundle = _read_forecast_bundle_latest()
    except Exception as e:
        print(f"[/stations] forecast bundle fallback error: {e}")
        bundle = {}

    if not bundle or "data" not in bundle:
        return []

    # Utilise l’horizon 15 si présent, sinon n’importe lequel non vide
    data = bundle.get("data", {})
    rows = data.get("15") or next((v for k, v in data.items() if isinstance(v, list) and v), [])
    if not rows:
        return []

    fc = pd.DataFrame(rows)

    # station_id (string) depuis station_id OU stationcode
    if "station_id" in fc.columns:
        station_id = fc["station_id"].astype("string")
    elif "stationcode" in fc.columns:
        station_id = fc["stationcode"].astype("string")
    else:
        return []

    if "capacity" in fc.columns:
        capacity = _to_int_series(fc["capacity"])
    else:
        capacity = _to_int_series(fc.get("capacity_bin", pd.Series([0] * len(fc))))

    bikes = pd.Series([0] * len(fc), dtype=int)  # pas de current bikes dans le bundle forecast

    out = pd.DataFrame(
        {
            "station_id": station_id.astype(str),
            "name": None,
            "lat": None,
            "lon": None,
            "capacity": capacity,
            "num_bikes_available": bikes,
            "num_docks_available": (capacity - bikes).clip(lower=0),
        }
    ).drop_duplicates(subset=["station_id"], keep="last")

    return _json_safe_df(out)