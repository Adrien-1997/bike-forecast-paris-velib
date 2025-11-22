# apps/api/routes/stations.py

"""Stations endpoint for Vélib’ Forecast.

This router exposes a `/stations` endpoint that returns a **minimal list of
stations** for the UI (map, dropdowns, etc.) with the following fields:

- station_id (string)
- name
- lat, lon
- capacity
- num_bikes_available
- num_docks_available

Strategy (data sources, in order of priority)
---------------------------------------------
1. Live snapshot (GBFS) via `core.snapshot_live.fetch_live_snapshot()`:
   - freshest source,
   - returns current bikes, capacity, coordinates, etc.

2. Fallback: `latest_forecast.json` bundle:
   - used if the live snapshot is unavailable,
   - only exposes capacity (no live bikes),
   - still allows the UI to build a static map of stations.

The endpoint is deliberately tolerant:
- if keys are missing or the bundle is malformed, it returns `[]` instead of
  raising errors;
- all outputs are JSON-safe (NaN/inf/null handled by `_json_safe_df`).
"""

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
    """Convert a DataFrame to a list of JSON-safe records.

    Behaviour:
    ----------
    - If `df` is None or empty → return `[]`.
    - Datetime columns:
        * converted to UTC,
        * formatted as ISO-8601 with `Z` suffix (`YYYY-MM-DDTHH:MM:SSZ`).
    - Inf / -Inf → replaced with NaN.
    - NaN, null-like values → converted to `None`.
    - NumPy scalars (`np.int64`, `np.float32`, etc.) → converted to native
      Python scalars via `.item()` when possible.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    list[dict]
        List of JSON-serializable records.
    """
    if df is None or df.empty:
        return []
    df = df.copy()

    # Datetimes → ISO 8601 Z
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            s = pd.to_datetime(df[c], errors="coerce", utc=True)
            df[c] = s.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Inf/NaN → None (via NaN first)
    df = df.replace([np.inf, -np.inf], np.nan)
    recs = df.to_dict(orient="records")

    for r in recs:
        for k, v in list(r.items()):
            if pd.isna(v):
                r[k] = None
                continue
            # Convert numpy scalars to plain Python types when possible
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

    Priority:
    ---------
      1) ENV `SERVING_FORECAST_PREFIX`
      2) `settings.SERVING_FORECAST_PREFIX`

    Returns
    -------
    str
        GCS prefix (without trailing slash), or empty string if not configured.
    """
    env_val = (os.environ.get("SERVING_FORECAST_PREFIX") or "").strip()
    if env_val:
        return env_val.rstrip("/")
    return (getattr(settings, "SERVING_FORECAST_PREFIX", "") or "").rstrip("/")


def _bundle_uri() -> str:
    """Build the URI / path for `latest_forecast.json`.

    Returns
    -------
    str
        - `gs://.../latest_forecast.json` when a GCS prefix is configured;
        - `"latest_forecast.json"` (local file) when no prefix is available.
    """
    base = _serving_forecast_prefix()
    # si base est vide → on tentera de lire un chemin local "latest_forecast.json"
    return f"{base}/latest_forecast.json" if base else "latest_forecast.json"


def _split_gs(uri: str) -> Tuple[str, str]:
    """Split a `gs://bucket/key` URI into `(bucket, key)`.

    Parameters
    ----------
    uri : str
        GCS URI starting with `gs://`.

    Returns
    -------
    tuple[str, str]
        `(bucket, key)`.

    Raises
    ------
    AssertionError
        If the URI does not start with `gs://`.
    """
    assert uri.startswith("gs://"), f"Invalid GCS URI: {uri}"
    b, k = uri[5:].split("/", 1)
    return b, k


def _read_forecast_bundle_latest() -> dict:
    """
    Read `latest_forecast.json` bundle and return it as a dict.

    Expected shape (minimal example):
    ---------------------------------
    {
      "generated_at": "...",
      "horizons": [15, 60],
      "data": {
        "15": [...],
        "60": [...]
      }
    }

    Returns
    -------
    dict
        Parsed JSON bundle, or `{}` if:
        - file / blob is missing,
        - JSON is invalid,
        - GCS is unavailable.
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
    """Convert a Series to `int` dtype with safe coercion and default fill.

    Parameters
    ----------
    s : pd.Series
        Input series (any dtype).
    default : int, default 0
        Value used to fill NaNs after coercion.

    Returns
    -------
    pd.Series
        Integer series (`int`) with NaNs replaced by `default`.
    """
    v = pd.to_numeric(s, errors="coerce").fillna(default)
    return v.astype(int)


# ───────────────────────── Route: /stations ─────────────────────────
@router.get("/stations")
def list_stations():
    """
    Liste minimale pour la carte :
      station_id (string), name, lat, lon, capacity,
      num_bikes_available, num_docks_available.

    Data sources (priority):
    ------------------------
      1) **Live snapshot** (GBFS via `fetch_live_snapshot()`):
         - returns the freshest information,
         - includes current bikes and capacity per station.

      2) **Fallback**: `latest_forecast.json` bundle:
         - used when live snapshot is unavailable or fails,
         - no "live bikes" count (bikes = 0 by construction),
         - still exposes `station_id` + `capacity` to build the map.

    Returns
    -------
    list[dict]
        JSON-safe list of station records, as produced by `_json_safe_df`.
        Returns `[]` on any failure / missing data.
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
