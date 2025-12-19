# api/routes/stations.py

"""Stations endpoint for Vélib’ Forecast.

This router exposes a lightweight `/serving/stations` endpoint that returns a
**minimal list of stations** for the UI (map, dropdowns, etc.) with:

- `station_id` (string)
- `name`
- `lat`, `lon`
- `capacity`
- `num_bikes_available`
- `num_docks_available`

Strategy (data sources, in order of priority)
---------------------------------------------
1) Live snapshot (GBFS) via `core.snapshot_live.fetch_live_snapshot()`:
   - freshest source,
   - provides current bikes + capacity + coordinates.

2) Fallback: forecast bundle JSON (`.../serving/forecast/latest_forecast.json`):
   - used if the live snapshot is unavailable,
   - bikes/docks default to 0 when missing,
   - still enables the UI to render a (degraded) station list.

Failure model
-------------
- On any failure (network / parsing / missing blob / unexpected type),
  the endpoint returns an empty list `[]` (never raises).

Caching
-------
- The endpoint sets `Cache-Control: no-store` to avoid client-side caching.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Response

from core.snapshot_live import fetch_live_snapshot

# google-cloud-storage is optional: only used when STORAGE_BACKEND != "local"
try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    storage = None  # type: ignore

router = APIRouter(prefix="/serving/stations", tags=["serving-stations"])

# ───────────────────────────── Backend selection (local vs GCS) ─────────────────────────────

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "gcs").lower()
USE_LOCAL = STORAGE_BACKEND == "local"

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("DATA_ROOT", REPO_ROOT / "data"))

GCS_LOCAL_ROOT_PREFIX = (
    os.getenv(
        "GCS_LOCAL_ROOT_PREFIX",
        "gs://velib-forecast-472820_cloudbuild/velib/",
    ).rstrip("/")
    + "/"
)


def _split_gs(uri: str) -> Tuple[str, str]:
    """Split a `gs://bucket/key` URI into `(bucket, key)`."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Bad GCS URI: {uri}")
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def _local_path_from_gs(uri: str) -> Path:
    """Map a velib gs:// URI to local DATA_ROOT path (STORAGE_BACKEND=local)."""
    if not uri.startswith(GCS_LOCAL_ROOT_PREFIX):
        raise FileNotFoundError(f"Local mapping failed for URI: {uri}")
    rel = uri[len(GCS_LOCAL_ROOT_PREFIX) :]
    return DATA_ROOT / rel.lstrip("/")


def _read_json_any(uri: str) -> Optional[dict]:
    """Read a JSON document from either local disk or GCS, depending on STORAGE_BACKEND.

    Returns None on any error.
    """
    if not uri:
        return None

    if USE_LOCAL:
        try:
            path = _local_path_from_gs(uri)
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    if storage is None or not uri.startswith("gs://"):
        return None
    try:
        bkt, key = _split_gs(uri)
        cli = storage.Client()
        blob = cli.bucket(bkt).blob(key)
        if not blob.exists():
            return None
        data = blob.download_as_bytes()
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def _json_sanitize(obj: Any) -> Any:
    """Replace NaN/±Inf with None recursively (JSON-safe)."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


# ───────────────────────────── Helpers: fallback bundle ─────────────────────────────

def _forecast_bundle_uri_or_empty() -> str:
    """Return the gs:// URI for the latest forecast bundle (or empty string).

    Priority:
    - ENV `GCS_FORECAST_PREFIX` or `SERVING_FORECAST_PREFIX`
    Output:
    - `<prefix>/latest_forecast.json`
    """
    raw = (
        os.environ.get("GCS_FORECAST_PREFIX", "")
        or os.environ.get("SERVING_FORECAST_PREFIX", "")
    )
    base = raw.strip().strip("'\"").rstrip("/")
    if not base.startswith("gs://"):
        return ""
    return f"{base}/latest_forecast.json"


def _stations_from_bundle(doc: dict) -> List[dict]:
    """Extract a degraded stations list from a `latest_forecast.json`-like bundle."""
    data = doc.get("data")
    if not isinstance(data, dict):
        return []

    # prefer horizon "15" then any non-empty list
    rows = data.get("15")
    if not isinstance(rows, list) or not rows:
        rows = next((v for v in data.values() if isinstance(v, list) and v), [])
    if not rows:
        return []

    out: Dict[str, dict] = {}

    for r in rows:
        if not isinstance(r, dict):
            continue

        sid = r.get("station_id") or r.get("stationcode") or r.get("id")
        if sid is None:
            continue
        sid = str(sid)

        cap = r.get("capacity")
        try:
            cap_i = int(cap) if cap is not None else 0
        except Exception:
            cap_i = 0

        # If the bundle contains coordinates/names, keep them; otherwise None
        rec = out.get(sid, {})
        rec.update(
            {
                "station_id": sid,
                "name": r.get("name", rec.get("name")),
                "lat": r.get("lat", rec.get("lat")),
                "lon": r.get("lon", rec.get("lon")),
                "capacity": cap_i,
                "num_bikes_available": 0,
                "num_docks_available": max(cap_i, 0),
            }
        )
        out[sid] = rec

    return list(out.values())


# ───────────────────────────── Route ─────────────────────────────

@router.get("")
def list_stations(response: Response) -> List[dict]:
    """Return a minimal list of stations for the UI.

    Notes
    -----
    - Primary source: live GBFS snapshot (freshest).
    - Fallback: latest forecast bundle (degraded).
    - On any failure: returns [].
    """
    # Always reflect latest state → no caching.
    response.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"

    # 1) Primary source: live snapshot
    try:
        df = fetch_live_snapshot()
    except Exception as e:  # pragma: no cover
        print(f"[stations] fetch_live_snapshot() failed: {e}")
        df = None

    if df is not None and hasattr(df, "to_dict") and getattr(df, "empty", True) is False:
        try:
            recs = df.to_dict(orient="records")
        except Exception as e:  # pragma: no cover
            print(f"[stations] to_dict(orient='records') failed: {e}")
            recs = []

        out: Dict[str, dict] = {}
        for r in recs:
            if not isinstance(r, dict):
                continue

            sid = r.get("station_id") or r.get("stationcode") or r.get("id")
            if sid is None:
                continue
            sid = str(sid)

            # canonical fields in your snapshot schema are: bikes, capacity, lat, lon, name
            cap = r.get("capacity")
            bikes = r.get("bikes", r.get("num_bikes_available"))

            try:
                cap_i = int(cap) if cap is not None else 0
            except Exception:
                cap_i = 0
            try:
                bikes_i = int(bikes) if bikes is not None else 0
            except Exception:
                bikes_i = 0

            out[sid] = {
                "station_id": sid,
                "name": r.get("name"),
                "lat": r.get("lat"),
                "lon": r.get("lon"),
                "capacity": cap_i,
                "num_bikes_available": max(bikes_i, 0),
                "num_docks_available": max(cap_i - bikes_i, 0),
            }

        if out:
            return _json_sanitize(list(out.values()))

    # 2) Fallback: latest_forecast.json bundle
    uri = _forecast_bundle_uri_or_empty()
    if not uri:
        return []

    doc = _read_json_any(uri)
    if not isinstance(doc, dict):
        return []

    stations = _stations_from_bundle(doc)
    return _json_sanitize(stations) if stations else []