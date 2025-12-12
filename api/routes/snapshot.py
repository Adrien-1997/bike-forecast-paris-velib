# api/routes/snapshot.py

"""Live snapshot endpoint for Vélib’ Forecast.

This router exposes a lightweight `/snapshot` endpoint that returns the
**current state of the Vélib' network** (sans météo) using:

- `core.snapshot_live.fetch_live_snapshot()` :
  - queries Smovengo GBFS endpoints,
  - merges `station_status` + `station_information`,
  - produces a DataFrame in the internal standard schema.

The API simply converts this DataFrame to a list of JSON records
(`orient="records"`) consumable by the frontend or other services.

Failure model
-------------
- If `fetch_live_snapshot()` fails (network error, parsing error, etc.),
  the endpoint returns an empty list `[]`.
- If the return value is not a pandas DataFrame (no `.to_dict`), it also
  returns `[]`.

This keeps `/snapshot` **safe and predictable**, even when Smovengo is down
or the machine is offline.
"""

from __future__ import annotations

from typing import Any, List

from fastapi import APIRouter
from core.snapshot_live import fetch_live_snapshot

router = APIRouter(prefix="/snapshot", tags=["snapshot"])


@router.get("")
def get_snapshot() -> List[dict]:
    """Return the current live Vélib' snapshot (without weather).

    Response
    --------
    A JSON array of records, each record containing (minimal expected schema):

    - `ts_utc`    : measurement timestamp (UTC naive),
    - `tbin_utc`  : timestamp rounded to 5 minutes (UTC naive),
    - `station_id`: station identifier (string),
    - `bikes`     : total number of available bikes,
    - `capacity`  : total station capacity,
    - `mechanical`: number of mechanical bikes,
    - `ebike`     : number of electric bikes,
    - `status`    : station status (e.g. "OK", "CLOSED"),
    - `lat`, `lon`: geographic coordinates,
    - `name`      : station name.

    Notes
    -----
    - On any failure (network / parsing / unexpected return type), the endpoint
      returns an empty list `[]` instead of raising an error, so that callers
      can degrade gracefully.
    """
    try:
        df = fetch_live_snapshot()
    except Exception as e:  # pragma: no cover
        # We log to stdout but keep the API behaviour stable.
        print(f"[snapshot] fetch_live_snapshot() failed: {e}")
        return []

    if not hasattr(df, "to_dict"):
        # Defensive: unexpected type
        print("[snapshot] fetch_live_snapshot() did not return a DataFrame-like object")
        return []

    try:
        return df.to_dict(orient="records")
    except Exception as e:  # pragma: no cover
        print(f"[snapshot] to_dict(orient='records') failed: {e}")
        return []