# apps/api/routes/forecast.py
from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd
from fastapi import APIRouter, Body, HTTPException, Query, Response
from google.api_core.exceptions import NotFound  # type: ignore

from api.core.forecast_reader import load_latest_forecast
from api.core.settings import settings

router = APIRouter(prefix="/forecast", tags=["forecast"])


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _parse_station_ids_csv(raw: Optional[str]) -> Optional[List[str]]:
    """
    Parse une liste CSV '6294, 6352' -> ['6294','6352'].
    Conserve des strings (pas de cast int) pour éviter toute perte (zéros, formats).
    """
    if not raw:
        return None
    vals: List[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            vals.append(tok)
    return vals or None


def _parse_station_ids_payload(payload: Any) -> Optional[List[str]]:
    """
    Extrait une liste d'IDs de stations depuis un body JSON.
    Accepté:
      - liste directe: ["6294","6352"]
      - dict avec une des clés: station_ids, ids, stations, stationcode(s), codes
    -> retourne une liste de strings (sans cast en int)
    """
    if payload is None:
        return None

    # liste brute
    if isinstance(payload, list):
        out = [str(x).strip() for x in payload if str(x).strip()]
        return out or None

    # dict
    if isinstance(payload, dict):
        candidate_keys = (
            "station_ids", "ids", "stations",
            "stationcode", "stationcodes", "codes"
        )
        for key in candidate_keys:
            v = payload.get(key)
            if isinstance(v, list):
                out = [str(x).strip() for x in v if str(x).strip()]
                if out:
                    return out
            elif isinstance(v, (str, int)):  # autoriser un seul id
                s = str(v).strip()
                if s:
                    return [s]
    return None


def _validate_h(h: int) -> None:
    supported = {int(x.strip()) for x in (settings.FORECAST_SUPPORTED or "15").split(",") if x.strip()}
    if h not in supported:
        raise HTTPException(status_code=400, detail=f"h must be in {sorted(supported)}")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@router.get("/latest")
def get_latest_forecast(
    h: int = Query(15, description="Horizon en minutes (ex: 15, 60)"),
    station_id: Optional[str] = Query(
        None,
        description="Liste CSV d'identifiants station ex: '6294,6352' (string, pas besoin d'être numérique)"
    ),
):
    """
    Renvoie les dernières prévisions pour un horizon donné.
    Filtrage optionnel sur une liste CSV de station_id (en string).
    """
    _validate_h(h)
    station_ids = _parse_station_ids_csv(station_id)

    try:
        df = load_latest_forecast(h, station_ids=station_ids)
    except NotFound:
        return Response(status_code=204)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to load forecast: {e}")

    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")


@router.get("")
def forecast_alias(
    h: int = Query(15, description="Horizon en minutes (ex: 15, 60)"),
    station_id: Optional[str] = Query(None, description="Liste CSV ex: '6294,6352'"),
):
    return get_latest_forecast(h=h, station_id=station_id)


@router.post("/batch")
def forecast_batch(
    payload: Any = Body(default=None),
    h: int = Query(15, description="Horizon en minutes (ex: 15, 60)"),
):
    """
    Renvoie les prévisions pour une liste d'IDs postés dans le body.
    Body accepté:
      - ["6294","6352"]
      - {"station_ids": ["6294","6352"]}
      - {"ids": [...]}, {"stations": [...]}
      - compat: {"stationcode": [...]}, {"stationcodes": [...]}, {"codes": [...]}
    Tous les IDs sont traités comme des strings.
    """
    _validate_h(h)
    station_ids = _parse_station_ids_payload(payload)

    try:
        df = load_latest_forecast(h, station_ids=station_ids)
    except NotFound:
        return Response(status_code=204)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to load forecast: {e}")

    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")
