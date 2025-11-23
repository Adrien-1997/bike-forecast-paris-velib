# api/routes/snapshot.py

"""Live snapshot endpoint for Vélib’ Forecast.

This router exposes a lightweight `/snapshot` endpoint that returns the
**current state of the Vélib' network** (sans météo) en s'appuyant sur :

- `core.snapshot_live.fetch_live_snapshot()` :
  - interroge les endpoints GBFS Smovengo,
  - fusionne `station_status` + `station_information`,
  - produit un DataFrame au format interne standard.

L’API convertit simplement ce DataFrame en une liste de dicts JSON
(`orient="records"`) consommable par le frontend ou d'autres services.
"""

from fastapi import APIRouter
from core.snapshot_live import fetch_live_snapshot

router = APIRouter(prefix="/snapshot", tags=["snapshot"])


@router.get("")
def get_snapshot():
    """Return the current live Vélib' snapshot (without weather).

    Response
    --------
    A JSON array of records, each record containing (schéma minimal attendu) :

    - `ts_utc`    : timestamp de mesure (UTC naïf),
    - `tbin_utc`  : timestamp arrondi à 5 minutes (UTC naïf),
    - `station_id`: identifiant de station (string),
    - `bikes`     : nombre total de vélos disponibles,
    - `capacity`  : capacité totale de la station,
    - `mechanical`: nombre de vélos mécaniques,
    - `ebike`     : nombre de vélos électriques,
    - `status`    : statut de la station (ex: "OK", "CLOSED"),
    - `lat`, `lon`: coordonnées géographiques,
    - `name`      : nom de la station.

    Notes
    -----
    - En cas d'échec ou si `fetch_live_snapshot()` ne renvoie pas un DataFrame,
      l’endpoint retourne simplement `[]`.
    """
    df = fetch_live_snapshot()
    return df.to_dict(orient="records") if hasattr(df, "to_dict") else []
