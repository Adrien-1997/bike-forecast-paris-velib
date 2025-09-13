# src/velib_client.py
"""
Client pour récupérer les snapshots Vélib' via API Opendata v2.1
et retourner un DataFrame normalisé prêt à insérer dans DuckDB.
"""

import requests
import pandas as pd
import datetime as dt

BASE = (
    "https://parisdata.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "velib-disponibilite-en-temps-reel/records"
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _boolish(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"oui", "yes", "true", "1"}:
        return True
    if s in {"non", "no", "false", "0"}:
        return False
    return None


def _fetch_page(offset=0, limit=100, active_only=True):
    params = {
        "limit": limit,
        "offset": offset,
        "select": (
            "stationcode,name,numbikesavailable,numdocksavailable,"
            "mechanical,ebike,capacity,coordonnees_geo,"
            "is_installed,is_renting,is_returning"
        ),
    }
    if active_only:
        params["where"] = "is_installed='OUI' AND is_renting='OUI'"

    r = requests.get(BASE, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("results", [])


# ----------------------------------------------------------------------
# API publique
# ----------------------------------------------------------------------
def fetch_snapshot(active_only=True) -> pd.DataFrame:
    """
    Récupère un snapshot complet (pagination 100 par 100).
    Retourne un DataFrame avec colonnes uniformisées.
    """
    frames, offset, limit = [], 0, 100
    while True:
        rows = _fetch_page(offset=offset, limit=limit, active_only=active_only)
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        offset += limit
        # garde-fou Explore v2.1 (offset+limit < 10000 sans group_by)
        if offset + limit >= 10000:
            break

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Timestamp snapshot (UTC naïf)
    ts = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).astimezone(dt.timezone.utc).replace(tzinfo=None)
    df["ts_utc"] = ts

    # Champs géographiques
    if "coordonnees_geo" in df.columns:
        geo = df["coordonnees_geo"]
        df["lat"] = geo.apply(lambda g: (g or {}).get("lat") if isinstance(g, dict) else None)
        df["lon"] = geo.apply(lambda g: (g or {}).get("lon") if isinstance(g, dict) else None)
    else:
        df["lat"], df["lon"] = None, None

    # Typage / normalisation
    for c in ["numbikesavailable", "numdocksavailable", "mechanical", "ebike", "capacity"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["is_installed", "is_renting", "is_returning"]:
        if c in df:
            df[c] = df[c].map(_boolish)

    # Colonnes finales
    keep = [
        "ts_utc", "stationcode", "name",
        "numbikesavailable", "numdocksavailable",
        "mechanical", "ebike", "capacity",
        "is_installed", "is_renting", "is_returning",
        "lat", "lon"
    ]
    for k in keep:
        if k not in df:
            df[k] = None

    return df[keep].reset_index(drop=True)


# Alias explicite pour ingestion.py
def fetch_snapshot_df() -> pd.DataFrame:
    return fetch_snapshot(active_only=True)


if __name__ == "__main__":
    df = fetch_snapshot()
    print(df.head())
    print(f"[velib_client] snapshot complet: {len(df)} stations")
