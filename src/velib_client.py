# src/velib_client.py  — snapshot complet avec pagination (limit=100)
import requests
import pandas as pd
import datetime as dt

BASE = (
    "https://parisdata.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "velib-disponibilite-en-temps-reel/records"
)

def _boolish(x):
    if isinstance(x, bool): return x
    if x is None: return None
    s = str(x).strip().lower()
    if s in {"oui","yes","true","1"}: return True
    if s in {"non","no","false","0"}: return False
    return None

def _fetch_page(offset=0, limit=100, active_only=True):
    params = {
        "limit": limit,
        "offset": offset,
        "select": (
            "stationcode,name,numbikesavailable,numdocksavailable,"
            "mechanical,ebike,capacity,coordonnees_geo,is_installed,is_renting,is_returning"
        ),
    }
    if active_only:
        params["where"] = "is_installed='OUI' AND is_renting='OUI'"
    r = requests.get(BASE, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("results", [])

def fetch_snapshot(active_only=True) -> pd.DataFrame:
    """Récupère tout le snapshot (pagination 100 par 100)."""
    frames, offset, limit = [], 0, 100
    while True:
        rows = _fetch_page(offset=offset, limit=limit, active_only=active_only)
        if not rows: break
        frames.append(pd.DataFrame(rows))
        offset += limit
        # garde-fou Explore v2.1 (offset+limit < 10000 sans group_by)
        if offset + limit >= 10000:
            break

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    ts = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    df["ts_utc"] = ts

    # Champs geo
    geo = df.get("coordonnees_geo")
    if geo is not None:
        df["lat"] = geo.apply(lambda g: (g or {}).get("lat"))
        df["lon"] = geo.apply(lambda g: (g or {}).get("lon"))
    else:
        df["lat"] = None
        df["lon"] = None

    # Typage / normalisation
    for c in ["numbikesavailable","numdocksavailable","mechanical","ebike","capacity"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["is_installed","is_renting","is_returning"]:
        if c in df: df[c] = df[c].map(_boolish)

    # Colonnes finales
    keep = [
        "ts_utc","stationcode","name","numbikesavailable","numdocksavailable",
        "mechanical","ebike","capacity","is_installed","is_renting","is_returning","lat","lon"
    ]
    for k in keep:
        if k not in df: df[k] = None
    return df[keep]
