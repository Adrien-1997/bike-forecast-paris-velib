# apps/api/routes/history.py
from fastapi import APIRouter, Query
import pandas as pd
from google.cloud import storage
from api.core.features_live import _latest_parquet

router = APIRouter(prefix="/history", tags=["history"])

@router.get("")
def history(
    station_id: str = Query(..., description="Identifiant station"),
    from_: str = Query(..., description="UTC start, ex: 2025-09-29T00:00:00Z"),
    to: str = Query(..., description="UTC end, ex: 2025-09-30T00:00:00Z"),
):
    try:
        client = storage.Client()
        parquet_path = _latest_parquet(client)
        df = pd.read_parquet(parquet_path, engine="pyarrow")

        if df.empty:
            return []

        # Parsing bornes temporelles
        from_ts = pd.to_datetime(from_, utc=True, errors="coerce")
        to_ts = pd.to_datetime(to, utc=True, errors="coerce")
        if pd.isna(from_ts) or pd.isna(to_ts):
            return {"error": "Invalid date format. Use ISO 8601 (YYYY-MM-DDTHH:MM:SSZ)"}

        # Filtrage station
        if "station_id" in df.columns:
            mask = df["station_id"].astype(str) == str(station_id)
        elif "stationcode" in df.columns:
            mask = df["stationcode"].astype(str) == str(station_id)
        else:
            return {"error": "Parquet missing station_id/stationcode column"}

        # Filtrage temporel (selon colonne dispo)
        ts_col = None
        if "tbin_latest" in df.columns:
            ts_col = "tbin_latest"
        elif "ts_utc_latest" in df.columns:
            ts_col = "ts_utc_latest"
        elif "ts_utc" in df.columns:
            ts_col = "ts_utc"

        if ts_col:
            ts_vals = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            mask &= (ts_vals >= from_ts) & (ts_vals <= to_ts)

        sub = df.loc[mask]
        if sub.empty:
            return []

        # Construire historique
        records = []
        for _, row in sub.iterrows():
            cap = int(row.get("capacity", row.get("capacity_bin", 0)) or 0)
            bikes = int(
                row.get("num_bikes_available", row.get("bikes_latest", 0)) or 0
            )
            ratio = float(bikes / cap) if cap > 0 else None
            records.append(
                {
                    "tbin_utc": str(row.get(ts_col)) if ts_col else None,
                    "nb_velos_bin": bikes,
                    "capacity_bin": cap,
                    "occ_ratio_bin": ratio,
                }
            )

        return records

    except Exception as e:
        return {"error": str(e)}
