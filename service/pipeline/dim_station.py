# pipeline/dim_station.py
import os
import duckdb
import requests
import pandas as pd
from datetime import datetime, timezone

URL_INFO = "https://velib-metropole-opendata.smovengo.cloud/opendata/Velib_Metropole/station_information.json"

# Env:
# - DB_LOCAL : /tmp/velib_reporting.duckdb  (déjà téléchargée par gcs_job.py)

def fetch_station_info():
    r = requests.get(URL_INFO, timeout=30)
    r.raise_for_status()
    js = r.json()
    rows = []
    for s in (js.get("data", {}).get("stations") or []):
        try:
            rows.append({
                "station_id": int(s["station_id"]),
                "name": s.get("name"),
                "lat": float(s["lat"]) if s.get("lat") is not None else None,
                "lon": float(s["lon"]) if s.get("lon") is not None else None,
                "capacity": int(s["capacity"]) if s.get("capacity") is not None else None,
            })
        except Exception:
            # Skip lignes malformées
            continue
    return rows

def ensure_table(con: duckdb.DuckDBPyConnection):
    con.execute("""
    CREATE SCHEMA IF NOT EXISTS dim;
    CREATE TABLE IF NOT EXISTS dim.dim_station (
      station_id   BIGINT,
      name         VARCHAR,
      lat          DOUBLE,
      lon          DOUBLE,
      capacity     INTEGER,
      status_init  VARCHAR,
      valid_from   TIMESTAMP,
      valid_to     TIMESTAMP,     -- NULL = actif
      is_current   BOOLEAN
    );
    """)
    # Index "complets" (DuckDB ne supporte pas les indexes partiels)
    con.execute("CREATE INDEX IF NOT EXISTS idx_dim_station_sid ON dim.dim_station(station_id);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_dim_station_current ON dim.dim_station(is_current);")

def scd2_upsert(con: duckdb.DuckDBPyConnection, now_utc, rows):
    # Convertir en DataFrame (DuckDB n'accepte pas une simple liste ici)
    df_in = pd.DataFrame(rows, columns=["station_id","name","lat","lon","capacity"])
    if df_in.empty:
        print("[dim_station] no rows from API, nothing to upsert.")
        return

    # Typage propre (éviter les 'object')
    df_in["station_id"] = pd.to_numeric(df_in["station_id"], errors="coerce").astype("Int64")
    for c in ["lat","lon"]:
        df_in[c] = pd.to_numeric(df_in[c], errors="coerce")
    df_in["capacity"] = pd.to_numeric(df_in["capacity"], errors="coerce").astype("Int64")

    con.register("incoming", df_in)

    con.execute("""
    CREATE OR REPLACE TEMP VIEW tmp_incoming AS
    SELECT
      CAST(station_id AS BIGINT) AS station_id,
      name,
      CAST(lat AS DOUBLE)       AS lat,
      CAST(lon AS DOUBLE)       AS lon,
      CAST(capacity AS INTEGER) AS capacity
    FROM incoming;
    """)

    # Expire les lignes actuelles si changement détecté
    con.execute("""
    UPDATE dim.dim_station d
    SET valid_to = ?, is_current = FALSE
    FROM tmp_incoming i
    WHERE d.is_current = TRUE
      AND d.station_id = i.station_id
      AND (
           COALESCE(d.name,'')        <> COALESCE(i.name,'')
        OR COALESCE(d.lat,  -1e9)     <> COALESCE(i.lat,  -1e9)
        OR COALESCE(d.lon,  -1e9)     <> COALESCE(i.lon,  -1e9)
        OR COALESCE(d.capacity, -1)   <> COALESCE(i.capacity, -1)
      );
    """, [now_utc])

    # Insère nouvelles versions (nouvelles stations ou celles expirées)
    con.execute("""
    INSERT INTO dim.dim_station (
      station_id, name, lat, lon, capacity, status_init,
      valid_from, valid_to, is_current
    )
    SELECT
      i.station_id, i.name, i.lat, i.lon, i.capacity, NULL,
      ?, NULL, TRUE
    FROM tmp_incoming i
    LEFT JOIN dim.dim_station d
      ON d.station_id = i.station_id AND d.is_current = TRUE
    WHERE d.station_id IS NULL
       OR EXISTS (
           SELECT 1 FROM dim.dim_station dx
           WHERE dx.station_id = i.station_id AND dx.valid_to = ?
       );
    """, [now_utc, now_utc])

def main():
    db_local = os.environ.get("DB_LOCAL", "/tmp/velib_reporting.duckdb")
    con = duckdb.connect(db_local)
    con.execute("PRAGMA threads=4;")
    ensure_table(con)

    rows = fetch_station_info()
    now_utc = datetime.now(timezone.utc).replace(microsecond=0).replace(tzinfo=None)  # naïf UTC

    scd2_upsert(con, now_utc, rows)

    con.close()
    print(f"[dim_station] OK — upsert {len(rows)} lignes (SCD-2)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
