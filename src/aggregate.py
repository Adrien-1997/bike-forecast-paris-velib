# src/aggregate.py
import os
import time
import duckdb
import pandas as pd
from pathlib import Path
from src.weather import fetch_history, fetch_forecast
from src.utils_io import resolve_path, get_export_path

CON = duckdb.connect("warehouse.duckdb")

def _to_utc_naive(x):
    dt = pd.to_datetime(x, errors="coerce", utc=True)
    if isinstance(dt, pd.Series):
        return dt.dt.tz_localize(None)
    return dt.tz_localize(None)

def _to_utc_naive_floor_hour(x):
    dt = pd.to_datetime(x, errors="coerce", utc=True)
    if isinstance(dt, pd.Series):
        return dt.dt.floor("h").dt.tz_localize(None)
    return dt.floor("h").tz_localize(None)

def occupancy_5min(with_weather: bool = True) -> pd.DataFrame:
    """
    Agrégat unique au pas 5 min (par station), avec jointure météo horaire.
    Colonnes clés: tbin_utc, hour_utc, stationcode, nb_velos_bin, nb_bornes_bin,
                   capacity_bin, occ_ratio_bin, temp_C, precip_mm, wind_mps, lat, lon.
    """
    q = """
    WITH base AS (
      SELECT
        ts_utc::TIMESTAMP                       AS ts_utc,
        stationcode,
        name,
        COALESCE(numbikesavailable,0)::INTEGER  AS bikes,
        COALESCE(numdocksavailable,0)::INTEGER  AS docks,
        NULLIF(capacity,0)::INTEGER             AS capacity_raw,
        try_cast(lat AS DOUBLE)                 AS lat,
        try_cast(lon AS DOUBLE)                 AS lon
      FROM velib_snapshots
    ),
    enriched AS (
      SELECT
        *,
        CASE
          WHEN capacity_raw IS NOT NULL THEN capacity_raw
          WHEN (bikes + docks) > 0 THEN (bikes + docks)
          ELSE NULL
        END AS capacity_est,
        CASE
          WHEN capacity_raw IS NOT NULL AND capacity_raw > 0 THEN bikes::DOUBLE / capacity_raw
          WHEN (bikes + docks) > 0 THEN bikes::DOUBLE / (bikes + docks)
          ELSE NULL
        END AS occ_ratio_snap
      FROM base
    ),
    binned AS (
      SELECT
        make_timestamp(
          year(ts_utc), month(ts_utc), day(ts_utc),
          hour(ts_utc),
          CAST(5 * floor(minute(ts_utc) / 5.0) AS INTEGER),
          0
        )                                        AS tbin_utc,
        stationcode,
        any_value(name)                          AS name,
        CAST(avg(bikes) AS INTEGER)              AS nb_velos_bin,
        CAST(avg(docks) AS INTEGER)              AS nb_bornes_bin,
        max(capacity_est)                        AS capacity_bin,
        any_value(lat)                           AS lat,
        any_value(lon)                           AS lon,
        avg(occ_ratio_snap)                      AS occ_ratio_bin_snap_avg
      FROM enriched
      GROUP BY 1,2
    )
    SELECT
      tbin_utc,
      stationcode,
      name,
      nb_velos_bin,
      nb_bornes_bin,
      capacity_bin,
      CASE
        WHEN capacity_bin IS NOT NULL AND capacity_bin > 0
          THEN nb_velos_bin::DOUBLE / capacity_bin
        WHEN (nb_velos_bin + nb_bornes_bin) > 0
          THEN nb_velos_bin::DOUBLE / (nb_velos_bin + nb_bornes_bin)
        ELSE NULL
      END AS occ_ratio_bin,
      lat, lon
    FROM binned
    ORDER BY tbin_utc, stationcode;
    """
    df = CON.execute(q).fetchdf()
    if df.empty:
        return df

    df["tbin_utc"] = _to_utc_naive(df["tbin_utc"])
    df["hour_utc"] = _to_utc_naive_floor_hour(df["tbin_utc"])

    if with_weather:
        # Historique météo sur la plage couverte
        try:
            w = fetch_history(df["hour_utc"].min(), df["hour_utc"].max())
        except Exception:
            w = None
        if w is not None and not w.empty:
            w["hour_utc"] = _to_utc_naive_floor_hour(w["hour_utc"])
            df = df.merge(w[["hour_utc","temp_C","precip_mm","wind_mps"]],
                          on="hour_utc", how="left")
        # Backfill forecast si trous récents
        if df[["temp_C","precip_mm","wind_mps"]].isna().any(axis=1).any():
            try:
                wf = fetch_forecast(pd.to_datetime(df["hour_utc"].max()), 24)
                if wf is not None and not wf.empty:
                    wf["hour_utc"] = _to_utc_naive_floor_hour(wf["hour_utc"])
                    df = df.merge(wf[["hour_utc","temp_C","precip_mm","wind_mps"]],
                                  on="hour_utc", how="left", suffixes=("", "_fx"))
                    for c in ["temp_C","precip_mm","wind_mps"]:
                        if f"{c}_fx" in df:
                            df[c] = df[c].fillna(df[f"{c}_fx"])
                    df.drop(columns=[c for c in ["temp_C_fx","precip_mm_fx","wind_mps_fx"] if c in df],
                            inplace=True, errors="ignore")
            except Exception:
                pass

    # Bornage ratio
    df["occ_ratio_bin"] = pd.to_numeric(df["occ_ratio_bin"], errors="coerce").clip(0, 1)
    return df

if __name__ == "__main__":
    # --- dossier des exports (on écrit localement ; la CI poussera vers HF) ---
    DOCS_EXPORTS = os.path.join("docs", "exports")
    os.makedirs(DOCS_EXPORTS, exist_ok=True)
    parquet_path = os.path.join(DOCS_EXPORTS, "velib.parquet")

    # --- agrégat 5 min + météo ---
    new = occupancy_5min(with_weather=True)
    if new.empty:
        print("[aggregate] Aucun nouveau point à agréger.")
        raise SystemExit(0)

    # --- charger l'existant (local OU Hugging Face) puis concat / dédupli ---
    try:
        old_path = get_export_path("velib.parquet")  # local si présent, sinon HF
        old = pd.read_parquet(old_path)
        # normaliser les timestamps (naïfs UTC)
        for c in ("tbin_utc", "hour_utc"):
            if c in old.columns:
                old[c] = pd.to_datetime(old[c], utc=True).dt.tz_localize(None)
        df = pd.concat([old, new], ignore_index=True)
        print(f"[aggregate] ancien parquet chargé depuis: {old_path}")
    except Exception as e:
        print(f"[aggregate] Pas d'existant ou lecture échouée ({e}) → on repart du nouveau")
        df = new.copy()

    # --- déduplication clé (tbin_utc, stationcode) ---
    df["tbin_utc"] = pd.to_datetime(df["tbin_utc"], utc=True).dt.tz_localize(None)
    df = (
        df.sort_values(["tbin_utc", "stationcode"])
          .drop_duplicates(subset=["tbin_utc", "stationcode"], keep="last")
    )

    # --- fenêtre glissante (90 jours) pour contenir la taille ---
    try:
        tmax = df["tbin_utc"].max()
        cutoff = (tmax - pd.Timedelta(days=90)).floor("5min")
        df = df[df["tbin_utc"] >= cutoff].copy()
    except Exception:
        pass

    # --- écrire parquet uniquement ---
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        # fallback robuste via DuckDB si pyarrow/fastparquet indisponible
        duckdb.register("out_tbl", df)
        duckdb.sql(f"COPY out_tbl TO '{parquet_path}' (FORMAT PARQUET);")

    print(f"[aggregate] OK → {parquet_path} (rows={len(df)})")