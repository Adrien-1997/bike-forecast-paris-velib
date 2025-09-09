# src/aggregate.py
import os
import time
import duckdb
import pandas as pd
from src.weather import fetch_history, fetch_forecast

CON = duckdb.connect("warehouse.duckdb")

def _to_utc_naive_hour(x):
    dt = pd.to_datetime(x, errors="coerce", utc=True)
    # Series vs DatetimeIndex
    try:
        return dt.dt.floor("h").dt.tz_localize(None)
    except AttributeError:
        return dt.floor("h").tz_localize(None)

def hourly_occupancy(with_weather: bool = True) -> pd.DataFrame:
    """
    Agrégat horaire par station avec:
      - nb_velos_hour (moyenne arrondie)
      - nb_bornes_hour (moyenne arrondie)
      - capacity_hour (max/heure)
      - occ_ratio_hour = nb_velos_hour / capacity_hour (fallback si capacité manquante)
      - (compat) bikes_avg, docks_avg conservés
    + Jointure météo (historique + backfill prévision si trous récents).
    """
    q = """
    WITH base AS (
      SELECT
        ts_utc::TIMESTAMP                         AS ts_utc,
        stationcode,
        name,
        COALESCE(numbikesavailable,0)::INTEGER    AS bikes,
        COALESCE(numdocksavailable,0)::INTEGER    AS docks,
        NULLIF(capacity,0)::INTEGER               AS capacity_raw,
        try_cast(lat AS DOUBLE)                   AS lat,
        try_cast(lon AS DOUBLE)                   AS lon
      FROM velib_snapshots
    ),
    enriched AS (
      SELECT
        *,
        /* Capacité estimée si non fournie : bikes + docks quand > 0 */
        CASE
          WHEN capacity_raw IS NOT NULL THEN capacity_raw
          WHEN (bikes + docks) > 0 THEN (bikes + docks)
          ELSE NULL
        END AS capacity_est,
        /* Ratio d'occupation observé au snapshot */
        CASE
          WHEN capacity_raw IS NOT NULL AND capacity_raw > 0 THEN bikes::DOUBLE / capacity_raw
          WHEN (bikes + docks) > 0 THEN bikes::DOUBLE / (bikes + docks)
          ELSE NULL
        END AS occ_ratio_snap
      FROM base
    ),
    hourly AS (
      SELECT
        date_trunc('hour', ts_utc)                AS hour_utc,
        stationcode,
        any_value(name)                           AS name,
        CAST(avg(bikes) AS INTEGER)               AS nb_velos_hour,
        CAST(avg(docks) AS INTEGER)               AS nb_bornes_hour,
        /* on prend la capacité la plus haute observée sur l'heure */
        max(capacity_est)                         AS capacity_hour,
        any_value(lat)                            AS lat,
        any_value(lon)                            AS lon,
        /* pour compat et diagnostic */
        avg(occ_ratio_snap)                       AS occ_ratio_hour_snap_avg
      FROM enriched
      GROUP BY 1,2
    )
    SELECT
      hour_utc,
      stationcode,
      name,
      nb_velos_hour,
      nb_bornes_hour,
      capacity_hour,
      /* Ratio recalculé avec la capacité d'heure (plus stable) */
      CASE
        WHEN capacity_hour IS NOT NULL AND capacity_hour > 0
          THEN nb_velos_hour::DOUBLE / capacity_hour
        WHEN (nb_velos_hour + nb_bornes_hour) > 0
          THEN nb_velos_hour::DOUBLE / (nb_velos_hour + nb_bornes_hour)
        ELSE NULL
      END AS occ_ratio_hour,
      /* alias pour compatibilité aval */
      nb_velos_hour::DOUBLE                      AS bikes_avg,
      nb_bornes_hour::DOUBLE                     AS docks_avg,
      lat, lon
    FROM hourly
    ORDER BY hour_utc, stationcode;
    """
    df = CON.execute(q).fetchdf()
    if df.empty:
        return df

    # Normalise clé de jointure : UTC naïf, arrondi à l’heure
    df["hour_utc"] = _to_utc_naive_hour(df["hour_utc"])

    if with_weather:
        # 1) Historique météo sur la fenêtre utile
        w = pd.DataFrame(columns=["hour_utc","temp_C","precip_mm","wind_mps"])
        try:
            wf = fetch_history(df["hour_utc"].min(), df["hour_utc"].max())
            if wf is not None and not wf.empty:
                w = wf.copy()
        except Exception as e:
            print(f"[weather] fetch_history failed: {e}")

        if not w.empty:
            w["hour_utc"] = _to_utc_naive_hour(w["hour_utc"])
            df = df.merge(w, on="hour_utc", how="left")
        else:
            for c in ["temp_C","precip_mm","wind_mps"]:
                if c not in df.columns:
                    df[c] = pd.NA

        # 2) Backfill via prévision pour combler des trous récents
        try:
            if df[["temp_C","precip_mm","wind_mps"]].isna().any(axis=1).any():
                fx_start = pd.to_datetime(df["hour_utc"].max())  # déjà naïf
                wf = fetch_forecast(fx_start, 24)
                if wf is not None and not wf.empty:
                    wf["hour_utc"] = _to_utc_naive_hour(wf["hour_utc"])
                    df = df.merge(wf, on="hour_utc", how="left", suffixes=("", "_fx"))
                    for c in ["temp_C","precip_mm","wind_mps"]:
                        if c in df.columns and f"{c}_fx" in df.columns:
                            df[c] = df[c].fillna(df[f"{c}_fx"])
                    drop = [c for c in ["temp_C_fx","precip_mm_fx","wind_mps_fx"] if c in df.columns]
                    if drop:
                        df.drop(columns=drop, inplace=True)
        except Exception as e:
            print(f"[weather] forecast backfill skipped: {e}")

    # Cohérence finale : clamp ratio dans [0,1]
    if "occ_ratio_hour" in df.columns:
        df["occ_ratio_hour"] = pd.to_numeric(df["occ_ratio_hour"], errors="coerce").clip(lower=0, upper=1)

    return df

def _safe_write_csv(df: pd.DataFrame, path: str, attempts: int = 6, delay: float = 1.0):
    """
    Écriture CSV Windows-friendly : fichier temporaire + os.replace + retries.
    """
    tmp = f"{path}.tmp"
    for i in range(attempts):
        try:
            df.to_csv(tmp, index=False)
            os.replace(tmp, path)  # remplace de manière atomique quand possible
            print(f"[aggregate] CSV écrit → {path}")
            return
        except PermissionError:
            # Nettoyage du tmp si besoin puis retry
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            print(f"[aggregate] CSV verrouillé (tentative {i+1}/{attempts}). Retry dans {delay}s…")
            time.sleep(delay)
        except Exception as e:
            print(f"[aggregate] CSV erreur inattendue: {e}")
            break
    print("[aggregate] CSV encore verrouillé → saut de l’écriture pour cette fois.")

if __name__ == "__main__":
    os.makedirs("exports", exist_ok=True)
    out = hourly_occupancy(with_weather=True)
    if not out.empty:
        # Parquet (pyarrow/fastparquet sinon fallback DuckDB)
        try:
            out.to_parquet("exports/velib_hourly.parquet", index=False)
        except Exception:
            duckdb.register("out_tbl", out)
            duckdb.sql("COPY out_tbl TO 'exports/velib_hourly.parquet' (FORMAT PARQUET);")
        # CSV robuste (si verrouillé, on skip sans planter)
        _safe_write_csv(out, "exports/velib_hourly.csv")
    print("OK hourly -> exports/velib_hourly.parquet (et .csv si dispo)")
