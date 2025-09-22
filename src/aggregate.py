# src/aggregate.py
import os
import pandas as pd
from pathlib import Path
from src.weather import fetch_history, fetch_forecast
from src.utils_io import get_export_path

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

def occupancy_5min(snapshot_df: pd.DataFrame, with_weather: bool = True) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame()

    # Arrondis temps
    snapshot_df["tbin_utc"] = snapshot_df["ts_utc"].dt.floor("5min")
    snapshot_df["hour_utc"] = snapshot_df["ts_utc"].dt.floor("h")

    agg = (
        snapshot_df
        .groupby(["tbin_utc", "stationcode"])
        .agg(
            name=("name", "first"),
            nb_velos_bin=("numbikesavailable", "mean"),
            nb_bornes_bin=("numdocksavailable", "mean"),
            capacity_bin=("capacity", "max"),
            lat=("lat", "first"),
            lon=("lon", "first"),
        )
        .reset_index()
    )

    # hour_utc pour la jointure météo
    agg["hour_utc"] = agg["tbin_utc"].dt.floor("h")

    # Types propres
    agg["nb_velos_bin"] = agg["nb_velos_bin"].round().astype("Int64")
    agg["nb_bornes_bin"] = agg["nb_bornes_bin"].round().astype("Int64")
    agg["capacity_bin"] = agg["capacity_bin"].astype("Int64")

    # Ratio d'occupation (évite les truthiness sur pd.NA)
    def _occ(r):
        cap = r.capacity_bin
        bikes = r.nb_velos_bin
        docks = r.nb_bornes_bin
        if pd.notna(cap) and cap > 0:
            return float(bikes) / float(cap)
        total = (0 if pd.isna(bikes) else int(bikes)) + (0 if pd.isna(docks) else int(docks))
        if total > 0:
            return float(0 if pd.isna(bikes) else int(bikes)) / float(total)
        return None

    agg["occ_ratio_bin"] = pd.to_numeric(agg.apply(_occ, axis=1), errors="coerce").clip(0, 1)

    # --- Météo (historique + forecast) --------------------------------------
    if with_weather:
        # Colonnes météo de base
        for c in ["temp_C", "precip_mm", "wind_mps"]:
            if c not in agg.columns:
                agg[c] = pd.NA

        # 1) Historique
        try:
            start, end = agg["hour_utc"].min(), agg["hour_utc"].max()
            w = fetch_history(start, end)
        except Exception:
            w = None

        if w is not None and not w.empty:
            w["hour_utc"] = _to_utc_naive_floor_hour(w["hour_utc"])
            cols = [c for c in ["hour_utc", "temp_C", "precip_mm", "wind_mps"] if c in w.columns]
            agg = agg.merge(w[cols], on="hour_utc", how="left", suffixes=("", "_hist"))
            # coalescence base <- hist
            for c in ["temp_C", "precip_mm", "wind_mps"]:
                ch = f"{c}_hist"
                if ch in agg.columns:
                    agg[c] = pd.to_numeric(agg[c], errors="coerce")
                    agg[ch] = pd.to_numeric(agg[ch], errors="coerce")
                    agg[c] = agg[c].fillna(agg[ch])
            agg.drop(columns=[c for c in ["temp_C_hist", "precip_mm_hist", "wind_mps_hist"] if c in agg.columns],
                     inplace=True, errors="ignore")

        # 2) Forecast si trous
        try:
            need_fx = agg[["temp_C", "precip_mm", "wind_mps"]].isna().any(axis=1).any()
        except KeyError:
            need_fx = True

        if need_fx:
            try:
                wf = fetch_forecast(pd.to_datetime(agg["hour_utc"].max()), 24)
            except Exception:
                wf = None
            if wf is not None and not wf.empty:
                wf["hour_utc"] = _to_utc_naive_floor_hour(wf["hour_utc"])
                cols = [c for c in ["hour_utc", "temp_C", "precip_mm", "wind_mps"] if c in wf.columns]
                agg = agg.merge(wf[cols], on="hour_utc", how="left", suffixes=("", "_fx"))
                for c in ["temp_C", "precip_mm", "wind_mps"]:
                    cf = f"{c}_fx"
                    if cf in agg.columns:
                        agg[c] = pd.to_numeric(agg[c], errors="coerce")
                        agg[cf] = pd.to_numeric(agg[cf], errors="coerce")
                        agg[c] = agg[c].fillna(agg[cf])
                agg.drop(columns=[c for c in ["temp_C_fx", "precip_mm_fx", "wind_mps_fx"] if c in agg.columns],
                         inplace=True, errors="ignore")

        # 3) Sécurité: nettoie résiduels *_x/_y
        for base in ["temp_C", "precip_mm", "wind_mps"]:
            bx, by = f"{base}_x", f"{base}_y"
            if bx in agg.columns or by in agg.columns:
                for extra in [bx, by]:
                    if extra in agg.columns:
                        agg[extra] = pd.to_numeric(agg[extra], errors="coerce")
                if base not in agg.columns:
                    agg[base] = pd.NA
                agg[base] = pd.to_numeric(agg[base], errors="coerce")
                if bx in agg.columns:
                    agg[base] = agg[base].fillna(agg[bx])
                if by in agg.columns:
                    agg[base] = agg[base].fillna(agg[by])
                agg.drop(columns=[c for c in [bx, by] if c in agg.columns], inplace=True, errors="ignore")

    # Colonnes finales (ordre stable)
    cols_first = [
        "tbin_utc", "hour_utc", "stationcode", "name",
        "nb_velos_bin", "nb_bornes_bin", "capacity_bin", "occ_ratio_bin",
        "lat", "lon", "temp_C", "precip_mm", "wind_mps"
    ]
    rest = [c for c in agg.columns if c not in cols_first]
    return agg[cols_first + rest]


# aggregate.py


def main(input_path: Path, output_path: Path):
    # Charger le snapshot ingéré
    df = pd.read_parquet(input_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # Fenêtre temporelle min/max
    start_ts, end_ts = df["ts"].min(), df["ts"].max()
    print(f"[aggregate] snapshot window: {start_ts} → {end_ts}")

    # Récup historique météo
    try:
        w = fetch_history(start_ts, end_ts)
    except Exception as e:
        print(f"[aggregate] weather fetch_history failed: {e}")
        w = None

    # 👉 log taille history météo
    print(f"[aggregate] weather hist rows: {0 if (w is None or w.empty) else len(w)}")

    # Est-ce qu’il faut du forecast ?
    need_fx = end_ts > pd.Timestamp.utcnow()
    print(f"[aggregate] need forecast: {need_fx}")

    wf = None
    if need_fx:
        try:
            wf = fetch_forecast(end_ts, horizon_h=24)
        except Exception as e:
            print(f"[aggregate] weather fetch_forecast failed: {e}")
            wf = None

        # 👉 log taille forecast météo
        print(f"[aggregate] weather forecast rows: {0 if (wf is None or wf.empty) else len(wf)}")

    # Fusion history + forecast
    if w is not None and wf is not None and not wf.empty:
        w = pd.concat([w, wf], ignore_index=True).drop_duplicates("hour_utc")
    elif wf is not None and not wf.empty and (w is None or w.empty):
        w = wf

    if w is None or w.empty:
        print("[aggregate] WARNING: no weather data available")
    else:
        print(f"[aggregate] merged weather rows: {len(w)}")

    # Join météo → snapshots
    if w is not None and not w.empty:
        df["hour_utc"] = df["ts"].dt.floor("h").dt.tz_localize(None)
        df = df.merge(w, on="hour_utc", how="left")

    # Sauvegarde
    df.to_parquet(output_path, index=False)
    print(f"[aggregate] saved {len(df)} rows → {output_path}")

if __name__ == "__main__":
    main()
