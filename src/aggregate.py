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

    # bins temps
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

    # ✅ nécessaire pour la jointure météo
    agg["hour_utc"] = agg["tbin_utc"].dt.floor("h")

    # types + ratio
    agg["nb_velos_bin"] = agg["nb_velos_bin"].astype("Int64")
    agg["nb_bornes_bin"] = agg["nb_bornes_bin"].astype("Int64")
    agg["occ_ratio_bin"] = agg.apply(
        lambda r: r.nb_velos_bin / r.capacity_bin
        if r.capacity_bin and r.capacity_bin > 0 else (
            r.nb_velos_bin / (r.nb_velos_bin + r.nb_bornes_bin)
            if (r.nb_velos_bin + r.nb_bornes_bin) > 0 else None
        ),
        axis=1,
    )
    agg["occ_ratio_bin"] = pd.to_numeric(agg["occ_ratio_bin"], errors="coerce").clip(0, 1)

    if with_weather:
        # ✅ évite KeyError si pas encore mergé
        for c in ["temp_C", "precip_mm", "wind_mps"]:
            if c not in agg.columns:
                agg[c] = pd.NA

        # Historique
        try:
            w = fetch_history(agg["hour_utc"].min(), agg["hour_utc"].max())
        except Exception:
            w = None
        if w is not None and not w.empty:
            w["hour_utc"] = _to_utc_naive_floor_hour(w["hour_utc"])
            cols = [c for c in ["hour_utc","temp_C","precip_mm","wind_mps"] if c in w.columns]
            agg = agg.merge(w[cols], on="hour_utc", how="left")

        # Faut-il compléter par la prévision ?
        need_fx = agg[["temp_C","precip_mm","wind_mps"]].isna().any(axis=1).any()
        if need_fx:
            try:
                wf = fetch_forecast(pd.to_datetime(agg["hour_utc"].max()), 24)
            except Exception:
                wf = None
            if wf is not None and not wf.empty:
                wf["hour_utc"] = _to_utc_naive_floor_hour(wf["hour_utc"])
                cols = [c for c in ["hour_utc","temp_C","precip_mm","wind_mps"] if c in wf.columns]
                agg = agg.merge(wf[cols], on="hour_utc", how="left", suffixes=("", "_fx"))
                for c in ["temp_C","precip_mm","wind_mps"]:
                    if f"{c}_fx" in agg.columns:
                        agg[c] = agg[c].fillna(agg[f"{c}_fx"])
                agg.drop(columns=[c for c in ["temp_C_fx","precip_mm_fx","wind_mps_fx"] if c in agg.columns],
                         inplace=True, errors="ignore")

    return agg


if __name__ == "__main__":
    from src.ingest import ingest_once

    DOCS_EXPORTS = os.path.join("docs", "exports")
    os.makedirs(DOCS_EXPORTS, exist_ok=True)
    parquet_path = os.path.join(DOCS_EXPORTS, "velib.parquet")

    snaps = ingest_once()
    new = occupancy_5min(snaps, with_weather=True)
    if new.empty:
        print("[aggregate] Aucun nouveau point.")
        raise SystemExit(0)

    try:
        old_path = get_export_path("velib.parquet")
        old = pd.read_parquet(old_path)
        for c in ("tbin_utc", "hour_utc"):
            if c in old.columns:
                old[c] = pd.to_datetime(old[c], utc=True).dt.tz_localize(None)
        df = pd.concat([old, new], ignore_index=True)
    except Exception as e:
        print(f"[aggregate] Pas d'existant ({e}) → repartir du nouveau")
        df = new.copy()

    df = (
        df.sort_values(["tbin_utc", "stationcode"])
          .drop_duplicates(subset=["tbin_utc", "stationcode"], keep="last")
    )

    try:
        tmax = df["tbin_utc"].max()
        cutoff = (tmax - pd.Timedelta(days=90)).floor("5min")
        df = df[df["tbin_utc"] >= cutoff].copy()
    except Exception:
        pass

    df.to_parquet(parquet_path, index=False)
    print(f"[aggregate] OK → {parquet_path} (rows={len(df)})")
