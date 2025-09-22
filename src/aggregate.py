import os
import pandas as pd
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
    """
    Agrège les snapshots Vélib au pas 5 min (par station),
    calcule le ratio d'occupation, et joint la météo (historique + prévision).
    """
    if snapshot_df.empty:
        return pd.DataFrame()

    # Bins temporels
    snapshot_df["tbin_utc"] = snapshot_df["ts_utc"].dt.floor("5min")
    snapshot_df["hour_utc"] = snapshot_df["ts_utc"].dt.floor("h")

    # Agrégat 5 min
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

    # Reconstitue hour_utc (clé de jointure météo)
    agg["hour_utc"] = agg["tbin_utc"].dt.floor("h")

    # Types + ratio
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
        weather_cols = ["temp_C", "precip_mm", "wind_mps"]

        # Historique météo
        try:
            w = fetch_history(agg["hour_utc"].min(), agg["hour_utc"].max())
        except Exception:
            w = None
        if w is not None and not w.empty:
            w["hour_utc"] = _to_utc_naive_floor_hour(w["hour_utc"])
            cols = [c for c in ["hour_utc", *weather_cols] if c in w.columns]
            agg = agg.merge(w[cols], on="hour_utc", how="left")

        # ⚠️ Garantir la présence des colonnes météo avant de tester les NaN
        for c in weather_cols:
            if c not in agg.columns:
                agg[c] = pd.NA

        # Compléter via la prévision s'il reste des trous
        need_fx = agg[weather_cols].isna().any(axis=1).any()
        if need_fx:
            try:
                wf = fetch_forecast(pd.to_datetime(agg["hour_utc"].max()), 24)
            except Exception:
                wf = None
            if wf is not None and not wf.empty:
                wf["hour_utc"] = _to_utc_naive_floor_hour(wf["hour_utc"])
                cols = [c for c in ["hour_utc", *weather_cols] if c in wf.columns]
                agg = agg.merge(wf[cols], on="hour_utc", how="left", suffixes=("", "_fx"))
                for c in weather_cols:
                    fx = f"{c}_fx"
                    if fx in agg.columns:
                        agg[c] = agg[c].fillna(agg[fx])
                drop_cols = [f"{c}_fx" for c in weather_cols if f"{c}_fx" in agg.columns]
                if drop_cols:
                    agg.drop(columns=drop_cols, inplace=True, errors="ignore")

    return agg


if __name__ == "__main__":
    # Exécution "standalone" : ingest → aggregate → écrit docs/exports/velib.parquet
    from src.ingest import ingest_once

    DOCS_EXPORTS = os.path.join("docs", "exports")
    os.makedirs(DOCS_EXPORTS, exist_ok=True)
    parquet_path = os.path.join(DOCS_EXPORTS, "velib.parquet")

    snaps = ingest_once()
    new = occupancy_5min(snaps, with_weather=True)
    if new.empty:
        print("[aggregate] Aucun nouveau point.")
        raise SystemExit(0)

    # Concat avec historique (local ou HF via utils_io)
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

    # Dédup + purge 90 jours
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
