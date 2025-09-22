# src/aggregate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
import numpy as np
import pandas as pd

# =============================================================================
# Utils dates
# =============================================================================

def _ensure_datetime_utc_naive(series: pd.Series) -> pd.Series:
    # Naive UTC in ns
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt.dt.tz_localize(None).astype("datetime64[ns]")

def _make_bins(dt_series: pd.Series, minutes: int, strategy: str = "left") -> pd.Series:
    dt = pd.to_datetime(dt_series, errors="coerce", utc=True)
    if strategy == "right":
        b = (dt.dt.ceil(f"{minutes}min") - pd.to_timedelta(minutes, unit="min")).dt.tz_localize(None)
    else:
        b = dt.dt.floor(f"{minutes}min").dt.tz_localize(None)
    return b.astype("datetime64[ns]")

# =============================================================================
# Weather (optionnel)
# =============================================================================

def _normalize_weather(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return pd.DataFrame(columns=["hour_utc", "temp_C", "precip_mm", "wind_mps"])
    dd = df.copy()
    # heure
    dt_col = None
    for c in ["hour_utc", "time_utc", "time", "datetime", "timestamp"]:
        if c in dd.columns:
            dt_col = c
            break
    if dt_col is None:
        if isinstance(dd.index, pd.DatetimeIndex):
            dd = dd.reset_index().rename(columns={"index": "hour_utc"})
            dt_col = "hour_utc"
        else:
            return pd.DataFrame(columns=["hour_utc", "temp_C", "precip_mm", "wind_mps"])
    dd["hour_utc"] = pd.to_datetime(dd[dt_col], errors="coerce", utc=True).dt.floor("h").dt.tz_localize(None).astype("datetime64[ns]")
    # valeurs
    dd["temp_C"]   = pd.to_numeric(dd.get("temp_C"), errors="coerce")
    dd["precip_mm"]= pd.to_numeric(dd.get("precip_mm"), errors="coerce")
    w = dd.get("wind_mps")
    if w is None:
        w = dd.get("wind_speed_10m")
    dd["wind_mps"] = pd.to_numeric(w, errors="coerce")
    out = dd[["hour_utc", "temp_C", "precip_mm", "wind_mps"]].drop_duplicates("hour_utc").sort_values("hour_utc")
    return out

def _merge_weather_tolerant(
    agg: pd.DataFrame,
    weather_history_df: Optional[pd.DataFrame] = None,
    weather_forecast_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    # import paresseux
    if weather_history_df is None or weather_forecast_df is None:
        try:
            from src.weather import fetch_history, fetch_forecast  # type: ignore
        except Exception:
            fetch_history = fetch_forecast = None  # type: ignore

    out = agg.copy()
    out["hour_utc"] = pd.to_datetime(out["hour_utc"]).astype("datetime64[ns]")
    for c in ["temp_C", "precip_mm", "wind_mps"]:
        if c not in out.columns:
            out[c] = pd.NA

    if (weather_history_df is None and 'fetch_history' in locals() and fetch_history) or \
       (weather_forecast_df is None and 'fetch_forecast' in locals() and fetch_forecast):
        # Fenêtre unique
        start = out["hour_utc"].min()
        end   = out["hour_utc"].max()
        try:
            h = _normalize_weather(weather_history_df if weather_history_df is not None else fetch_history(start, end))  # type: ignore
        except Exception:
            h = pd.DataFrame(columns=["hour_utc", "temp_C", "precip_mm", "wind_mps"])
        try:
            f = _normalize_weather(weather_forecast_df if weather_forecast_df is not None else fetch_forecast(end, 36))  # type: ignore
        except Exception:
            f = pd.DataFrame(columns=["hour_utc", "temp_C", "precip_mm", "wind_mps"])
    else:
        h = _normalize_weather(weather_history_df)
        f = _normalize_weather(weather_forecast_df)

    # On NE PERD JAMAIS de lignes : on merge seulement les hour_utc valides
    mask_valid = out["hour_utc"].notna()
    left_na = out.loc[~mask_valid].copy()
    left_ok = out.loc[ mask_valid].copy()

    if h.empty and f.empty:
        # Pas de météo → garder NaN et recomposer
        return pd.concat([left_ok, left_na], axis=0, ignore_index=True).sort_values(
            ["tbin_utc","stationcode"]
        ).reset_index(drop=True)

    # merge history (nearest ±30min)
    if not h.empty and not left_ok.empty:
        left_ok = pd.merge_asof(
            left_ok.sort_values("hour_utc").reset_index(),
            h.sort_values("hour_utc"),
            on="hour_utc",
            direction="nearest",
            tolerance=pd.Timedelta("30min"),
            suffixes=("", "_h"),
        ).set_index("index").sort_index()
        for c in ["temp_C", "precip_mm", "wind_mps"]:
            left_ok[c] = left_ok[c].fillna(left_ok.get(f"{c}_h"))
            if f"{c}_h" in left_ok.columns:
                left_ok.drop(columns=[f"{c}_h"], inplace=True, errors="ignore")

    # merge forecast (nearest ±30min)
    if not f.empty and not left_ok.empty:
        left_ok = pd.merge_asof(
            left_ok.sort_values("hour_utc").reset_index(),
            f.sort_values("hour_utc"),
            on="hour_utc",
            direction="nearest",
            tolerance=pd.Timedelta("30min"),
            suffixes=("", "_f"),
        ).set_index("index").sort_index()
        for c in ["temp_C", "precip_mm", "wind_mps"]:
            left_ok[c] = left_ok[c].fillna(left_ok.get(f"{c}_f"))
            if f"{c}_f" in left_ok.columns:
                left_ok.drop(columns=[f"{c}_f"], inplace=True, errors="ignore")

    out2 = pd.concat([left_ok, left_na], axis=0, ignore_index=True)
    out2["temp_C"]    = pd.to_numeric(out2["temp_C"], errors="coerce")
    out2["precip_mm"] = pd.to_numeric(out2["precip_mm"], errors="coerce")
    out2["wind_mps"]  = pd.to_numeric(out2["wind_mps"], errors="coerce")
    return out2.sort_values(["tbin_utc", "stationcode"]).reset_index(drop=True)

# =============================================================================
# Core aggregation
# =============================================================================

@dataclass
class OccupancyConfig:
    res_minutes: int = 5
    round_strategy: str = "left"   # 'left' (floor) or 'right' (ceil)
    with_weather: bool = False     # CLOUD: par défaut OFF (perf). Mets WITH_WEATHER=1 pour activer.

def occupancy_5min(
    df_snapshots: pd.DataFrame,
    config: OccupancyConfig = OccupancyConfig(),
    weather_history_df: Optional[pd.DataFrame] = None,
    weather_forecast_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Sortie colonnes garanties :
    tbin_utc, stationcode, name, nb_velos_bin, nb_bornes_bin, capacity_bin,
    lat, lon, hour_utc, occ_ratio_bin, temp_C, precip_mm, wind_mps
    """
    if df_snapshots is None or df_snapshots.empty:
        return pd.DataFrame(columns=[
            "tbin_utc","stationcode","name",
            "nb_velos_bin","nb_bornes_bin","capacity_bin",
            "lat","lon","hour_utc","occ_ratio_bin",
            "temp_C","precip_mm","wind_mps",
        ])

    df = df_snapshots.copy()

    # timestamp
    ts_col = next((c for c in ["t","ts","timestamp","date","datetime","recorded_at","t_utc","time_utc","ts_utc"]
                   if c in df.columns), None)
    if ts_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "timestamp"})
            ts_col = "timestamp"
        else:
            raise ValueError("No timestamp column found in snapshots.")
    df["t_utc"] = _ensure_datetime_utc_naive(df[ts_col])

    # station id / name
    if "stationcode" not in df.columns:
        alt = next((c for c in ["station_code","code","id","station_id"] if c in df.columns), None)
        if alt: df = df.rename(columns={alt:"stationcode"})
        else:   raise ValueError("Missing 'stationcode' column.")
    if "name" not in df.columns: df["name"] = pd.NA

    # counts
    vel_col = next((c for c in ["nb_velos_bin","nb_velos","bikes","numbikesavailable","num_bikes_available"] if c in df.columns), None)
    bor_col = next((c for c in ["nb_bornes_bin","nb_bornes","docks","numdocksavailable","num_docks_available"] if c in df.columns), None)
    cap_col = next((c for c in ["capacity_bin","capacity","cap"] if c in df.columns), None)
    if vel_col is None or bor_col is None:
        raise ValueError("Missing bike/dock count columns.")

    df["nb_velos_bin"]  = pd.to_numeric(df[vel_col], errors="coerce")
    df["nb_bornes_bin"] = pd.to_numeric(df[bor_col], errors="coerce")
    df["capacity_bin"]  = pd.to_numeric(df[cap_col], errors="coerce") if cap_col is not None else (df["nb_velos_bin"] + df["nb_bornes_bin"])

    # geo
    if "lat" not in df.columns or "lon" not in df.columns:
        if "latitude" in df.columns and "longitude" in df.columns:
            df = df.rename(columns={"latitude":"lat","longitude":"lon"})
        else:
            df["lat"] = pd.NA; df["lon"] = pd.NA

    # time bins
    df["tbin_utc"] = _make_bins(df["t_utc"], minutes=config.res_minutes, strategy=config.round_strategy)
    df["hour_utc"] = df["t_utc"].dt.floor("h").astype("datetime64[ns]")

    # agrégation (dernier état dans le bin)
    # astuce perf: on se contente de trier par t_utc une seule fois
    df = df.sort_values("t_utc")
    agg = df.groupby(["tbin_utc","stationcode"], as_index=False).agg({
        "name":"last",
        "nb_velos_bin":"last",
        "nb_bornes_bin":"last",
        "capacity_bin":"last",
        "lat":"last",
        "lon":"last",
        "hour_utc":"last",
    })

    # metrique
    with np.errstate(divide="ignore", invalid="ignore"):
        occ = np.where(agg["capacity_bin"] > 0, agg["nb_velos_bin"]/agg["capacity_bin"], np.nan)
    agg["occ_ratio_bin"] = pd.Series(occ).clip(0,1)

    # météo (optionnelle)
    if config.with_weather:
        agg = _merge_weather_tolerant(agg, weather_history_df, weather_forecast_df)
    else:
        for c in ["temp_C","precip_mm","wind_mps"]:
            if c not in agg.columns: agg[c] = pd.NA

    # ordre des colonnes
    cols = ["tbin_utc","stationcode","name",
            "nb_velos_bin","nb_bornes_bin","capacity_bin",
            "lat","lon","hour_utc","occ_ratio_bin",
            "temp_C","precip_mm","wind_mps"]
    agg = agg[cols].sort_values(["tbin_utc","stationcode"]).reset_index(drop=True)
    return agg

# =============================================================================
# Orchestrateur: ingest -> aggregate -> export (SHARD) -> push
# =============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    try:
        from src.ingest import ingest_once  # type: ignore
    except Exception:
        from ingest import ingest_once  # type: ignore

    # ingest
    df_raw = ingest_once()
    if df_raw is None or df_raw.empty:
        print("[aggregate] No raw data — exiting 0.")
        sys.exit(0)

    # config météo via env
    with_weather = os.environ.get("WITH_WEATHER","0") in ("1","true","True","yes","YES")
    cfg = OccupancyConfig(res_minutes=5, with_weather=with_weather)

    df_agg = occupancy_5min(df_raw, cfg)

    # SHARD horodaté (append, pas replace)
    # tbin le plus récent pour nommer le shard
    latest = pd.to_datetime(df_agg["tbin_utc"].max())
    ts_str = latest.strftime("%Y%m%d_%H%M") if pd.notna(latest) else pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")
    day_str = latest.strftime("%Y-%m-%d") if pd.notna(latest) else pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    outdir = Path(f"exports/shards/dt={day_str}")
    outdir.mkdir(parents=True, exist_ok=True)
    shard_path = outdir / f"velib_{ts_str}.parquet"

    df_agg.to_parquet(shard_path, index=False)
    print(f"[aggregate] wrote shard {shard_path} ({len(df_agg)} rows)")

    # push du shard uniquement (append)
    os.environ["PUSH_SRC"] = str(shard_path)
    # destination côté Hub: même arborescence
    os.environ["PUSH_DEST"] = str(shard_path).replace("\\", "/")

    # lancer le push
    try:
        from tools.push_hf import main as push_main  # type: ignore
    except Exception:
        from push_hf import main as push_main  # type: ignore
    rc = push_main()
    sys.exit(0 if (rc is None or rc == 0) else 1)
