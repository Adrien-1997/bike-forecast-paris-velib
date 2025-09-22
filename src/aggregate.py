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

def _to_utc_naive_floor_hour(x) -> pd.Series:
    dt = pd.to_datetime(x, errors="coerce", utc=True)
    return dt.dt.floor("h").dt.tz_localize(None).astype("datetime64[ns]")

def _ensure_datetime_utc_naive(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt.dt.tz_localize(None).astype("datetime64[ns]")

def _ensure_ns(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").astype("datetime64[ns]")

# =============================================================================
# Weather helpers (robust normalization + tolerant merge)
# =============================================================================

def _normalize_weather(df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Normalize any weather dataframe to:
    [hour_utc (naive, floored hour, datetime64[ns]), temp_C, precip_mm, wind_mps].
    """
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return pd.DataFrame(columns=["hour_utc", "temp_C", "precip_mm", "wind_mps"])

    dd = df.copy()

    # Datetime detection
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

    dd["hour_utc"] = _to_utc_naive_floor_hour(dd[dt_col])

    # Temperature (°C)
    temp = next((c for c in ["temp_C", "temperature", "temp", "air_temperature"] if c in dd.columns), None)
    dd["temp_C"] = pd.to_numeric(dd[temp], errors="coerce") if temp else pd.NA

    # Precipitation (mm)
    pr = next((c for c in ["precip_mm", "precipitation_mm", "rain_mm", "precip"] if c in dd.columns), None)
    dd["precip_mm"] = pd.to_numeric(dd[pr], errors="coerce") if pr else pd.NA

    # Wind (→ m/s)
    wd = next((c for c in ["wind_mps", "wind_speed_mps", "wind_kph", "wind_kmh", "wind_speed", "wind"] if c in dd.columns), None)
    if wd:
        vals = pd.to_numeric(dd[wd], errors="coerce")
        if wd in ["wind_kph", "wind_kmh"]:
            dd["wind_mps"] = vals / 3.6
        else:
            mean_val = vals.mean(skipna=True)
            dd["wind_mps"] = vals / 3.6 if (pd.notna(mean_val) and mean_val > 30) else vals
    else:
        dd["wind_mps"] = pd.NA

    out = dd[["hour_utc", "temp_C", "precip_mm", "wind_mps"]].drop_duplicates("hour_utc").sort_values("hour_utc")
    out["hour_utc"] = out["hour_utc"].astype("datetime64[ns]")
    return out


def _merge_weather_tolerant(
    agg: pd.DataFrame,
    weather_history_df: Optional[pd.DataFrame] = None,
    weather_forecast_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge history + forecast on hour_utc using merge_asof with ±30min tolerance.
    If weather_*_df are None, tries project-level fetchers.
    """
    # Lazy import to avoid hard dependency when running tests
    if weather_history_df is None or weather_forecast_df is None:
        try:
            from src.weather import fetch_history, fetch_forecast  # type: ignore
        except Exception:
            fetch_history = fetch_forecast = None  # type: ignore

    if weather_history_df is None and 'fetch_history' in locals() and fetch_history:
        try:
            weather_history_df = fetch_history(agg["hour_utc"].min(), agg["hour_utc"].max())
        except Exception:
            weather_history_df = None

    if weather_forecast_df is None and 'fetch_forecast' in locals() and fetch_forecast:
        try:
            weather_forecast_df = fetch_forecast(pd.to_datetime(agg["hour_utc"].max()), 36)
        except Exception:
            weather_forecast_df = None

    h = _normalize_weather(weather_history_df)
    f = _normalize_weather(weather_forecast_df)

    out = agg.copy()
    out["hour_utc"] = _ensure_ns(out["hour_utc"])

    for c in ["temp_C", "precip_mm", "wind_mps"]:
        if c not in out.columns:
            out[c] = pd.NA

    # Merge history first (fill past)
    if not h.empty:
        h["hour_utc"] = _ensure_ns(h["hour_utc"])
        out = pd.merge_asof(
            out.sort_values("hour_utc"),
            h.sort_values("hour_utc"),
            on="hour_utc",
            direction="nearest",
            tolerance=pd.Timedelta("30min"),
            suffixes=("", "_h"),
        )
        for c in ["temp_C", "precip_mm", "wind_mps"]:
            out[c] = out[c].fillna(out.get(f"{c}_h"))
            if f"{c}_h" in out.columns:
                out.drop(columns=[f"{c}_h"], inplace=True, errors="ignore")

    # Merge forecast next (fill future)
    if not f.empty:
        f["hour_utc"] = _ensure_ns(f["hour_utc"])
        out = pd.merge_asof(
            out.sort_values("hour_utc"),
            f.sort_values("hour_utc"),
            on="hour_utc",
            direction="nearest",
            tolerance=pd.Timedelta("30min"),
            suffixes=("", "_f"),
        )
        for c in ["temp_C", "precip_mm", "wind_mps"]:
            out[c] = out[c].fillna(out.get(f"{c}_f"))
            if f"{c}_f" in out.columns:
                out.drop(columns=[f"{c}_f"], inplace=True, errors="ignore")

    out["temp_C"] = pd.to_numeric(out["temp_C"], errors="coerce")
    out["precip_mm"] = pd.to_numeric(out["precip_mm"], errors="coerce")
    out["wind_mps"] = pd.to_numeric(out["wind_mps"], errors="coerce")
    return out


# =============================================================================
# Core aggregation
# =============================================================================

@dataclass
class OccupancyConfig:
    res_minutes: int = 5
    round_strategy: str = "left"   # 'left' (floor) or 'right' (ceil)
    with_weather: bool = True


def _make_bins(dt_series: pd.Series, minutes: int, strategy: str = "left") -> pd.Series:
    dt = pd.to_datetime(dt_series, errors="coerce", utc=True)
    if strategy == "right":
        b = (dt.dt.ceil(f"{minutes}min") - pd.to_timedelta(minutes, unit="min")).dt.tz_localize(None)
    else:
        b = dt.dt.floor(f"{minutes}min").dt.tz_localize(None)
    return b.astype("datetime64[ns]")


def occupancy_5min(
    df_snapshots: pd.DataFrame,
    config: OccupancyConfig = OccupancyConfig(),
    weather_history_df: Optional[pd.DataFrame] = None,
    weather_forecast_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Aggregate raw station snapshots to N-minute bins and compute occupancy & capacity.
    Output columns:
      tbin_utc, stationcode, name, nb_velos_bin, nb_bornes_bin, capacity_bin,
      lat, lon, hour_utc, occ_ratio_bin, temp_C, precip_mm, wind_mps
    """
    if df_snapshots is None or df_snapshots.empty:
        return pd.DataFrame(
            columns=[
                "tbin_utc", "stationcode", "name",
                "nb_velos_bin", "nb_bornes_bin", "capacity_bin",
                "lat", "lon", "hour_utc", "occ_ratio_bin",
                "temp_C", "precip_mm", "wind_mps",
            ]
        )

    df = df_snapshots.copy()

    # Detect timestamp column (inclut 'ts_utc')
    ts_col = next(
        (c for c in ["t", "ts", "timestamp", "date", "datetime", "recorded_at", "t_utc", "time_utc", "ts_utc"]
         if c in df.columns),
        None,
    )
    if ts_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "timestamp"})
            ts_col = "timestamp"
        else:
            raise ValueError(
                "No timestamp column found in snapshots "
                "(expected one of: t, ts, timestamp, date, datetime, recorded_at, t_utc, time_utc, ts_utc)."
            )

    df["t_utc"] = _ensure_datetime_utc_naive(df[ts_col])

    # Station identifiers / geometry
    if "stationcode" not in df.columns:
        alt = next((c for c in ["station_code", "code", "id", "station_id"] if c in df.columns), None)
        if alt:
            df = df.rename(columns={alt: "stationcode"})
        else:
            raise ValueError("Missing 'stationcode' column in snapshots.")

    if "name" not in df.columns:
        df["name"] = pd.NA

    # Counts
    vélos_cands = ["nb_velos_bin", "nb_velos", "bikes", "numbikesavailable", "num_bikes_available"]
    bornes_cands = ["nb_bornes_bin", "nb_bornes", "docks", "numdocksavailable", "num_docks_available"]
    cap_cands = ["capacity_bin", "capacity", "cap"]

    vel_col = next((c for c in vélos_cands if c in df.columns), None)
    bor_col = next((c for c in bornes_cands if c in df.columns), None)
    cap_col = next((c for c in cap_cands if c in df.columns), None)

    if vel_col is None or bor_col is None:
        raise ValueError("Missing bike/dock count columns (expected some of: nb_velos[_bin]/numbikesavailable, nb_bornes[_bin]/numdocksavailable).")

    df["nb_velos_bin"] = pd.to_numeric(df[vel_col], errors="coerce")
    df["nb_bornes_bin"] = pd.to_numeric(df[bor_col], errors="coerce")
    df["capacity_bin"] = pd.to_numeric(df[cap_col], errors="coerce") if cap_col is not None else (df["nb_velos_bin"] + df["nb_bornes_bin"])

    # Geometry
    if "lat" not in df.columns or "lon" not in df.columns:
        if "latitude" in df.columns and "longitude" in df.columns:
            df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
        else:
            df["lat"] = pd.NA
            df["lon"] = pd.NA

    # Time bins
    df["tbin_utc"] = _make_bins(df["t_utc"], minutes=config.res_minutes, strategy=config.round_strategy)
    df["hour_utc"] = df["t_utc"].dt.floor("h").astype("datetime64[ns]")

    # Aggregate per bin/station (last state in bin)
    agg = (
        df.sort_values("t_utc")
          .groupby(["tbin_utc", "stationcode"], as_index=False)
          .agg({
              "name": "last",
              "nb_velos_bin": "last",
              "nb_bornes_bin": "last",
              "capacity_bin": "last",
              "lat": "last",
              "lon": "last",
              "hour_utc": "last",
          })
    )

    # Derived metrics
    with np.errstate(divide="ignore", invalid="ignore"):
        occ = np.where(agg["capacity_bin"] > 0, agg["nb_velos_bin"] / agg["capacity_bin"], np.nan)
    agg["occ_ratio_bin"] = pd.Series(occ).clip(0, 1)

    # Weather merge
    if config.with_weather:
        agg = _merge_weather_tolerant(agg, weather_history_df, weather_forecast_df)
    else:
        for c in ["temp_C", "precip_mm", "wind_mps"]:
            if c not in agg.columns:
                agg[c] = pd.NA

    # Stable column order
    cols = [
        "tbin_utc", "stationcode", "name",
        "nb_velos_bin", "nb_bornes_bin", "capacity_bin",
        "lat", "lon", "hour_utc", "occ_ratio_bin",
        "temp_C", "precip_mm", "wind_mps",
    ]
    cols = [c for c in cols if c in agg.columns]
    agg = agg[cols].sort_values(["tbin_utc", "stationcode"]).reset_index(drop=True)
    return agg


# =============================================================================
# Orchestrateur (ingest -> aggregate -> export -> push HF)
# =============================================================================
if __name__ == "__main__":
    # bootstrap sys.path project root
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # imports runtime
    try:
        from src.ingest import ingest_once  # type: ignore
    except Exception:
        from ingest import ingest_once  # type: ignore

    # 1) ingest
    df_raw = ingest_once()
    if df_raw is None or df_raw.empty:
        print("[aggregate] No raw data — exiting 0.")
        sys.exit(0)

    # 2) aggregate
    cfg = OccupancyConfig(res_minutes=5, with_weather=True)
    df_agg = occupancy_5min(df_raw, cfg)

    # 3) export (chemin attendu par push)
    outdir = Path("exports")
    outdir.mkdir(parents=True, exist_ok=True)
    out_pq = outdir / "velib.parquet"
    df_agg.to_parquet(out_pq, index=False)
    print(f"[aggregate] wrote {out_pq} ({len(df_agg)} rows)")

    # 4) push HF
    try:
        # autoriser override par env si besoin
        os.environ.setdefault("PUSH_SRC", str(out_pq))
        try:
            from tools.push_hf import main as push_main  # type: ignore
        except Exception:
            # fallback si structure différente
            from push_hf import main as push_main  # type: ignore
        push_main()
    except SystemExit as e:
        # respecter code de retour du push
        raise
    except Exception as e:
        # ne pas masquer l’erreur pour Cloud Run
        print(f"[aggregate] push_hf failed: {e}")
        sys.exit(1)
