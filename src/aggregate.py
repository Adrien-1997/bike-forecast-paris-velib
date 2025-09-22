# aggregate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


# =============================================================================
# Weather helpers (robust normalization + tolerant merge)
# =============================================================================

def _normalize_weather(df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Normalize any weather dataframe to the schema:
    [hour_utc (datetime-naive, floored to hour), temp_C, precip_mm, wind_mps].
    Accepts several possible input column names; converts wind from km/h to m/s if needed.
    """
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return pd.DataFrame(columns=["hour_utc", "temp_C", "precip_mm", "wind_mps"])

    dd = df.copy()

    # --- Datetime detection
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

    dd["hour_utc"] = pd.to_datetime(dd[dt_col], errors="coerce", utc=True)
    dd["hour_utc"] = dd["hour_utc"].dt.floor("h").dt.tz_localize(None)

    # --- Temperature (°C)
    temp_candidates = ["temp_C", "temperature", "temp", "air_temperature"]
    temp = next((c for c in temp_candidates if c in dd.columns), None)
    dd["temp_C"] = pd.to_numeric(dd[temp], errors="coerce") if temp else pd.NA

    # --- Precipitation (mm)
    precip_candidates = ["precip_mm", "precipitation_mm", "rain_mm", "precip"]
    pr = next((c for c in precip_candidates if c in dd.columns), None)
    dd["precip_mm"] = pd.to_numeric(dd[pr], errors="coerce") if pr else pd.NA

    # --- Wind (→ m/s)
    wind_candidates = ["wind_mps", "wind_speed_mps", "wind_kph", "wind_kmh", "wind_speed", "wind"]
    wd = next((c for c in wind_candidates if c in dd.columns), None)
    if wd:
        vals = pd.to_numeric(dd[wd], errors="coerce")
        # Heuristic + explicit units
        if wd in ["wind_kph", "wind_kmh"]:
            dd["wind_mps"] = vals / 3.6
        else:
            mean_val = vals.mean(skipna=True)
            if mean_val is not None and pd.notna(mean_val) and mean_val > 30:
                dd["wind_mps"] = vals / 3.6
            else:
                dd["wind_mps"] = vals
    else:
        dd["wind_mps"] = pd.NA

    out = dd[["hour_utc", "temp_C", "precip_mm", "wind_mps"]].drop_duplicates("hour_utc")
    out = out.sort_values("hour_utc")
    return out


def _merge_weather_tolerant(
    agg: pd.DataFrame,
    weather_history_df: Optional[pd.DataFrame] = None,
    weather_forecast_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge history + forecast onto `agg` on hour_utc using merge_asof with ±30min tolerance.
    If weather_*_df are None, tries to call project-level functions fetch_history() and fetch_forecast().
    """
    # Try dynamic fetchers if needed
    if weather_history_df is None:
        try:
            # expected signature: fetch_history(start_dt, end_dt) -> DataFrame
            weather_history_df = fetch_history(agg["hour_utc"].min(), agg["hour_utc"].max())  # type: ignore[name-defined]
        except Exception:
            weather_history_df = None

    if weather_forecast_df is None:
        try:
            # expected signature: fetch_forecast(start_dt, hours_ahead=36) -> DataFrame
            weather_forecast_df = fetch_forecast(pd.to_datetime(agg["hour_utc"].max()), 36)  # type: ignore[name-defined]
        except Exception:
            weather_forecast_df = None

    # Normalize
    h = _normalize_weather(weather_history_df)
    f = _normalize_weather(weather_forecast_df)

    out = agg.copy()
    for c in ["temp_C", "precip_mm", "wind_mps"]:
        if c not in out.columns:
            out[c] = pd.NA

    # Merge history first (fill past)
    if not h.empty:
        out = pd.merge_asof(
            out.sort_values("hour_utc"),
            h,
            on="hour_utc",
            direction="nearest",
            tolerance=pd.Timedelta("30min"),
            suffixes=("", "_h"),
        )
        for c in ["temp_C", "precip_mm", "wind_mps"]:
            out[c] = out[c].fillna(out[f"{c}_h"])
            out.drop(columns=[f"{c}_h"], inplace=True, errors="ignore")

    # Merge forecast next (fill future)
    if not f.empty:
        out = pd.merge_asof(
            out.sort_values("hour_utc"),
            f,
            on="hour_utc",
            direction="nearest",
            tolerance=pd.Timedelta("30min"),
            suffixes=("", "_f"),
        )
        for c in ["temp_C", "precip_mm", "wind_mps"]:
            out[c] = out[c].fillna(out[f"{c}_f"])
            out.drop(columns=[f"{c}_f"], inplace=True, errors="ignore")

    # enforce numeric types
    out["temp_C"] = pd.to_numeric(out["temp_C"], errors="coerce")
    out["precip_mm"] = pd.to_numeric(out["precip_mm"], errors="coerce")
    out["wind_mps"] = pd.to_numeric(out["wind_mps"], errors="coerce")

    return out


# =============================================================================
# Core aggregation
# =============================================================================

@dataclass
class OccupancyConfig:
    """
    Configuration for occupancy aggregation.
    """
    res_minutes: int = 5                     # bin size for tbin_utc
    round_strategy: str = "left"             # 'left' (floor) or 'right' (ceil)
    with_weather: bool = True                # attach weather or not


def _ensure_datetime_utc_naive(series: pd.Series) -> pd.Series:
    """Convert to UTC then drop tz, keep as naive datetime64[ns]."""
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt.dt.tz_localize(None)


def _make_bins(dt_series: pd.Series, minutes: int, strategy: str = "left") -> pd.Series:
    """Create time bins at given minute frequency; 'left' floors, 'right' ceils."""
    dt = pd.to_datetime(dt_series, errors="coerce", utc=True)
    if strategy == "right":
        b = (dt.dt.ceil(f"{minutes}min") - pd.to_timedelta(minutes, unit="min")).dt.tz_localize(None)
    else:
        b = dt.dt.floor(f"{minutes}min").dt.tz_localize(None)
    return b


def occupancy_5min(
    df_snapshots: pd.DataFrame,
    config: OccupancyConfig = OccupancyConfig(),
    weather_history_df: Optional[pd.DataFrame] = None,
    weather_forecast_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Aggregate raw station snapshots to 5-min (configurable) bins and compute occupancy & capacity.
    Expected minimum columns in df_snapshots:
      - timestamp (any name acceptable; see candidates below)
      - stationcode
      - nb_velos (or *_bin), nb_bornes (or *_bin), capacity (optional)
      - lat, lon
      - name (optional)

    Output columns:
      tbin_utc, stationcode, name, nb_velos_bin, nb_bornes_bin, capacity_bin,
      lat, lon, hour_utc, occ_ratio_bin,
      [temp_C, precip_mm, wind_mps] if with_weather
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

    # --- Detect timestamp column
    ts_col = next(
        (c for c in ["t", "ts", "timestamp", "date", "datetime", "recorded_at", "t_utc", "time_utc"]
         if c in df.columns),
        None,
    )
    if ts_col is None:
        # fallback: try index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "timestamp"})
            ts_col = "timestamp"
        else:
            raise ValueError("No timestamp column found in snapshots (expected one of: t, ts, timestamp, date, datetime, recorded_at, t_utc).")

    df["t_utc"] = _ensure_datetime_utc_naive(df[ts_col])

    # --- Station identifiers / geometry
    if "stationcode" not in df.columns:
        alt = next((c for c in ["station_code", "code", "id", "station_id"] if c in df.columns), None)
        if alt:
            df = df.rename(columns={alt: "stationcode"})
        else:
            raise ValueError("Missing 'stationcode' column in snapshots.")

    if "name" not in df.columns:
        df["name"] = pd.NA

    # --- Counts
    vélos_cands = ["nb_velos_bin", "nb_velos", "bikes", "num_bikes_available"]
    bornes_cands = ["nb_bornes_bin", "nb_bornes", "docks", "num_docks_available"]
    cap_cands = ["capacity_bin", "capacity", "cap"]

    vel_col = next((c for c in vélos_cands if c in df.columns), None)
    bor_col = next((c for c in bornes_cands if c in df.columns), None)
    cap_col = next((c for c in cap_cands if c in df.columns), None)

    if vel_col is None or bor_col is None:
        raise ValueError("Missing bike/dock count columns (expected some of: nb_velos[_bin], nb_bornes[_bin]).")

    df["nb_velos_bin"] = pd.to_numeric(df[vel_col], errors="coerce")
    df["nb_bornes_bin"] = pd.to_numeric(df[bor_col], errors="coerce")

    if cap_col is not None:
        df["capacity_bin"] = pd.to_numeric(df[cap_col], errors="coerce")
    else:
        df["capacity_bin"] = df["nb_velos_bin"] + df["nb_bornes_bin"]

    # --- Geometry
    if "lat" not in df.columns or "lon" not in df.columns:
        if "latitude" in df.columns and "longitude" in df.columns:
            df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
        else:
            df["lat"] = pd.NA
            df["lon"] = pd.NA

    # --- Time bins
    df["tbin_utc"] = _make_bins(df["t_utc"], minutes=config.res_minutes, strategy=config.round_strategy)
    df["hour_utc"] = df["t_utc"].dt.floor("h")

    # --- Aggregate per bin/station (take last known state in the bin)
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

    # --- Derived metrics
    with np.errstate(divide="ignore", invalid="ignore"):
        occ = np.where(agg["capacity_bin"] > 0, agg["nb_velos_bin"] / agg["capacity_bin"], np.nan)
    agg["occ_ratio_bin"] = pd.to_numeric(occ)
    agg["occ_ratio_bin"] = agg["occ_ratio_bin"].clip(lower=0, upper=1)

    # --- Weather merge
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