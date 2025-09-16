# tools/build_network_overview.py
# Page builder — "Réseau / Aperçu du réseau"
# Produces: KPIs tables, a day-profile chart (today vs median of past weeks),
# and a snapshot map (penury/saturation) from events.parquet.
#
# CLI:
#   python tools/build_network_overview.py --events docs/exports/events.parquet --last-days 7 --tz Europe/Paris
#
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import math
import json
import numpy as np
import pandas as pd

# Optional but useful for the snapshot map (saved as HTML)
try:
    import folium
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

import matplotlib.pyplot as plt

# --------------------------- Paths & helpers ---------------------------

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
FIGS_DIR = ASSETS / "figs" / "network" / "overview"
TABLES_DIR = ASSETS / "tables" / "network" / "overview"
MAPS_DIR = ASSETS / "maps"

def _mkdirs() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

def _read_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[overview] Missing events file: {path}")
    df = pd.read_parquet(path)
    # Normalize expected columns
    # Time
    for tc in ("ts", "tbin_utc"):
        if tc in df.columns:
            df["ts"] = pd.to_datetime(df[tc], errors="coerce")
            break
    if "ts" not in df.columns:
        raise KeyError("[overview] No 'ts' or 'tbin_utc' column in events.parquet")
    df["ts"] = df["ts"].dt.floor("15min")

    # Station key
    sid = None
    for c in ("station_id", "stationcode", "stationCode", "id"):
        if c in df.columns:
            sid = c
            break
    if sid is None:
        raise KeyError("[overview] No station identifier column found (station_id/stationcode)")
    df["station_id"] = df[sid].astype(str)

    # Bikes (available)
    bikes_col = None
    for c in ("bikes", "nb_velos_bin", "velos_disponibles", "numBikesAvailable", "n_bikes"):
        if c in df.columns:
            bikes_col = c
            break
    if bikes_col is None:
        raise KeyError("[overview] No bikes column found (bikes/nb_velos_bin/velos_disponibles)")
    df["bikes"] = pd.to_numeric(df[bikes_col], errors="coerce").fillna(0).astype(float)

    # Capacity and/or docks available
    cap_col = None
    for c in ("capacity", "cap", "dock_count", "n_docks"):
        if c in df.columns:
            cap_col = c
            break
    docks_col = None
    for c in ("docks", "docks_disponibles", "numDocksAvailable", "places_disponibles"):
        if c in df.columns:
            docks_col = c
            break

    if docks_col is not None:
        df["docks_avail"] = pd.to_numeric(df[docks_col], errors="coerce").fillna(np.nan).astype(float)
    elif cap_col is not None:
        cap = pd.to_numeric(df[cap_col], errors="coerce").astype(float)
        df["docks_avail"] = (cap - df["bikes"]).where(cap.notna(), np.nan)
        df["capacity"] = cap
    else:
        # No docks/capacity info → we can still compute bike availability, but not dock metrics
        df["docks_avail"] = np.nan

    # Station metadata
    for c in ("name", "station_name", "label"):
        if c in df.columns:
            df["name"] = df[c].astype(str)
            break
    if "name" not in df.columns:
        df["name"] = df["station_id"]

    latcol = None
    for c in ("lat", "latitude"):
        if c in df.columns:
            latcol = c
            break
    loncol = None
    for c in ("lon", "lng", "longitude"):
        if c in df.columns:
            loncol = c
            break
    df["lat"] = pd.to_numeric(df.get(latcol, np.nan), errors="coerce")
    df["lon"] = pd.to_numeric(df.get(loncol, np.nan), errors="coerce")

    return df[["ts", "station_id", "name", "lat", "lon", "bikes", "docks_avail"]].copy()

def _to_local_date(s: pd.Series, tz: str | None) -> pd.Series:
    """Convert naive UTC timestamps to a local date for grouping (display only)."""
    if tz:
        # Treat input as UTC-naive → localize UTC → convert to tz → take date
        return (s.dt.tz_localize("UTC").dt.tz_convert(tz).dt.date)
    return s.dt.date  # still UTC-based date

def _kpi_snapshot(df: pd.DataFrame, tz: str | None) -> dict:
    """KPIs at the latest timestamp available (closest snapshot)."""
    tmax = df["ts"].max()
    snap = df[df["ts"] == tmax].copy()
    total_stations = df["station_id"].nunique()
    active_stations = snap["station_id"].nunique()
    offline_stations = total_stations - active_stations

    bikes = pd.to_numeric(snap["bikes"], errors="coerce")
    docks = pd.to_numeric(snap["docks_avail"], errors="coerce")

    # Availability flags
    bike_avail = (bikes > 0)
    dock_avail = (docks > 0) if docks.notna().any() else pd.Series([np.nan] * len(snap), index=snap.index)

    # Structural states
    penury = (bikes == 0)
    saturation = (docks == 0) if docks.notna().any() else pd.Series([np.nan] * len(snap), index=snap.index)

    kpi = {
        "timestamp_snapshot_utc": tmax.isoformat(),
        "active_stations": int(active_stations),
        "offline_stations": int(offline_stations),
        "total_stations": int(total_stations),
        "bike_availability_rate": float(bike_avail.mean() * 100) if len(bike_avail) else np.nan,
        "dock_availability_rate": float(dock_avail.mean() * 100) if dock_avail.notna().any() else np.nan,
        "penury_rate": float(penury.mean() * 100) if len(penury) else np.nan,
        "saturation_rate": float(saturation.mean() * 100) if saturation.notna().any() else np.nan,
    }
    return kpi

def _coverage(df: pd.DataFrame, last_days: int) -> float:
    """Approximate global coverage over the last_days window (fraction of present bins).
    For each station, compute (#observed bins / #expected bins) then average.
    """
    if last_days <= 0:
        return np.nan
    tmax = df["ts"].max()
    tmin = tmax - pd.Timedelta(days=last_days)
    window = df[(df["ts"] > tmin) & (df["ts"] <= tmax)].copy()
    if window.empty:
        return np.nan

    # expected bins per station ≈ ceil(window_minutes/15) + 1
    expected_per_station = math.ceil((tmax - tmin).total_seconds() / 60 / 15) + 1
    obs = window.groupby("station_id")["ts"].nunique().clip(upper=expected_per_station)
    cov = (obs / expected_per_station).mean()
    return float(round(cov * 100, 2))

def _volatility_today(df: pd.DataFrame, tz: str | None) -> float:
    """Median within-day std of bikes per station for the 'today' bucket (by display tz)."""
    if df.empty:
        return np.nan
    # define 'today' (by tz) using the last timestamp
    tmax = df["ts"].max()
    if tz:
        local_day = tmax.tz_localize("UTC").tz_convert(tz).date()
    else:
        local_day = tmax.date()
    # keep rows that fall on that local day
    mask = _to_local_date(df["ts"], tz) == local_day
    today = df[mask]
    if today.empty:
        return np.nan
    vol = today.groupby("station_id")["bikes"].std(ddof=0)
    if vol.empty:
        return 0.0
    return float(round(vol.median(), 3))

def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _save_fig(path: Path) -> None:
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()

# --------------------------- Charts & Map ---------------------------

def build_day_profile_chart(df: pd.DataFrame, tz: str | None, out_png: Path, weeks_back: int = 3) -> None:
    """Plot 'today' curve vs median day-of-week profile computed over past `weeks_back` weeks.
    Value shown = share of stations with at least 1 bike (robust & comparable).
    """
    if df.empty:
        # create an empty placeholder figure
        plt.figure(figsize=(8, 4))
        plt.title("Day profile (no data)")
        _save_fig(out_png)
        return

    # Local time for grouping (display only)
    local_dt = df["ts"].dt.tz_localize("UTC").dt.tz_convert(tz) if tz else df["ts"]
    df = df.assign(
        date_local=local_dt.dt.date,
        time_local=local_dt.dt.time,
        dow_local=local_dt.dt.dayofweek,  # Monday=0
        hhmm=local_dt.dt.strftime("%H:%M"),
    ).copy()

    # Today bucket by local date (use last timestamp's local date)
    last_local_date = df["date_local"].iloc[df["ts"].idxmax()]  # pick the last row's local date
    df_today = df[df["date_local"] == last_local_date].copy()
    # Reference window: previous `weeks_back` same weekdays
    target_dow = int(df_today["dow_local"].iloc[0]) if not df_today.empty else None
    if target_dow is not None:
        ref = df[(df["dow_local"] == target_dow) & (df["date_local"] < last_local_date)].copy()
    else:
        ref = df.iloc[0:0].copy()

    # Compute share of stations with bikes>0 by hh:mm
    def share_has_bike(dfg: pd.DataFrame) -> pd.Series:
        # 1) part de stations avec ≥1 vélo à chaque timestamp (index unique)
        g = dfg.groupby("ts")["bikes"].apply(lambda s: (s > 0).mean())

        # 2) mapping ts -> hh:mm sans doublons (un seul hh:mm par ts)
        hhmm_map = dfg.drop_duplicates("ts").set_index("ts")["hhmm"]
        hhmm = hhmm_map.reindex(g.index)  # g.index est unique → OK

        # 3) regrouper par hh:mm et moyenner
        out = g.groupby(hhmm).mean()

        # 4) tri par ordre temporel
        order = sorted(out.index, key=lambda s: (int(s[:2]) * 60 + int(s[3:])))
        return out.reindex(order)


    today_curve = share_has_bike(df_today) if not df_today.empty else pd.Series(dtype=float)
    # For reference: compute median across same weekday in the last `weeks_back` weeks
    if not ref.empty:
        # limit to last N weeks worth of days of same weekday
        unique_days = sorted(ref["date_local"].unique())[-weeks_back:]
        ref = ref[ref["date_local"].isin(unique_days)]
        # per day curve, then median across days
        curves = []
        for d in sorted(ref["date_local"].unique()):
            curves.append(share_has_bike(ref[ref["date_local"] == d]))
        ref_curve = pd.concat(curves, axis=1).median(axis=1) if curves else pd.Series(dtype=float)
    else:
        ref_curve = pd.Series(dtype=float)

    # Plot
    plt.figure(figsize=(10, 4.5))
    if not ref_curve.empty:
        plt.plot(ref_curve.index, ref_curve.values, label="Median (past weeks)", linewidth=2)
    if not today_curve.empty:
        plt.plot(today_curve.index, today_curve.values, label="Today", linewidth=2)
    plt.ylim(0, 1)
    plt.ylabel("Share of stations with ≥1 bike")
    plt.xlabel("Local time")
    plt.title("Day profile: Today vs median (same weekday)")
    plt.legend(loc="best")
    plt.xticks(rotation=0)
    _save_fig(out_png)

def build_snapshot_map(df: pd.DataFrame, out_html: Path) -> None:
    """Folium map at the latest snapshot highlighting penury (bikes==0) and saturation (docks==0)."""
    if not HAS_FOLIUM:
        print("[overview] Folium not installed — skipping map.")
        return
    if df.empty or df["lat"].isna().all() or df["lon"].isna().all():
        print("[overview] No lat/lon in events — skipping map.")
        return

    tmax = df["ts"].max()
    snap = df[df["ts"] == tmax].copy()
    # Center map
    lat0 = snap["lat"].median()
    lon0 = snap["lon"].median()
    m = folium.Map(location=[float(lat0), float(lon0)], zoom_start=12, tiles="cartodbpositron")

    for _, r in snap.iterrows():
        bikes = float(r.get("bikes", np.nan))
        docks = r.get("docks_avail", np.nan)
        name = str(r.get("name", r["station_id"]))
        sid = r["station_id"]

        # Color logic
        if not np.isnan(bikes) and bikes == 0:
            color = "red"         # penury
        elif not pd.isna(docks) and float(docks) == 0.0:
            color = "black"       # saturation
        else:
            color = "blue"        # ok/other

        tooltip = f"{name} • {sid}\nBikes={bikes:.0f} • Docks={'' if pd.isna(docks) else int(docks)}"
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.8,
            tooltip=tooltip
        ).add_to(m)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    print(f"[overview] Map saved → {out_html.resolve()}")

# --------------------------- Main ---------------------------

def main(events_path: Path, last_days: int, tz: str | None) -> None:
    _mkdirs()
    df = _read_events(events_path)
    if df.empty:
        raise SystemExit("[overview] events.parquet is empty")

    # --- KPIs snapshot ---
    kpi = _kpi_snapshot(df, tz=tz)
    kpi["coverage_last_days_pct"] = _coverage(df, last_days=last_days)
    kpi["volatility_today_median_bikes_std"] = _volatility_today(df, tz=tz)

    # Write KPI JSON & CSV
    _write_json(TABLES_DIR / "kpis_today.json", kpi)
    pd.DataFrame([kpi]).to_csv(TABLES_DIR / "kpis_today.csv", index=False)
    print("[overview] KPIs written.")

    # --- Day profile chart (Today vs median past weeks on same weekday) ---
    out_png = FIGS_DIR / "day_profile_today_vs_median.png"
    build_day_profile_chart(df, tz=tz, out_png=out_png, weeks_back=3)
    print(f"[overview] Day profile chart → {out_png.resolve()}")

    # --- Optional: Top tension zones table (penury/saturation rates over the last N days) ---
    if last_days > 0:
        tmax = df["ts"].max()
        tmin = tmax - pd.Timedelta(days=last_days)
        win = df[(df["ts"] > tmin) & (df["ts"] <= tmax)].copy()
        if not win.empty:
            # Rates by station
            by_station = win.groupby("station_id").agg(
                name=("name", "last"),
                lat=("lat", "last"),
                lon=("lon", "last"),
                n=("ts", "nunique"),
                penury_rate=("bikes", lambda s: (s == 0).mean()),
                sat_rate=("docks_avail", lambda s: (s == 0).mean() if s.notna().any() else np.nan),
                bike_avail_rate=("bikes", lambda s: (s > 0).mean()),
                dock_avail_rate=("docks_avail", lambda s: (s > 0).mean() if s.notna().any() else np.nan),
            ).reset_index()
            by_station = by_station.sort_values(["penury_rate", "sat_rate"], ascending=[False, False])
            by_station.to_csv(TABLES_DIR / "stations_tension_last_days.csv", index=False)
            print("[overview] stations_tension_last_days.csv written.")

    # --- Snapshot map (HTML) ---
    build_snapshot_map(df, out_html=MAPS_DIR / "network_overview.html")

    # --- Small “health” CSVs to assist the markdown page ---
    # Availability snapshot distribution (counts of penury/saturation/ok)
    tmax = df["ts"].max()
    snap = df[df["ts"] == tmax].copy()
    bikes = (snap["bikes"] > 0)
    docks = (snap["docks_avail"] > 0) if snap["docks_avail"].notna().any() else pd.Series([np.nan]*len(snap), index=snap.index)

    dist = pd.DataFrame({
        "metric": ["bike_avail", "dock_avail", "penury", "saturation"],
        "value": [
            float(bikes.mean() * 100) if len(snap) else np.nan,
            float(docks.mean() * 100) if docks.notna().any() else np.nan,
            float((~bikes).mean() * 100) if len(snap) else np.nan,
            float(((snap["docks_avail"] == 0).mean() * 100) if snap["docks_avail"].notna().any() else np.nan),
        ]
    })
    dist.to_csv(TABLES_DIR / "snapshot_distribution.csv", index=False)

    print("[overview] Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build 'Network / Overview' assets from events.parquet")
    ap.add_argument("--events", type=Path, required=True, help="Path to docs/exports/events.parquet")
    ap.add_argument("--last-days", type=int, default=7, help="Window for recent KPIs (coverage, tension tables)")
    ap.add_argument("--tz", type=str, default=None, help="Display timezone (data stays UTC)")
    args = ap.parse_args()
    main(events_path=args.events, last_days=args.last_days, tz=args.tz)
