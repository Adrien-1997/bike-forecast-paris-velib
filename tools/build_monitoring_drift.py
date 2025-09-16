# tools/build_monitoring_drift.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import folium
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

# --------------------------- Paths ---------------------------

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
ASSETS = ROOT / "docs" / "assets"
TABLES_DIR = ASSETS / "tables" / "monitoring" / "drift"
FIGS_DIR = ASSETS / "figs" / "monitoring" / "drift"
MAPS_DIR = ROOT / "docs" / "maps"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)
MAPS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------- IO Helpers ---------------------------

def _read_parquet(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[drift] File not found: {p}")
    return pd.read_parquet(p)

# --------------------------- Core helpers ---------------------------

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize expected columns: ts, station_id, lat, lon, bikes, docks_avail, capacity_src, occ_ratio."""
    if df.empty:
        return df

    # ts column
    tcol = None
    for c in ("ts", "tbin_utc", "timestamp"):
        if c in df.columns:
            tcol = c
            break
    if tcol is None:
        raise KeyError("[drift] Missing time column (ts/tbin_utc/timestamp)")
    df["ts"] = pd.to_datetime(df[tcol], errors="coerce").dt.floor("15min")

    # station_id
    sid = None
    for c in ("station_id", "stationcode", "stationCode", "id"):
        if c in df.columns:
            sid = c
            break
    if sid is None:
        raise KeyError("[drift] Missing station_id/stationcode")
    df["station_id"] = df[sid].astype(str)

    # bikes / docks_avail / capacity_src
    # Try to infer; if not present, create with NaN so downstream selection never KeyErrors.
    if "bikes" not in df.columns:
        for c in ("num_bikes_available", "n_bikes", "available_bikes"):
            if c in df.columns:
                df["bikes"] = pd.to_numeric(df[c], errors="coerce")
                break
    if "bikes" not in df.columns:
        df["bikes"] = np.nan

    if "docks_avail" not in df.columns:
        for c in ("num_docks_available", "n_docks", "available_docks"):
            if c in df.columns:
                df["docks_avail"] = pd.to_numeric(df[c], errors="coerce")
                break
    if "docks_avail" not in df.columns:
        df["docks_avail"] = np.nan

    if "capacity_src" not in df.columns:
        for c in ("capacity", "capacity_est", "cap"):
            if c in df.columns:
                df["capacity_src"] = pd.to_numeric(df[c], errors="coerce")
                break
    if "capacity_src" not in df.columns:
        df["capacity_src"] = np.nan

    # lat/lon best-effort inference
    if "lat" not in df.columns or "lon" not in df.columns:
        for la, lo in (("latitude", "longitude"), ("lat", "lng"), ("lat", "long")):
            if la in df.columns and lo in df.columns:
                df["lat"] = pd.to_numeric(df[la], errors="coerce")
                df["lon"] = pd.to_numeric(df[lo], errors="coerce")
                break
    if "lat" not in df.columns:
        df["lat"] = np.nan
    if "lon" not in df.columns:
        df["lon"] = np.nan

    # occ_ratio = bikes / capacity (with capacity estimate fallback)
    if "occ_ratio" not in df.columns:
        cap_est = _estimate_capacity(df[["station_id", "bikes", "docks_avail", "capacity_src"]].copy())
        df = df.merge(cap_est, on="station_id", how="left")
        cap = df["capacity_src"].fillna(df["capacity_est"])
        with np.errstate(divide="ignore", invalid="ignore"):
            occ = (pd.to_numeric(df["bikes"], errors="coerce") / cap).replace([np.inf, -np.inf], np.nan)
        df["occ_ratio"] = occ.clip(lower=0, upper=1)

    return df[["ts", "station_id", "lat", "lon", "bikes", "docks_avail", "capacity_src", "occ_ratio"]].copy()

def _to_local(df: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    if tz:
        dt = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_convert(tz)
    else:
        dt = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    return df.assign(date_local=dt.dt.date, dow=dt.dt.dayofweek, hour=dt.dt.hour)

def _psi_continuous(ref: pd.Series, cur: pd.Series, bins: int = 20, eps: float = 1e-9) -> float:
    a = pd.to_numeric(ref, errors="coerce").dropna()
    b = pd.to_numeric(cur, errors="coerce").dropna()
    if a.empty or b.empty:
        return np.nan
    q = np.unique(np.nanquantile(a, np.linspace(0, 1, bins + 1)))
    if len(q) < 3:
        q = np.linspace(a.min(), a.max(), bins + 1)
    ca, _ = np.histogram(a, bins=q)
    cb, _ = np.histogram(b, bins=q)
    pa = (ca / max(1, ca.sum())).astype(float) + eps
    pb = (cb / max(1, cb.sum())).astype(float) + eps
    return float(np.sum((pa - pb) * np.log(pa / pb)))

def _ks_stat(ref: pd.Series, cur: pd.Series) -> float:
    a = pd.to_numeric(ref, errors="coerce").dropna()
    b = pd.to_numeric(cur, errors="coerce").dropna()
    if a.empty or b.empty:
        return np.nan
    q = np.unique(np.nanquantile(pd.concat([a, b]), np.linspace(0, 1, 201)))
    ca, _ = np.histogram(a, bins=q)
    cb, _ = np.histogram(b, bins=q)
    cdfa = np.cumsum(ca) / max(1, ca.sum())
    cdfb = np.cumsum(cb) / max(1, cb.sum())
    return float(np.max(np.abs(cdfa - cdfb)))

def _delta_mean_var(ref: pd.Series, cur: pd.Series) -> tuple[float, float]:
    a = pd.to_numeric(ref, errors="coerce").dropna()
    b = pd.to_numeric(cur, errors="coerce").dropna()
    if a.empty or b.empty:
        return (np.nan, np.nan)
    dm = (b.mean() - a.mean()) / (a.std(ddof=1) + 1e-9)
    dv = (b.var(ddof=1) - a.var(ddof=1)) / (a.var(ddof=1) + 1e-9)
    return (float(dm), float(dv))

# --------------------------- Capacity estimation ---------------------------

def _estimate_capacity(win: pd.DataFrame) -> pd.Series:
    """Estimate station capacity: prefer capacity_src; else 0.98-quantile of bikes+docks; else bikes."""
    def est(g: pd.DataFrame) -> float:
        cap = g["capacity_src"].dropna().max() if "capacity_src" in g.columns else np.nan
        if pd.notna(cap) and cap > 0:
            return float(cap)
        if "docks_avail" in g.columns and g["docks_avail"].notna().any():
            s = (g.get("bikes", pd.Series(dtype=float)).clip(lower=0) +
                 g["docks_avail"].clip(lower=0)).dropna()
            if len(s):
                return float(s.quantile(0.98))
        b = g.get("bikes", pd.Series(dtype=float)).clip(lower=0).dropna()
        return float(b.quantile(0.98)) if len(b) else np.nan

    cols = [c for c in ["station_id", "bikes", "docks_avail", "capacity_src"] if c in win.columns]
    return (win[cols]
            .groupby("station_id", group_keys=False)
            .apply(est)
            .rename("capacity_est"))

def _assign_zone(df: pd.DataFrame) -> pd.Series:
    for c in ("arrondissement", "arr", "zone", "district"):
        if c in df.columns:
            return df[c].astype(str)

    # grid ~1km (lat/lon rounded to 0.01) — only for valid coords
    lat = pd.to_numeric(df.get("lat"), errors="coerce")
    lon = pd.to_numeric(df.get("lon"), errors="coerce")
    mask = lat.notna() & lon.notna()

    z = pd.Series(index=df.index, dtype=object, name="zone")
    lat_rounded = (lat[mask] * 100).round() / 100.0
    lon_rounded = (lon[mask] * 100).round() / 100.0
    z.loc[mask] = lat_rounded.astype(str) + "," + lon_rounded.astype(str)
    return z

# --------------------------- Windowing ---------------------------

def _split_windows(df: pd.DataFrame, current_days: int, reference_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.iloc[0:0].copy(), df.iloc[0:0].copy()
    tmax = df["ts"].max()
    t_cur_start = tmax - pd.Timedelta(days=current_days)
    t_ref_end = t_cur_start
    t_ref_start = t_ref_end - pd.Timedelta(days=reference_days)
    ref = df[(df["ts"] >= t_ref_start) & (df["ts"] < t_ref_end)].copy()
    cur = df[(df["ts"] >= t_cur_start) & (df["ts"] <= tmax)].copy()
    return ref, cur

# --------------------------- Main drift computation ---------------------------

def compute_drift(events: pd.DataFrame, current_days: int, reference_days: int, tz: Optional[str]) -> dict:
    df = _ensure_columns(events)
    df = _to_local(df, tz)

    # Windows
    ref, cur = _split_windows(df, current_days=current_days, reference_days=reference_days)

    # Daily aggregation by station
    def agg(df_):
        return (df_.groupby(["date_local", "station_id"])
                  .agg(occ_ratio=("occ_ratio", "mean"),
                       bikes=("bikes", "mean"),
                       docks_avail=("docks_avail", "mean"),
                       lat=("lat", "median"),
                       lon=("lon", "median"))
                  .reset_index())

    ref = agg(ref)
    cur = agg(cur)

    # PSI/KS/Δ for continuous features
    feats = ["occ_ratio", "bikes", "docks_avail"]
    rows_psi, rows_ks, rows_delta = [], [], []
    for f in feats:
        rows_psi.append({"feature": f, "psi": _psi_continuous(ref[f], cur[f])})
        rows_ks.append({"feature": f, "ks": _ks_stat(ref[f], cur[f])})
        dm, dv = _delta_mean_var(ref[f], cur[f])
        rows_delta.append({"feature": f, "delta_mean": dm, "delta_var": dv})

    psi_df = pd.DataFrame(rows_psi)
    ks_df = pd.DataFrame(rows_ks)
    d_df = pd.DataFrame(rows_delta)

    # PSI global daily (use occ_ratio mean per day across stations)
    by_day = (df.groupby("date_local")["occ_ratio"]
                .apply(lambda s: float(np.nanmean(s))).reset_index()
                .sort_values("date_local"))
    # EMA of daily PSI proxy
    alpha = 2 / (7 + 1.0)
    ema, last = [], None
    for _, r in by_day.iterrows():
        x = r["occ_ratio"]
        if pd.isna(x):
            ema.append(np.nan)
            continue
        last = x if last is None else (alpha * x + (1 - alpha) * last)
        ema.append(last)
    psi_daily_ema = pd.DataFrame({"date_local": by_day["date_local"], "psi_ema": ema})

    # Summary + alerts
    psi_top = psi_df.sort_values("psi", ascending=False).reset_index(drop=True)
    psi_global = float(psi_df.loc[psi_df["feature"] == "occ_ratio", "psi"].values[0]) if not psi_df.empty else np.nan
    alerts = []
    if np.isfinite(psi_global) and psi_global >= 0.25:
        alerts.append({"level": "high", "code": "psi_global_high", "text": f"High global PSI ({psi_global:.3f})"})
    elif np.isfinite(psi_global) and psi_global >= 0.1:
        alerts.append({"level": "medium", "code": "psi_global_medium", "text": f"Moderate global PSI ({psi_global:.3f})"})

    summary = pd.DataFrame([{
        "psi_global": psi_global,
        "top_feature": psi_top.iloc[0]["feature"] if not psi_top.empty else None,
        "top_feature_psi": float(psi_top.iloc[0]["psi"]) if not psi_top.empty else np.nan,
    }])

    # Map (zones) — build zones first, then draw if Folium available and coords are valid
    if HAS_FOLIUM and (ref["lat"].notna().any() or cur["lat"].notna().any()):
        ref = ref.assign(zone=_assign_zone(ref))
        cur = cur.assign(zone=_assign_zone(cur))

        def psi_zone(df_ref: pd.DataFrame, df_cur: pd.DataFrame) -> pd.DataFrame:
            rows = []
            for z, rsub in df_ref.groupby("zone"):
                if pd.isna(z):  # skip invalid zones
                    continue
                csub = df_cur[df_cur["zone"] == z]
                if csub.empty:
                    continue
                rows.append({"zone": z, "psi": _psi_continuous(rsub["occ_ratio"], csub["occ_ratio"])})
            return pd.DataFrame(rows)

        pz = psi_zone(ref, cur)
        cent = (ref.dropna(subset=["lat", "lon"]).groupby("zone")[["lat", "lon"]].median().reset_index())
        pz = pz.merge(cent, on="zone", how="left").dropna(subset=["lat", "lon"])

        if not pz.empty:
            lat0, lon0 = float(pz["lat"].median()), float(pz["lon"].median())
            m = folium.Map(location=[lat0, lon0], zoom_start=12, tiles="cartodbpositron")
            for _, r in pz.iterrows():
                psi = float(r["psi"]) if np.isfinite(r["psi"]) else 0.0
                rad = 3 + min(12, max(0.0, psi) * 20.0)
                folium.CircleMarker(
                    location=[float(r["lat"]), float(r["lon"])],
                    radius=rad,
                    color="red" if psi >= 0.25 else ("orange" if psi >= 0.1 else "blue"),
                    fill=True, fill_opacity=0.8,
                    tooltip=f"zone={r['zone']} • PSI={psi:.3f}"
                ).add_to(m)
            m.save(str(MAPS_DIR / "drift_by_zone.html"))

    return {
        "psi_df": psi_df,
        "ks_df": ks_df,
        "deltas_df": d_df,
        "psi_daily_ema": psi_daily_ema,
        "summary": summary,
        "alerts": alerts,
    }

# --------------------------- Optional: target drift ---------------------------

def compute_target_drift(perf: pd.DataFrame, current_days: int, reference_days: int, tz: Optional[str]) -> Optional[pd.DataFrame]:
    if perf is None or perf.empty:
        return None
    perf = _to_local(perf, tz)
    ref, cur = _split_windows(perf, current_days=current_days, reference_days=reference_days)
    if ref.empty or cur.empty:
        return None
    ks = _ks_stat(ref["y_true"], cur["y_true"])
    dm, dv = _delta_mean_var(ref["y_true"], cur["y_true"])
    return pd.DataFrame([{"ks": float(ks), "delta_mean": float(dm), "delta_var": float(dv)}])

# --------------------------- Plots ---------------------------

def _plot_top_features(psi_df: pd.DataFrame, out: Path, top_n: int = 10):
    if psi_df is None or psi_df.empty:
        return
    d = psi_df.sort_values("psi", ascending=False).head(top_n)
    plt.figure(figsize=(8, 4.5))
    plt.bar(d["feature"], d["psi"])
    plt.title("Top drifted features (PSI)")
    plt.ylabel("PSI")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()

def _plot_psi_ema(ema_df: pd.DataFrame, out: Path):
    if ema_df is None or ema_df.empty:
        return
    plt.figure(figsize=(8, 3.8))
    plt.plot(pd.to_datetime(ema_df["date_local"]), ema_df["psi_ema"])
    plt.title("Global PSI (EMA)")
    plt.ylabel("EMA(occ_ratio)")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()

# --------------------------- Exports ---------------------------

def _export_tables(res: dict):
    res["psi_df"].to_csv(TABLES_DIR / "psi_by_feature.csv", index=False)
    res["ks_df"].to_csv(TABLES_DIR / "ks_by_feature.csv", index=False)
    res["deltas_df"].to_csv(TABLES_DIR / "deltas_by_feature.csv", index=False)
    res["psi_daily_ema"].to_csv(TABLES_DIR / "psi_global_daily_ema.csv", index=False)
    res["summary"].to_csv(TABLES_DIR / "drift_summary.csv", index=False)

def _export_figs(res: dict):
    _plot_top_features(res["psi_df"], FIGS_DIR / "psi_top_features.png")
    _plot_psi_ema(res["psi_daily_ema"], FIGS_DIR / "psi_global_ema.png")

# --------------------------- CLI ---------------------------

def main(
    events_path: str,
    current_days: int,
    reference_days: int,
    tz: Optional[str],
    perf_path: Optional[str] = None,
):
    events = _read_parquet(events_path)
    events = _ensure_columns(events)

    res = compute_drift(events, current_days=current_days, reference_days=reference_days, tz=tz)
    _export_tables(res)
    _export_figs(res)

    if perf_path:
        try:
            perf = _read_parquet(perf_path)
            tgt = compute_target_drift(perf, current_days=current_days, reference_days=reference_days, tz=tz)
            if tgt is not None:
                tgt.to_csv(TABLES_DIR / "target_drift.csv", index=False)
        except Exception as e:
            print(f"[drift] target drift skipped: {e}")

    print("[drift] Done.")
    print(f"[drift] Tables -> {TABLES_DIR}")
    print(f"[drift] Figures -> {FIGS_DIR}")
    if (MAPS_DIR / "drift_by_zone.html").exists():
        print(f"[drift] Map -> {MAPS_DIR / 'drift_by_zone.html'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="docs/exports/events.parquet")
    ap.add_argument("--current-days", type=int, default=7)
    ap.add_argument("--reference-days", type=int, default=28)
    ap.add_argument("--tz", default="Europe/Paris")
    ap.add_argument("--perf", default=None, help="docs/exports/perf.parquet (optional)")
    args = ap.parse_args()

    main(
        events_path=args.events,
        current_days=args.current_days,
        reference_days=args.reference_days,
        tz=args.tz,
        perf_path=args.perf,
    )
