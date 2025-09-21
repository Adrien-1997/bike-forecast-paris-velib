# tools/build_monitoring_model_health.py
# Page builder — "Monitoring / Model health"
#
# Outputs (docs/assets/*):
#   tables/monitoring/model_health/
#     - daily_metrics.csv
#     - window_metrics.csv
#     - error_by_station_7d.csv
#     - error_by_hour_7d.csv
#     - error_by_cluster_7d.csv        (if clusters table exists)
#     - error_by_zone_7d.csv           (if lat/lon available)
#     - calibration_global_7d.csv
#     - calibration_by_hour_7d.csv
#     - coverage_j1_j7_j28.csv
#     - residuals_acf.csv
#     - top_degrading_stations.csv
#     - alerts_summary.csv
#   figs/monitoring/model_health/
#     - mae_rmse_daily.png
#     - lift_daily.png
#     - coverage_daily.png
#     - calibration_beta_by_hour_7d.png
#     - residuals_acf.png
#     - top_degrading_stations.png
#   maps/
#     - model_error_by_station_7d.html  (if lat/lon available)
#
# Page Markdown rendue :
#   docs/monitoring/model-health.md
#
# CLI:
#   python tools/build_monitoring_model_health.py \
#       --perf docs/exports/perf.parquet \
#       --last-days 30 --tz Europe/Paris --horizon 15 --bin-min 5
#
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional for maps
try:
    import folium
    from branca.element import Element  # pour injecter la légende HTML
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False
    Element = None


# --------------------------- Paths ---------------------------

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
FIGS_DIR = ASSETS / "figs" / "monitoring" / "model_health"
TABLES_DIR = ASSETS / "tables" / "monitoring" / "model_health"
MAPS_DIR = ASSETS / "maps"
OUT_MD = DOCS / "monitoring" / "model-health.md"
STATION_CLUSTERS = ASSETS / "tables" / "network" / "stations" / "station_clusters.csv"


# --------------------------- Utils ---------------------------
def _load_any_station_coords() -> Optional[pd.DataFrame]:
    """
    Cherche dans docs/assets/tables/network/stations/ un CSV contenant
    (station_id, lat, lon). Retourne le premier trouvé, sinon None.
    """
    base = STATION_CLUSTERS.parent  # .../network/stations/
    if not base.exists():
        return None
    candidates = sorted([p for p in base.glob("*.csv") if p.is_file()])
    for p in candidates:
        try:
            df = pd.read_csv(p, dtype={"station_id": str})
            cols = {c.lower(): c for c in df.columns}
            if "station_id" in {k.lower() for k in df.columns} and \
               "lat" in {k.lower() for k in df.columns} and \
               "lon" in {k.lower() for k in df.columns}:
                sid = cols.get("station_id", "station_id")
                lat = cols.get("lat", "lat")
                lon = cols.get("lon", "lon")
                out = df[[sid, lat, lon]].rename(columns={sid: "station_id", lat: "lat", lon: "lon"})
                out["station_id"] = out["station_id"].astype(str)
                out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
                out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
                return out.dropna(subset=["lat", "lon"])
        except Exception:
            continue
    return None

def _ensure_latlon(perf: pd.DataFrame) -> pd.DataFrame:
    """
    Si lat/lon manquants ou vides dans perf, tente une jointure avec un
    registre stations (station_id, lat, lon) trouvé automatiquement.
    """
    needs_merge = ("lat" not in perf.columns) or ("lon" not in perf.columns) \
                  or perf["lat"].isna().all() or perf["lon"].isna().all()
    if not needs_merge:
        return perf
    reg = _load_any_station_coords()
    if reg is None or reg.empty:
        return perf
    return perf.merge(reg, on="station_id", how="left")


def _mkdirs() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

def _save_fig(path: Path) -> None:
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()

def _read_perf(path: Path, bin_min: int = 5) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[model-health] Not found: {path}")
    df = pd.read_parquet(path)

    # ts
    if "ts" not in df.columns:
        raise KeyError("[model-health] Missing 'ts' column")
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce").dt.floor(f"{int(bin_min)}min")

    # station_id
    sid = None
    for c in ("station_id", "stationcode", "stationCode", "id"):
        if c in df.columns:
            sid = c
            break
    if sid is None:
        raise KeyError("[model-health] Missing station identifier (station_id/stationcode)")
    df["station_id"] = df[sid].astype(str)

    # y_true / preds
    if "y_true" not in df.columns:
        raise KeyError("[model-health] Missing 'y_true'")
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df.get("y_pred", np.nan), errors="coerce")
    if "y_pred_baseline" in df.columns:
        df["y_pred_baseline"] = pd.to_numeric(df["y_pred_baseline"], errors="coerce")
    else:
        # fallback: baseline = y_true (lift sera NaN si base==model)
        df["y_pred_baseline"] = df["y_true"]

    # optional meta
    if "lat" in df.columns: df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    if "lon" in df.columns: df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    if "name" in df.columns: df["name"] = df["name"].astype(str)

    return df

def _localize(df: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    if tz:
        ldt = df["ts"].dt.tz_localize("UTC").dt.tz_convert(tz)
        return df.assign(date_local=ldt.dt.date, dow=ldt.dt.dayofweek, hour=ldt.dt.hour)
    return df.assign(date_local=df["ts"].dt.date, dow=df["ts"].dt.dayofweek, hour=df["ts"].dt.hour)

def _rel_from_md(md_path: Path, target: Path) -> str:
    """Chemin relatif depuis md_path vers target, compatible MkDocs (use_directory_urls: true)."""
    md_rel = Path(md_path).resolve().relative_to(DOCS.resolve())
    parts = md_rel.with_suffix("").parts  # ('monitoring','model-health') / ('index',)
    depth = len(parts) if parts[-1] != "index" else len(parts) - 1
    prefix = "../" * max(depth, 0)
    rel_from_docs = Path(target).resolve().relative_to(DOCS.resolve()).as_posix()
    return (prefix + rel_from_docs).replace("//", "/")


# ---------- Robust metrics (no warnings on empty/NaN slices) ----------

def _metrics(y_true: pd.Series, y_hat: pd.Series) -> dict:
    """Compute MAE/RMSE/bias/var on finite pairs only. Returns NaN if no valid pair."""
    yt = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float, copy=False)
    yh = pd.to_numeric(y_hat,  errors="coerce").to_numpy(dtype=float, copy=False)

    mask = np.isfinite(yt) & np.isfinite(yh)
    if not np.any(mask):
        return {"mae": np.nan, "rmse": np.nan, "me": np.nan, "var_resid": np.nan}

    e = yt[mask] - yh[mask]
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    me = float(np.mean(e))
    var_resid = float(np.var(e)) if e.size >= 2 else np.nan
    return {"mae": mae, "rmse": rmse, "me": me, "var_resid": var_resid}

def _lift(mae_base: float, mae_model: float) -> float:
    if not np.isfinite(mae_base) or not np.isfinite(mae_model) or mae_base == 0:
        return np.nan
    return float((mae_base - mae_model) / mae_base)

def _safe_div(num: pd.Series | np.ndarray, den: pd.Series | np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(num, np.nan, dtype=float)
    m = np.isfinite(den) & (den != 0) & np.isfinite(num)
    out[m] = num[m] / den[m]
    return out

def _acf(series: pd.Series, nlags: int) -> np.ndarray:
    x = series.dropna().astype(float).values
    n = len(x)
    if n == 0:
        return np.array([])
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom == 0:
        return np.array([np.nan] * (nlags + 1))
    ac = [1.0]
    for k in range(1, nlags+1):
        if k >= n:
            ac.append(np.nan); continue
        ac.append(float(np.dot(x[:-k], x[k:]) / denom))
    return np.array(ac)

def _assign_zone(df: pd.DataFrame) -> pd.Series:
    for c in ("arrondissement","arr","zone","district"):
        if c in df.columns: return df[c].astype(str)
    if "lat" in df.columns and "lon" in df.columns:
        lat_rounded = (df["lat"] * 100).round() / 100.0
        lon_rounded = (df["lon"] * 100).round() / 100.0
        return (lat_rounded.astype(str) + "," + lon_rounded.astype(str)).rename("zone")
    return pd.Series(["unknown"] * len(df), index=df.index, name="zone")


# --------------------------- Core computations ---------------------------

def daily_metrics(df: pd.DataFrame, last_days: int) -> pd.DataFrame:
    d = (df.groupby("date_local")
            .apply(lambda g: pd.Series({
                "mae_model": _metrics(g["y_true"], g["y_pred"])["mae"],
                "mae_baseline": _metrics(g["y_true"], g["y_pred_baseline"])["mae"],
                "rmse_model": _metrics(g["y_true"], g["y_pred"])["rmse"],
                "rmse_baseline": _metrics(g["y_true"], g["y_pred_baseline"])["rmse"],
                "bias_model": _metrics(g["y_true"], g["y_pred"])["me"],
                "coverage_pred_pct": float(g["y_pred"].notna().mean() * 100.0),
                "n": int(len(g))
            }))).reset_index(names="date")
    d["lift_vs_baseline"] = d.apply(lambda r: _lift(r["mae_baseline"], r["mae_model"]), axis=1)
    if last_days and last_days > 0:
        d = d.sort_values("date").tail(last_days)
    return d

def window_slice(df: pd.DataFrame, days: int) -> pd.DataFrame:
    tmax = df["ts"].max()
    tmin = tmax - pd.Timedelta(days=days)
    return df[(df["ts"] > tmin) & (df["ts"] <= tmax)].copy()

def window_metrics(df: pd.DataFrame, days: int) -> dict:
    if days <= 0 or df.empty:
        return {}
    win = window_slice(df, days)
    if win.empty:
        return {}
    m_model = _metrics(win["y_true"], win["y_pred"])
    m_base  = _metrics(win["y_true"], win["y_pred_baseline"])
    cov = float(win["y_pred"].notna().mean() * 100.0)
    return {
        "days": days,
        "mae_model": m_model["mae"], "rmse_model": m_model["rmse"], "bias_model": m_model["me"],
        "mae_baseline": m_base["mae"], "rmse_baseline": m_base["rmse"], "bias_baseline": m_base["me"],
        "lift_vs_baseline": _lift(m_base["mae"], m_model["mae"]),
        "coverage_pred_pct": cov,
        "rows": int(len(win)),
        "stations": int(win["station_id"].nunique()),
        "ts_min": win["ts"].min().isoformat(), "ts_max": win["ts"].max().isoformat()
    }

def segments_7d(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    w7 = window_slice(df, 7)
    out: Dict[str, pd.DataFrame] = {}

    by_station = (w7.groupby("station_id")
                    .apply(lambda g: pd.Series({
                        "mae_model": _metrics(g["y_true"], g["y_pred"])["mae"],
                        "mae_baseline": _metrics(g["y_true"], g["y_pred_baseline"])["mae"],
                        "rmse_model": _metrics(g["y_true"], g["y_pred"])["rmse"],
                        "coverage_pred_pct": float(g["y_pred"].notna().mean() * 100.0),
                        "n": int(len(g))
                    }))).reset_index()
    by_station["lift_vs_baseline"] = by_station.apply(lambda r: _lift(r["mae_baseline"], r["mae_model"]), axis=1)
    out["station"] = by_station

    by_hour = (w7.groupby("hour")
                 .apply(lambda g: pd.Series({
                     "mae_model": _metrics(g["y_true"], g["y_pred"])["mae"],
                     "mae_baseline": _metrics(g["y_true"], g["y_pred_baseline"])["mae"],
                     "coverage_pred_pct": float(g["y_pred"].notna().mean() * 100.0),
                     "n": int(len(g))
                 }))).reset_index()
    by_hour["lift_vs_baseline"] = by_hour.apply(lambda r: _lift(r["mae_baseline"], r["mae_model"]), axis=1)
    out["hour"] = by_hour

    if STATION_CLUSTERS.exists():
        clusters = pd.read_csv(STATION_CLUSTERS, dtype={"station_id": str})
        if "cluster" in clusters.columns:
            tmp = w7.merge(clusters[["station_id","cluster"]], on="station_id", how="left")
            if tmp["cluster"].notna().any():
                by_cluster = (tmp.dropna(subset=["cluster"])
                                .groupby("cluster")
                                .apply(lambda g: pd.Series({
                                    "mae_model": _metrics(g["y_true"], g["y_pred"])["mae"],
                                    "mae_baseline": _metrics(g["y_true"], g["y_pred_baseline"])["mae"],
                                    "coverage_pred_pct": float(g["y_pred"].notna().mean() * 100.0),
                                    "n": int(len(g))
                                }))).reset_index()
                by_cluster["lift_vs_baseline"] = by_cluster.apply(lambda r: _lift(r["mae_baseline"], r["mae_model"]), axis=1)
                out["cluster"] = by_cluster

    if "lat" in w7.columns and "lon" in w7.columns and w7["lat"].notna().any():
        w7 = w7.assign(zone=_assign_zone(w7))
        by_zone = (w7.groupby("zone")
                     .apply(lambda g: pd.Series({
                         "mae_model": _metrics(g["y_true"], g["y_pred"])["mae"],
                         "mae_baseline": _metrics(g["y_true"], g["y_pred_baseline"])["mae"],
                         "coverage_pred_pct": float(g["y_pred"].notna().mean() * 100.0),
                         "n": int(len(g))
                     }))).reset_index()
        by_zone["lift_vs_baseline"] = by_zone.apply(lambda r: _lift(r["mae_baseline"], r["mae_model"]), axis=1)
        out["zone"] = by_zone

    return out

def calibration_7d(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    w7 = window_slice(df, 7)
    w7 = w7[w7["y_pred"].notna()].copy()
    if w7.empty or len(w7) < 2:
        return pd.DataFrame(columns=["alpha","beta","n"]), pd.DataFrame(columns=["hour","alpha","beta","n"])

    b, a = np.polyfit(w7["y_pred"].astype(float), w7["y_true"].astype(float), 1)
    cal_glob = pd.DataFrame([{"alpha": float(a), "beta": float(b), "n": int(len(w7))}])

    def _fit_ab(g: pd.DataFrame) -> Tuple[float, float]:
        if len(g) > 1:
            coeffs = np.polyfit(g["y_pred"].astype(float), g["y_true"].astype(float), 1)
            return float(coeffs[1]), float(coeffs[0])
        return np.nan, np.nan

    by_hour = (w7.groupby("hour")
                .apply(lambda g: pd.Series({
                    "alpha": _fit_ab(g)[0],
                    "beta":  _fit_ab(g)[1],
                    "n": int(len(g))
                }))).reset_index()
    return cal_glob, by_hour

def coverage_recent(df: pd.DataFrame) -> pd.DataFrame:
    last_day = df["date_local"].max()
    j1 = df[df["date_local"] == last_day]
    cov_j1 = float(j1["y_pred"].notna().mean() * 100.0) if not j1.empty else np.nan
    w7 = window_slice(df, 7)
    w28 = window_slice(df, 28)
    cov7 = float(w7["y_pred"].notna().mean() * 100.0) if not w7.empty else np.nan
    cov28 = float(w28["y_pred"].notna().mean() * 100.0) if not w28.empty else np.nan
    return pd.DataFrame([{"coverage_j1_pct": cov_j1, "coverage_7d_pct": cov7, "coverage_28d_pct": cov28,
                          "last_day": str(last_day)}])

def residuals_acf(df: pd.DataFrame, nlags_steps: int = 48, bin_min: int = 5) -> pd.DataFrame:
    mask = df["y_pred"].notna()
    e = (df.loc[mask].groupby("ts").apply(lambda g: float((g["y_true"] - g["y_pred"]).mean()))).sort_index()
    ac = _acf(e, nlags=nlags_steps)
    return pd.DataFrame({f"lag_{int(bin_min)}min": list(range(len(ac))), "acf": ac})

def top_degrading_stations(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    seg = segments_7d(df)
    if "station" not in seg or seg["station"].empty:
        return pd.DataFrame()
    w7 = seg["station"]
    w28 = window_slice(df, 28)
    by_st_28 = (w28.groupby("station_id")
                  .apply(lambda g: pd.Series({
                      "mae_28d": _metrics(g["y_true"], g["y_pred"])["mae"],
                      "mae_base_28d": _metrics(g["y_true"], g["y_pred_baseline"])["mae"],
                      "n_28d": int(len(g))
                  }))).reset_index()
    merged = w7.merge(by_st_28, on="station_id", how="left")

    merged["delta_mae_pct"] = _safe_div(merged["mae_model"] - merged["mae_28d"], merged["mae_28d"]) * 100.0
    merged["lift_28d"] = _safe_div(merged["mae_base_28d"] - merged["mae_28d"], merged["mae_base_28d"])
    merged["delta_lift_pts"] = (merged["lift_vs_baseline"] - merged["lift_28d"]) * 100.0

    out = (merged.sort_values(["delta_mae_pct","delta_lift_pts"], ascending=[False, True])
                  .head(top_n)
                  .reset_index(drop=True))
    return out


# --------------------------- Alerts & Policies ---------------------------

def compute_alerts(daily: pd.DataFrame, win7: dict, win28: dict) -> pd.DataFrame:
    degr = None
    if win7 and win28 and np.isfinite(win7.get("mae_model", np.nan)) and np.isfinite(win28.get("mae_model", np.nan)):
        cond_mae = (win7["mae_model"] - win28["mae_model"]) / (win28["mae_model"] + 1e-12) > 0.10
        cond_lift = (win7.get("lift_vs_baseline", np.nan) < (win28.get("lift_vs_baseline", np.nan) - 0.05))
        degr = bool(cond_mae and cond_lift)

    cov_last = daily.sort_values("date").iloc[-1]["coverage_pred_pct"] if not daily.empty else np.nan
    coverage_alert = bool(np.isfinite(cov_last) and cov_last < 99.0)

    last10 = daily.sort_values("date").tail(10).copy()
    if not last10.empty and win28 and np.isfinite(win28.get("mae_model", np.nan)):
        mae28 = win28["mae_model"]; lift28 = win28.get("lift_vs_baseline", np.nan)
        last10["degr_day"] = (last10["mae_model"] > 1.1 * mae28) & (last10["lift_vs_baseline"] < (lift28 - 0.05))
        retrain_gate = bool(last10["degr_day"].sum() >= 3)
    else:
        retrain_gate = False

    last3 = daily.sort_values("date").tail(3)
    if not last3.empty:
        cond_cov = (last3["coverage_pred_pct"] < 95.0).sum() >= 3
        cond_lift = (last3["lift_vs_baseline"] < 0.0).sum() >= 3
        fallback = bool(cond_cov or cond_lift)
    else:
        fallback = False

    return pd.DataFrame([{
        "degradation_window_alert": degr,
        "coverage_alert_j1": coverage_alert,
        "calibration_alert": None,  # set later
        "retrain_gate": retrain_gate,
        "fallback_to_baseline": fallback,
        "mae7": win7.get("mae_model", np.nan) if win7 else np.nan,
        "mae28": win28.get("mae_model", np.nan) if win28 else np.nan,
        "lift7": win7.get("lift_vs_baseline", np.nan) if win7 else np.nan,
        "lift28": win28.get("lift_vs_baseline", np.nan) if win28 else np.nan,
        "coverage_last_pct": cov_last
    }])


# --------------------------- Figures ---------------------------

def plot_daily_series(daily: pd.DataFrame) -> None:
    if daily.empty:
        return
    plt.figure(figsize=(10, 3.4))
    plt.plot(daily["date"].astype(str), daily["mae_model"], label="MAE model")
    plt.plot(daily["date"].astype(str), daily["mae_baseline"], label="MAE baseline")
    plt.title("Daily MAE — model vs baseline")
    plt.xlabel("Date"); plt.ylabel("MAE")
    plt.legend(loc="best")
    _save_fig(FIGS_DIR / "mae_rmse_daily.png")

    plt.figure(figsize=(10, 3.2))
    plt.plot(daily["date"].astype(str), daily["lift_vs_baseline"] * 100.0, marker="o")
    plt.axhline(0.0, linewidth=1)
    plt.title("Daily lift (points)")
    plt.xlabel("Date"); plt.ylabel("points (100×lift)")
    _save_fig(FIGS_DIR / "lift_daily.png")

    plt.figure(figsize=(10, 3.0))
    plt.plot(daily["date"].astype(str), daily["coverage_pred_pct"], marker="o")
    plt.axhline(99.0, linewidth=1)
    plt.axhline(95.0, linewidth=1)
    plt.title("Prediction coverage (%)")
    plt.xlabel("Date"); plt.ylabel("%")
    _save_fig(FIGS_DIR / "coverage_daily.png")

def plot_calibration_by_hour(cal_hour: pd.DataFrame) -> None:
    if cal_hour.empty:
        return
    plt.figure(figsize=(8.5, 3.2))
    plt.plot(cal_hour["hour"], cal_hour["beta"], marker="o", label="β (slope)")
    plt.axhline(1.0, linewidth=1)
    plt.title("Calibration — slope by hour (7d)")
    plt.xlabel("Hour"); plt.ylabel("β")
    plt.legend(loc="best")
    _save_fig(FIGS_DIR / "calibration_beta_by_hour_7d.png")

def plot_acf(acf_df: pd.DataFrame, bin_min: int = 5) -> None:
    if acf_df.empty:
        return
    plt.figure(figsize=(8, 3.0))
    lag_col = f"lag_{int(bin_min)}min"
    if lag_col not in acf_df.columns:  # garde-fou
        lag_col = acf_df.columns[0]
    plt.bar(acf_df[lag_col], acf_df["acf"])
    plt.title("ACF of mean residuals")
    plt.xlabel(f"Lag ({int(bin_min)}-min steps)"); plt.ylabel("Autocorr.")
    _save_fig(FIGS_DIR / "residuals_acf.png")

def plot_top_degrading(df: pd.DataFrame) -> None:
    if df.empty:
        return
    top = df.head(20)
    plt.figure(figsize=(10, 4.0))
    labels = top["station_id"].astype(str)
    plt.barh(labels.iloc[::-1], top["delta_mae_pct"].iloc[::-1])
    plt.title("Top degrading stations (ΔMAE 7d vs 28d, %)")
    plt.xlabel("ΔMAE (%)")
    _save_fig(FIGS_DIR / "top_degrading_stations.png")

def map_error_by_station_7d(df: pd.DataFrame) -> None:
    # Enrichit lat/lon si nécessaire
    df = _ensure_latlon(df)

    w7 = window_slice(df, 7)
    if not HAS_FOLIUM:
        print("[model-health/map] Folium absent → carte non générée.")
        return
    if "lat" not in w7.columns or "lon" not in w7.columns:
        print("[model-health/map] Colonnes lat/lon absentes → carte non générée.")
        return
    if w7["lat"].isna().all() or w7["lon"].isna().all():
        print("[model-health/map] Aucune coordonnée valide → carte non générée.")
        return

    by_st = (w7.dropna(subset=["lat","lon"])
               .groupby(["station_id","lat","lon"], as_index=False)
               .apply(lambda g: pd.Series({
                   "mae": _metrics(g["y_true"], g["y_pred"])["mae"],
                   "bias": _metrics(g["y_true"], g["y_pred"])["me"],
                   "n": int(len(g))
               }))
               .reset_index(drop=True))

    if by_st.empty:
        print("[model-health/map] Pas de points agrégés → carte non générée.")
        return

    # Stats pour la légende
    ms = by_st["mae"].astype(float).dropna()
    mae_min = float(ms.min()) if not ms.empty else float("nan")
    mae_med = float(ms.median()) if not ms.empty else float("nan")
    mae_max = float(ms.max()) if not ms.empty else float("nan")

    lat0 = float(by_st["lat"].median()); lon0 = float(by_st["lon"].median())
    m = folium.Map(location=[lat0, lon0], zoom_start=12, tiles="cartodbpositron")

    for _, r in by_st.iterrows():
        col = "red" if (pd.notna(r['bias']) and r["bias"] > 0) else "blue"
        rad = 3 + (float(r["mae"]) if pd.notna(r["mae"]) else 0.0)  # taille ∝ MAE (capée ci-dessous)
        rad = max(3.0, min(13.0, rad))
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=rad, color=col, fill=True, fill_opacity=0.85,
            tooltip=f"{r['station_id']} • MAE={_fmt_float(r['mae'],2)} • biais={_fmt_float(r['bias'],2)} • n={int(r['n'])}"
        ).add_to(m)

    # Légende (HTML injecté)
    if Element is not None:
        legend_html = f"""
<div style="
  position: fixed; bottom: 20px; right: 20px; z-index: 9999;
  background: white; padding: 10px 12px; border: 1px solid #bbb;
  border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.2);
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  font-size: 12px; line-height: 1.25;">
  <div style="font-weight: 600; margin-bottom: 6px;">Légende</div>
  <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
    <span style="width:12px;height:12px;border-radius:50%;background:#e41a1c;display:inline-block;border:1px solid #999;"></span>
    <span>Biais &gt; 0 (rouge)</span>
  </div>
  <div style="display:flex; align-items:center; gap:6px; margin-bottom:6px;">
    <span style="width:12px;height:12px;border-radius:50%;background:#377eb8;display:inline-block;border:1px solid #999;"></span>
    <span>Biais ≤ 0 (bleu)</span>
  </div>
  <div style="margin-top:4px;">Taille ∝ MAE (capée à 13 px)</div>
  <div style="margin-top:4px; opacity:.8;">MAE min/med/max&nbsp;: {mae_min:.2f} / {mae_med:.2f} / {mae_max:.2f}</div>
</div>
""".strip()
        m.get_root().html.add_child(Element(legend_html))

    out_path = MAPS_DIR / "model_error_by_station_7d.html"
    m.save(str(out_path))
    print(f"[model-health/map] Carte → {out_path}")


# --------------------------- Markdown rendering ---------------------------

MD_TEMPLATE = """# Santé du modèle

**Objectif.** Surveiller la **performance dans le temps**, la **calibration** et la **couverture** des prédictions, pour décider d’un **ré-entraînement** ou d’un **fallback**.

## Questions auxquelles la page répond
- L’erreur (MAE/RMSE) se **dégrade-t-elle** ? Où (heures, clusters, zones) ?
- Le **lift** vs baseline reste-t-il positif et stable ?
- La **calibration** est-elle correcte globalement et par segments ?
- La **couverture** (`% y_pred` disponible) est-elle conforme ?

## Indicateurs clés
- **MAE / RMSE / biais** par jour et par segments (heure, station, cluster, zone).
- **Lift** vs persistance = `(MAE_base − MAE_model) / MAE_base`.
- **Calibration** : pente/intercept `y_true ~ y_pred` (global & segments).
- **Couverture prédictive** : part d’horodatages avec `y_pred`.
- **Stabilité résiduelle** : variance des résidus, auto-corrélation.

---

## Résumé rapide (fenêtres récentes)
- **Fenêtre 7j** — MAE: **{mae7}**, Lift: **{lift7}**, Couverture: **{cov7}%**  
- **Fenêtre 28j** — MAE: **{mae28}**, Lift: **{lift28}**, Couverture: **{cov28}%**  
- **Dernier jour** — Couverture prédictive: **{cov_last}%**  
- **Alertes** : Dégradation {alert_deg} · Couverture {alert_cov} · Calibration {alert_cal} · Fallback {alert_fb} · Gating retrain {alert_retrain}

> **Plage temporelle analysée** : {ts_min} → {ts_max} (timezone: **{tz}**) · Horizon: **{horizon_label}** · Série quotidienne exportée: **{last_days} jours**

---

## Visualisations
### Séries (MAE / lift / couverture)
![MAE]({fig_mae})
![Lift]({fig_lift})
![Couverture]({fig_cov})

### Calibration (pente β par heure, 7j)
![Calibration β]({fig_calib})

### Stabilité des résidus
![ACF]({fig_acf})

### Top stations en dégradation
![Top stations]({fig_topdeg})

### Carte d’erreur (7 derniers jours)
{map_block}


---

## Téléchargements & tables
- **Fenêtres** : `{win_csv}`
- **Jours** : `{daily_csv}`
- **Segments 7j** : stations `{seg_st_csv}`, heures `{seg_hr_csv}`{seg_cl_csv}{seg_zo_csv}
- **Calibration** : global `{cal_glob_csv}` ; par heure `{cal_hr_csv}`
- **Couverture** : `{cov_csv}`
- **ACF résidus** : `{acf_csv}`
- **Top dégradations** : `{topdeg_csv}`
- **Synthèse alertes** : `{alerts_csv}`

---

## Seuils / Politiques (par défaut)
- **Dégradation** : MAE_7j − MAE_28j > 10 % **et** lift_7j < lift_28j − 5 pts → **Alerte**.
- **Couverture** : < 99 % sur J-1 → **Alerte**.
- **Calibration** : |pente−1| > 0,1 **ou** |intercept| > 0,5 → **Alerte**.
- **Gating ré-entraînement** : 3 alertes “Dégradation” sur 10 jours → **Planifier retrain**.
- **Fallback** : couverture < 95 % **ou** lift < 0 sur 3 jours → activer **baseline** le temps du correctif.

> **Limites** — La performance agrégée peut masquer des **poches locales** de dégradation → toujours lire les découpages.
"""

def _fmt_float(x: float, nd=3) -> str:
    return "—" if not np.isfinite(x) else f"{x:.{nd}f}"

def _fmt_pct(x: float, nd=2) -> str:
    return "—" if not np.isfinite(x) else f"{x:.{nd}f}"

def _flag(b: Optional[bool]) -> str:
    if b is True: return "⚠️"
    if b is False: return "✅"
    return "—"

def _write_markdown(
    tz: Optional[str],
    horizon: Optional[int],
    last_days: int,
    win7: dict,
    win28: dict,
    alerts: pd.DataFrame,
    have_cluster: bool,
    have_zone: bool,
    bin_min: int,   # réservé si besoin d'afficher le pas plus tard
) -> None:
    # Chemins relatifs depuis la page MD
    fig_mae   = _rel_from_md(OUT_MD, FIGS_DIR / "mae_rmse_daily.png")
    fig_lift  = _rel_from_md(OUT_MD, FIGS_DIR / "lift_daily.png")
    fig_cov   = _rel_from_md(OUT_MD, FIGS_DIR / "coverage_daily.png")
    fig_calib = _rel_from_md(OUT_MD, FIGS_DIR / "calibration_beta_by_hour_7d.png")
    fig_acf   = _rel_from_md(OUT_MD, FIGS_DIR / "residuals_acf.png")
    fig_topdeg= _rel_from_md(OUT_MD, FIGS_DIR / "top_degrading_stations.png")

    # Carte: construire un bloc conditionnel
    map_path   = MAPS_DIR / "model_error_by_station_7d.html"
    map_exists = map_path.exists()
    if map_exists:
        map_rel = _rel_from_md(OUT_MD, map_path)
        map_block = f"""
<div style="margin: 0.5rem 0;">
  <iframe src="{map_rel}" style="width:100%;height:520px;border:0" loading="lazy" title="Carte erreur 7j"></iframe>
</div>
<p><a href="{map_rel}" target="_blank" rel="noopener">Ouvrir la carte dans un nouvel onglet ↗</a></p>
""".strip()
    else:
        map_block = "_Carte non disponible (coords manquantes ou Folium absent)._"

    daily_csv = _rel_from_md(OUT_MD, TABLES_DIR / "daily_metrics.csv")
    win_csv   = _rel_from_md(OUT_MD, TABLES_DIR / "window_metrics.csv")
    seg_st_csv= _rel_from_md(OUT_MD, TABLES_DIR / "error_by_station_7d.csv")
    seg_hr_csv= _rel_from_md(OUT_MD, TABLES_DIR / "error_by_hour_7d.csv")
    seg_cl_csv= f", clusters `{_rel_from_md(OUT_MD, TABLES_DIR / 'error_by_cluster_7d.csv')}`" if have_cluster else ""
    seg_zo_csv= f", zones `{_rel_from_md(OUT_MD, TABLES_DIR / 'error_by_zone_7d.csv')}`" if have_zone else ""
    cal_glob_csv = _rel_from_md(OUT_MD, TABLES_DIR / "calibration_global_7d.csv")
    cal_hr_csv   = _rel_from_md(OUT_MD, TABLES_DIR / "calibration_by_hour_7d.csv")
    cov_csv      = _rel_from_md(OUT_MD, TABLES_DIR / "coverage_j1_j7_j28.csv")
    acf_csv      = _rel_from_md(OUT_MD, TABLES_DIR / "residuals_acf.csv")
    topdeg_csv   = _rel_from_md(OUT_MD, TABLES_DIR / "top_degrading_stations.csv")
    alerts_csv   = _rel_from_md(OUT_MD, TABLES_DIR / "alerts_summary.csv")

    # Valeurs
    mae7   = _fmt_float(win7.get("mae_model", np.nan))
    mae28  = _fmt_float(win28.get("mae_model", np.nan))
    lift7  = _fmt_float(win7.get("lift_vs_baseline", np.nan))
    lift28 = _fmt_float(win28.get("lift_vs_baseline", np.nan))
    cov7   = _fmt_pct(win7.get("coverage_pred_pct", np.nan))
    cov28  = _fmt_pct(win28.get("coverage_pred_pct", np.nan))
    ts_min = win28.get("ts_min", "—") or "—"
    ts_max = win28.get("ts_max", "—") or "—"
    tzlabel = tz or "UTC"
    horlabel = f"{int(horizon)} min" if horizon is not None else "—"

    alert_row = alerts.iloc[0] if not alerts.empty else None
    alert_deg = _flag(alert_row["degradation_window_alert"]) if alert_row is not None else "—"
    alert_cov = _flag(alert_row["coverage_alert_j1"])       if alert_row is not None else "—"
    alert_cal = _flag(alert_row.get("calibration_alert"))   if alert_row is not None else "—"
    alert_fb  = _flag(alert_row["fallback_to_baseline"])    if alert_row is not None else "—"
    alert_rt  = _flag(alert_row["retrain_gate"])            if alert_row is not None else "—"
    cov_last  = _fmt_pct(alert_row["coverage_last_pct"])    if alert_row is not None else "—"

    md = MD_TEMPLATE.format(
        mae7=mae7, mae28=mae28, lift7=lift7, lift28=lift28,
        cov7=cov7, cov28=cov28, cov_last=cov_last,
        alert_deg=alert_deg, alert_cov=alert_cov, alert_cal=alert_cal, alert_fb=alert_fb, alert_retrain=alert_rt,
        ts_min=ts_min, ts_max=ts_max, tz=tzlabel, horizon_label=horlabel, last_days=last_days,
        fig_mae=fig_mae, fig_lift=fig_lift, fig_cov=fig_cov, fig_calib=fig_calib, fig_acf=fig_acf, fig_topdeg=fig_topdeg,
        # fichiers
        win_csv=win_csv, daily_csv=daily_csv,
        seg_st_csv=seg_st_csv, seg_hr_csv=seg_hr_csv, seg_cl_csv=seg_cl_csv, seg_zo_csv=seg_zo_csv,
        cal_glob_csv=cal_glob_csv, cal_hr_csv=cal_hr_csv, cov_csv=cov_csv, acf_csv=acf_csv, topdeg_csv=topdeg_csv, alerts_csv=alerts_csv,
        map_block=map_block,
    )

    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    print(f"[model-health] Markdown -> {OUT_MD}")

# --------------------------- Main ---------------------------

def main(perf_path: Path, last_days: int, tz: Optional[str], horizon: Optional[int], bin_min: int) -> None:
    _mkdirs()

    # Load & prepare
    perf = _read_perf(perf_path, bin_min=bin_min)

    # Optional horizon filter (prefer 'horizon_min', else 'horizon' if present)
    if horizon is not None:
        hcol = "horizon_min" if "horizon_min" in perf.columns else ("horizon" if "horizon" in perf.columns else None)
        if hcol is not None:
            perf = perf[pd.to_numeric(perf[hcol], errors="coerce") == float(horizon)].copy()

    perf = _localize(perf, tz=tz)

    # Daily metrics
    daily = daily_metrics(perf, last_days=last_days)
    daily.to_csv(TABLES_DIR / "daily_metrics.csv", index=False)

    # 7d / 28d windows
    win7 = window_metrics(perf, days=7)
    win28 = window_metrics(perf, days=28)
    rows = [w for w in (win7, win28) if w]
    if rows:
        pd.DataFrame(rows).to_csv(TABLES_DIR / "window_metrics.csv", index=False)

    # Segments (7d)
    segs = segments_7d(perf)
    have_cluster = "cluster" in segs
    have_zone = "zone" in segs
    if "station" in segs: segs["station"].to_csv(TABLES_DIR / "error_by_station_7d.csv", index=False)
    if "hour" in segs:    segs["hour"].to_csv(TABLES_DIR / "error_by_hour_7d.csv", index=False)
    if have_cluster:      segs["cluster"].to_csv(TABLES_DIR / "error_by_cluster_7d.csv", index=False)
    if have_zone:         segs["zone"].to_csv(TABLES_DIR / "error_by_zone_7d.csv", index=False)

    # Calibration (7d)
    cal_glob, cal_hour = calibration_7d(perf)
    if not cal_glob.empty:
        cal_glob.to_csv(TABLES_DIR / "calibration_global_7d.csv", index=False)
    if not cal_hour.empty:
        cal_hour.to_csv(TABLES_DIR / "calibration_by_hour_7d.csv", index=False)

    # Coverage J-1/J-7/J-28
    kov = coverage_recent(perf)
    kov.to_csv(TABLES_DIR / "coverage_j1_j7_j28.csv", index=False)

    # Residual stability (ACF)
    acf_df = residuals_acf(perf, nlags_steps=48, bin_min=bin_min)
    if not acf_df.empty:
        acf_df.to_csv(TABLES_DIR / "residuals_acf.csv", index=False)

    # Top degrading stations
    top_deg = top_degrading_stations(perf, top_n=30)
    if not top_deg.empty:
        top_deg.to_csv(TABLES_DIR / "top_degrading_stations.csv", index=False)

    # Alerts
    alerts = compute_alerts(daily, win7, win28)
    # Fill calibration alert with global α/β (7j)
    if not cal_glob.empty:
        a = float(cal_glob.iloc[0]["alpha"]); b = float(cal_glob.iloc[0]["beta"])
        cal_ok = (np.isfinite(b) and abs(b - 1.0) <= 0.1) and (np.isfinite(a) and abs(a) <= 0.5)
        alerts.loc[:, "calibration_alert"] = (not cal_ok)
        alerts.loc[:, "calibration_alpha"] = a
        alerts.loc[:, "calibration_beta"] = b
    alerts.to_csv(TABLES_DIR / "alerts_summary.csv", index=False)

    # Figures
    plot_daily_series(daily)
    plot_calibration_by_hour(cal_hour if not cal_hour.empty else pd.DataFrame())
    plot_acf(acf_df if not acf_df.empty else pd.DataFrame(), bin_min=bin_min)
    plot_top_degrading(top_deg if not top_deg.empty else pd.DataFrame())

    # Map (optional)
    map_error_by_station_7d(perf)

    # Markdown page
    _write_markdown(tz=tz, horizon=horizon, last_days=last_days,
                    win7=win7 or {}, win28=win28 or {}, alerts=alerts,
                    have_cluster=have_cluster, have_zone=have_zone, bin_min=bin_min)

    print("[monitoring/model-health] Done.")
    print(f"[monitoring/model-health] Daily metrics → {TABLES_DIR / 'daily_metrics.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build 'Monitoring / Model health' assets from perf.parquet")
    ap.add_argument("--perf", type=Path, required=True, help="Path to docs/exports/perf.parquet")
    ap.add_argument("--last-days", type=int, default=30, help="Length of the daily series to export")
    ap.add_argument("--tz", type=str, default=None, help="Timezone for hour/day aggregations (e.g. Europe/Paris)")
    ap.add_argument("--horizon", type=int, default=None, help="Optional horizon (minutes) to filter on, if present")
    ap.add_argument("--bin-min", type=int, default=5, help="Ingestion step (minutes) for ts flooring & ACF labeling")
    args = ap.parse_args()

    main(perf_path=args.perf, last_days=args.last_days, tz=args.tz,
         horizon=args.horizon, bin_min=args.bin_min)
