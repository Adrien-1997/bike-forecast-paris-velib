# tools/build_station_profiles.py
# Génère des pages station (Markdown) et figures :
#   - sparkline.png (7j)            [y_true only]
#   - hourly.png (profil horaire)   [y_true vs y_pred DISP aligné]
#   - obs_vs_pred.png (24h)         [y_true vs y_pred DISP aligné]
#   - residual_hist.png             [résiduels avec y_pred DISP aligné]
#
# Règles :
# - Données en UTC ; conversion tz uniquement pour l'affichage (--tz).
# - Comparaison stricte y_pred(T) vs y_true(T) dans les datasets.
# - Fallback y_pred -> y_pred_baseline si besoin.
# - Auto-align d'affichage : si un lag résiduel est détecté (corr max à k≠0),
#   on crée une colonne y_pred_disp = y_pred.shift(k) pour les FIGURES uniquement.
# - CI-safe : backend Agg, protections colonnes manquantes, pas d'échec hard.

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless/CI-safe
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXPORTS = DOCS / "exports"
ASSETS = DOCS / "assets"
STATIONS_DIR = DOCS / "stations"
IMG_STATIONS_DIR = DOCS / "assets" / "figs" / "stations"   # <== nouvel emplacement figures

plt.rcParams.update({"figure.autolayout": True})


# ------------------------- utils -------------------------

def _to_local(ts: pd.Series, tz: str | None) -> pd.Series:
    if tz:
        return pd.to_datetime(ts, utc=True, errors="coerce").dt.tz_convert(tz)
    return pd.to_datetime(ts, utc=False, errors="coerce")


def _ensure_dirs():
    STATIONS_DIR.mkdir(parents=True, exist_ok=True)
    IMG_STATIONS_DIR.mkdir(parents=True, exist_ok=True)


def _best_lag(a: pd.Series, b: pd.Series, max_steps: int = 8) -> tuple[int, float]:
    best_r, best_k = -np.inf, 0
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    for k in range(-max_steps, max_steps + 1):
        r = a.corr(b.shift(k))
        if pd.notna(r) and r > best_r:
            best_r, best_k = r, k
    if best_r == -np.inf:
        return 0, np.nan
    return best_k, float(best_r)


def _safe_group_cov(df: pd.DataFrame) -> pd.Series:
    if "y_pred" not in df.columns:
        return df.groupby("station_id").size().astype(float).mul(0.0)
    return df.groupby("station_id").apply(lambda g: g["y_pred"].notna().mean())


def _station_selector(df: pd.DataFrame, k: int, by: str) -> list[str]:
    if df.empty:
        return []
    cov = _safe_group_cov(df).fillna(0.0)
    vol = df.groupby("station_id")["y_true"].std().fillna(0.0) if "y_true" in df.columns else cov*0.0
    cnt = df.groupby("station_id").size()
    if by == "volatility":
        score = vol
    elif by == "coverage":
        score = cov
    else:
        score = cnt.astype(float)
    score = score * (1.0 + cov)
    return score.sort_values(ascending=False).head(k).index.astype(str).tolist()


def _fallback_pred_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "y_pred" not in out.columns or out["y_pred"].dropna().empty:
        if "y_pred" not in out.columns:
            out["y_pred"] = np.nan
        if "y_pred_baseline" in out.columns:
            print("[WARN] station: y_pred empty → using baseline")
            out["y_pred"] = out["y_pred"].fillna(out["y_pred_baseline"])
    return out


def _apply_display_lag(df_s: pd.DataFrame, max_steps: int = 8) -> tuple[pd.DataFrame, int, float]:
    disp = df_s.copy()
    lag, r = 0, np.nan
    if "y_true" in disp.columns and "y_pred" in disp.columns:
        sub = disp.dropna(subset=["y_true", "y_pred"])
        if len(sub) >= 20:
            lag, r = _best_lag(sub["y_true"], sub["y_pred"], max_steps=max_steps)
    disp["y_pred_disp"] = disp.get("y_pred")
    if lag != 0 and "y_pred" in disp.columns:
        disp["y_pred_disp"] = disp["y_pred"].shift(lag)
    return disp, lag, r


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def _rel_from_station_md(png_path: Path) -> str:
    return Path("..") / png_path.relative_to(DOCS)


# ------------------------- plots par station -------------------------

def plot_sparkline(df_s: pd.DataFrame, tz: str | None, out: Path):
    if df_s.empty or "y_true" not in df_s.columns:
        return
    x = _to_local(df_s["ts"], tz)
    plt.figure(figsize=(8, 2))
    plt.plot(x, df_s["y_true"], linewidth=1)
    plt.title("Observed (y_true) — last window")
    plt.xlabel(f"Time ({tz or 'UTC'})")
    _savefig(out)


def plot_hourly_profile(df_s_disp: pd.DataFrame, tz: str | None, out: Path, lag_info: tuple[int, float] | None):
    if df_s_disp.empty:
        return
    tloc = _to_local(df_s_disp["ts"], tz)
    df = df_s_disp.copy()
    df["hour"] = tloc.dt.hour
    cols = ["y_true"] + (["y_pred_disp"] if "y_pred_disp" in df.columns else (["y_pred"] if "y_pred" in df.columns else []))
    prof = df.groupby("hour")[cols].mean().dropna(how="all")
    if prof.empty:
        return
    plt.figure(figsize=(8, 3))
    if "y_true" in prof.columns:
        plt.plot(prof.index, prof["y_true"], label="Observed")
    ypred_col = "y_pred_disp" if "y_pred_disp" in prof.columns else ("y_pred" if "y_pred" in prof.columns else None)
    if ypred_col:
        plt.plot(prof.index, prof[ypred_col], label="Predicted")
    title = "Hourly profile (mean)"
    if lag_info:
        k, r = lag_info
        if not np.isnan(r):
            title += f" — lag*={k}×15min, corr={r:.3f}"
    plt.title(title)
    plt.xlabel("Hour"); plt.ylabel("Bikes")
    plt.xticks(range(0, 24, 2))
    plt.legend()
    _savefig(out)


def plot_obs_vs_pred_24h(df_s_disp: pd.DataFrame, tz: str | None, out: Path, lag_info: tuple[int, float] | None):
    if df_s_disp.empty or "y_true" not in df_s_disp.columns:
        return
    tmax = df_s_disp["ts"].max()
    sub = df_s_disp[df_s_disp["ts"] >= (tmax - pd.Timedelta(hours=24))].copy()
    ypred_col = "y_pred_disp" if "y_pred_disp" in sub.columns else ("y_pred" if "y_pred" in sub.columns else None)
    if ypred_col is None:
        return
    sub = sub.dropna(subset=["y_true", ypred_col])
    if sub.empty:
        return

    x = _to_local(sub["ts"], tz)
    plt.figure(figsize=(8, 3))
    plt.plot(x, sub["y_true"], label="Observed (y_true)")
    plt.plot(x, sub[ypred_col], label="Predicted (y_pred)")
    title = "Last 24h"
    if lag_info:
        k, r = lag_info
        if not np.isnan(r):
            title += f" — lag*={k}×15min, corr={r:.3f}"
    plt.title(title)
    plt.xlabel(f"Time ({tz or 'UTC'})"); plt.ylabel("Bikes")
    plt.legend()
    _savefig(out)


def plot_residual_hist(df_s_disp: pd.DataFrame, out: Path):
    ypred_col = "y_pred_disp" if "y_pred_disp" in df_s_disp.columns else ("y_pred" if "y_pred" in df_s_disp.columns else None)
    if df_s_disp.empty or ypred_col is None or "y_true" not in df_s_disp.columns:
        return
    sub = df_s_disp.dropna(subset=["y_true", ypred_col]).copy()
    if sub.empty:
        return
    resid = (sub[ypred_col] - sub["y_true"]).astype(float)
    plt.figure(figsize=(6, 3))
    plt.hist(resid, bins=40)
    plt.title("Residuals (y_pred - y_true) [display-aligned]")
    plt.xlabel("Residual"); plt.ylabel("Count")
    _savefig(out)


# ------------------------- page markdown -------------------------

def write_md(station_id: str, name: str | None, files: dict[str, Path], out_md: Path, lag_info: tuple[int, float] | None):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    title = f"Station {station_id}" + (f" — {name}" if name else "")
    lag_line = ""
    if lag_info:
        k, r = lag_info
        if not np.isnan(r):
            lag_line = f"\n> Display alignment applied: `lag* = {k} × 15min`, corr ≈ {r:.3f}\n"

    spark_rel = _rel_from_station_md(files["spark"]).as_posix()
    hourly_rel = _rel_from_station_md(files["hourly"]).as_posix()
    ovsp_rel = _rel_from_station_md(files["ovsp"]).as_posix()
    resid_rel = _rel_from_station_md(files["resid"]).as_posix()

    md = f"""# {title}
{lag_line}
Figures:

- Sparkline (7d): ![]({spark_rel})
- Hourly profile: ![]({hourly_rel})
- Observed vs Predicted (24h): ![]({ovsp_rel})
- Residuals: ![]({resid_rel})
"""
    out_md.write_text(md, encoding="utf-8")


# ------------------------- main -------------------------

def _run(events_path: Path, perf_path: Path, last_days: int, hours: int, select_k: int, by: str, tz: str | None, auto_align_display: bool = True) -> int:
    _ensure_dirs()

    try:
        perf = pd.read_parquet(perf_path)
    except Exception as e:
        print(f"[ERR] cannot read perf: {e}")
        return 0
    if "ts" not in perf.columns or "station_id" not in perf.columns:
        print("[ERR] perf is missing required columns ['ts','station_id']")
        return 0

    perf["ts"] = pd.to_datetime(perf["ts"], utc=False, errors="coerce")
    perf["station_id"] = perf["station_id"].astype(str)

    perf_full = perf.copy()
    if last_days and last_days > 0:
        tmax = perf_full["ts"].max()
        if pd.notna(tmax):
            perf_full = perf_full[perf_full["ts"] >= (tmax - pd.Timedelta(days=last_days))].copy()
    if perf_full.empty:
        print("[WARN] No data in last_days window; nothing to build.")
        return 0

    has_pred = ("y_pred" in perf_full.columns) and perf_full["y_pred"].notna().any()
    if not has_pred and "y_pred_baseline" in perf_full.columns:
        print("[WARN] y_pred empty globally → using baseline everywhere")
        perf_full["y_pred"] = perf_full["y_pred_baseline"]

    selected = _station_selector(perf_full, k=select_k, by=by)
    if not selected:
        selected = (perf_full["station_id"].value_counts().head(select_k).index.astype(str).tolist())
    print(f"[INFO] Selected {len(selected)} stations: {selected[:6]}{'...' if len(selected)>6 else ''}")

    perf_plot = perf_full.copy()
    if hours and hours > 0:
        tmax = perf_plot["ts"].max()
        perf_plot = perf_plot[perf_plot["ts"] >= (tmax - pd.Timedelta(hours=hours))].copy()

    names = {}
    try:
        ev = pd.read_parquet(events_path, columns=["station_id", "name"])
        ev = ev.dropna(subset=["station_id"])
        ev["station_id"] = ev["station_id"].astype(str)
        names = ev.drop_duplicates(["station_id"], keep="last").set_index("station_id")["name"].to_dict()
    except Exception:
        print("[INFO] No station names found in events (optional).")

    ok_pages = 0
    for sid in selected:
        try:
            df_s = perf_plot[perf_plot["station_id"] == str(sid)].copy()
            if df_s.empty:
                print(f"[WARN] station {sid}: no data in plotting window; skip")
                continue
            df_s = _fallback_pred_if_needed(df_s)

            if auto_align_display:
                df_disp, k, r = _apply_display_lag(df_s, max_steps=8)
                lag_info = (k, r)
            else:
                df_disp, lag_info = df_s.copy(), None

            imgdir = IMG_STATIONS_DIR / f"{sid}"
            files = {
                "spark": imgdir / "sparkline.png",
                "hourly": imgdir / "hourly.png",
                "ovsp":   imgdir / "obs_vs_pred.png",
                "resid":  imgdir / "residual_hist.png",
            }

            plot_sparkline(df_s, tz, files["spark"])
            plot_hourly_profile(df_disp, tz, files["hourly"], lag_info)
            plot_obs_vs_pred_24h(df_disp, tz, files["ovsp"], lag_info)
            plot_residual_hist(df_disp, files["resid"])

            write_md(sid, names.get(str(sid)), files, STATIONS_DIR / f"{sid}.md", lag_info)
            ok_pages += 1
        except Exception as e:
            print(f"[ERR] station {sid}: {e}; skipping")

    print("[OK] Station profiles generated:")
    print(f" - pages: {ok_pages} -> {STATIONS_DIR / '*.md'}")
    print(" - figs per station: sparkline.png | hourly.png | obs_vs_pred.png | residual_hist.png")
    return ok_pages


def main(events_path: Path, perf_path: Path, last_days: int, hours: int, select_k: int, by: str, tz: str | None, auto_align_display: bool = True):
    try:
        _run(events_path=events_path, perf_path=perf_path, last_days=last_days, hours=hours, select_k=select_k, by=by, tz=tz, auto_align_display=auto_align_display)
        return 0
    except Exception as e:
        print(f"[ERR] build_station_profiles failed: {e}")
        traceback.print_exc()
        return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build per-station pages with figures")
    ap.add_argument("--events", type=Path, default=EXPORTS / "events.parquet")
    ap.add_argument("--perf", type=Path, default=EXPORTS / "perf.parquet")
    ap.add_argument("--last-days", type=int, default=7)
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--select", type=int, default=12)
    ap.add_argument("--by", type=str, default="volatility", choices=["volatility", "coverage", "count"])
    ap.add_argument("--tz", type=str, default=None)
    ap.add_argument("--no-auto-align-display", action="store_true")
    args = ap.parse_args()
    sys.exit(
        main(
            events_path=args.events,
            perf_path=args.perf,
            last_days=args.last_days,
            hours=args.hours,
            select_k=args.select,
            by=args.by,
            tz=args.tz,
            auto_align_display=not args.no_auto_align_display,
        )
    )
