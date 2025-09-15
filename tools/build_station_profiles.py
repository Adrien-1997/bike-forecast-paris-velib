# tools/build_station_profiles.py
# Génère des pages station (Markdown) et figures :
#   - sparkline.png (7j)
#   - hourly.png (profil horaire moyen)
#   - obs_vs_pred.png (24h, auto-align display si décalage détecté)
#   - residual_hist.png
#
# Règles :
# - Données en UTC ; conversion tz uniquement pour l'affichage (--tz).
# - Comparaison stricte y_pred(T) vs y_true(T).
# - Fallback y_pred -> y_pred_baseline si besoin.
# - Auto-align d'affichage : décale y_pred dans les FIGURES si un lag résiduel est détecté.
# - CI-safe : backend Agg, protections contre colonnes manquantes, et pas d'échec hard.

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

plt.rcParams.update({"figure.autolayout": True})


# ------------------------- utils -------------------------

def _to_local(ts: pd.Series, tz: str | None) -> pd.Series:
    if tz:
        # suppose ts en UTC naive → localise en UTC puis convertit
        return pd.to_datetime(ts, utc=True, errors="coerce").dt.tz_convert(tz)
    return pd.to_datetime(ts, utc=False, errors="coerce")


def _ensure_dirs():
    STATIONS_DIR.mkdir(parents=True, exist_ok=True)


def _best_lag(a: pd.Series, b: pd.Series, max_steps: int = 8) -> tuple[int, float]:
    """Trouve le décalage (en steps de 15 min) qui maximise corr(y_true, y_pred.shift(k))."""
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
    """Couverture y_pred par station, sans exiger l'existence de la colonne y_pred."""
    if "y_pred" not in df.columns:
        return df.groupby("station_id").size().astype(float).mul(0.0)
    return df.groupby("station_id").apply(lambda g: g["y_pred"].notna().mean())


def _station_selector(df: pd.DataFrame, k: int, by: str) -> list[str]:
    """Sélectionne k stations selon un critère, en pénalisant la faible couverture y_pred."""
    if df.empty:
        return []
    cov = _safe_group_cov(df).fillna(0.0)
    vol = df.groupby("station_id")["y_true"].std(min_count=10).fillna(0.0) if "y_true" in df.columns else cov*0.0
    cnt = df.groupby("station_id").size()
    if by == "volatility":
        score = vol
    elif by == "coverage":
        score = cov
    else:
        score = cnt.astype(float)
    score = score * (1.0 + cov)  # boost stations bien couvertes en y_pred
    return score.sort_values(ascending=False).head(k).index.astype(str).tolist()


def _fallback_pred_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Si y_pred est vide pour une station, utiliser la baseline (sans écraser les valeurs déjà présentes)."""
    out = df.copy()
    if "y_pred" not in out.columns or out["y_pred"].dropna().empty:
        if "y_pred" not in out.columns:
            out["y_pred"] = np.nan
        if "y_pred_baseline" in out.columns:
            print("[WARN] station: y_pred empty → using baseline")
            out["y_pred"] = out["y_pred"].fillna(out["y_pred_baseline"])
    return out


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


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


def plot_hourly_profile(df_s: pd.DataFrame, tz: str | None, out: Path):
    if df_s.empty:
        return
    tloc = _to_local(df_s["ts"], tz)
    df = df_s.copy()
    df["hour"] = tloc.dt.hour
    cols = [c for c in ["y_true", "y_pred"] if c in df.columns]
    if not cols:
        return
    prof = df.groupby("hour")[cols].mean().dropna(how="all")
    if prof.empty:
        return
    plt.figure(figsize=(8, 3))
    if "y_true" in prof.columns:
        plt.plot(prof.index, prof["y_true"], label="Observed")
    if "y_pred" in prof.columns:
        plt.plot(prof.index, prof["y_pred"], label="Predicted")
    plt.title("Hourly profile (mean)")
    plt.xlabel("Hour"); plt.ylabel("Bikes")
    plt.xticks(range(0, 24, 2))
    plt.legend()
    _savefig(out)


def plot_obs_vs_pred_24h(df_s: pd.DataFrame, tz: str | None, out: Path, auto_align_display: bool = True):
    """
    Trace y_true vs y_pred sur 24h pour une station.
    Si auto_align_display=True, détecte un lag résiduel et décale y_pred pour l'AFFICHAGE uniquement.
    """
    if df_s.empty or "y_true" not in df_s.columns:
        return
    tmax = df_s["ts"].max()
    sub = df_s[df_s["ts"] >= (tmax - pd.Timedelta(hours=24))].copy()
    sub = sub.dropna(subset=["y_true"])
    if sub.empty or "y_pred" not in sub.columns:
        return
    sub = sub.dropna(subset=["y_pred"])
    if sub.empty:
        return

    lag, r = 0, np.nan
    disp = sub.copy()
    if auto_align_display:
        lag, r = _best_lag(sub["y_true"], sub["y_pred"], max_steps=8)
        if lag != 0:
            disp["y_pred"] = disp["y_pred"].shift(lag)
            disp = disp.dropna(subset=["y_true", "y_pred"])

    if disp.empty:
        return

    x = _to_local(disp["ts"], tz)
    plt.figure(figsize=(8, 3))
    plt.plot(x, disp["y_true"], label="Observed (y_true)")
    plt.plot(x, disp["y_pred"], label="Predicted (y_pred)")
    title = "Last 24h"
    if auto_align_display and not np.isnan(r):
        title += f" — lag*={lag} steps (15min), corr={r:.3f}"
    plt.title(title)
    plt.xlabel(f"Time ({tz or 'UTC'})"); plt.ylabel("Bikes")
    plt.legend()
    _savefig(out)


def plot_residual_hist(df_s: pd.DataFrame, out: Path):
    if df_s.empty or "y_pred" not in df_s.columns or "y_true" not in df_s.columns:
        return
    sub = df_s.dropna(subset=["y_true", "y_pred"]).copy()
    if sub.empty:
        return
    resid = (sub["y_pred"] - sub["y_true"]).astype(float)
    plt.figure(figsize=(6, 3))
    plt.hist(resid, bins=40)
    plt.title("Residuals (y_pred - y_true)")
    plt.xlabel("Residual"); plt.ylabel("Count")
    _savefig(out)


# ------------------------- page markdown -------------------------

def write_md(station_id: str, name: str | None, files: dict[str, Path], out_md: Path):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    title = f"Station {station_id}" + (f" — {name}" if name else "")
    md = f"""# {title}

Figures:

- Sparkline (7d): ![]({files['spark'].as_posix()})
- Hourly profile: ![]({files['hourly'].as_posix()})
- Observed vs Predicted (24h): ![]({files['ovsp'].as_posix()})
- Residuals: ![]({files['resid'].as_posix()})
"""
    out_md.write_text(md, encoding="utf-8")


# ------------------------- main -------------------------

def _run(
    events_path: Path,
    perf_path: Path,
    last_days: int,
    hours: int,
    select_k: int,
    by: str,
    tz: str | None,
    auto_align_display: bool = True,
) -> int:
    _ensure_dirs()

    # Lecture perf
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

    # Fenêtre LAST_DAYS (sélection)
    perf_full = perf.copy()
    if last_days and last_days > 0:
        tmax = perf_full["ts"].max()
        if pd.notna(tmax):
            perf_full = perf_full[perf_full["ts"] >= (tmax - pd.Timedelta(days=last_days))].copy()
    if perf_full.empty:
        print("[WARN] No data in last_days window; nothing to build.")
        return 0

    # fallback global si y_pred vide
    has_pred = ("y_pred" in perf_full.columns) and perf_full["y_pred"].notna().any()
    if not has_pred and "y_pred_baseline" in perf_full.columns:
        print("[WARN] y_pred empty globally → using baseline everywhere")
        perf_full["y_pred"] = perf_full["y_pred_baseline"]

    # Sélection stations
    selected = _station_selector(perf_full, k=select_k, by=by)
    if not selected:
        selected = (perf_full["station_id"].value_counts().head(select_k).index.astype(str).tolist())
    print(f"[INFO] Selected {len(selected)} stations: {selected[:6]}{'...' if len(selected)>6 else ''}")

    # Fenêtre HOURS (plots)
    perf_plot = perf_full.copy()
    if hours and hours > 0:
        tmax = perf_plot["ts"].max()
        perf_plot = perf_plot[perf_plot["ts"] >= (tmax - pd.Timedelta(hours=hours))].copy()

    # Noms stations (facultatif)
    names = {}
    try:
        ev = pd.read_parquet(events_path, columns=["station_id", "name"])
        ev = ev.dropna(subset=["station_id"])
        ev["station_id"] = ev["station_id"].astype(str)
        names = ev.drop_duplicates(["station_id"], keep="last").set_index("station_id")["name"].to_dict()
    except Exception:
        print("[INFO] No station names found in events (optional).")

    # Génération par station
    ok_pages = 0
    for sid in selected:
        try:
            df_s = perf_plot[perf_plot["station_id"] == str(sid)].copy()
            if df_s.empty:
                print(f"[WARN] station {sid}: no data in plotting window; skip")
                continue
            df_s = _fallback_pred_if_needed(df_s)

            base = STATIONS_DIR / f"{sid}"
            files = {
                "spark": base / "sparkline.png",
                "hourly": base / "hourly.png",
                "ovsp":   base / "obs_vs_pred.png",
                "resid":  base / "residual_hist.png",
            }

            plot_sparkline(df_s, tz, files["spark"])
            plot_hourly_profile(df_s, tz, files["hourly"])
            plot_obs_vs_pred_24h(df_s, tz, files["ovsp"], auto_align_display=auto_align_display)
            plot_residual_hist(df_s, files["resid"])

            write_md(sid, names.get(str(sid)), files, base.with_suffix(".md"))
            ok_pages += 1
        except Exception as e:
            print(f"[ERR] station {sid}: {e}; skipping")

    print("[OK] Station profiles generated:")
    print(f" - pages: {ok_pages} -> {STATIONS_DIR / '*.md'}")
    print(" - figs per station: sparkline.png | hourly.png | obs_vs_pred.png | residual_hist.png")
    return ok_pages


def main(
    events_path: Path,
    perf_path: Path,
    last_days: int,
    hours: int,
    select_k: int,
    by: str,
    tz: str | None,
    auto_align_display: bool = True,
):
    try:
        _run(
            events_path=events_path,
            perf_path=perf_path,
            last_days=last_days,
            hours=hours,
            select_k=select_k,
            by=by,
            tz=tz,
            auto_align_display=auto_align_display,
        )
        # Toujours succès pour ne pas casser la CI (les autres assets ont de la valeur)
        return 0
    except Exception as e:
        print(f"[ERR] build_station_profiles failed: {e}")
        traceback.print_exc()
        # Ne pas faire échouer le pipeline
        return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build per-station pages with figures")
    ap.add_argument("--events", type=Path, default=EXPORTS / "events.parquet")
    ap.add_argument("--perf", type=Path, default=EXPORTS / "perf.parquet")
    ap.add_argument("--last-days", type=int, default=7, help="Fenêtre jours (pour stats/volatilité)")
    ap.add_argument("--hours", type=int, default=48, help="Fenêtre d'affichage pour les figures")
    ap.add_argument("--select", type=int, default=12, help="Nombre de stations à documenter")
    ap.add_argument("--by", type=str, default="volatility", choices=["volatility", "coverage", "count"])
    ap.add_argument("--tz", type=str, default=None, help="Affichage (ex: Europe/Paris). Données restent en UTC.")
    ap.add_argument("--no-auto-align-display", action="store_true", help="Désactive l'auto-alignement d'affichage.")
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
