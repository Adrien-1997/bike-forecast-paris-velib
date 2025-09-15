# tools/build_station_profiles.py
# Génère des pages station (Markdown) et figures :
#   - sparkline.png (7j)
#   - hourly.png (profil horaire moyen)
#   - obs_vs_pred.png (24h, avec auto-align display si décalage détecté)
#   - residual_hist.png
#
# Règles :
# - Les données restent en UTC ; la conversion de fuseau est uniquement pour l'affichage.
# - La performance compare STRICTEMENT y_pred(T) vs y_true(T).
# - Si y_pred est manquant pour une station, fallback sur y_pred_baseline (persistance).
# - Auto-align d'affichage : si un lag résiduel est détecté (corr max à k≠0), on
#   décale y_pred pour la figure (les fichiers parquet restent inchangés).
#
# Usage :
#   python tools/build_station_profiles.py --events docs/exports/events.parquet \
#       --perf docs/exports/perf.parquet --last-days 7 --hours 48 --select 12 \
#       --by volatility --tz Europe/Paris --no-auto-align-display

from __future__ import annotations

import argparse
from pathlib import Path
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


def _station_selector(df: pd.DataFrame, k: int, by: str) -> list[str]:
    """Sélectionne k stations selon un critère, en pénalisant la faible couverture y_pred."""
    if df.empty:
        return []
    cov = df.groupby("station_id")["y_pred"].apply(
        lambda s: s.notna().mean() if "y_pred" in df.columns else 0.0
    )
    vol = df.groupby("station_id")["y_true"].std(min_count=10).fillna(0.0)
    cnt = df.groupby("station_id").size()
    if by == "volatility":
        score = vol
    elif by == "coverage":
        score = cov
    else:
        score = cnt
    score = score * (1.0 + cov.fillna(0.0))  # boost stations bien couvertes en y_pred
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
    if df_s.empty:
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
    prof = df.groupby("hour")[["y_true", "y_pred"]].mean().dropna()
    if prof.empty:
        return
    plt.figure(figsize=(8, 3))
    plt.plot(prof.index, prof["y_true"], label="Observed")
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
    if df_s.empty:
        return
    tmax = df_s["ts"].max()
    sub = df_s[df_s["ts"] >= (tmax - pd.Timedelta(hours=24))].dropna(subset=["y_true", "y_pred"]).copy()
    if sub.empty:
        return

    # Auto-align d'affichage si besoin (ne modifie pas df_s).
    lag, r = 0, np.nan
    disp = sub.copy()
    if auto_align_display:
        lag, r = _best_lag(sub["y_true"], sub["y_pred"], max_steps=8)
        if lag != 0:
            disp = disp.copy()
            disp["y_pred"] = disp["y_pred"].shift(lag)
            # drop na introduits par le shift
            disp = disp.dropna(subset=["y_true", "y_pred"])

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
    _ensure_dirs()

    perf = pd.read_parquet(perf_path)
    perf["ts"] = pd.to_datetime(perf["ts"], utc=False, errors="coerce")
    perf["station_id"] = perf["station_id"].astype(str)

    # --- Fenêtre LAST_DAYS pour la sélection (riche) ---
    perf_full = perf.copy()
    if last_days and last_days > 0:
        tmax = perf_full["ts"].max()
        if pd.notna(tmax):
            perf_full = perf_full[perf_full["ts"] >= (tmax - pd.Timedelta(days=last_days))].copy()

    # fallback y_pred global si vraiment vide
    has_pred = ("y_pred" in perf_full.columns) and perf_full["y_pred"].notna().any()
    if not has_pred and "y_pred_baseline" in perf_full.columns:
        print("[WARN] y_pred empty globally → using baseline everywhere")
        perf_full["y_pred"] = perf_full["y_pred_baseline"]

    if perf_full.empty:
        print("[WARN] No data in last_days window; nothing to build.")
        return

    # Sélection stations
    selected = _station_selector(perf_full, k=select_k, by=by)
    if not selected:
        selected = (perf_full["station_id"].value_counts().head(select_k).index.astype(str).tolist())
    print(f"[INFO] Selected {len(selected)} stations: {selected[:6]}{'...' if len(selected)>6 else ''}")

    # --- Fenêtre HOURS pour tracer (plus étroite) ---
    perf_plot = perf_full.copy()
    if hours and hours > 0:
        tmax = perf_plot["ts"].max()
        perf_plot = perf_plot[perf_plot["ts"] >= (tmax - pd.Timedelta(hours=hours))].copy()

    # info station (name) depuis events
    try:
        ev = pd.read_parquet(events_path, columns=["station_id", "name"]).dropna()
        ev["station_id"] = ev["station_id"].astype(str)
        names = ev.drop_duplicates(["station_id"], keep="last").set_index("station_id")["name"].to_dict()
    except Exception:
        names = {}
    if not names:
        print("[INFO] No station names found in events; pages will use IDs only.")

    for sid in selected:
        df_s = perf_plot[perf_plot["station_id"] == str(sid)].copy()
        if df_s.empty:
            continue
        df_s = _fallback_pred_if_needed(df_s)

        # chemins de sortie
        base = STATIONS_DIR / f"{sid}"
        files = {
            "spark": base / "sparkline.png",
            "hourly": base / "hourly.png",
            "ovsp":   base / "obs_vs_pred.png",
            "resid":  base / "residual_hist.png",
        }

        # figures
        plot_sparkline(df_s, tz, files["spark"])
        plot_hourly_profile(df_s, tz, files["hourly"])
        plot_obs_vs_pred_24h(df_s, tz, files["ovsp"], auto_align_display=auto_align_display)
        plot_residual_hist(df_s, files["resid"])

        # markdown
        write_md(sid, names.get(str(sid)), files, base.with_suffix(".md"))

    print("[OK] Station profiles generated:")
    print(f" - pages: {len(selected)} -> {STATIONS_DIR / '*.md'}")
    print(" - figs per station: sparkline.png | hourly.png | obs_vs_pred.png | residual_hist.png")


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
