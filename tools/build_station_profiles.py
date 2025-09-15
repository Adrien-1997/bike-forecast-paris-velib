# tools/build_station_profiles.py
# Génère des pages station (Markdown) et figures :
#   - sparkline.png (7j)
#   - hourly.png (profil horaire moyen)
#   - obs_vs_pred.png (24h)
#   - residual_hist.png
#
# Règles :
# - Les données restent en UTC ; la conversion de fuseau n'est utilisée que pour l'affichage.
# - La performance compare STRICTEMENT y_pred(T) vs y_true(T).
# - Si y_pred est manquant pour une station, fallback sur y_pred_baseline (persistance).
#
# Usage :
#   python tools/build_station_profiles.py --events docs/exports/events.parquet \
#       --perf docs/exports/perf.parquet --last-days 7 --hours 48 --select 12 --by volatility --tz Europe/Paris

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
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
        return pd.to_datetime(ts, utc=True, errors="coerce").dt.tz_convert(tz)
    return pd.to_datetime(ts, utc=False, errors="coerce")


def _ensure_dirs():
    STATIONS_DIR.mkdir(parents=True, exist_ok=True)


def _station_selector(df: pd.DataFrame, k: int, by: str) -> list[str]:
    """Sélectionne k stations selon un critère simple sur la dernière fenêtre."""
    g = df.copy()
    g["date"] = g["ts"].dt.date
    # volatilité = std des bikes par jour (proxy de variabilité)
    vol = g.groupby("station_id")["y_true"].std().fillna(0.0)
    # couverture en y_pred
    cov = g.groupby("station_id")["y_pred"].apply(lambda s: s.notna().mean()).fillna(0.0)

    if by == "volatility":
        score = vol
    elif by == "coverage":
        score = cov
    else:
        # nombre d'observations
        score = g.groupby("station_id").size()

    top = score.sort_values(ascending=False).head(k).index.astype(str).tolist()
    return top


def _fallback_pred_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Si y_pred est vide pour une station, use baseline. Log visuel."""
    out = df.copy()
    # on ne remplace que les NaN (pour garder le modèle quand il existe)
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
    if df_s.empty: return
    x = _to_local(df_s["ts"], tz)
    plt.figure(figsize=(8, 2))
    plt.plot(x, df_s["y_true"], linewidth=1)
    plt.title("Observed (y_true) — last window")
    plt.xlabel(f"Time ({tz or 'UTC'})")
    _savefig(out)


def plot_hourly_profile(df_s: pd.DataFrame, tz: str | None, out: Path):
    if df_s.empty: return
    tloc = _to_local(df_s["ts"], tz)
    df = df_s.copy()
    df["hour"] = tloc.dt.hour
    prof = df.groupby("hour")[["y_true", "y_pred"]].mean().dropna()
    if prof.empty: return
    plt.figure(figsize=(8, 3))
    plt.plot(prof.index, prof["y_true"], label="Observed")
    plt.plot(prof.index, prof["y_pred"], label="Predicted")
    plt.title("Hourly profile (mean)")
    plt.xlabel("Hour"); plt.ylabel("Bikes")
    plt.xticks(range(0, 24, 2))
    plt.legend()
    _savefig(out)


def plot_obs_vs_pred_24h(df_s: pd.DataFrame, tz: str | None, out: Path):
    if df_s.empty: return
    tmax = df_s["ts"].max()
    sub = df_s[df_s["ts"] >= (tmax - pd.Timedelta(hours=24))].dropna(subset=["y_true", "y_pred"]).copy()
    if sub.empty: return
    x = _to_local(sub["ts"], tz)
    plt.figure(figsize=(8, 3))
    plt.plot(x, sub["y_true"], label="Observed (y_true)")
    plt.plot(x, sub["y_pred"], label="Predicted (y_pred)")
    plt.title("Last 24h")
    plt.xlabel(f"Time ({tz or 'UTC'})"); plt.ylabel("Bikes")
    plt.legend()
    _savefig(out)


def plot_residual_hist(df_s: pd.DataFrame, out: Path):
    sub = df_s.dropna(subset=["y_true", "y_pred"]).copy()
    if sub.empty: return
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

def main(events_path: Path, perf_path: Path, last_days: int, hours: int, select_k: int, by: str, tz: str | None):
    _ensure_dirs()

    perf = pd.read_parquet(perf_path)
    perf["ts"] = pd.to_datetime(perf["ts"], utc=False, errors="coerce")
    perf["station_id"] = perf["station_id"].astype(str)

    # filtrage fenêtre
    if last_days and last_days > 0:
        tmax = perf["ts"].max()
        if pd.notna(tmax):
            perf = perf[perf["ts"] >= (tmax - pd.Timedelta(days=last_days))].copy()

    # fallback y_pred par station si besoin
    has_pred = ("y_pred" in perf.columns) and perf["y_pred"].notna().any()
    if not has_pred and "y_pred_baseline" in perf.columns:
        print("[WARN] y_pred empty globally → using baseline everywhere")
        perf["y_pred"] = perf["y_pred_baseline"]

    # restreindre à la fenêtre horaires pour les graphes si demandé
    if hours and hours > 0:
        tmax = perf["ts"].max()
        perf = perf[perf["ts"] >= (tmax - pd.Timedelta(hours=hours))].copy()

    # info station (name) depuis events
    try:
        ev = pd.read_parquet(events_path, columns=["station_id", "name"]).dropna()
        ev["station_id"] = ev["station_id"].astype(str)
        # dernier nom connu par station
        names = ev.drop_duplicates(["station_id"], keep="last").set_index("station_id")["name"].to_dict()
    except Exception:
        names = {}

    # sélection des stations à profiler
    selected = _station_selector(perf, k=select_k, by=by)

    for sid in selected:
        df_s = perf[perf["station_id"] == str(sid)].copy()
        df_s = _fallback_pred_if_needed(df_s)

        # chemins
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
        plot_obs_vs_pred_24h(df_s, tz, files["ovsp"])
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
    args = ap.parse_args()
    main(events_path=args.events, perf_path=args.perf, last_days=args.last_days,
         hours=args.hours, select_k=args.select, by=args.by, tz=args.tz)
