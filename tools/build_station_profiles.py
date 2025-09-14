# tools/build_station_profiles.py
# Étape 6/6 — Profils de stations (portfolio-ready)
# Génère des fiches Markdown par station avec KPIs + visuels :
#   - sparkline d'occupation (7–30j)
#   - profil horaire moyen
#   - Observé vs Prédit (48h)
#   - Histogramme des résidus
#   - tableau KPI (mean, std, parts <10%/>90%, MAE, RMSE, biais)
#
# Entrées:
#   --events docs/exports/events.parquet  (ou .csv)
#   --perf   docs/exports/perf.parquet    (ou .csv)
#   --last-days 7  (fenêtre pour usage)
#   --hours 48     (fenêtre pour obs vs préd station)
#   --select 12    (nombre de stations à publier)
#   --by volatility|error|tension|random (stratégie de sélection)
#   --stations 31607 10005 ... (forçage liste)
#
# Sorties:
#   - docs/stations/<station_id>.md
#   - docs/stations/index.md
#   - docs/assets/figs/stations/<station_id>/*.png
#
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIGS = DOCS / "assets" / "figs" / "stations"
PAGES = DOCS / "stations"


def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    ensure_dir(path)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def read_events(path: Path, last_days: int) -> pd.DataFrame:
    if path.exists():
        ev = pd.read_parquet(path) if str(path).lower().endswith(".parquet") else pd.read_csv(path)
    else:
        # fallback sur velib.parquet via datasets.load_normalized
        from datasets import load_normalized
        ev, _, _, _, _ = load_normalized(DOCS / "exports" / "velib.parquet", horizon_minutes=60, last_days=last_days)

    need = {"ts", "station_id", "bikes", "capacity"}
    miss = need - set(ev.columns)
    if miss:
        raise ValueError(f"events missing columns: {miss}")

    ev["ts"] = pd.to_datetime(ev["ts"], utc=False, errors="coerce")
    ev = ev.sort_values(["station_id", "ts"]).reset_index(drop=True)

    cap = pd.to_numeric(ev["capacity"], errors="coerce").replace(0, np.nan)
    ev["bikes"] = pd.to_numeric(ev["bikes"], errors="coerce")
    ev["occ"] = (ev["bikes"]/cap).clip(0,1)

    if last_days and last_days > 0:
        tmax = ev["ts"].max()
        tmin = tmax - pd.Timedelta(days=last_days)
        ev = ev[ev["ts"].between(tmin, tmax)].copy()

    return ev


def read_perf(path: Path, hours: int) -> pd.DataFrame:
    if path.exists():
        perf = pd.read_parquet(path) if str(path).lower().endswith(".parquet") else pd.read_csv(path)
    else:
        from datasets import load_normalized
        _, perf, _, _, _ = load_normalized(DOCS / "exports" / "velib.parquet", horizon_minutes=60, last_days=None)

    need = {"ts", "station_id", "y_true", "y_pred"}
    miss = need - set(perf.columns)
    if miss:
        raise ValueError(f"perf missing columns: {miss}")

    perf["ts"] = pd.to_datetime(perf["ts"], utc=False, errors="coerce")
    perf = perf.sort_values(["station_id", "ts"]).reset_index(drop=True)

    if hours and hours > 0:
        tmax = perf["ts"].max()
        tmin = tmax - pd.Timedelta(hours=hours)
        perf = perf[perf["ts"] >= tmin].copy()

    # numeric safe
    perf["y_true"] = pd.to_numeric(perf["y_true"], errors="coerce")
    perf["y_pred"] = pd.to_numeric(perf["y_pred"], errors="coerce")

    return perf


def compute_kpis(ev: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    # KPIs par station
    k1 = (ev.groupby("station_id")["occ"]
            .agg(occ_mean="mean", occ_median="median", occ_std="std")
            .reset_index())

    occ = ev[["station_id","occ"]].copy()
    occ["low"] = occ["occ"].lt(0.10).fillna(False).astype(int)
    occ["high"] = occ["occ"].gt(0.90).fillna(False).astype(int)
    k2 = (occ.groupby("station_id")[["low","high"]].mean()
             .rename(columns={"low":"share_low","high":"share_high"})
             .reset_index())

    if perf is not None and not perf.empty:
        e = perf.assign(err=lambda d: d["y_pred"] - d["y_true"])
        k3 = (e.groupby("station_id")["err"]
                .agg(bias="mean")
                .reset_index())
        k4 = (e.assign(ae=lambda d: d["err"].abs())
                .groupby("station_id")["ae"]
                .agg(MAE="mean", RMSE=lambda s: float(np.sqrt(np.nanmean((s.values)**2))))
                .reset_index())
        k = k1.merge(k2, on="station_id", how="left").merge(k3, on="station_id", how="left").merge(k4, on="station_id", how="left")
    else:
        k = k1.merge(k2, on="station_id", how="left")
        k["bias"] = np.nan; k["MAE"] = np.nan; k["RMSE"] = np.nan

    # name/lat/lon si dispo
    meta_cols = [c for c in ["name","lat","lon","capacity"] if c in ev.columns]
    meta = ev.drop_duplicates("station_id")[["station_id"] + meta_cols]
    k = k.merge(meta, on="station_id", how="left")
    return k


def select_station_ids(kpis: pd.DataFrame, n: int, by: str, explicit: Optional[List[str]]) -> List[str]:
    if explicit:
        return [str(s) for s in explicit if str(s) in kpis["station_id"].astype(str).values]

    df = kpis.copy()
    by = (by or "volatility").lower()
    if by == "error" and "MAE" in df.columns:
        order = df.sort_values("MAE", ascending=False)
    elif by == "tension":
        order = df.assign(tension=df["share_low"].fillna(0) + df["share_high"].fillna(0)).sort_values("tension", ascending=False)
    elif by == "random":
        order = df.sample(frac=1.0, random_state=42)
    else:  # volatility
        order = df.sort_values("occ_std", ascending=False)
    return order["station_id"].astype(str).head(n).tolist()


# --------- plots per station ---------

def plot_sparkline(ev_s: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(8, 2.2))
    plt.plot(ev_s["ts"], ev_s["occ"])
    plt.title("Occupancy — sparkline")
    plt.xlabel("Time"); plt.ylabel("Occ.")
    save_fig(out)


def plot_hourly_profile(ev_s: pd.DataFrame, out: Path) -> None:
    tmp = ev_s.copy()
    tmp["hour"] = tmp["ts"].dt.hour
    prof = tmp.groupby("hour")["occ"].agg(["mean","median"]).reset_index()
    plt.figure(figsize=(6.5, 3.2))
    plt.plot(prof["hour"], prof["mean"], marker="o", label="mean")
    plt.plot(prof["hour"], prof["median"], marker="s", label="median")
    plt.title("Hourly profile")
    plt.xlabel("Hour"); plt.ylabel("Occ.")
    plt.xticks(range(0,24,2)); plt.legend()
    save_fig(out)


def plot_obs_vs_pred(perf_s: pd.DataFrame, out: Path) -> None:
    if perf_s is None or perf_s.empty:
        # placeholder
        plt.figure(figsize=(7, 2.8)); plt.title("No prediction data"); save_fig(out); return
    plt.figure(figsize=(8, 3.2))
    plt.plot(perf_s["ts"], perf_s["y_true"], label="Observed")
    plt.plot(perf_s["ts"], perf_s["y_pred"], label="Predicted")
    plt.title("Observed vs Predicted (last window)")
    plt.xlabel("Time"); plt.ylabel("Bikes")
    plt.legend()
    save_fig(out)


def plot_residual_hist(perf_s: pd.DataFrame, out: Path) -> None:
    if perf_s is None or perf_s.empty:
        plt.figure(figsize=(6, 3)); plt.title("No residuals"); save_fig(out); return
    e = perf_s["y_pred"] - perf_s["y_true"]
    plt.figure(figsize=(6, 3))
    plt.hist(e.dropna(), bins=30)
    plt.title("Residuals histogram")
    plt.xlabel("Residual"); plt.ylabel("Count")
    save_fig(out)


# --------- pages ---------

def write_station_page(st_id: str, k: pd.Series, fig_dir: Path, out_md: Path) -> None:
    name = k.get("name", st_id)
    cap = k.get("capacity", np.nan)
    occ_mean = k.get("occ_mean", np.nan)
    occ_std  = k.get("occ_std", np.nan)
    share_low = k.get("share_low", np.nan)
    share_high = k.get("share_high", np.nan)
    mae = k.get("MAE", np.nan)
    rmse = k.get("RMSE", np.nan)
    bias = k.get("bias", np.nan)

    txt = f'''
# Station {name} ({st_id})

**Synthèse rapide**
- Capacité : {cap if pd.notna(cap) else '—'}
- Occupation moyenne : {occ_mean:.2f} — variabilité (std) : {occ_std:.2f}
- Sous-tension (<10%) : {share_low*100:.1f}% — Surtension (>90%) : {share_high*100:.1f}%
- MAE : {mae:.2f} — RMSE : {rmse:.2f} — Biais : {bias:.2f}

## Occupation — sparkline
![sparkline](../assets/figs/stations/{st_id}/sparkline.png)

## Profil horaire (moyenne & médiane)
![hourly](../assets/figs/stations/{st_id}/hourly.png)

## Observé vs Prédit (fenêtre récente)
![ovsp](../assets/figs/stations/{st_id}/obs_vs_pred.png)

## Résidus (histogramme)
![resid](../assets/figs/stations/{st_id}/residual_hist.png)
'''.strip()

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(txt, encoding="utf-8")



def write_index(pages: List[Tuple[str, str]], out_md: Path, strategy: str) -> None:
    lines = ["# Profils de stations", "", f"_Stratégie de sélection : **{strategy}**_.", ""]
    for st_id, title in pages:
        lines.append(f"- [{title} — {st_id}](./{st_id}.md)")
    ensure_dir(out_md)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------- main ---------

def main():
    ap = argparse.ArgumentParser(description="Build station profile pages (figs + markdown).")
    ap.add_argument("--events", type=Path, default=DOCS / "exports" / "events.parquet")
    ap.add_argument("--perf", type=Path, default=DOCS / "exports" / "perf.parquet")
    ap.add_argument("--last-days", type=int, default=7)
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--select", type=int, default=12)
    ap.add_argument("--by", type=str, default="volatility", choices=["volatility","error","tension","random"])
    ap.add_argument("--stations", nargs="*", default=None, help="Explicit station ids to include")
    args = ap.parse_args()

    ev = read_events(args.events, last_days=args.last_days)
    perf = read_perf(args.perf, hours=args.hours)

    kpis = compute_kpis(ev, perf)
    chosen = select_station_ids(kpis, n=args.select, by=args.by, explicit=args.stations)

    pages: List[Tuple[str, str]] = []

    for st_id in chosen:
        ev_s = ev[ev["station_id"].astype(str) == str(st_id)].copy()
        perf_s = perf[perf["station_id"].astype(str) == str(st_id)].copy()

        # Figures
        fig_dir = FIGS / str(st_id)
        plot_sparkline(ev_s, fig_dir / "sparkline.png")
        plot_hourly_profile(ev_s, fig_dir / "hourly.png")
        plot_obs_vs_pred(perf_s, fig_dir / "obs_vs_pred.png")
        plot_residual_hist(perf_s, fig_dir / "residual_hist.png")

        # Page
        krow = kpis[kpis["station_id"].astype(str) == str(st_id)].iloc[0]
        title = krow.get("name", st_id)
        write_station_page(st_id, krow, fig_dir, PAGES / f"{st_id}.md")
        pages.append((str(st_id), str(title)))

    # Index
    write_index(pages, PAGES / "index.md", strategy=args.by)

    print("[OK] Station profiles generated:")
    print(f" - pages: {len(pages)} → docs/stations/*.md")
    print(" - figs per station: sparkline.png | hourly.png | obs_vs_pred.png | residual_hist.png")

if __name__ == "__main__":
    main()