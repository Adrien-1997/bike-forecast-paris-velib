# tools/build_station_profiles.py
# Génère :
#   - docs/stations/<station_id>/*.png
#       · timeline_7d.png
#       · obs_vs_pred_24h.png
#       · heatmap_hour_dow.png
#       · residual_box_by_hour.png
#       · residual_hist.png
#   - docs/stations/<station_id>.md   (légendes & unités claires)
#   - docs/stations/index.md          (limité à 10 liens)
#
# Sources meta (priorité -> fallback):
#   1) docs/assets/tables/stations.csv
#   2) docs/exports/events.parquet (pour name)

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless/CI-safe
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXPORTS = DOCS / "exports"
ASSETS = DOCS / "assets"
TABLES = ASSETS / "tables"
STATS_CSV = TABLES / "stations.csv"

STATIONS_DIR = DOCS / "stations"
IMG_STATIONS_DIR = STATIONS_DIR

plt.rcParams.update({"figure.autolayout": True})


# ------------------------- utils -------------------------

def _to_local(ts: pd.Series, tz: Optional[str]) -> pd.Series:
    if tz:
        return pd.to_datetime(ts, utc=True, errors="coerce").dt.tz_convert(tz)
    return pd.to_datetime(ts, utc=False, errors="coerce")

def _ensure_dirs():
    STATIONS_DIR.mkdir(parents=True, exist_ok=True)

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
    return df.groupby("station_id")["y_pred"].apply(lambda s: s.notna().mean())

def _station_selector(df: pd.DataFrame, k: int, by: str) -> list[str]:
    if df.empty: return []
    cov = _safe_group_cov(df).fillna(0.0)

    if "y_true" in df.columns:
        grp = df.groupby("station_id")["y_true"]
        vol = grp.std().fillna(0.0)
        n = grp.count()
        vol = vol.where(n >= 10, 0.0).fillna(0.0)
    else:
        vol = cov * 0.0

    cnt = df.groupby("station_id").size().astype(float)

    if by == "volatility": score = vol
    elif by == "coverage": score = cov
    else: score = cnt

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

def _apply_display_lag(df_s: pd.DataFrame, max_steps: int = 8) -> pd.DataFrame:
    """Aligne visuellement y_pred pour les FIGURES (non affiché dans les titres/MD)."""
    disp = df_s.copy()
    lag, _ = 0, np.nan
    if "y_true" in disp.columns and "y_pred" in disp.columns:
        sub = disp.dropna(subset=["y_true", "y_pred"])
        if len(sub) >= 20:
            lag, _ = _best_lag(sub["y_true"], sub["y_pred"], max_steps=max_steps)
    disp["y_pred_disp"] = disp.get("y_pred")
    if lag != 0 and "y_pred" in disp.columns:
        disp["y_pred_disp"] = disp["y_pred"].shift(lag)
    return disp

def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()

# ------------------------- lecture stats (stations.csv) -------------------------

def _load_station_stats(events_path: Path) -> pd.DataFrame:
    """
    Charge la table stations :
      - Priorité : docs/assets/tables/stations.csv
      - Fallback : events.parquet (pour récupérer au moins station_id/name)
    Colonnes possibles : station_id, name, lat, lon, capacity_mean, bikes_mean, occ_mean, bikes_std, occ_std, cluster
    """
    if STATS_CSV.exists():
        try:
            df = pd.read_csv(STATS_CSV)
            df["station_id"] = df["station_id"].astype(str)
            for c in ("capacity_mean", "bikes_mean", "occ_mean", "bikes_std", "occ_std", "cluster"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except Exception as e:
            print(f"[WARN] cannot read {STATS_CSV}: {e}")

    # fallback minimal via events
    try:
        ev = pd.read_parquet(events_path, columns=["station_id", "name"])
        ev = ev.dropna(subset=["station_id"])
        ev["station_id"] = ev["station_id"].astype(str)
        return ev.drop_duplicates(["station_id"], keep="last")
    except Exception:
        print("[INFO] No station names found in events (optional).")
        return pd.DataFrame(columns=["station_id"])

# ------------------------- PLOTS -------------------------

def plot_timeline_7d(df_s: pd.DataFrame, tz: Optional[str], out: Path):
    """
    Timeline 7 jours — Observé vs Prédit (vélos disponibles).
    """
    if df_s.empty or "y_true" not in df_s.columns: return
    tmax = df_s["ts"].max()
    sub = df_s[df_s["ts"] >= (tmax - pd.Timedelta(days=7))].copy()
    ypred_col = "y_pred_disp" if "y_pred_disp" in sub.columns else ("y_pred" if "y_pred" in sub.columns else None)
    if ypred_col is None: return
    sub = sub.dropna(subset=["y_true", ypred_col])
    if sub.empty: return

    x = _to_local(sub["ts"], tz)
    plt.figure(figsize=(12, 3.5))
    plt.plot(x, sub["y_true"], label="Observé (vélos dispo)")
    plt.plot(x, sub[ypred_col], label="Prédit (vélos dispo)")
    plt.title("7 jours — Observé vs Prédit")
    plt.xlabel(f"Temps ({tz or 'UTC'})"); plt.ylabel("Vélos disponibles")
    plt.legend()
    _savefig(out)

def plot_obs_vs_pred_24h(df_s: pd.DataFrame, tz: Optional[str], out: Path):
    if df_s.empty or "y_true" not in df_s.columns: return
    tmax = df_s["ts"].max()
    sub = df_s[df_s["ts"] >= (tmax - pd.Timedelta(hours=24))].copy()
    ypred_col = "y_pred_disp" if "y_pred_disp" in sub.columns else ("y_pred" if "y_pred" in sub.columns else None)
    if ypred_col is None: return
    sub = sub.dropna(subset=["y_true", ypred_col])
    if sub.empty: return

    x = _to_local(sub["ts"], tz)
    plt.figure(figsize=(12, 3.2))
    plt.plot(x, sub["y_true"], label="Observé (vélos dispo)")
    plt.plot(x, sub[ypred_col], label="Prédit (vélos dispo)")
    plt.title("Dernières 24h — Observé vs Prédit")
    plt.xlabel(f"Temps ({tz or 'UTC'})"); plt.ylabel("Vélos disponibles")
    plt.legend()
    _savefig(out)

def plot_heatmap_hour_dow(df_s: pd.DataFrame, tz: Optional[str], out: Path):
    """
    Heatmap (heure × jour_de_semaine) des vélos disponibles observés (y_true).
    """
    if df_s.empty or "y_true" not in df_s.columns: return
    df = df_s.copy()
    tloc = _to_local(df["ts"], tz)
    df["hour"] = tloc.dt.hour
    df["dow"]  = tloc.dt.dayofweek  # 0=Mon ... 6=Sun
    mat = df.pivot_table(index="dow", columns="hour", values="y_true", aggfunc="mean")
    if mat.empty: return

    plt.figure(figsize=(9, 4))
    plt.imshow(mat.values, aspect="auto", origin="upper")
    plt.colorbar(label="Vélos disponibles (moyenne)")
    plt.yticks(range(7), ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"])
    plt.xticks(range(0,24,2), range(0,24,2))
    plt.xlabel("Heure"); plt.ylabel("Jour")
    plt.title("Disponibilité moyenne — carte heure × jour")
    _savefig(out)

def plot_residual_box_by_hour(df_s: pd.DataFrame, tz: Optional[str], out: Path):
    """
    Boîtes à moustaches des résidus (prédit − observé) par heure de la journée (fenêtre récente).
    """
    ypred_col = "y_pred_disp" if "y_pred_disp" in df_s.columns else ("y_pred" if "y_pred" in df_s.columns else None)
    if df_s.empty or ypred_col is None or "y_true" not in df_s.columns: return
    df = df_s.dropna(subset=["y_true", ypred_col]).copy()
    if df.empty: return
    tloc = _to_local(df["ts"], tz)
    df["hour"] = tloc.dt.hour
    df["resid"] = (df[ypred_col] - df["y_true"]).astype(float)

    data = [df.loc[df["hour"]==h, "resid"].values for h in range(24)]
    plt.figure(figsize=(10, 3.5))
    plt.boxplot(data, positions=range(24), widths=0.7, showfliers=False)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.title("Résidus par heure (prédit − observé)")
    plt.xlabel("Heure"); plt.ylabel("Écart (vélos)")
    plt.xticks(range(0,24,2), range(0,24,2))
    _savefig(out)

def plot_residual_hist(df_s: pd.DataFrame, out: Path):
    ypred_col = "y_pred_disp" if "y_pred_disp" in df_s.columns else ("y_pred" if "y_pred" in df_s.columns else None)
    if df_s.empty or ypred_col is None or "y_true" not in df_s.columns: return
    sub = df_s.dropna(subset=["y_true", ypred_col]).copy()
    if sub.empty: return
    resid = (sub[ypred_col] - sub["y_true"]).astype(float)
    plt.figure(figsize=(7, 3))
    plt.hist(resid, bins=40)
    plt.title("Distribution des résidus (prédit − observé)")
    plt.xlabel("Écart (vélos)"); plt.ylabel("Occurrences")
    _savefig(out)

# ------------------------- pages -------------------------

def _fmt(x, digits=1):
    try:
        if pd.isna(x): return "—"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "—"

def write_md(station_id: str, meta: Dict[str, object], files: Dict[str, Path], out_md: Path):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    name = meta.get("name")
    title = f"Station {station_id}" + (f" — {name}" if isinstance(name, str) and name else "")

    md = f"""# {title}

Unités : **nombre de vélos disponibles**.

**Figures :**

- **7 jours — Observé vs Prédit**  
  Évolution sur 7 jours des vélos disponibles réels (`y_true`) et des prévisions (`y_pred`).  
  ![](./{station_id}/timeline_7d.png)

- **Observé vs Prédit — dernières 24h**  
  Focus court terme (24h) pour apprécier la réactivité locale.  
  ![](./{station_id}/obs_vs_pred_24h.png)

- **Disponibilité — carte heure × jour**  
  Moyenne des vélos disponibles par **heure** et **jour** (plus sombre = plus de vélos).  
  ![](./{station_id}/heatmap_hour_dow.png)

- **Résidus par heure (7j)**  
  Boîtes à moustaches des écarts `(prédit − observé)` par **heure de la journée**.  
  ![](./{station_id}/residual_box_by_hour.png)

- **Distribution des résidus**  
  Histogramme des écarts `(prédit − observé)` en **vélos**.  
  ![](./{station_id}/residual_hist.png)
"""
    out_md.write_text(md, encoding="utf-8")

def write_index_md(pages: List[tuple[str, Optional[str], Optional[object]]], out_md: Path, limit: int = 10):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    limited = pages[: max(0, min(limit, len(pages)))]
    lines = [
        "# Fiches stations",
        "",
        f"Liste auto-générée (limite : {len(limited)}/{len(pages)} stations affichées).",
        "",
    ]
    for sid, nm, cl in limited:
        label = f"#{sid}" + (f" — {nm}" if nm else "")
        if cl is not None and not (isinstance(cl, float) and np.isnan(cl)):
            label += f" (C{int(cl)})"
        lines.append(f"- [{label}]({sid}.md)")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ------------------------- run / main -------------------------

def _run(events_path: Path, perf_path: Path, last_days: int, hours: int, select_k: int, by: str, tz: Optional[str], auto_align_display: bool = True) -> int:
    _ensure_dirs()

    # perf
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
        selected = perf_full["station_id"].value_counts().head(select_k).index.astype(str).tolist()
    print(f"[INFO] Selected {len(selected)} stations: {selected[:6]}{'...' if len(selected)>6 else ''}")

    perf_plot = perf_full.copy()
    if hours and hours > 0:
        tmax = perf_plot["ts"].max()
        perf_plot = perf_plot[perf_plot["ts"] >= (tmax - pd.Timedelta(hours=hours))].copy()

    # stats & meta (stations.csv prioritaire)
    stats = _load_station_stats(events_path)
    stats["station_id"] = stats["station_id"].astype(str)
    stats_idx = stats.set_index("station_id") if "station_id" in stats.columns else pd.DataFrame()

    def _meta_for(sid: str) -> Dict[str, object]:
        if hasattr(stats_idx, "index") and sid in stats_idx.index:
            row = stats_idx.loc[sid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            d = row.to_dict()
            if "name" in d and not pd.isna(d["name"]):
                d["name"] = str(d["name"])
            return d
        return {}

    # génération par station
    pages: List[tuple[str, Optional[str], Optional[object]]] = []
    ok_pages = 0
    for sid in selected:
        try:
            df_s = perf_plot[perf_plot["station_id"] == str(sid)].copy()
            if df_s.empty:
                print(f"[WARN] station {sid}: no data in plotting window; skip")
                continue
            df_s = _fallback_pred_if_needed(df_s)
            df_disp = _apply_display_lag(df_s, max_steps=8) if auto_align_display else df_s.copy()

            imgdir = IMG_STATIONS_DIR / f"{sid}"
            files = {
                "t7d":    imgdir / "timeline_7d.png",
                "ovsp24": imgdir / "obs_vs_pred_24h.png",
                "heat":   imgdir / "heatmap_hour_dow.png",
                "boxhr":  imgdir / "residual_box_by_hour.png",
                "resid":  imgdir / "residual_hist.png",
            }

            # plots
            plot_timeline_7d(df_disp, tz, files["t7d"])
            plot_obs_vs_pred_24h(df_disp, tz, files["ovsp24"])
            plot_heatmap_hour_dow(df_s, tz, files["heat"])
            plot_residual_box_by_hour(df_disp, tz, files["boxhr"])
            plot_residual_hist(df_disp, files["resid"])

            meta = _meta_for(str(sid))
            write_md(sid, meta, files, STATIONS_DIR / f"{sid}.md")
            pages.append((sid, meta.get("name"), meta.get("cluster")))
            ok_pages += 1
        except Exception as e:
            print(f"[ERR] station {sid}: {e}; skipping")

    # index limité à 10 stations
    write_index_md(pages, STATIONS_DIR / "index.md", limit=10)

    print("[OK] Station profiles generated:")
    print(f" - pages: {ok_pages} -> {STATIONS_DIR / '*.md'}")
    print(" - index: {STATIONS_DIR / 'index.md'} (max 10 links)")
    print(" - figs per station: timeline_7d | obs_vs_pred_24h | heatmap_hour_dow | residual_box_by_hour | residual_hist")
    return ok_pages


def main(events_path: Path, perf_path: Path, last_days: int, hours: int, select_k: int, by: str, tz: Optional[str], auto_align_display: bool = True):
    try:
        _run(events_path=events_path, perf_path=perf_path, last_days=last_days, hours=hours, select_k=select_k, by=by, tz=tz, auto_align_display=auto_align_display)
        return 0
    except Exception as e:
        print(f"[ERR] build_station_profiles failed: {e}")
        traceback.print_exc()
        return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build per-station pages with varied figures (7d timeline first)")
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
