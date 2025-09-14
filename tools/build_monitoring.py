# tools/build_monitoring.py
# Monitoring "data & modèle" — santé des données, drift PSI, importance (proxy), tendance d'erreur.
# Entrées :
#   - --events docs/exports/events.parquet (ou .csv)  [ts, station_id, bikes, capacity, ...]
#   - --perf   docs/exports/perf.parquet   (ou .csv)  [ts, station_id, y_true, y_pred, horizon_min]
#   - si absents, ils sont (re)construits depuis docs/exports/velib.parquet via datasets.load_normalized()
# Sorties (docs/assets) :
#   tables/data_health.csv
#   tables/psi_features.csv
#   tables/feature_importance_proxy.csv
#   tables/daily_error.csv
#   figs/mon_data_health.png
#   figs/mon_psi.png
#   figs/mon_feature_importance.png
#   figs/mon_error_trend.png
#
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_normalized

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIGS = DOCS / "assets" / "figs"
TABLES = DOCS / "assets" / "tables"


# ----------------- utils -----------------

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    ensure_dir(path)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def read_events(events_path: Optional[Path], last_days: Optional[int]) -> pd.DataFrame:
    if events_path and Path(events_path).exists():
        ev = pd.read_parquet(events_path) if str(events_path).lower().endswith(".parquet") else pd.read_csv(events_path)
    else:
        ev, _, _, _, _ = load_normalized(DOCS / "exports" / "velib.parquet", horizon_minutes=60, last_days=last_days)

    # Sanity
    need = {"ts", "station_id", "bikes", "capacity"}
    miss = need - set(ev.columns)
    if miss:
        raise ValueError(f"events missing columns: {miss}")

    ev["ts"] = pd.to_datetime(ev["ts"], utc=False, errors="coerce")
    ev = ev.sort_values(["station_id", "ts"]).reset_index(drop=True)

    cap = pd.to_numeric(ev["capacity"], errors="coerce").replace(0, np.nan)
    ev["bikes"] = pd.to_numeric(ev["bikes"], errors="coerce")
    ev["occ"] = (ev["bikes"]/cap).clip(0,1)

    return ev


def read_perf(perf_path: Optional[Path], horizon: int, last_days: Optional[int]) -> pd.DataFrame:
    if perf_path and Path(perf_path).exists():
        perf = pd.read_parquet(perf_path) if str(perf_path).lower().endswith(".parquet") else pd.read_csv(perf_path)
    else:
        _, perf, _, _, _ = load_normalized(DOCS / "exports" / "velib.parquet", horizon_minutes=horizon, last_days=last_days)

    need = {"ts", "station_id", "y_true", "y_pred"}
    miss = need - set(perf.columns)
    if miss:
        raise ValueError(f"perf missing columns: {miss}")
    perf["ts"] = pd.to_datetime(perf["ts"], utc=False, errors="coerce")
    perf = perf.sort_values(["station_id", "ts"]).reset_index(drop=True)
    perf["y_true"] = pd.to_numeric(perf["y_true"], errors="coerce")
    perf["y_pred"] = pd.to_numeric(perf["y_pred"], errors="coerce")
    if "horizon_min" not in perf.columns:
        perf["horizon_min"] = horizon
    return perf


# ----------------- Data health -----------------

def data_health(ev: pd.DataFrame) -> pd.DataFrame:
    """Complétude et qualité simple des champs clés."""
    total = len(ev)
    def share_na(s): return float(s.isna().mean())
    def share_zero(s): return float((s==0).mean())

    cap = pd.to_numeric(ev.get("capacity", np.nan), errors="coerce")
    bikes = pd.to_numeric(ev.get("bikes", np.nan), errors="coerce")

    rows = [
        {"field":"bikes", "missing": share_na(bikes), "share_zero": share_zero(bikes)},
        {"field":"capacity", "missing": share_na(cap), "share_zero": share_zero(cap)},
        {"field":"occ", "missing": share_na(ev["occ"])}
    ]

    # uptime station = part d'horodatages présents par station sur la fenêtre
    sta_counts = ev.groupby("station_id")["ts"].count()
    uptime = float(sta_counts.mean() / sta_counts.max()) if sta_counts.size > 0 else np.nan

    out = pd.DataFrame(rows)
    out.loc[len(out)] = {"field": "station_uptime_rel", "missing": np.nan, "share_zero": np.nan}
    out.loc[len(out)-1, "value"] = uptime
    return out


def plot_data_health(health: pd.DataFrame, out_fig: Path) -> None:
    # Barh sur 'missing' trié
    h = health.copy()
    if "missing" in h.columns:
        h = h.sort_values("missing", ascending=False)
    plt.figure(figsize=(7, 4))
    plt.barh(h["field"].astype(str), h.get("missing", 0).fillna(0))
    plt.title("Missingness (share)")
    save_fig(out_fig)


# ----------------- PSI drift -----------------

def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index entre deux séries numériques."""
    ref = pd.Series(ref).dropna().astype(float)
    cur = pd.Series(cur).dropna().astype(float)
    if ref.empty or cur.empty:
        return np.nan
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, qs))
    if len(edges) < 3:
        return 0.0
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_p = np.clip(ref_hist / max(ref_hist.sum(), 1), 1e-6, 1)
    cur_p = np.clip(cur_hist / max(cur_hist.sum(), 1), 1e-6, 1)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def compute_psi_table(ev: pd.DataFrame, ref_pct: float = 0.3) -> pd.DataFrame:
    n = len(ev)
    if n < 100:
        return pd.DataFrame(columns=["feature","psi"])

    ref = ev.iloc[: int(ref_pct * n)]
    cur = ev.iloc[int((1-ref_pct) * n) :]

    nums = []
    for c in ["bikes", "capacity", "occ"]:
        if c in ev.columns:
            nums.append(c)

    rows = []
    for c in nums:
        v = psi(ref[c].values, cur[c].values, bins=10)
        rows.append({"feature": c, "psi": v})
    df = pd.DataFrame(rows).sort_values("psi", ascending=False)
    return df


def plot_psi(df: pd.DataFrame, out_fig: Path) -> None:
    if df.empty:
        # placeholder
        plt.figure(figsize=(5,3))
        plt.title("PSI — insufficient data")
        save_fig(out_fig)
        return
    top = df.head(25)
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["psi"])
    plt.gca().invert_yaxis()
    plt.xlabel("PSI"); plt.title("Feature drift (PSI)")
    save_fig(out_fig)


# ----------------- Feature importance proxy -----------------

def feature_proxy_importance(ev: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    """
    Proxy |corr(feature, y_true)| avec features "simples" disponibles.
    Si pas de features additionnelles dans events, on crée : hour_sin/cos, dow, occ courante.
    """
    # Merge sur (ts, station_id) pour aligner features à t et y_true (t+H)
    base = perf[["ts","station_id","y_true"]].merge(
        ev[["ts","station_id","occ"]].copy(), on=["ts","station_id"], how="left"
    )

    # Features temporelles
    base["hour"] = base["ts"].dt.hour
    base["dow"] = base["ts"].dt.weekday
    base["hour_sin"] = np.sin(2*np.pi*base["hour"]/24)
    base["hour_cos"] = np.cos(2*np.pi*base["hour"]/24)

    # Ajouter si présentes dans events
    cand = []
    for c in ["bikes", "capacity"]:
        if c in ev.columns: cand.append(c)

    feat_cols = ["occ","hour_sin","hour_cos","dow"] + cand
    rows = []
    for c in feat_cols:
        try:
            v = abs(pd.to_numeric(base[c], errors="coerce").corr(base["y_true"]))
            if pd.notna(v):
                rows.append({"feature": c, "abs_corr": float(v)})
        except Exception:
            pass

    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)


def plot_feature_importance(imp: pd.DataFrame, out_fig: Path) -> None:
    if imp.empty:
        plt.figure(figsize=(5,3)); plt.title("No feature correlations"); save_fig(out_fig); return
    top = imp.head(20)
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["abs_corr"])
    plt.gca().invert_yaxis()
    plt.xlabel("|corr(feature, target)|")
    plt.title("Feature importance (correlation proxy)")
    save_fig(out_fig)


# ----------------- Error trend -----------------

def error_trend(perf: pd.DataFrame) -> pd.DataFrame:
    df = perf.copy()
    df["day"] = df["ts"].dt.date
    mae = (df.assign(err=(df["y_pred"] - df["y_true"]).abs())
             .groupby("day")["err"].mean().reset_index())
    mae["day"] = pd.to_datetime(mae["day"])
    return mae


def plot_error_trend(mae: pd.DataFrame, out_fig: Path) -> None:
    plt.figure(figsize=(9,4))
    plt.plot(mae["day"], mae["err"], marker="o")
    plt.title("MAE trend (daily)")
    plt.xlabel("Day"); plt.ylabel("MAE")
    save_fig(out_fig)


# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser(description="Build monitoring assets: data health, PSI drift, feature importance proxy, error trend.")
    ap.add_argument("--events", type=Path, default=DOCS / "exports" / "events.parquet", help="events.(parquet|csv)")
    ap.add_argument("--perf", type=Path, default=DOCS / "exports" / "perf.parquet", help="perf.(parquet|csv)")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--last-days", type=int, default=14)
    ap.add_argument("--ref-pct", type=float, default=0.3, help="Part de la fenêtre utilisée comme référence pour PSI (début et fin)")
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)

    ev = read_events(args.events, last_days=args.last_days)
    perf = read_perf(args.perf, horizon=args.horizon, last_days=args.last_days)

    # Data health
    health = data_health(ev)
    ensure_dir(TABLES / "data_health.csv"); health.to_csv(TABLES / "data_health.csv", index=False)
    plot_data_health(health, FIGS / "mon_data_health.png")

    # PSI
    psi_df = compute_psi_table(ev, ref_pct=args.ref_pct)
    ensure_dir(TABLES / "psi_features.csv"); psi_df.to_csv(TABLES / "psi_features.csv", index=False)
    plot_psi(psi_df, FIGS / "mon_psi.png")

    # Feature importance proxy
    imp = feature_proxy_importance(ev, perf)
    ensure_dir(TABLES / "feature_importance_proxy.csv"); imp.to_csv(TABLES / "feature_importance_proxy.csv", index=False)
    plot_feature_importance(imp, FIGS / "mon_feature_importance.png")

    # Error trend
    mae = error_trend(perf)
    ensure_dir(TABLES / "daily_error.csv"); mae.to_csv(TABLES / "daily_error.csv", index=False)
    plot_error_trend(mae, FIGS / "mon_error_trend.png")

    print("[OK] Monitoring assets generated:")
    print(" - figs: mon_data_health.png | mon_psi.png | mon_feature_importance.png | mon_error_trend.png")
    print(" - tbls: data_health.csv | psi_features.csv | feature_importance_proxy.csv | daily_error.csv")


if __name__ == "__main__":
    main()