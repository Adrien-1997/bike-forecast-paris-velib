# tools/build_usage.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# On s'appuie sur l'ingestion normalisée
from datasets import load_normalized

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIGS = DOCS / "assets" / "figs"
TABLES = DOCS / "assets" / "tables"
MAPS = DOCS / "assets" / "maps"

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def save_fig(path: Path) -> None:
    ensure_dir(path)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

def read_events(events_path: Optional[Path], last_days: Optional[int]) -> pd.DataFrame:
    """
    Charge events [ts, station_id, bikes, capacity, (lat,lon,name)]
    - Si events_path est fourni et existe, on le lit
    - Sinon on part de docs/exports/velib.parquet via load_normalized()
    """
    if events_path and Path(events_path).exists():
        if str(events_path).lower().endswith(".parquet"):
            ev = pd.read_parquet(events_path)
        else:
            ev = pd.read_csv(events_path)
    else:
        # fallback : construit depuis velib.parquet
        events, perf, meta, mapping, used = load_normalized(
            DOCS / "exports" / "velib.parquet",
            horizon_minutes=60,
            last_days=last_days
        )
        ev = events

    # Sanity
    need = {"ts", "station_id", "bikes", "capacity"}
    miss = need - set(ev.columns)
    if miss:
        raise ValueError(f"Events missing columns: {miss}")

    # Datetime & tri
    ev["ts"] = pd.to_datetime(ev["ts"], utc=False, errors="coerce")
    ev = ev.sort_values(["station_id", "ts"]).reset_index(drop=True)

    # Occupancy
    cap = ev["capacity"].replace(0, np.nan)
    ev["occ"] = (ev["bikes"] / cap).clip(0, 1)
    return ev

def kpis_usage(ev: pd.DataFrame) -> pd.DataFrame:
    occ = ev["occ"].dropna()
    out = pd.DataFrame([{
        "occ_mean": float(occ.mean()),
        "occ_median": float(occ.median()),
        "occ_iqr": float(occ.quantile(0.75) - occ.quantile(0.25)),
        "share_low_<10%": float((occ < 0.10).mean()),
        "share_high_>90%": float((occ > 0.90).mean()),
    }])
    return out

def daily_timeseries(ev: pd.DataFrame) -> pd.DataFrame:
    g = ev.set_index("ts").sort_index()
    ts = g["occ"].resample("D").agg(["mean", "median"])
    ts["q10"] = g["occ"].resample("D").quantile(0.10)
    ts["q90"] = g["occ"].resample("D").quantile(0.90)
    return ts.reset_index()

def plot_daily_timeseries(ts: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(9,4))
    plt.plot(ts["ts"], ts["mean"], label="mean")
    plt.plot(ts["ts"], ts["median"], label="median")
    plt.fill_between(ts["ts"], ts["q10"], ts["q90"], alpha=0.2, label="10–90%")
    plt.title("Occupancy — daily mean/median with 10–90% band")
    plt.xlabel("Date"); plt.ylabel("Occupancy")
    plt.legend()
    save_fig(out)

def plot_hourly_profile(ev: pd.DataFrame, out: Path) -> None:
    ev["hour"] = ev["ts"].dt.hour
    prof = ev.groupby("hour")["occ"].agg(["mean","median","std"]).reset_index()
    plt.figure(figsize=(8,4))
    plt.plot(prof["hour"], prof["mean"], marker="o", label="mean")
    plt.plot(prof["hour"], prof["median"], marker="s", label="median")
    plt.title("Hourly profile (mean & median)")
    plt.xlabel("Hour of day"); plt.ylabel("Occupancy")
    plt.xticks(range(0,24,2))
    plt.legend()
    save_fig(out)


def plot_heatmap_hour_dow(ev: pd.DataFrame, out: Path) -> None:
    ev["hour"] = ev["ts"].dt.hour
    ev["dow"] = ev["ts"].dt.weekday  # 0=Mon
    # Assure que 'occ' est bien numérique
    ev["occ"] = pd.to_numeric(ev["occ"], errors="coerce")
    mat = ev.groupby(["dow", "hour"])["occ"].mean().unstack(fill_value=np.nan)

    plt.figure(figsize=(9, 4.8))
    # Coerce -> float + NA-safe
    try:
        arr = mat.to_numpy(dtype=float)
    except Exception:
        arr = mat.values.astype(float, copy=False)

    # Si tableau vide ou full-NaN, on met un placeholder
    if arr.size == 0 or np.all(np.isnan(arr)):
        arr = np.zeros((1, 1), dtype=float)

    im = plt.imshow(arr, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks(ticks=range(7), labels=["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"])
    plt.xticks(ticks=range(24), labels=[str(h) for h in range(24)])
    plt.title("Occupancy heatmap — hour × weekday (mean)")
    plt.xlabel("Hour"); plt.ylabel("Day of week")
    save_fig(out)


def plot_station_variability(ev: pd.DataFrame, out: Path, tables_dir: Path) -> None:
    # agrégats de base
    st = ev.groupby("station_id")["occ"].agg(["mean", "std"]).reset_index()

    # masques NA-safe
    m_low = ev["occ"].lt(0.10)    # True/False/NA
    m_high = ev["occ"].gt(0.90)   # True/False/NA

    tmp = ev.assign(
        low=m_low.fillna(False).astype(int),   # NA -> False -> 0
        high=m_high.fillna(False).astype(int)
    )

    # parts (moyenne des 0/1) par station
    shares = (tmp.groupby("station_id")[["low", "high"]]
                .mean()
                .rename(columns={"low": "share_low", "high": "share_high"}))

    st = st.merge(shares, on="station_id", how="left")
    st["tension_score"] = st["share_low"].fillna(0) + st["share_high"].fillna(0)

    # export table
    (tables_dir / "station_stats.csv").parent.mkdir(parents=True, exist_ok=True)
    st.to_csv(tables_dir / "station_stats.csv", index=False)

    # scatter mean vs std
    plt.figure(figsize=(6.8, 6))
    plt.scatter(st["mean"], st["std"], s=8)
    plt.xlabel("Mean occupancy"); plt.ylabel("Std occupancy")
    plt.title("Stations: mean vs variability (higher std = more volatile)")
    save_fig(out)


def do_clustering(ev: pd.DataFrame, figs_dir: Path, tables_dir: Path, maps_dir: Path) -> None:
    # signature 24h par station (moyenne d'occupation par heure)
    ev["hour"] = ev["ts"].dt.hour
    feat = ev.groupby(["station_id","hour"])["occ"].mean().unstack(fill_value=0.0)
    # enrichissements
    feat["occ_mean"] = feat.mean(axis=1)
    feat["occ_std"]  = feat.std(axis=1)

    labels = None
    k = 5
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(feat.values)
        labels = km.labels_
        share = pd.Series(labels).value_counts(normalize=True).sort_index()
        # plot share
        plt.figure(figsize=(6,4))
        plt.bar(share.index.astype(str), share.values)
        plt.xlabel("Cluster"); plt.ylabel("Share"); plt.title("Cluster share")
        save_fig(figs_dir / "usage_clusters_share.png")

        # PCA embedding
        try:
            from sklearn.decomposition import PCA
            comp = PCA(n_components=2, random_state=42).fit_transform(feat.values)
            plt.figure(figsize=(6,5))
            plt.scatter(comp[:,0], comp[:,1], s=6)
            plt.title("Stations embedding (PCA 2D)")
            plt.xlabel("PC1"); plt.ylabel("PC2")
            save_fig(figs_dir / "usage_clusters_embedding.png")
        except Exception:
            pass

    except Exception:
        # pas de sklearn -> on tag tout en cluster -1
        labels = np.full(shape=(feat.shape[0],), fill_value=-1)

    # Export affectations
    clusters = pd.DataFrame({"station_id": feat.index.astype(str), "cluster": labels})
    ensure_dir(tables_dir / "station_clusters.csv")
    clusters.to_csv(tables_dir / "station_clusters.csv", index=False)

    # Carte folium si lat/lon dispo
    try:
        import folium
        st_meta = (ev.drop_duplicates("station_id")[["station_id","lat","lon","name"]]
                     if set(["lat","lon"]).issubset(ev.columns) else
                   ev.drop_duplicates("station_id")[["station_id"]])
        st_cl = st_meta.merge(clusters, on="station_id", how="left")
        m = folium.Map(location=[48.8566, 2.3522], zoom_start=12, tiles="cartodbpositron")
        for _, r in st_cl.iterrows():
            if {"lat","lon"}.issubset(st_cl.columns) and not (pd.isna(r["lat"]) or pd.isna(r["lon"])):
                folium.CircleMarker(
                    location=[float(r["lat"]), float(r["lon"])],
                    radius=4,
                    popup=f"{r.get('name', r['station_id'])} — C{int(r['cluster'])}",
                ).add_to(m)
        ensure_dir(maps_dir / "usage_map.html"); m.save(str(maps_dir / "usage_map.html"))
    except Exception:
        # ok, pas de carte
        pass

def main():
    ap = argparse.ArgumentParser(description="Build network usage report assets (figs + tables).")
    ap.add_argument("--events", type=Path, default=DOCS / "exports" / "events.parquet",
                    help="Chemin vers events.(parquet|csv). S'il n'existe pas, on utilisera velib.parquet via datasets.load_normalized().")
    ap.add_argument("--last-days", type=int, default=7, help="Fenêtre temporelle (jours) pour l'analyse.")
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    MAPS.mkdir(parents=True, exist_ok=True)

    ev = read_events(args.events, last_days=args.last_days)

    # --- KPI globaux ---
    kpi = kpis_usage(ev)
    ensure_dir(TABLES / "kpis_usage.csv"); kpi.to_csv(TABLES / "kpis_usage.csv", index=False)

    # --- Séries quotidiennes ---
    ts = daily_timeseries(ev)
    ensure_dir(TABLES / "usage_daily.csv"); ts.to_csv(TABLES / "usage_daily.csv", index=False)
    plot_daily_timeseries(ts, FIGS / "usage_daily_timeseries.png")

    # --- Profils horaires ---
    plot_hourly_profile(ev, FIGS / "usage_hourly_profile.png")

    # --- Heatmap heure × jour ---
    plot_heatmap_hour_dow(ev, FIGS / "usage_heatmap_hour_dow.png")

    # --- Variabilité par station ---
    plot_station_variability(ev, FIGS / "usage_station_variability.png", TABLES)

    # --- Clustering + Carte ---
    do_clustering(ev, FIGS, TABLES, MAPS)

    print("[OK] Usage assets generated:")
    print(" - figs: usage_daily_timeseries.png | usage_hourly_profile.png | usage_heatmap_hour_dow.png | usage_station_variability.png | usage_clusters_share.png | usage_clusters_embedding.png")
    print(" - map : assets/maps/usage_map.html (si folium dispo)")
    print(" - tbls: kpis_usage.csv | usage_daily.csv | station_stats.csv | station_clusters.csv")

if __name__ == "__main__":
    main()
