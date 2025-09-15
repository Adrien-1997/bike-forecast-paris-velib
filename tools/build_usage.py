# tools/build_usage.py
# Analyse d'usage (à partir de docs/exports/events.parquet) et génération d'assets :
#  FIGS :
#   - usage_daily_timeseries.png
#   - usage_hourly_profile.png
#   - usage_heatmap_hour_dow.png
#   - usage_station_variability.png
#   - usage_clusters_share.png
#   - usage_clusters_embedding.png
#  TABLES :
#   - kpis_usage.csv
#   - usage_daily.csv
#   - station_stats.csv
#   - station_clusters.csv
#  MAP (si folium dispo) :
#   - assets/maps/usage_map.html
#
# Règles :
# - Les données restent en UTC. La conversion de fuseau n'est utilisée que pour l'affichage (--tz).
# - On travaille sur la fenêtre des N derniers jours (--last-days).
# - Les clusters utilisent le profil horaire moyen par station (0-23). KMeans si dispo, fallback doux sinon.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXPORTS = DOCS / "exports"
ASSETS = DOCS / "assets"
FIGS = ASSETS / "figs"
TBLS = ASSETS / "tables"
MAPS = ASSETS / "maps"

plt.rcParams.update({"figure.autolayout": True})


# ------------------------- utils -------------------------

def _ensure_dirs():
    FIGS.mkdir(parents=True, exist_ok=True)
    TBLS.mkdir(parents=True, exist_ok=True)
    MAPS.mkdir(parents=True, exist_ok=True)


def _to_local(ts: pd.Series, tz: Optional[str]) -> pd.Series:
    if tz:
        return pd.to_datetime(ts, utc=True, errors="coerce").dt.tz_convert(tz)
    return pd.to_datetime(ts, utc=False, errors="coerce")


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


# ------------------------- KPI & agrégats -------------------------

def compute_kpis(events: pd.DataFrame) -> pd.DataFrame:
    # KPIs simples sur la fenêtre : nb stations actives, moyenne bikes, taux d'occupation moyen
    n_stations = events["station_id"].nunique()
    bikes_mean = float(events["bikes"].mean()) if "bikes" in events else np.nan
    occ_mean = float(events["occ"].mean()) if "occ" in events else np.nan
    cap_mean = float(events["capacity"].mean()) if "capacity" in events else np.nan
    return pd.DataFrame([{
        "stations": n_stations,
        "bikes_mean": bikes_mean,
        "occ_mean": occ_mean,
        "capacity_mean": cap_mean,
        "rows": int(len(events))
    }])


def daily_series(events: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    tloc = _to_local(events["ts"], tz)
    df = events.copy()
    df["date"] = tloc.dt.date
    # series quotidiennes (moyennes sur la journée)
    cols = [c for c in ["bikes", "occ", "capacity"] if c in df.columns]
    agg = df.groupby("date")[cols].mean().reset_index()
    agg["date"] = pd.to_datetime(agg["date"])
    return agg


def hourly_profile(events: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    tloc = _to_local(events["ts"], tz)
    df = events.copy()
    df["hour"] = tloc.dt.hour
    cols = [c for c in ["bikes", "occ"] if c in df.columns]
    prof = df.groupby("hour")[cols].mean().reset_index()
    return prof


def hour_dow_heatmap(events: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    tloc = _to_local(events["ts"], tz)
    df = events.copy()
    df["hour"] = tloc.dt.hour
    df["dow"] = tloc.dt.dayofweek
    # MAE-like sur l'usage ? Ici on trace la moyenne des bikes (ou occ) par (dow,hour)
    metric = "bikes" if "bikes" in df.columns else ("occ" if "occ" in df.columns else None)
    if metric is None:
        return pd.DataFrame()
    pivot = df.pivot_table(index="dow", columns="hour", values=metric, aggfunc="mean")
    return pivot


def station_variability(events: pd.DataFrame) -> pd.DataFrame:
    # variabilité par station (std des bikes & occ sur la fenêtre)
    cols = [c for c in ["bikes", "occ"] if c in events.columns]
    if not cols:
        return pd.DataFrame()
    g = (events.groupby("station_id")[cols]
               .agg(["mean", "std"])
               .reset_index())
    # aplatit MultiIndex
    g.columns = ["station_id"] + [f"{m}_{s}" for m, s in g.columns.tolist()[1:]]
    return g


def station_hourly_matrix(events: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    """Matrice (station x 24h) du profil horaire moyen de bikes (fallback occ)."""
    metric = "bikes" if "bikes" in events.columns else ("occ" if "occ" in events.columns else None)
    if metric is None:
        return pd.DataFrame()

    tloc = _to_local(events["ts"], tz)
    df = events.copy()
    df["hour"] = tloc.dt.hour
    mat = (df.groupby(["station_id", "hour"])[metric]
             .mean()
             .unstack(fill_value=np.nan)
             .reindex(columns=range(24)))
    # normalisation par station (optionnel) : met à l’échelle le profil
    mat = mat.div(mat.max(axis=1).replace(0, np.nan), axis=0)
    return mat


# ------------------------- Clustering (facultatif) -------------------------

def cluster_stations(hour_mat: pd.DataFrame, n_clusters: int = 6):
    """KMeans (si dispo). Fallback: quantiles arbitraux sur un score simple."""
    if hour_mat.empty:
        return pd.Series(dtype="int"), pd.DataFrame()

    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        X = hour_mat.fillna(0.0).values
        km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        # embedding 2D via PCA pour visualisation
        pca = PCA(n_components=2, random_state=42)
        emb = pca.fit_transform(X)
        emb = pd.DataFrame(emb, index=hour_mat.index, columns=["x", "y"])
        return pd.Series(labels, index=hour_mat.index), emb
    except Exception:
        # fallback : score = heure de pic moyen => quantiles
        peaks = hour_mat.idxmax(axis=1)
        # buckets par quantiles (0..n_clusters-1)
        q = pd.qcut(peaks, q=min(n_clusters, peaks.nunique()), labels=False, duplicates="drop")
        return q.astype(int), pd.DataFrame({"x": peaks.astype(float) % 24, "y": hour_mat.max(axis=1).values},
                                           index=hour_mat.index)


# ------------------------- Plots -------------------------

def plot_usage_daily(agg: pd.DataFrame, tz: Optional[str], out: Path):
    if agg.empty: return
    plt.figure(figsize=(10, 4))
    if "bikes" in agg: plt.plot(agg["date"], agg["bikes"], label="bikes (mean)")
    if "occ" in agg:   plt.plot(agg["date"], agg["occ"], label="occ (mean)")
    plt.title("Daily usage (mean per day)")
    plt.xlabel(f"Date ({tz or 'UTC'})"); plt.ylabel("Value")
    plt.legend()
    _savefig(out)


def plot_usage_hourly(prof: pd.DataFrame, out: Path):
    if prof.empty: return
    plt.figure(figsize=(10, 4))
    if "bikes" in prof: plt.plot(prof["hour"], prof["bikes"], label="bikes (mean)")
    if "occ" in prof:   plt.plot(prof["hour"], prof["occ"], label="occ (mean)")
    plt.title("Hourly profile (mean)")
    plt.xlabel("Hour of day"); plt.ylabel("Value")
    plt.xticks(range(0, 24, 2))
    plt.legend()
    _savefig(out)


def plot_heatmap_hour_dow(pivot: pd.DataFrame, out: Path):
    if pivot.empty: return
    plt.figure(figsize=(10, 4))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.yticks(range(7), ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    plt.xticks(range(24), range(24))
    plt.colorbar(label="Mean value")
    plt.title("Mean value by Hour x DayOfWeek")
    _savefig(out)


def plot_station_variability(stats: pd.DataFrame, out: Path):
    if stats.empty: return
    plt.figure(figsize=(8, 4))
    col = "bikes_std" if "bikes_std" in stats.columns else ("occ_std" if "occ_std" in stats.columns else None)
    if col is None: return
    plt.hist(stats[col].dropna(), bins=40)
    plt.title("Station variability (std over window)")
    plt.xlabel(col); plt.ylabel("Stations")
    _savefig(out)


def plot_clusters_share(labels: pd.Series, out: Path):
    if labels.empty: return
    plt.figure(figsize=(6, 4))
    share = labels.value_counts(normalize=True).sort_index()
    plt.bar(share.index.astype(str), (share * 100.0).values)
    plt.title("Cluster share (%)")
    plt.xlabel("Cluster"); plt.ylabel("% of stations")
    _savefig(out)


def plot_clusters_embedding(emb: pd.DataFrame, labels: pd.Series, out: Path):
    if emb.empty or labels.empty: return
    plt.figure(figsize=(6, 5))
    for lbl in sorted(labels.unique()):
        mask = labels == lbl
        plt.scatter(emb.loc[mask, "x"], emb.loc[mask, "y"], s=12, label=f"C{lbl}")
    plt.legend(markerscale=2)
    plt.title("Station embedding (clusters)")
    plt.xlabel("x"); plt.ylabel("y")
    _savefig(out)


# ------------------------- Folium map (optionnel) -------------------------

def make_map(events: pd.DataFrame, labels: pd.Series, out_html: Path):
    try:
        import folium
    except Exception:
        return  # folium non dispo → on skippe silencieusement

    # dernier snapshot par station pour coords & nom
    cols = [c for c in ["station_id","name","lat","lon"] if c in events.columns]
    if not {"station_id","lat","lon"}.issubset(cols):
        return
    last = (events.sort_values("ts")
                  .dropna(subset=["lat","lon"])
                  .drop_duplicates(subset=["station_id"], keep="last"))[cols].copy()
    last["cluster"] = labels.reindex(last["station_id"].astype(str)).values

    # centre moyen
    lat0 = float(last["lat"].mean())
    lon0 = float(last["lon"].mean())
    m = folium.Map(location=[lat0, lon0], zoom_start=12, tiles="cartodbpositron")

    def color(lbl):
        palette = ["blue","red","green","purple","orange","darkred","lightred","beige",
                   "darkblue","darkgreen","cadetblue","darkpurple","white","pink","lightblue",
                   "lightgreen","gray","black","lightgray"]
        try:
            return palette[int(lbl) % len(palette)]
        except Exception:
            return "gray"

    for _, r in last.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=4,
            color=color(r["cluster"]),
            fill=True,
            fill_opacity=0.8,
            popup=f"{r.get('name','?')} (#{r['station_id']}) — C{r['cluster']}"
        ).add_to(m)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))


# ------------------------- main -------------------------

def main(events_path: Path, last_days: int, tz: Optional[str], n_clusters: int):
    _ensure_dirs()

    events = pd.read_parquet(events_path)
    # types
    events["ts"] = pd.to_datetime(events["ts"], utc=False, errors="coerce")
    events["station_id"] = events["station_id"].astype(str)

    # fenêtre
    if last_days and last_days > 0:
        tmax = events["ts"].max()
        if pd.notna(tmax):
            events = events[events["ts"] >= (tmax - pd.Timedelta(days=last_days))].copy()

    # KPI & agrégats
    kpis = compute_kpis(events)
    kpis.to_csv(TBLS / "kpis_usage.csv", index=False)

    dly = daily_series(events, tz)
    dly.to_csv(TBLS / "usage_daily.csv", index=False)

    prof = hourly_profile(events, tz)

    heat = hour_dow_heatmap(events, tz)

    stats = station_variability(events)
    stats.to_csv(TBLS / "station_stats.csv", index=False)

    # Clustering profils horaires par station (facultatif)
    hour_mat = station_hourly_matrix(events, tz=None)  # normalisation indépendante de tz
    labels, emb = cluster_stations(hour_mat, n_clusters=n_clusters)
    if not labels.empty:
        pd.DataFrame({"station_id": labels.index.astype(str), "cluster": labels.values}).to_csv(
            TBLS / "station_clusters.csv", index=False
        )

    # PLots
    plot_usage_daily(dly, tz, FIGS / "usage_daily_timeseries.png")
    plot_usage_hourly(prof, FIGS / "usage_hourly_profile.png")
    plot_heatmap_hour_dow(heat, FIGS / "usage_heatmap_hour_dow.png")
    plot_station_variability(stats, FIGS / "usage_station_variability.png")
    plot_clusters_share(labels, FIGS / "usage_clusters_share.png")
    plot_clusters_embedding(emb, labels, FIGS / "usage_clusters_embedding.png")

    # Map (optionnelle)
    try:
        make_map(events, labels, MAPS / "usage_map.html")
        map_msg = " | map : assets/maps/usage_map.html (si folium dispo)"
    except Exception:
        map_msg = ""
    print("[OK] Usage assets generated:")
    print(" - figs: usage_daily_timeseries.png | usage_hourly_profile.png | "
          "usage_heatmap_hour_dow.png | usage_station_variability.png | usage_clusters_share.png | usage_clusters_embedding.png")
    if map_msg:
        print(" -" + map_msg)
    print(" - tbls: kpis_usage.csv | usage_daily.csv | station_stats.csv | station_clusters.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build usage analytics from events.parquet")
    ap.add_argument("--events", type=Path, default=EXPORTS / "events.parquet")
    ap.add_argument("--last-days", type=int, default=7)
    ap.add_argument("--tz", type=str, default=None, help="Affichage (ex: Europe/Paris). Données restent en UTC.")
    ap.add_argument("--clusters", type=int, default=6, help="Nombre de clusters KMeans (si sklearn dispo).")
    args = ap.parse_args()
    main(events_path=args.events, last_days=args.last_days, tz=args.tz, n_clusters=args.clusters)
