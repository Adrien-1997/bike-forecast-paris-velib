# tools/build_network_stations.py
# Page builder — "Réseau / Stations & profils"
# - Produit une table exploratoire (pénurie/saturation 7 & 30 j, volatilité, couverture, capacité estimée, distance centre)
# - Calcule un PROFIL 24 h par station (médiane par quart d'heure, sur ~28 j), normalise, CLUSTERS (K-Means par défaut)
# - Génère les figures (centroïdes 24 h, PCA 2D) et, si folium dispo, une carte des stations colorées par cluster
#
# CLI :
#   python tools/build_network_stations.py --events docs/exports/events.parquet --last-days 7 --clusters 6 --hours 48 --select 12 --by volatility --tz Europe/Paris
#
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json
import math
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional dependencies
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    HAS_SK = True
except Exception:
    HAS_SK = False

try:
    import folium
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False


# --------------------------- Paths & constants ---------------------------

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
FIGS_DIR = ASSETS / "figs" / "network" / "stations"
TABLES_DIR = ASSETS / "tables" / "network" / "stations"
MAPS_DIR = ASSETS / "maps"
STATIONS_DIR = DOCS / "stations"  # (facultatif) pour d’éventuelles fiches

PARIS_LAT, PARIS_LON = 48.8566, 2.3522  # distance au centre (indicatif)
PROFILE_WINDOW_DAYS = 28                # fenêtre pour le profil 24 h (médiane)


# --------------------------- Utils ---------------------------

def _mkdirs() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    STATIONS_DIR.mkdir(parents=True, exist_ok=True)

def _read_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"[stations] Introuvable: {path}")
    df = pd.read_parquet(path)

    # --- ts ---
    for tc in ("ts", "tbin_utc"):
        if tc in df.columns:
            df["ts"] = pd.to_datetime(df[tc], errors="coerce")
            break
    if "ts" not in df.columns:
        raise KeyError("[stations] Colonne temporelle manquante (ts/tbin_utc)")
    df["ts"] = df["ts"].dt.floor("15min")

    # --- station_id ---
    sid = None
    for c in ("station_id", "stationcode", "stationCode", "id"):
        if c in df.columns:
            sid = c
            break
    if sid is None:
        raise KeyError("[stations] Identifiant station manquant (station_id/stationcode)")
    df["station_id"] = df[sid].astype(str)

    # --- bikes ---
    bikes_col = None
    for c in ("bikes", "nb_velos_bin", "velos_disponibles", "numBikesAvailable", "n_bikes"):
        if c in df.columns:
            bikes_col = c
            break
    if bikes_col is None:
        raise KeyError("[stations] Colonne vélos manquante (bikes/nb_velos_bin/velos_disponibles)")
    df["bikes"] = pd.to_numeric(df[bikes_col], errors="coerce")

    # --- docks (si dispo) & capacity (si dispo) ---
    cap_col = None
    for c in ("capacity", "cap", "dock_count", "n_docks_total"):
        if c in df.columns:
            cap_col = c
            break
    docks_col = None
    for c in ("docks", "docks_disponibles", "numDocksAvailable", "places_disponibles"):
        if c in df.columns:
            docks_col = c
            break
    df["docks_avail"] = pd.to_numeric(df.get(docks_col, np.nan), errors="coerce")
    df["capacity_src"] = pd.to_numeric(df.get(cap_col, np.nan), errors="coerce")

    # --- meta ---
    name_col = None
    for c in ("name", "station_name", "label"):
        if c in df.columns:
            name_col = c
            break
    df["name"] = df.get(name_col, df["station_id"]).astype(str)

    df["lat"] = pd.to_numeric(df.get("lat", np.nan), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon", np.nan), errors="coerce")

    return df[["ts", "station_id", "name", "lat", "lon", "bikes", "docks_avail", "capacity_src"]].copy()

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    # robust to NaN
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return np.nan
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2)**2
    return float(2 * r * np.arcsin(np.sqrt(a)))

def _estimate_capacity(win: pd.DataFrame) -> pd.Series:
    """Capacité estimée par station. Priorité: capacity_src ; sinon ~quantile 0.98 de (bikes + docks_avail) si docks dispo ;
       sinon quantile 0.98 de bikes.
    """
    def est(g: pd.DataFrame) -> float:
        # Récupère la meilleure capacité déclarée (nullable-friendly)
        cap = g["capacity_src"].dropna().max() if "capacity_src" in g else np.nan
        try:
            cap_f = float(cap)
        except Exception:
            cap_f = np.nan

        if np.isfinite(cap_f) and cap_f > 0:
            return cap_f

        # Sinon, estimer via docks + bikes si dispo
        if g["docks_avail"].notna().any():
            s = (g["bikes"].clip(lower=0) + g["docks_avail"].clip(lower=0)).dropna()
            if len(s):
                return float(s.quantile(0.98))

        # Sinon, fallback sur bikes
        b = g["bikes"].clip(lower=0).dropna()
        if len(b):
            return float(b.quantile(0.98))

        return np.nan

    gb = win.groupby("station_id")
    try:
        # pandas ≥ 2.2 — évite le FutureWarning
        return gb.apply(est, include_groups=False)
    except TypeError:
        # pandas plus ancien
        return gb.apply(est)



def _expected_bins(tmin: pd.Timestamp, tmax: pd.Timestamp) -> int:
    return max(1, math.ceil((tmax - tmin).total_seconds() / 60 / 15) + 1)

def _station_stats(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """KPIs par station sur les X derniers jours."""
    if days <= 0 or df.empty:
        return pd.DataFrame()
    tmax = df["ts"].max()
    tmin = tmax - pd.Timedelta(days=days)
    win = df[(df["ts"] > tmin) & (df["ts"] <= tmax)].copy()
    if win.empty:
        return pd.DataFrame()

    # couverture = (#bins observés) / (#bins attendus) moyen sur la station
    exp_bins = _expected_bins(tmin, tmax)
    g = win.groupby("station_id", as_index=False)

    stats = g.agg(
        name=("name", "last"),
        lat=("lat", "last"),
        lon=("lon", "last"),
        n_bins=("ts", "nunique"),
        bikes_mean=("bikes", "mean"),
        bikes_std=("bikes", "std"),
        penury_rate=("bikes", lambda s: (s.fillna(0) <= 0).mean()),
        sat_rate=("docks_avail", lambda s: (s == 0).mean() if s.notna().any() else np.nan),
        dock_avail_rate=("docks_avail", lambda s: (s > 0).mean() if s.notna().any() else np.nan),
        bike_avail_rate=("bikes", lambda s: (s.fillna(0) > 0).mean()),
    )

    stats["coverage"] = stats["n_bins"].clip(0, exp_bins) / float(exp_bins)

    # capacité estimée
    cap = _estimate_capacity(win)
    stats = stats.merge(cap.rename("capacity_est"), left_on="station_id", right_index=True, how="left")

    # distance au centre
    stats["dist_center_km"] = stats.apply(
        lambda r: _haversine_km(r["lat"], r["lon"], PARIS_LAT, PARIS_LON), axis=1
    )

    # formatage
    for c in ("penury_rate", "sat_rate", "dock_avail_rate", "bike_avail_rate", "coverage"):
        if c in stats.columns:
            stats[c] = stats[c].astype(float)
    stats["bikes_std"] = stats["bikes_std"].fillna(0.0).astype(float)

    return stats[[
        "station_id", "name", "lat", "lon",
        "capacity_est",
        "penury_rate", "sat_rate",
        "bike_avail_rate", "dock_avail_rate",
        "bikes_std", "coverage",
        "dist_center_km", "n_bins"
    ]].copy()

def _save_fig(path: Path) -> None:
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()

def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# --------------------------- Profiles (24 h) & Clustering ---------------------------

def _build_24h_profile(df: pd.DataFrame, days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne:
       - profiles: index station_id, colonnes hh:mm (96), valeurs = médiane d'occupation (bikes/cap) sur la fenêtre
       - meta: station_id, name, lat, lon, capacity_est
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    tmax = df["ts"].max()
    tmin = tmax - pd.Timedelta(days=days)
    win = df[(df["ts"] > tmin) & (df["ts"] <= tmax)].copy()
    if win.empty:
        return pd.DataFrame(), pd.DataFrame()

    # capacité estimée par station
    cap = _estimate_capacity(win).rename("capacity_est")
    meta = win.groupby("station_id", as_index=False).agg(
        name=("name", "last"),
        lat=("lat", "last"),
        lon=("lon", "last"),
    ).merge(cap, left_on="station_id", right_index=True, how="left")

    # ratio d'occupation (0..1) pour comparabilité
    win = win.merge(cap.rename("capacity_est"), left_on="station_id", right_index=True, how="left")
    win["occ"] = (win["bikes"].clip(lower=0) / win["capacity_est"]).clip(upper=1.0)
    # fallback si cap manquante → normaliser par quantile des bikes au lieu de capacity_est
    fallback = win["occ"].isna()
    if fallback.any():
        q98 = (win.groupby("station_id")["bikes"]
                 .transform(lambda s: s.clip(lower=0).quantile(0.98) if len(s) else np.nan))
        win.loc[fallback, "occ"] = (win.loc[fallback, "bikes"].clip(lower=0) / q98).clip(upper=1.0)

    # hh:mm locale (reste UTC si tz non géré ici ; pour la comparabilité ça suffit)
    win["hhmm"] = win["ts"].dt.strftime("%H:%M")
    prof = (win.groupby(["station_id", "hhmm"])["occ"]
                .median()
                .unstack("hhmm")
                .sort_index(axis=1))

    # s'assurer d'avoir 96 colonnes triées
    def _hhmm_key(s: str) -> int:
        return int(s[:2]) * 60 + int(s[3:])
    target_cols = sorted(prof.columns, key=_hhmm_key)
    prof = prof.reindex(columns=target_cols)

    # clamp & fill
    prof = prof.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)

    return prof, meta

def _cluster_profiles(prof: pd.DataFrame, k: int) -> Tuple[pd.Series, dict, Optional[pd.DataFrame]]:
    """Clusterise les profils (lignes=stations, colonnes=96 points). Retourne:
       - labels: pd.Series[station_id] → cluster int
       - summary: dict (scores silhouette/DB, k ajusté...)
       - pca2: DataFrame avec x,y,cluster pour scatter (ou None si SK indispo)
    """
    summary = {"sklearn_available": HAS_SK, "k_requested": k}
    if not HAS_SK or prof.empty or prof.shape[0] < 2:
        return pd.Series(dtype="int"), summary, None

    n_samples = prof.shape[0]
    k_eff = max(2, min(k, n_samples))
    if k_eff != k:
        summary["k_effective"] = k_eff
    else:
        summary["k_effective"] = k

    X = prof.values.astype("float32")
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=k_eff, n_init="auto", random_state=42)
    labels = km.fit_predict(Xs)

    sil = float(silhouette_score(Xs, labels)) if n_samples > k_eff else float("nan")
    dbi = float(davies_bouldin_score(Xs, labels)) if k_eff >= 2 else float("nan")

    summary.update({
        "silhouette": sil,
        "davies_bouldin": dbi,
        "n_stations": int(n_samples),
        "n_features": int(prof.shape[1]),
    })

    # PCA 2D pour visualisation
    try:
        pca = PCA(n_components=2, random_state=42)
        emb = pca.fit_transform(Xs)
        pca2 = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1],
                             "cluster": labels}, index=prof.index).reset_index(names="station_id")
    except Exception:
        pca2 = None

    return pd.Series(labels, index=prof.index, name="cluster"), summary, pca2


# --------------------------- Figures & Map ---------------------------

def _plot_centroids(prof: pd.DataFrame, labels: pd.Series, out_png: Path) -> None:
    """Trace les centroïdes (occ 0..1) par cluster (courbes sur 96 points)."""
    if prof.empty or labels.empty:
        plt.figure(figsize=(8, 4)); plt.title("Centroids (no data)"); _save_fig(out_png); return

    df = prof.copy()
    df["cluster"] = labels
    groups = sorted(df["cluster"].unique())
    hhmm = list(prof.columns)

    plt.figure(figsize=(11, 5.5))
    for g in groups:
        centroid = df[df["cluster"] == g].drop(columns=["cluster"]).mean(axis=0).values
        plt.plot(hhmm, centroid, label=f"Cluster {g}", linewidth=2)
    plt.ylim(0, 1)
    plt.ylabel("Occupation (ratio)")
    plt.xlabel("Heure (hh:mm)")
    plt.title("Profils 24 h — Centroïdes par cluster")
    plt.xticks([hhmm[i] for i in range(0, len(hhmm), 8)])  # une étiquette toutes les 2 h
    plt.legend(loc="best", ncol=2)
    _save_fig(out_png)

def _plot_pca_scatter(pca2: Optional[pd.DataFrame], out_png: Path) -> None:
    if pca2 is None or pca2.empty:
        plt.figure(figsize=(6, 5)); plt.title("PCA (indisponible)"); _save_fig(out_png); return
    plt.figure(figsize=(7.5, 6))
    for g, sub in pca2.groupby("cluster"):
        plt.scatter(sub["x"], sub["y"], s=12, label=f"Cluster {g}", alpha=0.8)
    plt.title("Profils 24 h — PCA(2) par cluster")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(loc="best", fontsize=8, ncol=2)
    _save_fig(out_png)

def _map_clusters(meta: pd.DataFrame, labels: pd.Series | pd.DataFrame, out_html: Path) -> None:
    # Accepte Series (index=station_id, name='cluster') ou DataFrame avec colonne 'cluster'
    if isinstance(labels, pd.DataFrame):
        lab = labels["cluster"] if "cluster" in labels.columns else labels.iloc[:, 0]
    else:
        lab = labels
    df = meta.copy()

    # Merge cluster
    df = df.merge(lab.rename("cluster"), left_on="station_id", right_index=True, how="left")

    # Conversions numériques robustes
    for col in ("lat", "lon", "capacity_est", "capacity_src"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filtrer coordonnées valides
    n0 = len(df)
    df = df.dropna(subset=["lat", "lon"])
    df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]
    dropped = n0 - len(df)
    if dropped:
        print(f"[map] Skipped {dropped} station(s) without valid lat/lon.")
    if df.empty:
        print("[map] No stations with valid coordinates; skipping map export.")
        return

    # Carte centrée sur la médiane des coordonnées
    m = folium.Map(
        location=[float(df["lat"].median()), float(df["lon"].median())],
        zoom_start=12,
        tiles="cartodbpositron",
    )

    # Points
    for _, r in df.iterrows():
        cap = r["capacity_est"] if "capacity_est" in r else np.nan
        name = r["name"] if "name" in r and pd.notna(r["name"]) else str(r["station_id"])
        # Rayon borné pour rester lisible
        rad = 5.0 if pd.isna(cap) else float(np.clip(cap / 2.0, 3.0, 12.0))

        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=rad,
            weight=1,
            fill=True,
            fill_opacity=0.7,
            tooltip=f"{name} • cap≈{'' if pd.isna(cap) else int(cap)} • cluster={r['cluster']}",
        ).add_to(m)

    m.save(out_html)
    print(f"[map] Saved {out_html}")



# --------------------------- Main ---------------------------

def main(events_path: Path, last_days: int, k: int, hours: int, select: int, by: str, tz: str | None) -> None:
    _mkdirs()
    events = _read_events(events_path)
    if events.empty:
        raise SystemExit("[stations] events.parquet est vide")

    # ---------- 1) Tables exploratoires 7 j & 30 j ----------
    stats_7 = _station_stats(events, days=7)
    stats_30 = _station_stats(events, days=30)

    stats_7.to_csv(TABLES_DIR / "station_stats_7d.csv", index=False)
    stats_30.to_csv(TABLES_DIR / "station_stats_30d.csv", index=False)
    print("[stations] Tables 7j & 30j écrites.")

    # ---------- 2) Profils 24 h & Clustering ----------
    prof, meta = _build_24h_profile(events, days=PROFILE_WINDOW_DAYS)
    labels, summary, pca2 = _cluster_profiles(prof, k=k)

    # Sauvegarde résultats clustering
    if not prof.empty:
        # centroids table (pour docs)
        centroids = []
        for g in sorted(labels.dropna().unique()) if not labels.empty else []:
            centroid = prof[labels == g].mean(axis=0)
            tmp = pd.DataFrame({"hhmm": centroid.index, "cluster": int(g), "occ": centroid.values})
            centroids.append(tmp)
        centroids_df = pd.concat(centroids, axis=0) if centroids else pd.DataFrame(columns=["hhmm","cluster","occ"])
        centroids_df.to_csv(TABLES_DIR / "cluster_centroids_24h.csv", index=False)

        # station_clusters table
        clusters_df = meta.merge(labels.rename("cluster"), left_on="station_id", right_index=True, how="left")
        clusters_df.to_csv(TABLES_DIR / "station_clusters.csv", index=False)

        # distribution par cluster
        if not labels.empty:
            dist = labels.value_counts(dropna=True).sort_index().rename_axis("cluster").reset_index(name="count")
        else:
            dist = pd.DataFrame(columns=["cluster", "count"])
        dist.to_csv(TABLES_DIR / "cluster_distribution.csv", index=False)

        # scores JSON
        _write_json(TABLES_DIR / "clustering_summary.json", summary)

        # Figures
        _plot_centroids(prof, labels, FIGS_DIR / "centroids_24h.png")
        _plot_pca_scatter(pca2, FIGS_DIR / "clusters_pca.png")

        # Carte
        _map_clusters(meta, labels, MAPS_DIR / "network_stations_clusters.html")
    else:
        # profils indisponibles → écrire placeholders
        pd.DataFrame(columns=["hhmm","cluster","occ"]).to_csv(TABLES_DIR / "cluster_centroids_24h.csv", index=False)
        pd.DataFrame(columns=["station_id","name","lat","lon","capacity_est","cluster"]).to_csv(TABLES_DIR / "station_clusters.csv", index=False)
        _write_json(TABLES_DIR / "clustering_summary.json", {"sklearn_available": HAS_SK, "note": "profils indisponibles"})

    # ---------- 3) Sélection d’exemples (sparklines, optionnel) ----------
    # Simplement fournir une table d’exemples (IDs) selon le critère demandé, pour que la page MD les relie.
    sel_df = stats_7.copy()
    if not sel_df.empty:
        key = {"volatility": "bikes_std", "coverage": "coverage", "count": "n_bins"}[by]
        ascending = (by == "coverage")  # par défaut on veut top volatilité (desc), mais min couverture (asc) peut être pertinent
        sel = sel_df.sort_values(key, ascending=ascending).head(select)[["station_id","name"]]
        sel.to_csv(TABLES_DIR / f"selection_{by}.csv", index=False)

    print("[stations] Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build 'Network / Stations & profils' assets from events.parquet")
    ap.add_argument("--events", type=Path, required=True, help="Path to docs/exports/events.parquet")
    ap.add_argument("--last-days", type=int, default=7, help="Fenêtre récente pour la table (7 j par défaut)")
    ap.add_argument("--clusters", type=int, default=6, help="Nombre de clusters (K-Means)")
    ap.add_argument("--hours", type=int, default=48, help="(Réservé) fenêtre d'illustration station (heures)")
    ap.add_argument("--select", type=int, default=12, help="Nombre de stations à sélectionner pour exemples")
    ap.add_argument("--by", type=str, default="volatility", choices=["volatility","coverage","count"], help="Critère de sélection d'exemples")
    ap.add_argument("--tz", type=str, default=None, help="Fuseau d'affichage (non utilisé pour le clustering)")
    args = ap.parse_args()

    main(
        events_path=args.events,
        last_days=args.last_days,
        k=args.clusters,
        hours=args.hours,
        select=args.select,
        by=args.by,
        tz=args.tz,
    )
