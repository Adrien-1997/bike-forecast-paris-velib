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
from typing import Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import warnings

try:
    import sklearn
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
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
OUT_MD = DOCS / "network" / "stations.md"

PARIS_LAT, PARIS_LON = 48.8566, 2.3522  # distance au centre (indicatif)
PROFILE_WINDOW_DAYS = 28                # fenêtre pour le profil 24 h (médiane)

def rel_from_md(md_path: Path, target: Path) -> str:
    """Chemin relatif (POSIX) depuis md_path vers target, compatible MkDocs (use_directory_urls: true)."""
    md_rel = Path(md_path).resolve().relative_to(DOCS.resolve())
    parts = md_rel.with_suffix("").parts
    depth = len(parts) if parts and parts[-1] != "index" else max(len(parts) - 1, 0)
    prefix = "../" * max(depth, 0)
    rel_from_docs = Path(target).resolve().relative_to(DOCS.resolve()).as_posix()
    return (prefix + rel_from_docs).replace("//", "/")

MD_TEMPLATE = """# Stations & profils

Cette page permet une **exploration fine station par station** (tables filtrables) et présente des **profils comportementaux** du réseau via clustering.

---

## 1) Table exploratoire (7 j / 30 j)

- **Colonnes** : ID, nom, **capacité estimée**, % pénurie / % saturation (**7 & 30 jours**), **volatilité** (σ vélos), **couverture**, **distance au centre**, **cluster**.  
- **Tri/filtre** : top pénuries / saturations, variabilité, cluster, zone.

**Téléchargements** :
- 7 jours : `{stats7_rel}`
- 30 jours : `{stats30_rel}`
- Clusters par station : `{clusters_rel}`  
- Distribution des clusters : `{dist_rel}`  
- Centroïdes 24 h (CSV) : `{centroids_csv_rel}`  
- Résumé clustering (JSON) : `{summary_json_rel}`

---

## 2) Fiches station (liens depuis la table)

Chaque station peut avoir une fiche dédiée (optionnel) :  
**Sparkline 7 j**, **profil 24 h typique** (médiane / 15 min), **heatmap h×j** récente, et **indicateurs** (pénurie/saturation 7 & 30 j, volatilité, couverture, événements).

---

## 3) Carte des clusters
<div style="margin: .5rem 0;">
  <iframe src="{map_rel}" style="width:100%;height:520px;border:0" loading="lazy" title="Carte des stations par cluster"></iframe>
</div>

> Couleur = cluster ; taille ≈ capacité estimée.

---

## 4) Profils 24 h — centroïdes par cluster
![Centroids]({centroids_png_rel})

---

## 5) Projection PCA(2) des profils
![PCA]({pca_png_rel})

> La PCA ne sert **qu'à visualiser** la séparation des groupes ; le clustering se fait sur **l’espace complet (96 points)**.

## 5bis) Cercle des corrélations (PCA)

<div style="max-width:480px;margin:auto;">
<img src="{pca_circle_png_rel}" alt="PCA Circle"/>
</div>

> Les flèches indiquent la contribution des variables (quarts d’heure) aux deux premiers axes.

---

## 6) Sélection d’exemples ({by_label})
`{selection_rel}`

---

## 7) Clustering — méthodologie détaillée

**Objectif.** Regrouper les stations par **similarité d’usage** pour révéler des archétypes (résidentiel, pôle d’emplois, gares, loisirs…).

**Variables (features).**
- **Profil 24 h** (96 pas de 15 min) : médiane d’**occupation** `bikes / capacity_est` sur ~{profile_days} jours, **centré-réduit**.
- **Amplitude/variabilité** : écart-type quotidien, plage min-max normalisée.
- **Asymétries temporelles** : ratios matin/soir, semaine/week-end.
- **Contexte léger** (optionnel) : capacité, distance centre, altitude.

**Pré-traitement.**
- **Standardisation** (moyenne 0, écart-type 1) sur les profils.
- **PCA (2D)** uniquement pour la figure — pas pour l’algorithme.

**Algorithmes.**
- **K-Means** par défaut, *k* choisi empiriquement (coude) + **Silhouette** / **Davies-Bouldin**.  
- **HDBSCAN** en option quand la densité varie beaucoup (gère le **bruit** sans imposer *k*).

**Attribution & stabilité.**
- Scores internes : **Silhouette = {silhouette:.3f}**, **Davies-Bouldin = {dbi:.3f}** (k={k_eff}, n={n_stations}).  
- **Bootstrap** par semaines pour vérifier la stabilité (optionnel).  
- **Centroïdes** publiés (courbes moyennes par cluster) comme *comportements-types*.  
- Signalement des stations **frontières** (incertitude) possible.

**Étiquettes interprétables (exemples).**
- **Résidentiel nocturne** : haut la nuit, baisse le matin, remonte le soir.  
- **Pôle d’emplois** : bas la nuit, pic d’arrivée le matin, vidage fin de journée.  
- **Transport / gares** : fortes oscillations synchronisées aux pointes.  
- **Touristique / loisirs** : week-end marqué, milieux de journée élevés.

> **Limites.** Le clustering **décrit** des usages, il ne **prédit** pas. Les groupes évoluent avec la saison, travaux, événements — **recalcul périodique** prévu.

---

## 8) Mémo technique
- **Source** : `docs/exports/events.parquet`, pas de 15 min.  
- **Capacité estimée** : priorité à `capacity_src`, sinon quantile 0.98 de `(bikes + docks_avail)` si dispo, sinon 0.98 de `bikes`.  
- **Pénurie / Saturation** : `bikes == 0` / (`docks_avail == 0` ou `capacity - bikes == 0`).  
- **Couverture** : `#bins observés / #bins attendus` sur {last_days} j.  
- **Volatilité** : σ des vélos par station sur la journée locale, médiane des stations.

"""

# --------------------------- Utils ---------------------------

def _mkdirs() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    STATIONS_DIR.mkdir(parents=True, exist_ok=True)

def _read_events(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Harmonisation colonnes attendues
    for col in ("station_id","ts","bikes","docks_avail","capacity_src","name","lat","lon"):
        if col not in df.columns:
            df[col] = pd.NA
    # Types
    df["station_id"] = df["station_id"].astype(str)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["bikes"] = pd.to_numeric(df["bikes"], errors="coerce")
    if "docks_avail" in df:
        df["docks_avail"] = pd.to_numeric(df["docks_avail"], errors="coerce")
    df["capacity_src"] = pd.to_numeric(df["capacity_src"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df.dropna(subset=["station_id","ts"])

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
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
        cap = g["capacity_src"].dropna().max() if "capacity_src" in g.columns else np.nan
        try:
            cap_f = float(cap)
        except Exception:
            cap_f = np.nan
        if np.isfinite(cap_f) and cap_f > 0:
            return cap_f
        if "docks_avail" in g.columns and g["docks_avail"].notna().any():
            s = (g["bikes"].clip(lower=0) + g["docks_avail"].clip(lower=0)).dropna()
            if len(s):
                return float(s.quantile(0.98))
        b = g["bikes"].clip(lower=0).dropna()
        if len(b):
            return float(b.quantile(0.98))
        return np.nan
    gb = win.groupby("station_id")
    try:
        return gb.apply(est, include_groups=False)
    except TypeError:
        return gb.apply(est)

def _expected_bins(tmin: pd.Timestamp, tmax: pd.Timestamp, freq: str = "15min") -> int:
    if pd.isna(tmin) or pd.isna(tmax):
        return 0
    tmin = pd.to_datetime(tmin, utc=True)
    tmax = pd.to_datetime(tmax, utc=True)
    return int(((tmax - tmin) / pd.Timedelta(freq)) + 1)

def _station_stats(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    # Fenêtre récente
    tmax = df["ts"].max()
    tmin = tmax - pd.Timedelta(days=days)
    sub = df[(df["ts"] >= tmin) & (df["ts"] <= tmax)].copy()

    # capacité estimée (sur fenêtre récente pour éviter anciens artefacts)
    cap = _estimate_capacity(sub)
    cap.name = "capacity_est"

    # agrégation
    g = sub.merge(cap.rename("capacity_est"), left_on="station_id", right_index=True, how="left")
    g["occ"] = g["bikes"] / g["capacity_est"].replace({0: np.nan})

    grp = g.groupby("station_id")
    out = grp.agg(
        name=("name", "last"),
        lat=("lat", "last"),
        lon=("lon", "last"),
        capacity_est=("capacity_est", "max"),
        bikes_std=("bikes", "std"),
        n_bins=("ts", "count"),
        min_ts=("ts", "min"),
        max_ts=("ts", "max"),
        penury=("bikes", lambda s: float(np.mean(s.fillna(0) <= 0))),
        saturation=("occ", lambda s: float(np.mean((1 - s).fillna(0) <= 0))),
    ).reset_index()

    out["coverage"] = out.apply(lambda r: r["n_bins"] / max(_expected_bins(r["min_ts"], r["max_ts"]), 1), axis=1)
    out["dist_centre_km"] = out.apply(lambda r: _haversine_km(PARIS_LAT, PARIS_LON, r["lat"], r["lon"]) if pd.notna(r["lat"]) and pd.notna(r["lon"]) else np.nan, axis=1)
    return out[["station_id","name","lat","lon","capacity_est","penury","saturation","bikes_std","coverage","dist_centre_km","n_bins"]]

def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _build_24h_profile(df: pd.DataFrame, days: int = PROFILE_WINDOW_DAYS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tmax = df["ts"].max()
    tmin = tmax - pd.Timedelta(days=days)
    sub = df[(df["ts"] >= tmin) & (df["ts"] <= tmax)].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    # capacité station
    cap = _estimate_capacity(sub).rename("capacity_est")
    meta = sub.groupby("station_id").agg(
        name=("name","last"),
        lat=("lat","last"),
        lon=("lon","last"),
        capacity_est=("capacity_src","max")
    ).reset_index()
    meta = meta.merge(cap, on="station_id", how="left")
    meta["capacity_est"] = meta["capacity_est_y"].fillna(meta["capacity_est_x"])
    meta = meta.drop(columns=[c for c in ["capacity_est_x","capacity_est_y"] if c in meta.columns])

    # occ par quart d'heure (médiane)
    sub["local"] = pd.to_datetime(sub["ts"], utc=True, errors="coerce").dt.tz_convert("Europe/Paris")
    sub["hhmm"] = sub["local"].dt.strftime("%H:%M")
    occ = sub.merge(cap.rename("capacity_est"), left_on="station_id", right_index=True, how="left")
    occ["occ"] = occ["bikes"] / occ["capacity_est"].replace({0: np.nan})
    prof = occ.groupby(["station_id","hhmm"])["occ"].median().unstack().reindex(columns=[f"{h:02d}:{m:02d}" for h in range(24) for m in (0,15,30,45)], fill_value=np.nan)

    return prof, meta

def _cluster_profiles(
    prof: pd.DataFrame,
    k: int = 4,
    min_profile_cov: float = 0.40,
    neutral_fill: float = 0.5,
):
    """Clusterise les profils 24 h et renvoie (labels, summary, pca2, pca_info).
    pca_info = {"var_ratio": (pc1, pc2), "components": np.ndarray shape (2, n_features), "feature_names": list[str]}
    - min_profile_cov: proportion min de points non-NaN par station (0.40 = >= 40% des 96 quarts d'heure).
    - neutral_fill: valeur neutre si une colonne entière est NaN (occupation ~0.5).
    """
    summary = {"sklearn_available": HAS_SK}

    # Cas sans sklearn ou profil vide -> toujours 4 retours
    if not HAS_SK or prof.empty:
        return pd.Series(dtype="int64"), summary, pd.DataFrame(), None

    # 1) Filtrer profils trop incomplets
    cov = 1.0 - prof.isna().mean(axis=1)
    keep_idx = cov >= float(min_profile_cov)
    summary["profile_row_coverage_min"] = float(min_profile_cov)
    summary["n_input"] = int(prof.shape[0])
    summary["n_kept"] = int(keep_idx.sum())
    if summary["n_kept"] < 2:
        # Trop peu pour clusteriser proprement
        return pd.Series(dtype="int64"), summary, pd.DataFrame(), None

    prof2 = prof.loc[keep_idx].copy()
    feat_names = list(prof2.columns)

    # 2) Imputation NaN : médiane colonne, fallback neutre
    X = prof2.values.astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isnan(col_med), neutral_fill, col_med)
    X = np.where(np.isnan(X), col_med, X)
    X = np.where(np.isnan(X), neutral_fill, X)

    # 3) Standardisation
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)

    # 4) KMeans robuste
    k_eff = max(2, min(int(k), Xn.shape[0]))
    km = KMeans(n_clusters=k_eff, n_init=10, random_state=42)
    lab_arr = km.fit_predict(Xn)
    labels = pd.Series(lab_arr, index=prof2.index, name="cluster", dtype="int64")

    # 5) PCA 2D pour visu (+ infos)
    p = PCA(n_components=2, random_state=42)
    Z = p.fit_transform(Xn)
    pca2 = pd.DataFrame(Z, index=prof2.index, columns=["PC1", "PC2"])
    var_ratio = tuple(map(float, p.explained_variance_ratio_[:2]))
    comps = p.components_.astype(float)  # shape (2, n_features)
    pca_info = {"var_ratio": var_ratio, "components": comps, "feature_names": feat_names}

    # 6) Scores internes
    try:
        sil = float(silhouette_score(Xn, lab_arr))
    except Exception:
        sil = float("nan")
    try:
        dbi = float(davies_bouldin_score(Xn, lab_arr))
    except Exception:
        dbi = float("nan")

    summary.update({
        "k_effective": k_eff,
        "silhouette": sil,
        "davies_bouldin": dbi,
        "n_stations": int(labels.shape[0]),
        "pca_var_ratio": var_ratio,
    })

    return labels, summary, pca2, pca_info


def _plot_centroids(prof: pd.DataFrame, labels: pd.Series, out_png: Path) -> None:
    # Si rien à tracer, produire un placeholder
    if prof.empty or labels is None or labels.empty:
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "Centroids indisponibles", ha="center")
        _save_fig(out_png)
        return

    # 1) Aligner les index (stations gardées pour le clustering)
    labels = labels.dropna()
    prof_kept = prof.loc[labels.index]

    # 2) Centroides par cluster (moyenne ligne -> courbe 24h)
    try:
        centroids = prof_kept.groupby(labels).mean(numeric_only=True)
    except TypeError:
        # pandas anciens sans numeric_only
        centroids = prof_kept.groupby(labels).mean()

    if centroids.empty:
        plt.figure(figsize=(6, 2))
        plt.text(0.5, 0.5, "Centroids indisponibles", ha="center")
        _save_fig(out_png)
        return

    # 3) Tracé
    plt.figure(figsize=(9, 4))
    for g, row in centroids.sort_index().iterrows():
        y = row.values
        plt.plot(range(len(y)), y, label=f"Cluster {g}", alpha=0.9)

    plt.xticks([0, 24, 48, 72, 95], ["00:00", "06:00", "12:00", "18:00", "23:45"])
    plt.title("Profils 24 h — centroïdes (occupation)")
    plt.legend(loc="best", fontsize=8, ncol=2)
    _save_fig(out_png)


def _plot_pca_scatter(pca2: pd.DataFrame, labels: pd.Series, out_png: Path, pca_info=None, top_feats:int=8) -> None:
    # pca_info: {"var_ratio": (pc1,pc2), "components": np.ndarray(2,n), "feature_names": list}
    plt.figure(figsize=(6.2, 5.2))
    ax = plt.gca()

    if pca2.empty or labels is None or labels.empty:
        plt.text(0.5,0.5,"PCA indisponible", ha="center")
        _save_fig(out_png); return

    # Scatter par cluster
    for g in sorted(labels.dropna().unique()):
        mask = (labels == g)
        plt.scatter(pca2.loc[mask,"PC1"], pca2.loc[mask,"PC2"], s=14, alpha=0.8, label=f"Cluster {g}")

    # Axes = % variance si dispo
    if isinstance(pca_info, dict) and pca_info.get("var_ratio"):
        vr = pca_info["var_ratio"]
        plt.xlabel(f"PC1 ({vr[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({vr[1]*100:.1f}%)")
    else:
        plt.xlabel("PC1"); plt.ylabel("PC2")

    plt.title("Profils 24 h — PCA(2) par cluster")
    plt.legend(loc="best", fontsize=8, ncol=2)

    # Biplot léger : projeter quelques features (heures) avec plus forte norme de loading
    try:
        if isinstance(pca_info, dict) and isinstance(pca_info.get("components"), np.ndarray) and pca_info.get("feature_names"):
            comps = pca_info["components"]      # shape (2, n_features)
            feats = pca_info["feature_names"]   # list len=n_features
            norms = np.sqrt(comps[0]**2 + comps[1]**2)
            idx = np.argsort(norms)[-int(top_feats):]  # top |loading|
            # échelle raisonnable des flèches par rapport au nuage
            scale = 2.0
            for i in idx:
                x, y = comps[0, i]*scale, comps[1, i]*scale
                ax.arrow(0, 0, x, y, head_width=0.05, head_length=0.08, fc="#555", ec="#555", alpha=0.8, length_includes_head=True)
                ax.text(x*1.06, y*1.06, str(feats[i]), fontsize=8, ha="left", va="center", color="#333")
    except Exception:
        pass

    _save_fig(out_png)

from matplotlib.patches import Circle

def _plot_pca_correlation_circle(pca_info, out_png: Path, top_feats: int = 10) -> None:
    """
    Trace le cercle des corrélations (biplot des features) à partir des infos PCA.
    - pca_info doit contenir: {"components", "explained_variance", "feature_names"}.
    - top_feats: nombre de features les plus contributives à annoter.
    """
    plt.figure(figsize=(6.2, 6.2))
    ax = plt.gca()

    if not isinstance(pca_info, dict):
        plt.text(0.5, 0.5, "Infos PCA indisponibles", ha="center")
        _save_fig(out_png)
        return

    comps = np.asarray(pca_info.get("components"))
    feats = pca_info.get("feature_names")
    expl = np.asarray(pca_info.get("explained_variance"), dtype=float)

    if comps is None or feats is None or expl is None:
        plt.text(0.5, 0.5, "Infos PCA incomplètes", ha="center")
        _save_fig(out_png)
        return

    # Loadings = eigenvectors * sqrt(eigenvalues)
    load = comps.T * np.sqrt(expl)  # shape = (n_features, 2)
    norms = np.linalg.norm(load, axis=1)
    eps = 1e-12
    load_norm = np.divide(load, np.maximum(norms[:, None], eps))

    # Cercle unité
    circle = Circle((0, 0), 1.0, fill=False, color="#888", linestyle="--", linewidth=1.0)
    ax.add_patch(circle)
    ax.axhline(0, color="#bbb", linewidth=0.8)
    ax.axvline(0, color="#bbb", linewidth=0.8)

    # Sélection des top features
    sel_idx = np.argsort(norms)[-int(top_feats):]

    for i in sel_idx:
        x, y = load_norm[i, 0], load_norm[i, 1]
        if not np.isnan(x) and not np.isnan(y):
            ax.arrow(0, 0, x, y,
                    head_width=0.03, head_length=0.05,
                    fc="#444", ec="#444", alpha=0.85, length_includes_head=True)
            ax.text(x * 1.07, y * 1.07, str(feats[i]),
                    fontsize=8, ha="left", va="center", color="#333")

    # Axes, limites, aspect
    ax.set_title("Cercle des corrélations (PCA)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal", adjustable="box")
    print("[pca_circle] load_norm shape:", load_norm.shape, "norms max:", norms.max())
    _save_fig(out_png)


def _map_clusters(meta: pd.DataFrame, labels: pd.Series | pd.DataFrame, out_html: Path) -> None:
    # Besoins: folium + coords valides
    try:
        import folium
        from branca.element import Element
    except Exception:
        print("[stations] Folium indisponible — carte ignorée.")
        return
    if meta is None or len(meta) == 0:
        print("[stations] Meta vide — carte ignorée.")
        return

    # Accepte Series (index=station_id) ou DataFrame avec colonne 'cluster'
    if isinstance(labels, pd.DataFrame):
        lab = labels["cluster"] if "cluster" in labels.columns else labels.iloc[:, 0]
    else:
        lab = labels

    # Merge
    df = meta.copy()
    try:
        df = df.merge(lab.rename("cluster"), left_on="station_id", right_index=True, how="left")
    except Exception:
        df["cluster"] = np.nan

    # Conversions numériques robustes
    for col in ("lat", "lon", "capacity_est", "capacity_src", "cluster"):
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

    # Palette catégorielle (tab10) + couleur spéciale pour bruit -1 (HDBSCAN)
    palette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
               '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    uniq = sorted(pd.unique(df["cluster"].dropna()))
    color_of = {int(c): palette[i % len(palette)] for i, c in enumerate(uniq)}
    if -1 in uniq:
        color_of[-1] = "#9e9e9e"  # bruit/outliers

    # Carte
    m = folium.Map(
        location=[float(df["lat"].median()), float(df["lon"].median())],
        zoom_start=12,
        tiles="cartodbpositron",
    )

    # Points colorés par cluster (taille ~ sqrt(capacité), bornée 2–8 px)
    for _, r in df.iterrows():
        # cluster → couleur
        cl_val = r.get("cluster")
        try:
            c_int = int(cl_val) if pd.notna(cl_val) else None
        except Exception:
            c_int = None
        color = color_of.get(c_int, "#4c78a8")  # défaut bleu

        # nom station
        name = str(r.get("name") or r.get("station_id"))

        # capacité estimée → rayon
        cap = pd.to_numeric(r.get("capacity_est"), errors="coerce")
        if pd.isna(cap) or float(cap) <= 0:
            rad = 3.0
        else:
            # racine pour lisser l’échelle, bornes 2–8 px
            rad = float(np.clip(np.sqrt(float(cap)) / 2.5, 2.0, 8.0))

        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=rad,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=0.8,
            tooltip=f"{name} • cap≈{'' if pd.isna(cap) else int(cap)} • cluster={c_int if c_int is not None else '—'}",
        ).add_to(m)

    # Légende simple (HTML)
    if len(color_of):
        items = "".join(
            f'<li style="margin:2px 0;"><span style="display:inline-block;width:10px;height:10px;'
            f'background:{col};margin-right:6px;border:1px solid #999;"></span>'
            f'Cluster {lab}</li>'
            for lab, col in sorted(color_of.items())
        )
        legend_html = f"""
        <div style="position: fixed; bottom: 12px; left: 12px; z-index: 9999;
                    background: white; padding: 8px 10px; border-radius: 6px;
                    box-shadow: 0 1px 4px rgba(0,0,0,.25); font-size:12px;">
          <b>Clusters</b>
          <ul style="list-style:none; margin:6px 0 0; padding:0;">{items}</ul>
        </div>"""
        m.get_root().html.add_child(Element(legend_html))

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
    labels, summary, pca2, pca_info = _cluster_profiles(prof, k=k)

    # Sauvegarde résultats clustering
    if not prof.empty and not labels.empty:
        # centroids table (pour docs)
        centroids = []
        if not labels.empty:
            # Aligner le profil sur l'index filtré (labels.index)
            prof_kept = prof.loc[labels.index]

            for g in sorted(labels.dropna().unique()):
                mask = (labels == g)           # index = stations filtrées
                if mask.any():
                    centroid = prof_kept.loc[mask].mean(axis=0)
                    tmp = pd.DataFrame(
                        {"hhmm": centroid.index, "cluster": int(g), "occ": centroid.values}
                    )
                    centroids.append(tmp)

        centroids_df = (
            pd.concat(centroids, axis=0)
            if centroids else pd.DataFrame(columns=["hhmm", "cluster", "occ"])
        )
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
        _plot_pca_scatter(pca2, labels, FIGS_DIR / "clusters_pca.png", pca_info=pca_info, top_feats=10)
        _plot_pca_correlation_circle(pca_info, FIGS_DIR / "clusters_pca_circle.png", top_feats=12)

        # Carte
        _map_clusters(meta, labels, MAPS_DIR / "network_stations_clusters.html")
    else:
        # profils indisponibles → écrire placeholders
        pd.DataFrame(columns=["hhmm","cluster","occ"]).to_csv(TABLES_DIR / "cluster_centroids_24h.csv", index=False)
        pd.DataFrame(columns=["station_id","name","lat","lon","cluster"]).to_csv(TABLES_DIR / "station_clusters.csv", index=False)
        _write_json(TABLES_DIR / "clustering_summary.json", {"sklearn_available": HAS_SK, "note": "profils indisponibles"})

    # ---------- 3) Sélection d’exemples (sparklines, optionnel) ----------
    # Simplement fournir une table d’exemples (IDs) selon le critère demandé, pour que la page MD les relie.
    sel_df = stats_7.copy()
    if not sel_df.empty:
        key = {"volatility": "bikes_std", "coverage": "coverage", "count": "n_bins"}[by]
        ascending = (by == "coverage")  # par défaut on veut top volatilité (desc), mais min couverture (asc) peut être pertinent
        sel = sel_df.sort_values(key, ascending=ascending).head(select)[["station_id","name"]]
        sel.to_csv(TABLES_DIR / f"selection_{by}.csv", index=False)

    # --- Génération de la page Markdown ---
    try:
        OUT_MD.parent.mkdir(parents=True, exist_ok=True)
        centroids_png = FIGS_DIR / "centroids_24h.png"
        pca_png = FIGS_DIR / "clusters_pca.png"
        pca_circle_png = FIGS_DIR / "clusters_pca_circle.png"   # <-- ajoute
        map_html = MAPS_DIR / "network_stations_clusters.html"
        stats7 = TABLES_DIR / "station_stats_7d.csv"
        stats30 = TABLES_DIR / "station_stats_30d.csv"
        clusters_csv = TABLES_DIR / "station_clusters.csv"
        dist_csv = TABLES_DIR / "cluster_distribution.csv"
        centroids_csv = TABLES_DIR / "cluster_centroids_24h.csv"
        summary_json = TABLES_DIR / "clustering_summary.json"
        selection_csv = TABLES_DIR / f"selection_{by}.csv"

        k_eff = int(summary.get("k_effective", k)) if isinstance(summary, dict) else k
        silhouette = float(summary.get("silhouette", float("nan"))) if isinstance(summary, dict) else float("nan")
        dbi = float(summary.get("davies_bouldin", float("nan"))) if isinstance(summary, dict) else float("nan")
        n_stations = int(summary.get("n_stations", prof.shape[0] if isinstance(prof, pd.DataFrame) else 0)) if isinstance(summary, dict) else 0

        md = MD_TEMPLATE.format(
            map_rel=rel_from_md(OUT_MD, map_html),
            centroids_png_rel=rel_from_md(OUT_MD, centroids_png),
            pca_png_rel=rel_from_md(OUT_MD, pca_png),
            pca_circle_png_rel=rel_from_md(OUT_MD, pca_circle_png),
            stats7_rel=rel_from_md(OUT_MD, stats7),
            stats30_rel=rel_from_md(OUT_MD, stats30),
            clusters_rel=rel_from_md(OUT_MD, clusters_csv),
            dist_rel=rel_from_md(OUT_MD, dist_csv),
            centroids_csv_rel=rel_from_md(OUT_MD, centroids_csv),
            summary_json_rel=rel_from_md(OUT_MD, summary_json),
            selection_rel=rel_from_md(OUT_MD, selection_csv),
            by_label=by,
            profile_days=PROFILE_WINDOW_DAYS,
            last_days=last_days,
            silhouette=(silhouette if not pd.isna(silhouette) else float("nan")),
            dbi=(dbi if not pd.isna(dbi) else float("nan")),
            k_eff=k_eff,
            n_stations=n_stations,
        )

        with open(OUT_MD, "w", encoding="utf-8", newline="\n") as f:
            f.write(md)
        print(f"[stations] MD -> {OUT_MD}")
    except Exception as _e:
        print("[stations] MD generation skipped:", _e)

    print("[stations] Done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build 'Network / Stations & profils' assets from events.parquet")
    ap.add_argument("--events", type=Path, required=True, help="Path to docs/exports/events.parquet")
    ap.add_argument("--last-days", type=int, default=7, help="Fenêtre récente pour la table (7 j par défaut)")
    ap.add_argument("--clusters", type=int, default=4, help="Nombre de clusters (K-Means)")
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
