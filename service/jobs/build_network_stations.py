# service/jobs/build_network_stations.py

"""
Vélib’ Forecast — Clustering & profils des stations (LATEST ONLY).

Rôle
----
Ce job analyse le comportement des stations sur une fenêtre temporelle récente
et produit les artefacts JSON pour la page :

    Monitoring / Network / Stations

Entrée (GCS)
------------
- Exports d’évènements, un fichier par jour :
    {GCS_EXPORTS_PREFIX}/events_YYYY-MM-DD.parquet

Le job lit une fenêtre UTC stricte de `WINDOW_DAYS` jours et agrège, par station :

    - capacity_est        : capacité estimée (médiane de capacity si dispo)
    - volatility          : écart-type du taux d’occupation sur la fenêtre
    - penury_rate         : moyenne de is_penury
    - saturation_rate     : moyenne de is_saturation
    - coverage_pct        : fraction de time bins "attendus" effectivement vus
    - profile[24]         : taux d’occupation moyen par heure (0–23)

Puis il exécute un clustering KMeans sur ces profils 24D et une PCA pour la
visualisation.

Sorties (LATEST only)
---------------------
Toutes les sorties sont écrites sous :

    {GCS_MONITORING_PREFIX}/monitoring/network/stations/latest/

Fichiers :
- kpis.json
    * KPIs haut niveau : n_stations, k_effective, silhouette, davies_bouldin…
- centroids.json
    * profil moyen (24h) par cluster (ou profil global si clustering KO)
- pca_scatter.json
    * scores PCA : un point par station (PC1, PC2, cluster)
- pca_circle.json
    * composantes PCA : loadings pour chacune des 24 features horaires
- stats7.json
    * table compacte de stations (preview) avec volatilité / couverture / cluster

Un manifest est également produit :

- manifest.json
    * schema_version, generated_at, window_days, sources, artifacts…

Environnement
-------------
Obligatoire :
    GCS_EXPORTS_PREFIX    = gs://bucket/velib/exports
    GCS_MONITORING_PREFIX = gs://bucket/velib   (ou .../monitoring)

Optionnel :
    MON_LAST_DAYS      (int, défaut 14 via NETWORK_WINDOW_DAYS)
    NETWORK_WINDOW_DAYS (fallback legacy pour la taille de fenêtre)
    STATIONS_MAX       (int, défaut 3000, cap sur le nb de stations)
    MIN_BINS_KEEP      (int, défaut 50, nb min de time bins valides par station)
    NETWORK_K          (int, optionnel, force K pour KMeans)
    NETWORK_K_MIN      (int, défaut 2)
    NETWORK_K_MAX      (int, défaut 8)

Notes
-----
- Le JSON est nettoyé (NaN / ±Inf → null) via `_json_safe`.
- Toutes les sorties sont en mode LATEST ONLY (pas de dossiers datés).
- Clustering et PCA sont robustes : en cas d’échec, le job produit quand même
  des sorties "saines" avec des profils de repli.
"""

from __future__ import annotations
import os, re, json, sys
from io import BytesIO
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception as e:
    raise RuntimeError("pyarrow requis") from e

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage requis") from e

# ── ML (clustering / métriques)
try:
    from sklearn.cluster import KMeans  # type: ignore
    from sklearn.metrics import silhouette_score, davies_bouldin_score  # type: ignore
except Exception:
    raise RuntimeError("scikit-learn requis (pip install scikit-learn)")

SCHEMA_VERSION = "1.2"  # 1.1 -> 1.2 : LATEST only, manifest, ENV unifiés, JSON safe

# ──────────────────────────────────────────────────────────────────────────────
# Helpers ENV (unifiés)
# ──────────────────────────────────────────────────────────────────────────────
def _env(name: str, default=None):
    """
    Lit une variable d’environnement avec une valeur par défaut.

    Paramètres
    ----------
    name : str
        Nom de la variable d’environnement.
    default : Any
        Valeur de repli si la variable est absente ou vide.

    Retourne
    --------
    Any
        Valeur brute (str) lue dans l’ENV, ou la valeur par défaut.
    """
    v = os.environ.get(name)
    return v if (v is not None and v != "") else default

def _env_int(name: str, default: int) -> int:
    """
    Lit une variable d’environnement de type entier.

    En cas d’échec de parsing, retourne la valeur par défaut.

    Paramètres
    ----------
    name : str
        Nom de la variable d’environnement.
    default : int
        Valeur entière de repli.

    Retourne
    --------
    int
        Entier parsé ou valeur par défaut.
    """
    try:
        return int(_env(name, default))
    except Exception:
        return default

# ──────────────────────────────────────────────────────────────────────────────
# Helpers GCS
# ──────────────────────────────────────────────────────────────────────────────
def _split(gs: str) -> Tuple[str, str]:
    """
    Découpe une URI GCS `gs://bucket/path` en (bucket, key).

    Paramètres
    ----------
    gs : str
        URI GCS.

    Retourne
    --------
    (str, str)
        Nom de bucket et clé d’objet (sans slash final).

    Lève
    ----
    AssertionError
        Si l’URI ne commence pas par `gs://`.
    """
    assert gs.startswith("gs://"), f"bad GCS uri: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k.rstrip("/")

def _read_parquet_blob_to_df(blob: "storage.Blob") -> pd.DataFrame:
    """
    Lit un blob parquet GCS dans un DataFrame pandas.

    Paramètres
    ----------
    blob : google.cloud.storage.Blob
        Blob GCS pointant vers un fichier parquet.

    Retourne
    --------
    pandas.DataFrame
        DataFrame chargé depuis le contenu parquet.
    """
    buf = BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    tbl = pq.read_table(buf)
    return tbl.to_pandas()

def _list_event_blobs(exports_prefix: str, start_date: datetime, end_date: datetime) -> List["storage.Blob"]:
    """
    Liste les blobs `events_YYYY-MM-DD.parquet` sur une fenêtre de dates UTC.

    Le filtrage se fait sur la date encodée dans le nom de fichier.

    Paramètres
    ----------
    exports_prefix : str
        Préfixe GCS de base pour les exports (GCS_EXPORTS_PREFIX).
    start_date : datetime
        Début de fenêtre (inclus, UTC).
    end_date : datetime
        Fin de fenêtre (inclus, UTC).

    Retourne
    --------
    list[google.cloud.storage.Blob]
        Liste triée de blobs correspondant à `events_YYYY-MM-DD.parquet`
        dans la fenêtre demandée.
    """
    bkt, key_prefix = _split(exports_prefix)
    client = storage.Client()
    bucket = client.bucket(bkt)
    blobs = list(client.list_blobs(bucket, prefix=key_prefix.strip("/") + "/"))
    pat = re.compile(r"events_(\d{4}-\d{2}-\d{2})\.parquet$")
    out = []
    for bl in blobs:
        m = pat.search(bl.name)
        if not m:
            continue
        d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        if start_date.date() <= d <= end_date.date():
            out.append(bl)
    out.sort(key=lambda b: b.name)
    return out

def _json_safe(o):
    """
    Nettoie récursivement un objet pour sérialisation JSON.

    - dict/list : parcours récursif
    - float     : NaN/Inf → None
    - autre     : renvoyé tel quel

    Utilisé juste avant `json.dumps` pour garantir un JSON valide.
    """
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_json_safe(v) for v in o]
    if isinstance(o, float):
        return float(o) if np.isfinite(o) else None
    return o

def _upload_json_gs(obj: dict, gs_uri: str, log_prefix: str = "network.stations"):
    """
    Envoie un document JSON vers GCS, avec nettoyage et log minimal.

    Paramètres
    ----------
    obj : dict
        Payload JSON-sérialisable (passé à `_json_safe`).
    gs_uri : str
        URI GCS cible.
    log_prefix : str, défaut "network.stations"
        Préfixe pour les messages de log.
    """
    bkt, key = _split(gs_uri)
    data = json.dumps(_json_safe(obj), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    storage.Client().bucket(bkt).blob(key).upload_from_string(data, content_type="application/json")
    print(f"[{log_prefix}] wrote → {gs_uri} ({len(data):,} bytes)")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers cœur (features station / profils)
# ──────────────────────────────────────────────────────────────────────────────
def _safe_occ_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Calcule un taux d’occupation robuste par ligne : bikes / capacity.

    Priorité :
      - si la colonne `occ_ratio` existe et contient des valeurs non vides → on l’utilise ;
      - sinon, on calcule bikes / capacity :
            occ = bikes / capacity si capacity > 0
            sinon NaN

    Les valeurs sont bornées à [0, 1] et les valeurs invalides sont mises à NaN.
    """
    if "occ_ratio" in df.columns and not pd.isna(df["occ_ratio"]).all():
        s = pd.to_numeric(df["occ_ratio"], errors="coerce")
    else:
        cap = pd.to_numeric(df.get("capacity"), errors="coerce")
        bk  = pd.to_numeric(df.get("bikes"), errors="coerce")
        s = pd.Series(np.where((cap > 0) & bk.notna(), bk / cap, np.nan), index=df.index, dtype="float64")
    s = s.where((s >= 0) & (s <= 1), np.nan)
    return s

def _coverage_pct(n_bins: int, n_days: float) -> float:
    """
    Estime la couverture temporelle par station (en pourcentage).

    Paramètres
    ----------
    n_bins : int
        Nombre de time bins valides (lignes avec taux d’occupation valide).
    n_days : float
        Nombre de jours couverts par la fenêtre.

    Retourne
    --------
    float
        Couverture dans [0, 100]. On suppose 288 bins par jour (pas de 5 minutes).
    """
    expected = max(1.0, 288.0 * max(0.0, n_days))
    return float(100.0 * min(1.0, n_bins / expected))

def _profile24(df: pd.DataFrame, occ: pd.Series) -> List[Optional[float]]:
    """
    Construit un profil 24 valeurs : taux d’occupation moyen par heure (0–23).

    Paramètres
    ----------
    df : pandas.DataFrame
        Sous-ensemble de station contenant au moins `tbin_utc`.
    occ : pandas.Series
        Série de taux d’occupation alignée sur df.

    Retourne
    --------
    list[float | None]
        24 valeurs (une par heure) ; None pour les heures sans données valides.
    """
    d = df.copy()
    d["occ_ratio"] = pd.to_numeric(occ, errors="coerce")
    d["hour"] = pd.to_datetime(d["tbin_utc"], errors="coerce").dt.hour
    vals: List[Optional[float]] = []
    for h in range(24):
        s = pd.to_numeric(d.loc[d["hour"] == h, "occ_ratio"], errors="coerce")
        s = s[(s >= 0) & (s <= 1)]
        vals.append(float(s.mean()) if len(s) > 0 else None)
    return vals

def _days_span(df: pd.DataFrame) -> float:
    """
    Calcule le nombre de jours couverts par le DataFrame (bornes incluses).

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame contenant `tbin_utc`.

    Retourne
    --------
    float
        Nombre de jours entre min et max (inclus). 0.0 si tous les timestamps
        sont invalides.
    """
    ts = pd.to_datetime(df["tbin_utc"], errors="coerce")
    if ts.isna().all():
        return 0.0
    dmin = ts.min().normalize()
    dmax = ts.max().normalize()
    return float((dmax - dmin).days + 1)

def _qhour_labels() -> List[str]:
    """
    Construit les labels d’axe pour les 24 features horaires : "00:00", ..., "23:00".
    """
    return [f"{h:02d}:00" for h in range(24)]

# ──────────────────────────────────────────────────────────────────────────────
# PCA (optionnelle, via SVD numpy)
# ──────────────────────────────────────────────────────────────────────────────
def _pca_first2(profiles: List[List[float]]) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    Calcule une PCA simple sur des profils 24D et renvoie les 2 premières composantes.

    Paramètres
    ----------
    profiles : list[list[float]]
        Matrice (N, 24) de profils d’occupation (NaN autorisés).

    Retourne
    --------
    (scores, components, var_ratio)
        scores     : ndarray (N, 2), coordonnées des stations dans l’espace PC1/PC2
        components : ndarray (2, 24), loadings PCA par heure
        var_ratio  : (float, float), variance expliquée par PC1/PC2

    Notes
    -----
    - Les NaN sont centrés puis remplacés par 0.0 avant SVD.
    - PCA effectuée manuellement via SVD (sans dépendance sklearn).
    """
    X = np.asarray(profiles, dtype=np.float64)  # (N,24)
    mu = np.nanmean(X, axis=0, keepdims=True)
    Xc = X - mu
    Xc = np.nan_to_num(Xc, nan=0.0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigvals = (S ** 2) / max(1, (Xc.shape[0] - 1))
    total = float(np.sum(eigvals)) if eigvals.size else 1.0
    vr = (float(eigvals[0] / total) if eigvals.size > 0 else 0.0,
          float(eigvals[1] / total) if eigvals.size > 1 else 0.0)
    scores = U[:, :2] * S[:2]
    comps = Vt[:2, :]
    return scores, comps, vr

# ──────────────────────────────────────────────────────────────────────────────
# Helpers clustering
# ──────────────────────────────────────────────────────────────────────────────
def _impute_col_mean(X: np.ndarray) -> np.ndarray:
    """
    Remplace les NaN par la moyenne de colonne dans une matrice 2D.

    Paramètres
    ----------
    X : numpy.ndarray
        Matrice d’entrée avec éventuellement des NaN.

    Retourne
    --------
    numpy.ndarray
        Nouvelle matrice où les NaN ont été imputés par les moyennes de colonnes.
    """
    X = X.copy()
    col_means = np.nanmean(X, axis=0)
    idxs = np.where(np.isnan(X))
    X[idxs] = np.take(col_means, idxs[1])
    return X

def _run_kmeans_auto(
    X: np.ndarray,
    k_forced: Optional[int] = None,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42
) -> Tuple[np.ndarray, int, Optional[float], Optional[float], np.ndarray]:
    """
    Fit KMeans sur les profils et, si besoin, cherche le nombre de clusters optimal.

    Paramètres
    ----------
    X : numpy.ndarray
        Matrice (N, D) avec éventuellement des NaN.
    k_forced : int | None, défaut None
        Si fourni (>= 1), utilise directement cette valeur de K.
    k_min : int, défaut 2
        Nombre minimal de clusters à tester.
    k_max : int, défaut 8
        Nombre maximal de clusters à tester.
    random_state : int, défaut 42
        Graine aléatoire pour KMeans.

    Retourne
    --------
    labels : numpy.ndarray
        Label de cluster pour chaque échantillon.
    k_effective : int
        Nombre de clusters finalement retenu.
    silhouette : float | None
        Meilleur score de silhouette (plus c’est haut, mieux c’est). None si non calculé.
    davies_bouldin : float | None
        Indice de Davies–Bouldin (plus c’est bas, mieux c’est). None si non calculé.
    centroids : numpy.ndarray
        Centres de clusters finaux.

    Notes
    -----
    - Quand `k_forced` est None, la fonction teste K dans [k_min, k_max]
      et sélectionne le K avec la meilleure silhouette.
    - En cas d’échec complet, fallback sur une solution à un seul cluster.
    """
    """
    Retourne: labels, k_effective, silhouette, davies_bouldin, centroids
    """
    X_imp = _impute_col_mean(X)
    best_k = None
    best_sil = -1.0
    best_labels = None
    best_centroids = None
    best_dbi = None

    def _fit_k(k: int):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_imp)
        sil = silhouette_score(X_imp, labels) if len(set(labels)) > 1 else -1.0
        dbi = davies_bouldin_score(X_imp, labels)
        cents = km.cluster_centers_
        return labels, sil, dbi, cents

    if k_forced is not None and k_forced >= 1:
        labels, sil, dbi, cents = _fit_k(k_forced)
        return labels, int(k_forced), float(sil), float(dbi), cents

    for k in range(k_min, max(k_min, k_max) + 1):
        try:
            labels, sil, dbi, cents = _fit_k(k)
            if sil > best_sil:
                best_k = k
                best_sil = sil
                best_labels = labels
                best_centroids = cents
                best_dbi = dbi
        except Exception as e:
            print(f"[warn] KMeans k={k} failed: {e}")

    if best_labels is None:
        km = KMeans(n_clusters=1, random_state=random_state, n_init=10).fit(X_imp)
        return km.labels_, 1, None, None, km.cluster_centers_

    return best_labels, int(best_k), float(best_sil), float(best_dbi), best_centroids

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    """
    Point d’entrée CLI pour le job Network Stations (clustering, LATEST ONLY).

    Pipeline
    --------
    1. Lire la configuration depuis l’ENV :
         - GCS_EXPORTS_PREFIX, GCS_MONITORING_PREFIX
         - WINDOW_DAYS (MON_LAST_DAYS / NETWORK_WINDOW_DAYS)
         - STATIONS_MAX, MIN_BINS_KEEP
         - NETWORK_K / NETWORK_K_MIN / NETWORK_K_MAX
    2. Définir la fenêtre UTC stricte :
         start = 00:00 de (now - (WINDOW_DAYS - 1))
         end   = now (UTC)
    3. Lister et lire tous les `events_YYYY-MM-DD.parquet` de cette fenêtre.
       Si aucun fichier ou aucun DataFrame lisible, sortie anticipée (manifest seul).
    4. Normaliser schéma & types :
         - garantir la présence des colonnes clés (remplies à NA si absentes)
         - parser timestamps & numériques
         - filtrer sur [start, now]
    5. Grouper par station et calculer :
         - coverage, volatilité, taux de pénurie/saturation, profil 24h
         - ne garder que les stations avec au moins MIN_BINS_KEEP bins valides
         - limiter à STATIONS_MAX stations pour le clustering / l’UI
    6. Construire la matrice de profils (N, 24) et lancer :
         - KMeans (K auto ou forcé)
         - PCA (2D) pour la visualisation
    7. Construire les artefacts JSON :
         - kpis, centroids, pca_scatter, pca_circle, stats7
    8. Uploader tous les artefacts vers :
         {GCS_MONITORING_PREFIX}/monitoring/network/stations/latest/
       ainsi qu’un manifest.json.

    Retourne
    --------
    int
        Code de sortie (0 en cas de succès).
    """
    EXPORTS_PREFIX = _env("GCS_EXPORTS_PREFIX")     # gs://bucket/velib/exports
    MON_PREFIX     = _env("GCS_MONITORING_PREFIX")  # gs://bucket/velib (ou .../monitoring)
    if not (EXPORTS_PREFIX and str(EXPORTS_PREFIX).startswith("gs://")):
        raise RuntimeError("GCS_EXPORTS_PREFIX manquant ou invalide")
    if not (MON_PREFIX and str(MON_PREFIX).startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX manquant ou invalide")

    # Unification ENV (fallback sur anciens noms)
    WINDOW_DAYS   = _env_int("MON_LAST_DAYS", _env_int("NETWORK_WINDOW_DAYS", 14))
    STATIONS_MAX  = _env_int("STATIONS_MAX", 3000)
    MIN_BINS_KEEP = _env_int("MIN_BINS_KEEP", 50)
    K_FORCED      = _env("NETWORK_K")
    K_FORCED_INT  = int(K_FORCED) if (K_FORCED and K_FORCED.isdigit()) else None
    K_MIN         = _env_int("NETWORK_K_MIN", 2)
    K_MAX         = _env_int("NETWORK_K_MAX", 8)

    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=WINDOW_DAYS - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"[network.stations] window UTC: {start.date()} → {now.date()} (days={WINDOW_DAYS})")

    # 1) Lire les parquets évènementiels dans la fenêtre
    blobs = _list_event_blobs(EXPORTS_PREFIX, start, now)

    # Base LATEST only (normalisée)
    mon_base = MON_PREFIX.rstrip("/")
    if not mon_base.endswith("/monitoring"):
        mon_base = mon_base + "/monitoring"
    base_latest = f"{mon_base}/network/stations/latest"

    if not blobs:
        print("[network.stations] no event blobs in window — nothing to do")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": now.isoformat().replace("+00:00","Z"),
            "latest_prefix": base_latest,
            "window_days": int(WINDOW_DAYS),
            "sources": {"exports_prefix": EXPORTS_PREFIX},
            "artifacts": [],
        }
        _upload_json_gs(manifest, f"{base_latest}/manifest.json")
        return 0

    frames: List[pd.DataFrame] = []
    for bl in blobs:
        print(f"[read] {bl.name}")
        try:
            df = _read_parquet_blob_to_df(bl)
            frames.append(df)
        except Exception as e:
            print(f"[warn] failed to read {bl.name}: {e}")

    if not frames:
        print("[network.stations] no readable data — nothing to do")
        return 0

    ev = pd.concat(frames, ignore_index=True)
    if ev.empty:
        print("[network.stations] events empty — nothing to do")
        return 0

    # Colonnes minimales et typage (robuste)
    need = {"tbin_utc","station_id","bikes","capacity","lat","lon","name","is_penury","is_saturation","occ_ratio"}
    for c in need:
        if c not in ev.columns:
            ev[c] = pd.NA

    ev["tbin_utc"]      = pd.to_datetime(ev["tbin_utc"], utc=True, errors="coerce")
    ev["station_id"]    = ev["station_id"].astype("string")
    ev["bikes"]         = pd.to_numeric(ev["bikes"], errors="coerce")
    ev["capacity"]      = pd.to_numeric(ev["capacity"], errors="coerce")
    ev["lat"]           = pd.to_numeric(ev["lat"], errors="coerce")
    ev["lon"]           = pd.to_numeric(ev["lon"], errors="coerce")
    ev["name"]          = ev["name"].astype("string")
    ev["is_penury"]     = pd.to_numeric(ev["is_penury"], errors="coerce")
    ev["is_saturation"] = pd.to_numeric(ev["is_saturation"], errors="coerce")

    ev = ev.dropna(subset=["tbin_utc","station_id"]).copy()
    ev = ev[(ev["tbin_utc"] >= pd.Timestamp(start)) & (ev["tbin_utc"] <= pd.Timestamp(now))].copy()

    n_days = _days_span(ev)

    # 2) Agrégats par station
    recs: List[Dict] = []
    g = ev.groupby("station_id", dropna=True)

    for sid, grp in g:
        occ = _safe_occ_ratio(grp)
        occ_valid = occ[(occ >= 0) & (occ <= 1)]
        n_bins = int(occ_valid.notna().sum())
        if n_bins < MIN_BINS_KEEP:
            continue

        _name = grp["name"].dropna().astype("string").tail(1)
        name  = None if _name.empty else str(_name.iloc[0])

        _lat = pd.to_numeric(grp["lat"], errors="coerce").dropna().tail(1)
        _lon = pd.to_numeric(grp["lon"], errors="coerce").dropna().tail(1)
        lat  = None if _lat.empty else float(_lat.iloc[0])
        lon  = None if _lon.empty else float(_lon.iloc[0])

        capacity_est = float(pd.to_numeric(grp["capacity"], errors="coerce").median(skipna=True)) if grp["capacity"].notna().any() else None
        volatility   = float(np.nanstd(occ_valid)) if occ_valid.notna().sum() > 1 else None
        penury_rate  = float(np.nanmean(pd.to_numeric(grp["is_penury"], errors="coerce"))) if grp["is_penury"].notna().any() else None
        sat_rate     = float(np.nanmean(pd.to_numeric(grp["is_saturation"], errors="coerce"))) if grp["is_saturation"].notna().any() else None
        coverage     = _coverage_pct(n_bins=n_bins, n_days=n_days)

        profile = _profile24(grp, occ)  # 24 valeurs (float | None)

        recs.append({
            "station_id": str(sid),
            "name": name,
            "lat": lat,
            "lon": lon,
            "capacity_est": capacity_est,
            "volatility": volatility,
            "penury_rate": penury_rate,
            "saturation_rate": sat_rate,
            "bins_seen": n_bins,
            "coverage_pct": coverage,
            "profile": profile,
            "cluster": None,  # rempli après clustering
        })

    # Tri : d’abord stations les mieux couvertes (bins_seen), puis station_id
    recs.sort(key=lambda r: (-int(r.get("bins_seen") or 0), r.get("station_id") or ""))
    if len(recs) > STATIONS_MAX:
        recs = recs[:STATIONS_MAX]

    # 3) Clustering sur les profils (N,24)
    profiles_mat = []
    rec_idxs = []
    for i, r in enumerate(recs):
        prof = r.get("profile")
        if isinstance(prof, list) and len(prof) == 24:
            row = [ (float(v) if v is not None else np.nan) for v in prof ]
            profiles_mat.append(row)
            rec_idxs.append(i)

    k_effective = None
    sil_score = None
    dbi_score = None
    centroids_mat = None
    labels = None

    if len(profiles_mat) >= 5:
        X = np.asarray(profiles_mat, dtype=np.float64)
        try:
            labels, k_effective, sil_score, dbi_score, centroids_mat = _run_kmeans_auto(
                X, k_forced=K_FORCED_INT, k_min=K_MIN, k_max=K_MAX, random_state=42
            )
            for idx, lab in zip(rec_idxs, labels):
                recs[idx]["cluster"] = int(lab)
        except Exception as e:
            print(f"[warn] clustering failed, continue without clusters: {e}")

    # 4) PCA (optionnelle) pour la visualisation
    generated_at = now.isoformat().replace("+00:00","Z")
    x_labels = _qhour_labels()

    pca_scatter = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "var_ratio": None,
        "points": []  # [{station_id,name,cluster,PC1,PC2}]
    }
    pca_circle = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "feature_names": x_labels,
        "components": None,
        "var_ratio": None
    }

    if len(profiles_mat) >= 3:
        try:
            scores, comps, vr = _pca_first2(profiles_mat)
            pts = []
            for (i_idx, sc) in enumerate(scores):
                rec_i = rec_idxs[i_idx]
                r = recs[rec_i]
                pts.append({
                    "station_id": r["station_id"],
                    "name": r.get("name"),
                    "cluster": r.get("cluster"),
                    "PC1": float(sc[0]),
                    "PC2": float(sc[1])
                })
            pca_scatter["points"] = pts
            pca_scatter["var_ratio"] = [round(vr[0], 6), round(vr[1], 6)]
            pca_circle["components"] = [
                [float(x) for x in comps[0, :].tolist()],
                [float(x) for x in comps[1, :].tolist()]
            ]
            pca_circle["var_ratio"] = [round(vr[0], 6), round(vr[1], 6)]
        except Exception as e:
            print(f"[warn] PCA failed: {e}")

    # 5) Centroids JSON (un item par cluster, sinon profil moyen réseau)
    if centroids_mat is not None and labels is not None and k_effective and k_effective >= 1:
        centroids_payload = []
        for k in range(k_effective):
            y = [float(v) if np.isfinite(v) else None for v in centroids_mat[k].tolist()]
            centroids_payload.append({"cluster": int(k), "y": y})
    else:
        valid_profiles = [r["profile"] for r in recs if isinstance(r.get("profile"), list)]
        centroid = []
        for i in range(24):
            vals = [p[i] for p in valid_profiles if p[i] is not None]
            centroid.append(float(np.mean(vals)) if vals else None)
        centroids_payload = [{"cluster": 0, "y": centroid}]

    centroids = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "x_labels": x_labels,
        "centroids": centroids_payload
    }

    # KPIs globales pour la page
    kpis = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "n_stations": len(recs),
        "k_effective": int(k_effective) if k_effective else None,
        "silhouette": round(float(sil_score), 6) if sil_score is not None else None,
        "davies_bouldin": round(float(dbi_score), 6) if dbi_score is not None else None,
        "window_days": int(WINDOW_DAYS)
    }

    # Table preview (tronquée)
    rows_preview = []
    for r in recs[: min(1000, len(recs))]:
        rows_preview.append({
            "station_id": r["station_id"],
            "name": r.get("name"),
            "capacity_est": r.get("capacity_est"),
            "volatility": r.get("volatility"),
            "penury_rate": r.get("penury_rate"),
            "saturation_rate": r.get("saturation_rate"),
            "coverage_pct": r.get("coverage_pct"),
            "cluster": r.get("cluster")
        })
    stats7 = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "rows": rows_preview
    }

    # 6) Uploads — LATEST only + manifest
    _upload_json_gs(kpis,        f"{base_latest}/kpis.json")
    _upload_json_gs(centroids,   f"{base_latest}/centroids.json")
    _upload_json_gs(pca_scatter, f"{base_latest}/pca_scatter.json")
    _upload_json_gs(pca_circle,  f"{base_latest}/pca_circle.json")
    _upload_json_gs(stats7,      f"{base_latest}/stats7.json")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "latest_prefix": base_latest,
        "window_days": int(WINDOW_DAYS),
        "sources": {"exports_prefix": EXPORTS_PREFIX},
        "artifacts": [
            "kpis.json",
            "centroids.json",
            "pca_scatter.json",
            "pca_circle.json",
            "stats7.json"
        ],
    }
    _upload_json_gs(manifest, f"{base_latest}/manifest.json")

    print("[network.stations] done (latest only)")
    return 0

if __name__ == "__main__":
    sys.exit(main())