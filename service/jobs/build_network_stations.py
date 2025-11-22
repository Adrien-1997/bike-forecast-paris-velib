# service/jobs/build_network_stations.py

"""
Vélib’ Forecast — Network Stations clustering & profiles (LATEST ONLY).

Role
----
This job analyses per-station behaviour over a recent time window and produces
JSON artifacts for the **Monitoring / Network / Stations** page.

Input (GCS)
-----------
- Event exports, one file per day:
    {GCS_EXPORTS_PREFIX}/events_YYYY-MM-DD.parquet

The job reads a strict UTC window of `WINDOW_DAYS` days and aggregates, per
station:

    - capacity_est        : estimated capacity (median capacity if available)
    - volatility          : std-dev of occupancy ratio over the window
    - penury_rate         : mean(is_penury)
    - saturation_rate     : mean(is_saturation)
    - coverage_pct        : fraction of expected time bins seen
    - profile[24]         : average occupancy ratio per hour (0–23)

Then it runs a KMeans clustering on the 24-dimensional profiles and a PCA
projection for visualization.

Outputs (LATEST only)
---------------------
All outputs are written under:

    {GCS_MONITORING_PREFIX}/monitoring/network/stations/latest/

Files:
- kpis.json
    * high-level KPIs: n_stations, k_effective, silhouette, davies_bouldin…
- centroids.json
    * per-cluster average 24-hour profile (or global profile if clustering fails)
- pca_scatter.json
    * PCA scores: one point per station (PC1, PC2, cluster)
- pca_circle.json
    * PCA components: loadings for each of the 24 hourly features
- stats7.json
    * compact station table (preview) with volatility/coverage/cluster

A manifest is also produced:

- manifest.json
    * schema_version, generated_at, window_days, sources, artifacts…

Environment
-----------
Required:
    GCS_EXPORTS_PREFIX   = gs://bucket/velib/exports
    GCS_MONITORING_PREFIX= gs://bucket/velib   (or .../monitoring)

Optional:
    MON_LAST_DAYS   (int, default 14 via NETWORK_WINDOW_DAYS)
    NETWORK_WINDOW_DAYS (legacy fallback for window size)
    STATIONS_MAX    (int, default 3000, cap on number of stations kept)
    MIN_BINS_KEEP   (int, default 50, minimum valid time bins per station)
    NETWORK_K       (int, optional, force K for KMeans)
    NETWORK_K_MIN   (int, default 2)
    NETWORK_K_MAX   (int, default 8)

Notes
-----
- JSON is sanitized (NaN / ±Inf → null) via `_json_safe`.
- All outputs are LATEST ONLY (no dated folders).
- Clustering and PCA are robust: if they fail, the job still produces sane
  outputs with fallback profiles.
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

SCHEMA_VERSION = "1.2"  # 1.1 -> 1.2: LATEST only, manifest, ENV unifiés, JSON safe

# ──────────────────────────────────────────────────────────────────────────────
# ENV helpers (unifiés)
# ──────────────────────────────────────────────────────────────────────────────
def _env(name: str, default=None):
    """
    Read an environment variable with a default fallback.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : Any
        Value to return if the variable is absent or empty.

    Returns
    -------
    Any
        Raw string from environment or the default.
    """
    v = os.environ.get(name)
    return v if (v is not None and v != "") else default

def _env_int(name: str, default: int) -> int:
    """
    Read an integer-valued environment variable.

    If parsing fails, the provided default is returned.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : int
        Default integer value.

    Returns
    -------
    int
        Parsed integer or default.
    """
    try:
        return int(_env(name, default))
    except Exception:
        return default

# ──────────────────────────────────────────────────────────────────────────────
# GCS helpers
# ──────────────────────────────────────────────────────────────────────────────
def _split(gs: str) -> Tuple[str, str]:
    """
    Split a GCS URI `gs://bucket/path` into (bucket, key).

    Parameters
    ----------
    gs : str
        GCS URI.

    Returns
    -------
    (str, str)
        Bucket name and object key (without trailing slash).

    Raises
    ------
    AssertionError
        If the URI does not start with `gs://`.
    """
    assert gs.startswith("gs://"), f"bad GCS uri: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k.rstrip("/")

def _read_parquet_blob_to_df(blob: "storage.Blob") -> pd.DataFrame:
    """
    Read a Parquet blob from GCS into a pandas DataFrame.

    Parameters
    ----------
    blob : google.cloud.storage.Blob
        GCS blob pointing to a parquet file.

    Returns
    -------
    pandas.DataFrame
        DataFrame loaded from the parquet content.
    """
    buf = BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    tbl = pq.read_table(buf)
    return tbl.to_pandas()

def _list_event_blobs(exports_prefix: str, start_date: datetime, end_date: datetime) -> List["storage.Blob"]:
    """
    List `events_YYYY-MM-DD.parquet` blobs in a given UTC date window.

    Filtering is done on the date embedded in the filename.

    Parameters
    ----------
    exports_prefix : str
        Base GCS prefix for exports (GCS_EXPORTS_PREFIX).
    start_date : datetime
        Start of the window (inclusive, UTC).
    end_date : datetime
        End of the window (inclusive, UTC).

    Returns
    -------
    list[google.cloud.storage.Blob]
        Sorted list of blobs matching `events_YYYY-MM-DD.parquet`
        within the requested date range.
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
    Recursively sanitize an object for JSON serialization.

    - dict/list: recurse
    - float: NaN/Inf → None
    - other: returned as-is

    Used right before `json.dumps` to guarantee valid JSON.
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
    Upload a JSON document to GCS, with sanitization and minimal logging.

    Parameters
    ----------
    obj : dict
        JSON-serializable payload (will be passed through `_json_safe`).
    gs_uri : str
        Target GCS URI.
    log_prefix : str, default "network.stations"
        Prefix for log messages.
    """
    bkt, key = _split(gs_uri)
    data = json.dumps(_json_safe(obj), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    storage.Client().bucket(bkt).blob(key).upload_from_string(data, content_type="application/json")
    print(f"[{log_prefix}] wrote → {gs_uri} ({len(data):,} bytes)")

# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────
def _safe_occ_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Compute a robust occupancy ratio for each row: bikes / capacity.

    Priority:
      - if `occ_ratio` column is present and non-empty → use it;
      - otherwise, compute bikes / capacity:
            occ = bikes / capacity if capacity > 0
            else NaN

    Values are clipped to [0, 1] and invalid values are set to NaN.
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
    Estimate per-station temporal coverage as a percentage.

    Parameters
    ----------
    n_bins : int
        Number of valid time bins (rows with valid occupancy) for the station.
    n_days : float
        Number of days spanned by the window.

    Returns
    -------
    float
        Coverage in [0, 100]. Expected bins = 288 per day (5-minute bins).
    """
    expected = max(1.0, 288.0 * max(0.0, n_days))
    return float(100.0 * min(1.0, n_bins / expected))

def _profile24(df: pd.DataFrame, occ: pd.Series) -> List[Optional[float]]:
    """
    Compute a 24-value profile of mean occupancy per local hour (0–23).

    Parameters
    ----------
    df : pandas.DataFrame
        Station subset with at least `tbin_utc`.
    occ : pandas.Series
        Occupancy ratio series aligned with df.

    Returns
    -------
    list[float | None]
        24 values (one per hour); None for hours with no valid data.
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
    Compute how many days are covered by the dataframe (inclusive).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with `tbin_utc`.

    Returns
    -------
    float
        Number of days between min and max (inclusive). 0.0 if timestamps
        are all invalid.
    """
    ts = pd.to_datetime(df["tbin_utc"], errors="coerce")
    if ts.isna().all():
        return 0.0
    dmin = ts.min().normalize()
    dmax = ts.max().normalize()
    return float((dmax - dmin).days + 1)

def _qhour_labels() -> List[str]:
    """
    Build x-axis labels for 24 hourly features: "00:00", ..., "23:00".
    """
    return [f"{h:02d}:00" for h in range(24)]

# ──────────────────────────────────────────────────────────────────────────────
# PCA (optionnelle)
# ──────────────────────────────────────────────────────────────────────────────
def _pca_first2(profiles: List[List[float]]) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    Compute a simple PCA on 24-dimensional profiles and return first 2 PCs.

    Parameters
    ----------
    profiles : list[list[float]]
        Matrix of shape (N, 24) with occupancy profiles (NaN allowed).

    Returns
    -------
    (scores, components, var_ratio)
        scores     : ndarray (N, 2), station coordinates in PC1/PC2 space
        components : ndarray (2, 24), PCA loadings for each hour
        var_ratio  : (float, float), explained variance ratio for PC1/PC2

    Notes
    -----
    - NaN values are centred then replaced by 0.0 before SVD.
    - PCA is done manually via SVD (no sklearn dependency).
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
# Clustering helpers
# ──────────────────────────────────────────────────────────────────────────────
def _impute_col_mean(X: np.ndarray) -> np.ndarray:
    """
    Replace NaN by column means in a 2D matrix.

    Parameters
    ----------
    X : numpy.ndarray
        Input array with potential NaN values.

    Returns
    -------
    numpy.ndarray
        New array with NaN imputed by column means.
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
    Fit KMeans on profiles and (optionally) search an optimal number of clusters.

    Parameters
    ----------
    X : numpy.ndarray
        Data matrix (N, D) with NaN possibly present.
    k_forced : int | None, default None
        If provided and >= 1, use this value directly for KMeans.
    k_min : int, default 2
        Minimum number of clusters when searching.
    k_max : int, default 8
        Maximum number of clusters when searching.
    random_state : int, default 42
        Random seed for KMeans.

    Returns
    -------
    labels : numpy.ndarray
        Cluster label for each sample.
    k_effective : int
        Chosen number of clusters.
    silhouette : float | None
        Best silhouette score (higher is better). None if not computed.
    davies_bouldin : float | None
        Davies–Bouldin index (lower is better). None if not computed.
    centroids : numpy.ndarray
        Final cluster centers.

    Notes
    -----
    - When `k_forced` is None, the function tests K in [k_min, k_max]
      and selects the K with highest silhouette score.
    - If everything fails, it falls back to a single-cluster solution.
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
    CLI entrypoint for the Network Stations clustering job (LATEST ONLY).

    Pipeline
    --------
    1. Read configuration from env:
         - GCS_EXPORTS_PREFIX, GCS_MONITORING_PREFIX
         - WINDOW_DAYS (MON_LAST_DAYS / NETWORK_WINDOW_DAYS)
         - STATIONS_MAX, MIN_BINS_KEEP
         - NETWORK_K / NETWORK_K_MIN / NETWORK_K_MAX
    2. Define strict UTC window:
         start = 00:00 of (now - (WINDOW_DAYS - 1))
         end   = now (UTC)
    3. List and read all `events_YYYY-MM-DD.parquet` in this window.
       If none are found or no readable frames, exit early (manifest only).
    4. Normalize schema & dtypes:
         - ensure all core columns exist (filled with NA if missing)
         - parse timestamps & numerics
         - filter on [start, now]
    5. Group by station and compute:
         - coverage, volatility, penury/saturation rates, 24-hour profile
         - keep stations with at least MIN_BINS_KEEP valid bins
         - cap at STATIONS_MAX stations for clustering/UI
    6. Build the profiles matrix (N, 24) and run:
         - KMeans (auto or forced K)
         - PCA (2D) for visualization
    7. Build JSON artifacts:
         - kpis, centroids, pca_scatter, pca_circle, stats7
    8. Upload all artifacts to:
         {GCS_MONITORING_PREFIX}/monitoring/network/stations/latest/
       plus a manifest.json.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    EXPORTS_PREFIX = _env("GCS_EXPORTS_PREFIX")     # gs://bucket/velib/exports
    MON_PREFIX     = _env("GCS_MONITORING_PREFIX")  # gs://bucket/velib (ou .../monitoring)
    if not (EXPORTS_PREFIX and str(EXPORTS_PREFIX).startswith("gs://")):
        raise RuntimeError("GCS_EXPORTS_PREFIX manquant ou invalide")
    if not (MON_PREFIX and str(MON_PREFIX).startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX manquant ou invalide")

    # Unification ENV (fallbacks sur anciens noms)
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

    # colonnes minimales et types (robuste)
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

    # KPIs
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
