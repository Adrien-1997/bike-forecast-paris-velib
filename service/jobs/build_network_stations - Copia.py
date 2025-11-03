# service/jobs/build_network_stations.py
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
    raise RuntimeError("scikit-learn requis (pip install scikit-learn)")  # explicite

SCHEMA_VERSION = "1.1"

# ──────────────────────────────────────────────────────────────────────────────
# GCS helpers
# ──────────────────────────────────────────────────────────────────────────────

def _split(gs: str) -> Tuple[str, str]:
    assert gs.startswith("gs://"), f"bad GCS uri: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k.rstrip("/")

def _read_parquet_blob_to_df(blob: "storage.Blob") -> pd.DataFrame:
    buf = BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    tbl = pq.read_table(buf)
    return tbl.to_pandas()

def _list_event_blobs(exports_prefix: str, start_date: datetime, end_date: datetime) -> List["storage.Blob"]:
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

def _upload_json_gs(obj: dict, gs_uri: str):
    bkt, key = _split(gs_uri)
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    storage.Client().bucket(bkt).blob(key).upload_from_string(data, content_type="application/json")
    print(f"[network.stations] wrote → {gs_uri} ({len(data):,} bytes)")

# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_occ_ratio(df: pd.DataFrame) -> pd.Series:
    if "occ_ratio" in df.columns and not pd.isna(df["occ_ratio"]).all():
        s = pd.to_numeric(df["occ_ratio"], errors="coerce")
    else:
        cap = pd.to_numeric(df.get("capacity"), errors="coerce")
        bk  = pd.to_numeric(df.get("bikes"), errors="coerce")
        s = pd.Series(np.where((cap > 0) & bk.notna(), bk / cap, np.nan), index=df.index, dtype="float64")
    s = s.where((s >= 0) & (s <= 1), np.nan)
    return s

def _coverage_pct(n_bins: int, n_days: float) -> float:
    expected = max(1.0, 288.0 * max(0.0, n_days))
    return float(100.0 * min(1.0, n_bins / expected))

def _profile24(df: pd.DataFrame, occ: pd.Series) -> List[Optional[float]]:
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
    ts = pd.to_datetime(df["tbin_utc"], errors="coerce")
    if ts.isna().all():
        return 0.0
    dmin = ts.min().normalize()
    dmax = ts.max().normalize()
    return float((dmax - dmin).days + 1)

def _qhour_labels() -> List[str]:
    return [f"{h:02d}:00" for h in range(24)]

# ──────────────────────────────────────────────────────────────────────────────
# PCA (optionnelle)
# ──────────────────────────────────────────────────────────────────────────────

def _pca_first2(profiles: List[List[float]]) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
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
    """Impute NaN by column means (simple and stable for 0..1 ratios)."""
    X = X.copy()
    col_means = np.nanmean(X, axis=0)
    idxs = np.where(np.isnan(X))
    X[idxs] = np.take(col_means, idxs[1])
    return X

def _run_kmeans_auto(X: np.ndarray, k_forced: Optional[int] = None,
                     k_min: int = 2, k_max: int = 8, random_state: int = 42
                    ) -> Tuple[np.ndarray, int, Optional[float], Optional[float], np.ndarray]:
    """
    Retourne: labels, k_effective, silhouette, davies_bouldin, centroids
    """
    X_imp = _impute_col_mean(X)
    best_k = None
    best_sil = -1.0
    best_labels = None
    best_centroids = None
    best_dbi = None

    # fonction pour fitter un KMeans robuste
    def _fit_k(k: int):
        # n_init=10 (sklearn>=1.4 autorise 'auto', mais restons compatibles)
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_imp)
        # silhouette exige au moins 2 clusters et pas de singleton total
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
        # fallback trivial: 1 cluster (pas idéal mais évite le crash)
        km = KMeans(n_clusters=1, random_state=random_state, n_init=10).fit(X_imp)
        return km.labels_, 1, None, None, km.cluster_centers_

    return best_labels, int(best_k), float(best_sil), float(best_dbi), best_centroids

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    EXPORTS_PREFIX = os.environ.get("GCS_EXPORTS_PREFIX")    # gs://bucket/velib/exports
    MON_PREFIX     = os.environ.get("GCS_MONITORING_PREFIX") # gs://bucket/monitoring
    if not (EXPORTS_PREFIX and EXPORTS_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_EXPORTS_PREFIX manquant ou invalide")
    if not (MON_PREFIX and MON_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX manquant ou invalide")

    WINDOW_DAYS   = int(os.environ.get("NETWORK_WINDOW_DAYS", "14"))
    STATIONS_MAX  = int(os.environ.get("STATIONS_MAX", "3000"))
    MIN_BINS_KEEP = int(os.environ.get("MIN_BINS_KEEP", "50"))
    K_FORCED      = os.environ.get("NETWORK_K")  # optionnel
    K_FORCED_INT  = int(K_FORCED) if (K_FORCED and K_FORCED.isdigit()) else None
    K_MIN         = int(os.environ.get("NETWORK_K_MIN", "2"))
    K_MAX         = int(os.environ.get("NETWORK_K_MAX", "8"))

    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=WINDOW_DAYS - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    print(f"[network.stations] window UTC: {start.date()} → {now.date()} (days={WINDOW_DAYS})")

    # 1) Lire les parquets évènementiels dans la fenêtre
    blobs = _list_event_blobs(EXPORTS_PREFIX, start, now)
    if not blobs:
        print("[network.stations] no event blobs in window — nothing to do")
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

    # colonnes minimales et types
    need = {"tbin_utc","station_id","bikes","capacity","lat","lon","name","is_penury","is_saturation","occ_ratio"}
    for c in need:
        if c not in ev.columns:
            ev[c] = pd.NA
    ev["tbin_utc"]   = pd.to_datetime(ev["tbin_utc"], errors="coerce")
    ev["station_id"] = ev["station_id"].astype("string")
    ev["is_penury"]  = pd.to_numeric(ev["is_penury"], errors="coerce")
    ev["is_saturation"] = pd.to_numeric(ev["is_saturation"], errors="coerce")
    ev = ev.dropna(subset=["tbin_utc","station_id"]).copy()

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
            # Propager les labels vers recs
            for idx, lab in zip(rec_idxs, labels):
                recs[idx]["cluster"] = int(lab)
        except Exception as e:
            print(f"[warn] clustering failed, continue without clusters: {e}")

    # 4) PCA (optionnelle) pour la visualisation
    pca_scatter = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now.isoformat().replace("+00:00","Z"),
        "var_ratio": None,
        "points": []  # [{station_id,name,cluster,PC1,PC2}]
    }
    pca_circle = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now.isoformat().replace("+00:00","Z"),
        "feature_names": _qhour_labels(),
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
    x_labels = _qhour_labels()
    generated_at = now.isoformat().replace("+00:00", "Z")

    if centroids_mat is not None and labels is not None and k_effective and k_effective >= 1:
        # centroids_mat est (k,24) dans l'espace imputé/standard — on le laisse tel quel (moyennes horaires 0..1)
        centroids_payload = []
        for k in range(k_effective):
            y = [float(v) if np.isfinite(v) else None for v in centroids_mat[k].tolist()]
            centroids_payload.append({"cluster": int(k), "y": y})
    else:
        # fallback: centroid global (moyenne positionnelle)
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
        "window_days": WINDOW_DAYS
    }

    # Table preview
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

    # 6) Uploads (latest + versionnés)
    base_alias = f"{MON_PREFIX.rstrip('/')}/monitoring/network/stations/latest"
    base_ver   = f"{MON_PREFIX.rstrip('/')}/monitoring/network/stations/{now.strftime('%Y-%m-%dT%H-%M-%SZ')}"

    _upload_json_gs(kpis,        f"{base_alias}/kpis.json")
    _upload_json_gs(centroids,   f"{base_alias}/centroids.json")
    _upload_json_gs(pca_scatter, f"{base_alias}/pca_scatter.json")
    _upload_json_gs(pca_circle,  f"{base_alias}/pca_circle.json")
    _upload_json_gs(stats7,      f"{base_alias}/stats7.json")

    _upload_json_gs(kpis,        f"{base_ver}/kpis.json")
    _upload_json_gs(centroids,   f"{base_ver}/centroids.json")
    _upload_json_gs(pca_scatter, f"{base_ver}/pca_scatter.json")
    _upload_json_gs(pca_circle,  f"{base_ver}/pca_circle.json")
    _upload_json_gs(stats7,      f"{base_ver}/stats7.json")

    print("[network.stations] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
