# service/jobs/build_monitoring_drift.py
from __future__ import annotations
import os, sys, json
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

# SciPy est optionnel : si absent, on sortira KS sans p-value
try:
    from scipy.stats import ks_2samp  # type: ignore
except Exception:
    ks_2samp = None  # type: ignore

SCHEMA_VERSION = "1.0"
DEFAULT_FEATURES = ["occ_ratio", "temp_C", "precip_mm", "wind_mps", "capacity"]

# ─────────────── GCS helpers ───────────────
def _split(gs: str) -> Tuple[str, str]:
    assert gs.startswith("gs://"), f"bad GCS uri: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k.rstrip("/")

def _read_parquet_gs(gs_uri: str) -> pd.DataFrame:
    bkt, key = _split(gs_uri)
    buf = BytesIO()
    storage.Client().bucket(bkt).blob(key).download_to_file(buf)
    buf.seek(0)
    tbl = pq.read_table(buf)
    return tbl.to_pandas()

def _upload_json_gs(obj: dict, gs_uri: str):
    bkt, key = _split(gs_uri)
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    storage.Client().bucket(bkt).blob(key).upload_from_string(data, content_type="application/json")
    print(f"[drift] wrote → {gs_uri} ({len(data):,} bytes)")

# ─────────────── Time helpers ───────────────
def _anchor_day_utc() -> datetime:
    ad = os.environ.get("ANCHOR_DAY")
    if ad:
        # accepte "YYYY-MM-DD"
        return datetime.fromisoformat(ad).replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)

def _parse_window(s: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    # "YYYY-MM-DD..YYYY-MM-DD"
    a, b = s.split("..", 1)
    start = pd.Timestamp(a).tz_localize(None)
    end = pd.Timestamp(b).tz_localize(None)
    return start, end

def _default_windows(anchor: datetime) -> Tuple[Tuple[pd.Timestamp,pd.Timestamp], Tuple[pd.Timestamp,pd.Timestamp]]:
    # référence = mois précédent complet ; actuel = 7 derniers jours glissants
    # ex: anchor=2025-10-09 → ref=2025-09-01..2025-09-30 ; cur=2025-10-03..2025-10-09
    end_cur = anchor.date()
    start_cur = end_cur - timedelta(days=6)
    cur = (pd.Timestamp(start_cur), pd.Timestamp(end_cur))

    # mois précédent
    first_this = pd.Timestamp(f"{anchor.year}-{anchor.month:02d}-01")
    last_prev = first_this - pd.Timedelta(days=1)
    first_prev = pd.Timestamp(f"{last_prev.year}-{last_prev.month:02d}-01")
    ref = (first_prev, last_prev)
    return ref, cur

# ─────────────── Drift metrics ───────────────
def _nan_clean(x: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    return arr[~np.isnan(arr)]

def _common_bins(x: np.ndarray, y: np.ndarray, n_bins: int = 30) -> np.ndarray:
    # bornes robustes : [p1, p99] sur concat
    if x.size == 0 and y.size == 0:
        return np.linspace(0, 1, n_bins + 1)
    z = np.concatenate([x, y]) if (x.size and y.size) else (x if x.size else y)
    lo, hi = np.nanpercentile(z, [1, 99]) if z.size else (0.0, 1.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(z)) if z.size else 0.0, float(np.nanmax(z)) if z.size else 1.0
        if lo == hi:
            lo -= 0.5
            hi += 0.5
    return np.linspace(lo, hi, n_bins + 1)

def _hist_counts(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    # counts par bin (exclut NaN en amont)
    c, _ = np.histogram(x, bins=bins)
    return c.astype(float)

def _psi(counts_ref: np.ndarray, counts_cur: np.ndarray) -> float:
    # population stability index sur distributions discrètes
    p = counts_ref / max(1.0, counts_ref.sum())
    q = counts_cur / max(1.0, counts_cur.sum())
    # éviter log(0) → lissage epsilon
    eps = 1e-9
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum((p - q) * np.log(p / q)))

def _ks_stat_pvalue(x: np.ndarray, y: np.ndarray) -> Tuple[float, Optional[float]]:
    if x.size == 0 or y.size == 0:
        return 0.0, None
    # SciPy si dispo
    if ks_2samp is not None:
        s = ks_2samp(x, y, alternative="two-sided", mode="asymp")
        return float(s.statistic), float(s.pvalue)
    # fallback: seulement la stat KS via ECDF
    xs = np.sort(x); ys = np.sort(y)
    xx = np.concatenate([xs, ys])
    Fx = np.searchsorted(xs, xx, side="right") / xs.size
    Fy = np.searchsorted(ys, xx, side="right") / ys.size
    stat = float(np.max(np.abs(Fx - Fy)))
    return stat, None

# ─────────────── Core build ───────────────
def _slice_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    ts = pd.to_datetime(df["tbin_utc"], errors="coerce")
    m = (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))
    return df.loc[m].copy()

def _ensure_features(df: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    for c in feats:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def _drift_for_feature(ref: pd.Series, cur: pd.Series, n_bins: int = 30) -> Dict[str, object]:
    x = _nan_clean(ref)
    y = _nan_clean(cur)
    bins = _common_bins(x, y, n_bins=n_bins)
    h_ref = _hist_counts(x, bins)
    h_cur = _hist_counts(y, bins)
    psi = _psi(h_ref, h_cur)
    ks_stat, ks_p = _ks_stat_pvalue(x, y)
    # deltas moments
    out = {
        "psi": float(psi),
        "ks": float(ks_stat),
        "p_value": (None if ks_p is None else float(ks_p)),
        "delta_mean": (None if (x.size == 0 or y.size == 0) else float(np.nanmean(y) - np.nanmean(x))),
        "delta_std":  (None if (x.size == 0 or y.size == 0) else float(np.nanstd(y)  - np.nanstd(x))),
        "bins": bins.tolist(),
        "ref_counts": h_ref.astype(int).tolist(),
        "cur_counts": h_cur.astype(int).tolist(),
        "ref_n": int(x.size),
        "cur_n": int(y.size),
    }
    return out

# ─────────────── Main ───────────────
def main() -> int:
    EXPORTS_PREFIX = os.environ.get("GCS_EXPORTS_PREFIX")
    MON_PREFIX     = os.environ.get("GCS_MONITORING_PREFIX")
    if not (EXPORTS_PREFIX and EXPORTS_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_EXPORTS_PREFIX manquant ou invalide")
    if not (MON_PREFIX and MON_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX manquant ou invalide")

    # Paramètres
    FEATURES = [s.strip() for s in os.environ.get("DRIFT_FEATURES", ",".join(DEFAULT_FEATURES)).split(",") if s.strip()]
    N_BINS   = int(os.environ.get("DRIFT_BINS", "30"))
    anchor   = _anchor_day_utc()
    anchor_tag = anchor.strftime("%Y%m%d")

    ref_env = os.environ.get("REFERENCE_WINDOW")  # "YYYY-MM-DD..YYYY-MM-DD"
    cur_env = os.environ.get("CURRENT_WINDOW")
    if ref_env and cur_env:
        ref_win = _parse_window(ref_env)
        cur_win = _parse_window(cur_env)
    else:
        ref_win, cur_win = _default_windows(anchor)

    ref_str = f"{ref_win[0].date()}..{ref_win[1].date()}"
    cur_str = f"{cur_win[0].date()}..{cur_win[1].date()}"
    print(f"[drift] reference_window={ref_str} | current_window={cur_str}")
    print(f"[drift] features={FEATURES} bins={N_BINS}")

    events_uri = f"{EXPORTS_PREFIX.rstrip('/')}/events.parquet"
    print(f"[drift] read: {events_uri}")
    df = _read_parquet_gs(events_uri)
    if df.empty:
        print("[drift] events empty — nothing to do")
        return 0

    if "tbin_utc" not in df.columns:
        print("[drift] missing tbin_utc — abort")
        return 1
    df["tbin_utc"] = pd.to_datetime(df["tbin_utc"], errors="coerce")
    df = _ensure_features(df, FEATURES)

    ref_df = _slice_window(df, *ref_win)
    cur_df = _slice_window(df, *cur_win)

    # Calcul par feature
    features_out: List[Dict[str, object]] = []
    dists_out: List[Dict[str, object]] = []

    for feat in FEATURES:
        stats = _drift_for_feature(ref_df[feat], cur_df[feat], n_bins=N_BINS)
        flag = "ok"
        # seuils simples : PSI > 0.2 warn, > 0.3 alert ; KS > 0.2 warn/alert
        psi, ks = stats["psi"], stats["ks"]
        if psi is not None and psi >= 0.3 or ks is not None and ks >= 0.3:
            flag = "alert"
        elif psi is not None and psi >= 0.2 or ks is not None and ks >= 0.2:
            flag = "warn"

        features_out.append({
            "name": feat,
            "psi": stats["psi"],
            "ks": stats["ks"],
            "delta_mean": stats["delta_mean"],
            "delta_std": stats["delta_std"],
            "p_value": stats["p_value"],
            "flag": flag,
            "ref_n": stats["ref_n"],
            "cur_n": stats["cur_n"],
        })
        dists_out.append({
            "feature": feat,
            "bins": stats["bins"],
            "ref_counts": stats["ref_counts"],
            "cur_counts": stats["cur_counts"],
        })

    # Résumés
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "reference_window": ref_str,
        "current_window": cur_str,
        "features": features_out,
    }
    dist = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "reference_window": ref_str,
        "current_window": cur_str,
        "distributions": dists_out,
    }

    # Écritures alias + versionné
    base = f"{MON_PREFIX.rstrip('/')}/drift"
    _upload_json_gs(summary, f"{base}/drift_summary.json")
    _upload_json_gs(summary, f"{base}/drift_summary_{anchor_tag}.json")
    _upload_json_gs(dist,    f"{base}/drift_distributions.json")
    _upload_json_gs(dist,    f"{base}/drift_distributions_{anchor_tag}.json")

    print("[drift] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
