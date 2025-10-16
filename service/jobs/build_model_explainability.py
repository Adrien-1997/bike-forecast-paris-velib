# service/jobs/build_model_explainability.py
from __future__ import annotations
import os, re, sys, json
from io import BytesIO
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timezone

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

SCHEMA_VERSION = "1.2"  # + uncertainty fallback + station names

# ───────────────────────── GCS helpers ─────────────────────────

def _split(gs: str) -> Tuple[str, str]:
    assert gs.startswith("gs://"), f"bad GCS uri: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k.rstrip("/")

def _exists(gs: str) -> bool:
    b, k = _split(gs)
    return storage.Client().bucket(b).blob(k).exists()

def _list_under(prefix_gs: str, regex: Optional[re.Pattern]=None) -> List["storage.Blob"]:
    bkt, key = _split(prefix_gs)
    cli = storage.Client()
    blobs = list(cli.list_blobs(bkt, prefix=key.strip("/") + "/"))
    if regex is not None:
        blobs = [b for b in blobs if regex.search(b.name)]
    return blobs

def _read_parquet_blob_to_df(blob: "storage.Blob") -> pd.DataFrame:
    buf = BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return pq.read_table(buf).to_pandas()

def _upload_json_gs(obj: object, gs_uri: str):
    # JSON "safe": NaN/±Inf → null
    def _san(o):
        if isinstance(o, dict):  return {k: _san(v) for k, v in o.items()}
        if isinstance(o, list):  return [_san(v) for v in o]
        if isinstance(o, float): return float(o) if np.isfinite(o) else None
        return o
    safe = _san(obj)
    b, k = _split(gs_uri)
    data = json.dumps(safe, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    storage.Client().bucket(b).blob(k).upload_from_string(data, content_type="application/json")
    print(f"[model.explain] wrote → {gs_uri} ({len(data):,} bytes)")

# ───────────────────────── Readers ─────────────────────────

def _read_latest_perf(exports_prefix: str) -> Tuple[pd.DataFrame, Optional[str]]:
    base = exports_prefix.rstrip("/")
    main = f"{base}/perf.parquet"
    pat = re.compile(r"/perf_(\d{4}-\d{2}-\d{2})\.parquet$")

    if _exists(main):
        bkt, key = _split(main)
        df = _read_parquet_blob_to_df(storage.Client().bucket(bkt).blob(key))
        day = None
        for tcol in ("tbin_utc","ts","ts_utc"):
            if tcol in df.columns:
                ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
                if ts.notna().any():
                    day = ts.max().date().isoformat()
                    break
        return df, day

    blobs = _list_under(base, regex=pat)
    if not blobs:
        raise FileNotFoundError(f"no perf parquet under {base}")
    blobs.sort(key=lambda b: b.name)
    last = blobs[-1]
    m = pat.search(last.name)
    day = m.group(1) if m else None
    df = _read_parquet_blob_to_df(last)
    return df, day

# ───────────────────────── Utils ─────────────────────────

def _pick_cols(perf: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = set(perf.columns)

    def pick(cands):
        for c in cands:
            if c in cols:
                return c
        return None

    ts      = pick(["tbin_utc","ts","ts_utc","datetime"])
    sid     = pick(["station_id","stationcode","stationCode","id","station"])
    y_true  = "y_true" if "y_true" in cols else None
    y_pred  = pick(["y_pred_int","y_pred","yhat","prediction","pred"])
    lat     = pick(["lat","latitude"])
    lon     = pick(["lon","lng","longitude"])
    # incertitude (variantes usuelles)
    ylo     = pick(["yhat_lo","y_pred_lo","y_pred_lower","pred_lo","pi_lo","lo","lower"])
    yhi     = pick(["yhat_hi","y_pred_hi","y_pred_upper","pred_hi","pi_hi","hi","upper"])
    sigma   = pick(["yhat_std","pred_std","sigma","std_pred"])
    # nom station
    name    = pick(["name","station_name","nom","StationName","stationNom"])

    if not (ts and sid and y_true):
        raise KeyError(f"[explain] Colonnes minimales manquantes (ts={ts}, station={sid}, y_true={y_true})")
    return dict(ts=ts, station=sid, y_true=y_true, y_pred=y_pred,
                lat=lat, lon=lon, ylo=ylo, yhi=yhi, sigma=sigma, name=name)

def _to_local(s_utc_like: pd.Series, tzname: str) -> pd.Series:
    s = pd.to_datetime(s_utc_like, utc=True, errors="coerce")
    return s.dt.tz_convert(tzname)

def _metrics(y_true: pd.Series, y_hat: pd.Series) -> Dict[str, float]:
    e = (y_true - y_hat).astype(float)
    return {
        "mae": float(np.nanmean(np.abs(e))),
        "rmse": float(np.sqrt(np.nanmean(e**2))),
        "me": float(np.nanmean(e)),
    }

def _group_apply(gb, func):
    try:
        return gb.apply(func, include_groups=False)
    except TypeError:
        return gb.apply(func)

def _qq_points(x: np.ndarray) -> Tuple[List[float], List[float]]:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return [], []
    xs = np.sort((x - np.mean(x)) / (np.std(x) if np.std(x) > 0 else 1.0))
    n = xs.size
    p = (np.arange(1, n + 1) - 0.5) / n
    a = 0.147
    th = np.sign(2*p-1) * np.sqrt(np.sqrt((2/(np.pi*a) + np.log(1-(2*p-1)**2)/2.0)**2 - (np.log(1-(2*p-1)**2)/a)) - (2/(np.pi*a) + np.log(1-(2*p-1)**2)/2.0))
    return th.tolist(), xs.tolist()

def _acf(values: pd.Series, nlags: int = 144) -> List[float]:
    x = pd.Series(values).astype(float)
    x = x - x.mean()
    acf = np.zeros(nlags + 1)
    denom = (x**2).sum()
    if denom == 0 or len(x) == 0:
        return acf.tolist()
    for k in range(nlags + 1):
        num = (x.iloc[:-k or None] * x.shift(k).iloc[:-k or None]).sum() if k > 0 else denom
        acf[k] = num / denom
    return acf.tolist()

# ───────────────────────── Residuals ─────────────────────────

def build_residuals_docs(perf: pd.DataFrame, cols: Dict[str, Optional[str]], tz: str) -> Dict[str, object]:
    df = perf.copy()
    df["ts"] = pd.to_datetime(df[cols["ts"]], utc=True, errors="coerce")
    df["station_id"] = df[cols["station"]].astype("string")
    y = pd.to_numeric(df[cols["y_true"]], errors="coerce")
    yp = pd.to_numeric(df[cols["y_pred"]], errors="coerce") if cols["y_pred"] else np.nan
    df["y_pred"] = yp
    df["resid"] = (y - df["y_pred"]).astype(float)
    used = df.dropna(subset=["y_pred"]).copy()

    doc = {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z")}
    if used.empty:
        doc.update({"hist": [], "qq": {"th": [], "emp": []}, "acf": [], "hetero": [], "episodes": []})
        return doc

    hist_counts, bin_edges = np.histogram(used["resid"].values, bins=60)
    hist = [{"bin_left": float(bin_edges[i]), "bin_right": float(bin_edges[i+1]), "count": int(hist_counts[i])} for i in range(len(hist_counts))]

    th, emp = _qq_points(used["resid"].values)
    mean_by_ts = used.groupby("ts")["resid"].mean()
    acf_vals = _acf(mean_by_ts, nlags=144)

    q = pd.qcut(y, q=20, duplicates="drop")
    het = (_group_apply(
        used.assign(bin=q).groupby("bin", observed=False),
        lambda g: pd.Series({"mae": float(np.nanmean(np.abs(g["resid"]))), "n": int(len(g))})
    ).reset_index()).rename(columns={"bin":"quantile"})
    hetero = [{"quantile": str(r["quantile"]), "mae": float(r["mae"]), "n": int(r["n"])} for _, r in het.iterrows()]

    TH = 4.0
    def _episodes(g: pd.DataFrame) -> pd.Series:
        x = (np.abs(g["resid"].values) >= TH).astype(int)
        best = cur = 0
        for v in x:
            if v: cur += 1; best = max(best, cur)
            else: cur = 0
        return pd.Series({"max_run": int(best), "n": int(len(g))})
    episodes_df = (_group_apply(used.sort_values(["station_id","ts"]).groupby("station_id"), _episodes)
                   .reset_index().sort_values("max_run", ascending=False))
    episodes = [{"station_id": str(r["station_id"]), "max_run": int(r["max_run"]), "n": int(r["n"])} for _, r in episodes_df.iterrows()]

    doc.update({"hist": hist, "qq": {"th": th, "emp": emp}, "acf": acf_vals, "hetero": hetero, "episodes": episodes})
    return doc

# ───────────────────────── Calibration ─────────────────────────

def build_calibration_docs(perf: pd.DataFrame, cols: Dict[str, Optional[str]], tz: str) -> Dict[str, object]:
    df = perf.copy()
    df["ts"] = pd.to_datetime(df[cols["ts"]], utc=True, errors="coerce")
    lts = _to_local(df["ts"], tz)
    df["hour"] = lts.dt.hour
    y = pd.to_numeric(df[cols["y_true"]], errors="coerce")
    yp = pd.to_numeric(df[cols["y_pred"]], errors="coerce") if cols["y_pred"] else pd.Series(np.nan, index=df.index)

    name_col = cols.get("name")
    used = df.assign(y_true=y, y_pred=yp).dropna(subset=["y_pred"]).copy()

    out = {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z")}
    if used.empty:
        out.update({"fit": {"alpha": None, "beta": None, "n": 0}, "binned": [], "by_hour": [], "rel_error_levels": [], "bias_by_station": []})
        return out

    x = used["y_pred"].astype(float).values
    yy = used["y_true"].astype(float).values
    if np.isfinite(x).all() and np.isfinite(yy).all() and x.size > 1:
        b, a = np.polyfit(x, yy, 1)
        alpha, beta = float(a), float(b)
    else:
        alpha = beta = float("nan")

    q = pd.qcut(used["y_pred"], q=20, duplicates="drop")
    cal_bin = (_group_apply(
        used.groupby(q, observed=False),
        lambda g: pd.Series({"y_pred_mean": float(np.nanmean(g["y_pred"])),
                             "y_true_mean": float(np.nanmean(g["y_true"])),
                             "n": int(len(g))})
    ).reset_index(names=["bin"])).rename(columns={"bin":"quantile"})
    binned = [{"quantile": str(r["quantile"]), "y_pred_mean": float(r["y_pred_mean"]),
               "y_true_mean": float(r["y_true_mean"]), "n": int(r["n"])} for _, r in cal_bin.iterrows()]

    def _beta(g: pd.DataFrame) -> pd.Series:
        if len(g) < 2 or g["y_pred"].isna().all(): return pd.Series({"alpha": np.nan, "beta": np.nan, "n": len(g)})
        b, a = np.polyfit(g["y_pred"].astype(float), g["y_true"].astype(float), 1)
        return pd.Series({"alpha": float(a), "beta": float(b), "n": int(len(g))})
    by_hour = (_group_apply(used.groupby("hour"), _beta).reset_index())
    by_hour_doc = [{"hour": int(r["hour"]), "alpha": (None if not np.isfinite(r["alpha"]) else float(r["alpha"])),
                    "beta": (None if not np.isfinite(r["beta"]) else float(r["beta"])), "n": int(r["n"])} for _, r in by_hour.iterrows()]

    tert = pd.qcut(used["y_true"], q=3, duplicates="drop", labels=["Bas","Moyen","Haut"])
    rel = (_group_apply(
        used.assign(level=tert).groupby("level", observed=False),
        lambda g: pd.Series({"mape_like": float(np.nanmean(np.abs(g["y_true"]-g["y_pred"]) / np.maximum(1.0, g["y_true"]))), "n": int(len(g))})
    ).reset_index())
    rel_doc = [{"level": str(r["level"]), "mape_like": float(r["mape_like"]), "n": int(r["n"])} for _, r in rel.iterrows()]

    # ——— Ajout du nom de station
    bias_station = (
        used.assign(resid=(used["y_true"] - used["y_pred"]))
            .groupby(cols["station"])
            .agg(
                bias=("resid","mean"),
                n=("resid","size"),
                lat=(cols["lat"] if cols["lat"] else cols["station"], "last"),
                lon=(cols["lon"] if cols["lon"] else cols["station"], "last"),
                name=((name_col if name_col else cols["station"]), "last"),
            )
            .reset_index()
            .rename(columns={cols["station"]: "station_id"})
    )

    def _to_float_or_none(v):
        try:
            f = float(v)
            return f if np.isfinite(f) else None
        except Exception:
            return None

    bias_doc = [{
        "station_id": str(r["station_id"]),
        "name": (None if pd.isna(r["name"]) else str(r["name"])),
        "bias": (None if not np.isfinite(r["bias"]) else float(r["bias"])),
        "lat": _to_float_or_none(r["lat"]),
        "lon": _to_float_or_none(r["lon"]),
        "n": int(r["n"]),
    } for _, r in bias_station.iterrows()]

    out.update({
        "fit": {"alpha": (None if not np.isfinite(alpha) else alpha),
                "beta":  (None if not np.isfinite(beta)  else beta),
                "n": int(len(used))},
        "binned": binned,
        "by_hour": by_hour_doc,
        "rel_error_levels": rel_doc,
        "bias_by_station": bias_doc,
    })
    return out

# ───────────────────────── Uncertainty ─────────────────────────
def build_uncertainty_doc(perf: pd.DataFrame, cols: Dict[str, Optional[str]]) -> Dict[str, object]:
    """
    1) si ylo/yhi fournis → coverage direct
    2) sinon si sigma + y_pred → bornes paramétriques (Normal)
    3) sinon → bandes par quantiles de résidus (par heure locale si possible, sinon global)
    """
    out = {"schema_version": SCHEMA_VERSION,
           "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z")}
    NOM = float(os.environ.get("EXPLAIN_PI_NOMINAL", "0.90"))

    def _z_for_nominal(p: float) -> float:
        lut = {0.80:1.28155, 0.85:1.43953, 0.90:1.64485, 0.95:1.95996, 0.98:2.32635, 0.99:2.57583}
        if p in lut: return lut[p]
        keys = sorted(lut.keys())
        for i in range(len(keys)-1):
            a,b = keys[i], keys[i+1]
            if a <= p <= b:
                za, zb = lut[a], lut[b]
                t = (p-a)/(b-a)
                return za + t*(zb-za)
        return lut[0.90]

    df = perf.copy()
    y = pd.to_numeric(df[cols["y_true"]], errors="coerce")
    yp = pd.to_numeric(df[cols["y_pred"]], errors="coerce") if cols.get("y_pred") else pd.Series(np.nan, index=df.index)
    df = df.assign(y_true=y, y_pred=yp)
    df["ts"] = pd.to_datetime(df[cols["ts"]], utc=True, errors="coerce")

    # 1) bornes déjà présentes
    if cols.get("ylo") and cols.get("yhi"):
        lo = pd.to_numeric(df[cols["ylo"]], errors="coerce")
        hi = pd.to_numeric(df[cols["yhi"]], errors="coerce")
        used = df.assign(yhat_lo=lo, yhat_hi=hi).dropna(subset=["y_pred","yhat_lo","yhat_hi","y_true"])
        if used.empty:
            out.update({"method": "provided_bands", "nominal": None, "coverage": None})
            return out
        inside = ((used["y_true"] >= used["yhat_lo"]) & (used["y_true"] <= used["yhat_hi"])).astype(int)
        out.update({"method": "provided_bands", "nominal": None,
                    "coverage": {"empirical": float(inside.mean()), "n": int(len(used))}})
        return out

    # 2) sigma paramétrique
    if cols.get("sigma") and cols.get("y_pred"):
        sd = pd.to_numeric(perf[cols["sigma"]], errors="coerce")
        used = df.assign(sigma=sd).dropna(subset=["y_pred","sigma","y_true"])
        if not used.empty:
            z = _z_for_nominal(NOM)
            used["yhat_lo"] = used["y_pred"] - z*used["sigma"]
            used["yhat_hi"] = used["y_pred"] + z*used["sigma"]
            inside = ((used["y_true"] >= used["yhat_lo"]) & (used["y_true"] <= used["yhat_hi"])).astype(int)
            out.update({"method": "parametric_sigma", "nominal": NOM,
                        "coverage": {"empirical": float(inside.mean()), "n": int(len(used))}})
            return out

    # 3) bandes par quantiles de résidu (par heure locale si possible)
    if cols.get("y_pred"):
        resid = (df["y_true"] - df["y_pred"]).astype(float)
        used = df.assign(resid=resid).dropna(subset=["y_pred","resid","y_true"])
        if used.empty:
            out.update({"method": "residual_quantiles", "nominal": NOM, "coverage": None})
            return out

        qlo = (1.0 - NOM) / 2.0
        qhi = 1.0 - qlo

        tz = os.environ.get("EXPLAIN_TZ", "Europe/Paris")
        used["hour"] = used["ts"].dt.tz_convert(tz).dt.hour

        try:
            grp = used.groupby("hour")
            qtab = grp["resid"].quantile([qlo, qhi]).unstack(level=1).rename(columns={qlo:"qlo", qhi:"qhi"})
            used = used.join(qtab, on="hour")
            method = "residual_quantiles/by_hour"
        except Exception:
            # fallback global
            qlo_v = float(np.nanquantile(used["resid"].values, qlo))
            qhi_v = float(np.nanquantile(used["resid"].values, qhi))
            used["qlo"] = qlo_v
            used["qhi"] = qhi_v
            method = "residual_quantiles/global"

        used["yhat_lo"] = used["y_pred"] + used["qlo"]
        used["yhat_hi"] = used["y_pred"] + used["qhi"]
        inside = ((used["y_true"] >= used["yhat_lo"]) & (used["y_true"] <= used["yhat_hi"])).astype(int)
        out.update({"method": method, "nominal": NOM,
                    "coverage": {"empirical": float(inside.mean()), "n": int(len(used))}})
        return out

    # 4) rien à faire (pas de y_pred)
    out.update({"method": "none", "nominal": None, "coverage": None})
    return out

# ───────────────────────── Main ─────────────────────────

def main() -> int:
    EXPORTS_PREFIX = os.environ.get("GCS_EXPORTS_PREFIX")    # gs://.../exports
    MON_PREFIX     = os.environ.get("GCS_MONITORING_PREFIX") # gs://.../velib (ou .../monitoring)
    if not (EXPORTS_PREFIX and EXPORTS_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_EXPORTS_PREFIX manquant ou invalide")
    if not (MON_PREFIX and MON_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX manquant ou invalide")

    TZNAME       = os.environ.get("EXPLAIN_TZ", "Europe/Paris")

    now = datetime.now(timezone.utc)

    # Sortie: latest/ seulement
    mon_base = MON_PREFIX.rstrip("/")
    if not mon_base.endswith("/monitoring"):
        mon_base = mon_base + "/monitoring"
    base = f"{mon_base}/model/explainability/latest"

    # Lecture perf
    print(f"[model.explain] reading perf from {EXPORTS_PREFIX}")
    perf, perf_day = _read_latest_perf(EXPORTS_PREFIX)
    if perf.empty:
        print("[model.explain] perf is empty — nothing to do")
        return 0

    # Colonnes minimales + typage
    cols = _pick_cols(perf)
    perf[cols["ts"]] = pd.to_datetime(perf[cols["ts"]], utc=True, errors="coerce")
    perf[cols["station"]] = perf[cols["station"]].astype("string")
    perf[cols["y_true"]] = pd.to_numeric(perf[cols["y_true"]], errors="coerce")
    if cols["y_pred"]:
        perf[cols["y_pred"]] = pd.to_numeric(perf[cols["y_pred"]], errors="coerce")

    # Bornes temporelles
    ts = pd.to_datetime(perf[cols["ts"]], utc=True, errors="coerce")
    ts_min = ts.min(); ts_max = ts.max()
    n_rows = int(len(perf))
    n_stations = int(perf[cols["station"]].nunique())

    # Overview
    overview = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now.isoformat().replace("+00:00","Z"),
        "tz": TZNAME,
        "anchor_day_perf": perf_day,
        "perf_rows": n_rows,
        "perf_stations": n_stations,
        "ts_min_perf": (ts_min.isoformat().replace("+00:00","Z") if pd.notna(ts_min) else None),
        "ts_max_perf": (ts_max.isoformat().replace("+00:00","Z") if pd.notna(ts_max) else None),
        "has_y_pred": bool(cols["y_pred"]),
        "has_uncertainty": bool(cols["ylo"] and cols["yhi"] or cols.get("sigma")),
    }

    # Calculs
    residuals_doc   = build_residuals_docs(perf, cols, TZNAME)
    calibration_doc = build_calibration_docs(perf, cols, TZNAME)
    uncertainty_doc = build_uncertainty_doc(perf, cols)

    # Uploads (LATEST ONLY)
    _upload_json_gs(overview,         f"{base}/overview.json")
    _upload_json_gs(residuals_doc,    f"{base}/residuals.json")
    _upload_json_gs(calibration_doc,  f"{base}/calibration.json")
    _upload_json_gs(uncertainty_doc,  f"{base}/uncertainty.json")

    print("[model.explain] done (latest only)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
