# service/jobs/build_model_explainability.py
"""
Vélib' Forecast — Model Explainability (LATEST only, XGBoost native FI)
======================================================================

Génère des artefacts d'explicabilité légers à partir des exports de performance :
- residuals (hist/qq/acf/hétéro/épisodes)
- calibration (régression, binnings, by_hour, biais par station)
- uncertainty (coverage via bandes fournies, sigma, ou quantiles de résidus)
- feature_importance (***native XGBoost get_score*** : gain/weight/cover + shares)

Publication :
  {GCS_MONITORING_PREFIX}/monitoring/model/explainability/latest/h{H}/
+ manifest top-level à :
  {GCS_MONITORING_PREFIX}/monitoring/model/explainability/latest/manifest.json

ENV requis :
- GCS_EXPORTS_PREFIX          gs://.../velib/exports
- GCS_MONITORING_PREFIX       gs://.../velib

ENV optionnels :
- EXPLAIN_TZ                  default "Europe/Paris"
- EXPLAIN_HORIZONS            ex: "15,60" (minutes) — si absent → "15"
- EXPLAIN_WINDOW_DAYS         entier, 0 = pas de fenêtre (par défaut 0)

Tuning residuals (léger) :
- EXPLAIN_QQ_POINTS           default 512
- EXPLAIN_HIST_BINS           default 60
- EXPLAIN_ACF_NLAGS           default 144
- EXPLAIN_EPISODES_TOPK       default 64

Uncertainty :
- EXPLAIN_PI_NOMINAL          default 0.90

Feature Importance (XGBoost natif) :
- EXPLAIN_FI_TOPK_TO_PUBLISH  default 60 (0 = tout)
- MODEL_URI                   si un seul horizon
- MODEL_URI_{H}               ex: MODEL_URI_15, MODEL_URI_60
- MODEL_URI_TEMPLATE          ex: "gs://.../models/h{H}/latest.joblib"

Sortie : latest only (AUCUN dossier timestampé ici).
Exit code 0 si succès (même si entrée vide).
"""
from __future__ import annotations
import os, re, sys, json
from io import BytesIO
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
)

# ────────────────────────── Dépendances I/O ──────────────────────────
try:
    import pyarrow.parquet as pq  # type: ignore
except Exception as e:
    raise RuntimeError("pyarrow requis") from e

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage requis") from e

try:
    import xgboost as xgb  # type: ignore
    from joblib import load as joblib_load
except Exception as e:
    raise RuntimeError("xgboost et joblib requis pour la FI native") from e


# ────────────────────────── Constantes & schéma ──────────────────────
SCHEMA_VERSION = "1.5"   # 1.4 (surrogate) → 1.5 (XGB native only)
BIN_MIN = 5

RESID_QQ_POINTS     = int(os.environ.get("EXPLAIN_QQ_POINTS", "512"))
RESID_HIST_BINS     = int(os.environ.get("EXPLAIN_HIST_BINS", "60"))
RESID_ACF_NLAGS     = int(os.environ.get("EXPLAIN_ACF_NLAGS", "144"))
RESID_EPISODES_TOPK = int(os.environ.get("EXPLAIN_EPISODES_TOPK", "64"))

WINDOW_DAYS = int(os.environ.get("EXPLAIN_WINDOW_DAYS", "0"))
FI_TOPK_TO_PUBLISH = int(os.environ.get("EXPLAIN_FI_TOPK_TO_PUBLISH", "60"))


# ────────────────────────── Helpers GCS/Parquet ──────────────────────
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


# ────────────────────────── Lecture perf ─────────────────────────────
def _read_latest_perf(exports_prefix: str) -> Tuple[pd.DataFrame, Optional[str]]:
    base = exports_prefix.rstrip("/")
    main = f"{base}/perf.parquet"
    pat = re.compile(r"/perf_(\d{4}-\d{2}-\d{2})\.parquet$")

    if _exists(main):
        bkt, key = _split(main)
        df = _read_parquet_blob_to_df(storage.Client().bucket(bkt).blob(key))
        day = None
        for tcol in ("tbin_utc","ts","ts_utc","timestamp","datetime"):
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


# ────────────────────────── Colonnes & horizons ──────────────────────
def _pick_cols(perf: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = set(perf.columns)
    def pick(cands):
        for c in cands:
            if c in cols:
                return c
        return None

    ts      = pick(["tbin_utc","ts","ts_utc","timestamp","datetime"])
    sid     = pick(["station_id","stationcode","stationCode","id","station"])
    y_true  = "y_true" if "y_true" in cols else None
    y_pred  = pick(["y_pred_int","y_pred","yhat","prediction","pred"])
    lat     = pick(["lat","latitude"])
    lon     = pick(["lon","lng","longitude"])
    ylo     = pick(["yhat_lo","y_pred_lo","y_pred_lower","pred_lo","pi_lo","lo","lower"])
    yhi     = pick(["yhat_hi","y_pred_hi","y_pred_upper","pred_hi","pi_hi","hi","upper"])
    sigma   = pick(["yhat_std","pred_std","sigma","std_pred"])
    name    = pick(["name","station_name","nom","StationName","stationNom"])
    hbins   = pick(["horizon_bins","hbin","h_bins"])
    hmin    = pick(["horizon_min","hmin","h_min","horizonMin"])

    if not (ts and sid and y_true):
        raise KeyError(f"[explain] Colonnes minimales manquantes (ts={ts}, station={sid}, y_true={y_true})")

    return dict(ts=ts, station=sid, y_true=y_true, y_pred=y_pred,
                lat=lat, lon=lon, ylo=ylo, yhi=yhi, sigma=sigma, name=name,
                hbins=hbins, hmin=hmin)

def _normalize_types(perf: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    df = perf.copy()
    df[cols["ts"]] = pd.to_datetime(df[cols["ts"]], utc=True, errors="coerce")
    df[cols["station"]] = df[cols["station"]].astype("string")
    df[cols["y_true"]] = pd.to_numeric(df[cols["y_true"]], errors="coerce")
    if cols["y_pred"]:
        df[cols["y_pred"]] = pd.to_numeric(df[cols["y_pred"]], errors="coerce")
    if cols["ylo"]:
        df[cols["ylo"]] = pd.to_numeric(df[cols["ylo"]], errors="coerce")
    if cols["yhi"]:
        df[cols["yhi"]] = pd.to_numeric(df[cols["yhi"]], errors="coerce")
    if cols["sigma"]:
        df[cols["sigma"]] = pd.to_numeric(df[cols["sigma"]], errors="coerce")
    if cols["hbins"]:
        df[cols["hbins"]] = pd.to_numeric(df[cols["hbins"]], errors="coerce")
    if cols["hmin"]:
        df[cols["hmin"]] = pd.to_numeric(df[cols["hmin"]], errors="coerce")
    return df

def _normalize_hbin_series(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.Series:
    if cols.get("hbins") and cols["hbins"] in df.columns:
        s = pd.to_numeric(df[cols["hbins"]], errors="coerce")
        return s.round().astype("Int64")
    if cols.get("hmin") and cols["hmin"] in df.columns:
        m = pd.to_numeric(df[cols["hmin"]], errors="coerce")
        return (m / BIN_MIN).round().astype("Int64")
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")


# ────────────────────────── Utils (stats) ────────────────────────────
def _group_apply(gb, func):
    try:
        return gb.apply(func, include_groups=False)
    except TypeError:
        return gb.apply(func)

def _round_float(x: float, nd: int = 6) -> float:
    try:
        if not np.isfinite(x): return float("nan")
        return float(np.round(x, nd))
    except Exception:
        return float("nan")

def _round_list(xs, nd: int = 6):
    out = []
    for v in xs:
        if isinstance(v, (int, float, np.floating)):
            out.append(_round_float(float(v), nd))
        else:
            out.append(v)
    return out

def _qq_points(x: np.ndarray, m: int = RESID_QQ_POINTS) -> Tuple[List[float], List[float]]:
    x = x[np.isfinite(x)]
    if x.size == 0 or m <= 1:
        return [], []
    mu = np.mean(x); sd = np.std(x)
    if not np.isfinite(sd) or sd == 0.0: sd = 1.0
    z = (x - mu) / sd
    p = np.linspace(0.5 / m, 1.0 - 0.5 / m, m)
    emp = np.quantile(z, p).astype(float)
    a = 0.147
    t = 2 * p - 1.0
    t = np.clip(t, -0.999999, 0.999999)
    ln = np.log(1 - t * t)
    inner = (2 / (np.pi * a) + ln / 2.0)
    th = np.sign(t) * np.sqrt(np.sqrt(inner**2 - (ln / a)) - inner)
    return _round_list(th.tolist()), _round_list(emp.tolist())

def _acf(values: pd.Series, nlags: int = RESID_ACF_NLAGS) -> List[float]:
    x = pd.Series(values).astype(float)
    x = x - x.mean()
    acf = np.zeros(nlags + 1)
    denom = (x**2).sum()
    if denom == 0 or len(x) == 0:
        return acf.tolist()
    for k in range(nlags + 1):
        num = (x.iloc[:-k or None] * x.shift(k).iloc[:-k or None]).sum() if k > 0 else denom
        acf[k] = num / denom
    return _round_list(acf.tolist())


# ────────────────────────── Residuals ────────────────────────────────
def build_residuals_docs(perf: pd.DataFrame, cols: Dict[str, Optional[str]], tz: str) -> Dict[str, object]:
    df = perf.copy()
    df["ts"] = pd.to_datetime(df[cols["ts"]], utc=True, errors="coerce")
    df["station_id"] = df[cols["station"]].astype("string")
    y_true = pd.to_numeric(df[cols["y_true"]], errors="coerce")
    y_pred = pd.to_numeric(df[cols["y_pred"]], errors="coerce") if cols["y_pred"] else pd.Series(np.nan, index=df.index)

    used = df.assign(y_true=y_true, y_pred=y_pred).dropna(subset=["y_true", "y_pred"]).copy()
    base_doc = {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z")}
    if used.empty:
        return {**base_doc, "hist": [], "qq": {"th": [], "emp": []}, "acf": [], "hetero": [], "episodes": []}

    used["resid"] = (used["y_true"] - used["y_pred"]).astype(float)
    used = used[np.isfinite(used["resid"])].copy()
    if used.empty:
        return {**base_doc, "hist": [], "qq": {"th": [], "emp": []}, "acf": [], "hetero": [], "episodes": []}

    vals = used["resid"].to_numpy(dtype=float)
    max_abs = float(np.nanpercentile(np.abs(vals), 99)) if vals.size else 1.0
    if not np.isfinite(max_abs) or max_abs == 0: max_abs = 1.0
    hist_counts, bin_edges = np.histogram(vals, bins=RESID_HIST_BINS, range=(-max_abs, max_abs))
    hist = [{"bin_left": _round_float(float(bin_edges[i])),
             "bin_right": _round_float(float(bin_edges[i+1])),
             "count": int(hist_counts[i])} for i in range(len(hist_counts))]

    th, emp = _qq_points(vals, m=RESID_QQ_POINTS)
    mean_by_ts = used.groupby("ts")["resid"].mean()
    acf_vals = _acf(mean_by_ts, nlags=RESID_ACF_NLAGS)

    q = pd.qcut(used["y_true"], q=20, duplicates="drop")
    het = (_group_apply(
        used.assign(bin=q).groupby("bin", observed=False),
        lambda g: pd.Series({"mae": float(np.nanmean(np.abs(g["resid"]))), "n": int(len(g))})
    ).reset_index()).rename(columns={"bin":"quantile"})

    hetero = [{"quantile": f"q{i+1}", "mae": _round_float(float(r["mae"])), "n": int(r["n"])} for i, (_, r) in enumerate(het.iterrows())]

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
    episodes_df = episodes_df.loc[episodes_df["max_run"] > 0]
    if RESID_EPISODES_TOPK > 0:
        episodes_df = episodes_df.head(RESID_EPISODES_TOPK)
    episodes = [{"station_id": str(r["station_id"]), "max_run": int(r["max_run"]), "n": int(r["n"])}
                for _, r in episodes_df.iterrows()]

    return {**base_doc, "hist": hist, "qq": {"th": th, "emp": emp}, "acf": acf_vals, "hetero": hetero, "episodes": episodes}


# ────────────────────────── Calibration ──────────────────────────────
def build_calibration_docs(perf: pd.DataFrame, cols: Dict[str, Optional[str]], tz: str) -> Dict[str, object]:
    df = perf.copy()
    df["ts"] = pd.to_datetime(df[cols["ts"]], utc=True, errors="coerce")
    lts = df["ts"].dt.tz_convert(tz)
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
        if len(g) < 2 or g["y_pred"].isna().all():
            return pd.Series({"alpha": np.nan, "beta": np.nan, "n": len(g)})
        b, a = np.polyfit(g["y_pred"].astype(float), g["y_true"].astype(float), 1)
        return pd.Series({"alpha": float(a), "beta": float(b), "n": int(len(g))})
    by_hour = (_group_apply(used.groupby("hour"), _beta).reset_index())
    by_hour_doc = [{"hour": int(r["hour"]),
                    "alpha": (None if not np.isfinite(r["alpha"]) else float(r["alpha"])),
                    "beta":  (None if not np.isfinite(r["beta"])  else float(r["beta"])),
                    "n": int(r["n"])} for _, r in by_hour.iterrows()]

    tert = pd.qcut(used["y_true"], q=3, duplicates="drop", labels=["Bas","Moyen","Haut"])
    rel = (_group_apply(
        used.assign(level=tert).groupby("level", observed=False),
        lambda g: pd.Series({"mape_like": float(np.nanmean(np.abs(g["y_true"]-g["y_pred"]) / np.maximum(1.0, g["y_true"]))), "n": int(len(g))})
    ).reset_index())
    rel_doc = [{"level": str(r["level"]), "mape_like": float(r["mape_like"]), "n": int(r["n"])} for _, r in rel.iterrows()]

    bias_station = (
        used.assign(resid=(used["y_true"] - used["y_pred"]))
            .groupby(cols["station"])  # type: ignore[index]
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
            f = float(v);  return f if np.isfinite(f) else None
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


# ────────────────────────── Uncertainty ──────────────────────────────
def build_uncertainty_doc(perf: pd.DataFrame, cols: Dict[str, Optional[str]]) -> Dict[str, object]:
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

    if cols.get("ylo") and cols.get("yhi"):
        lo = pd.to_numeric(df[cols["ylo"]], errors="coerce")
        hi = pd.to_numeric(df[cols["yhi"]], errors="coerce")
        used = df.assign(yhat_lo=lo, yhat_hi=hi).dropna(subset=["y_pred","yhat_lo","yhat_hi","y_true"])
        if used.empty:
            out.update({"method": "provided_bands", "nominal": None, "coverage": None})
            return out
        inside = ((used["y_true"] >= used["yhat_lo"]) & (used["y_true"] <= used["yhat_hi"])).astype(int)
        out.update({"method": "provided_bands", "nominal": None, "coverage": {"empirical": float(inside.mean()), "n": int(len(used))}})
        return out

    if cols.get("sigma") and cols.get("y_pred"):
        sd = pd.to_numeric(perf[cols["sigma"]], errors="coerce")
        used = df.assign(sigma=sd).dropna(subset=["y_pred","sigma","y_true"])
        if not used.empty:
            z = _z_for_nominal(NOM)
            used["yhat_lo"] = used["y_pred"] - z*used["sigma"]
            used["yhat_hi"] = used["y_pred"] + z*used["sigma"]
            inside = ((used["y_true"] >= used["yhat_lo"]) & (used["y_true"] <= used["yhat_hi"])).astype(int)
            out.update({"method": "parametric_sigma", "nominal": NOM, "coverage": {"empirical": float(inside.mean()), "n": int(len(used))}})
            return out

    if cols.get("y_pred"):
        resid = (df["y_true"] - df["y_pred"]).astype(float)
        used = df.assign(resid=resid).dropna(subset=["y_pred","resid","y_true"])
        if used.empty:
            out.update({"method": "residual_quantiles", "nominal": NOM, "coverage": None})
            return out

        qlo = (1.0 - NOM) / 2.0; qhi = 1.0 - qlo
        tz = os.environ.get("EXPLAIN_TZ", "Europe/Paris")
        used["hour"] = used["ts"].dt.tz_convert(tz).dt.hour
        try:
            grp = used.groupby("hour")
            qtab = grp["resid"].quantile([qlo, qhi]).unstack(level=1).rename(columns={qlo:"qlo", qhi:"qhi"})
            used = used.join(qtab, on="hour"); method = "residual_quantiles/by_hour"
        except Exception:
            qlo_v = float(np.nanquantile(used["resid"].values, qlo))
            qhi_v = float(np.nanquantile(used["resid"].values, qhi))
            used["qlo"] = qlo_v; used["qhi"] = qhi_v; method = "residual_quantiles/global"

        used["yhat_lo"] = used["y_pred"] + used["qlo"]
        used["yhat_hi"] = used["y_pred"] + used["qhi"]
        inside = ((used["y_true"] >= used["yhat_lo"]) & (used["y_true"] <= used["yhat_hi"])).astype(int)
        out.update({"method": method, "nominal": NOM, "coverage": {"empirical": float(inside.mean()), "n": int(len(used))}})
        return out

    out.update({"method": "none", "nominal": None, "coverage": None})
    return out


# ────────────────────────── Feature Importance (XGB) ─────────────────
def _load_joblib_from_gcs(gs_uri: str):
    assert gs_uri.startswith("gs://")
    bkt, key = gs_uri[5:].split("/", 1)
    buf = BytesIO()
    storage.Client().bucket(bkt).blob(key).download_to_file(buf)
    buf.seek(0)
    return joblib_load(buf)

def _extract_booster(obj) -> xgb.Booster:
    if isinstance(obj, xgb.Booster):
        return obj
    if hasattr(obj, "get_booster"):
        return obj.get_booster()
    if hasattr(obj, "named_steps") and isinstance(obj.named_steps, dict):
        for step in obj.named_steps.values():
            if hasattr(step, "get_booster"):
                return step.get_booster()
            if isinstance(step, xgb.Booster):
                return step
    if isinstance(obj, dict):
        for k in ("model","xgb_model","estimator","booster"):
            v = obj.get(k)
            if v is None: continue
            if isinstance(v, xgb.Booster): return v
            if hasattr(v, "get_booster"): return v.get_booster()
    raise TypeError(f"Cannot extract Booster from type={type(obj)}")

def _resolve_model_uri(hmin: int, horizons: List[int]) -> Optional[str]:
    # 1) MODEL_URI_{H}
    env_key = f"MODEL_URI_{hmin}"
    if os.environ.get(env_key): return os.environ[env_key]
    # 2) MODEL_URI si un seul horizon
    if len(horizons) == 1 and os.environ.get("MODEL_URI"):
        return os.environ["MODEL_URI"]
    # 3) MODEL_URI_TEMPLATE avec {H}
    templ = os.environ.get("MODEL_URI_TEMPLATE")
    if templ and "{H" in templ:
        return templ.replace("{H}", str(hmin)).replace("{h}", str(hmin))
    return None

def build_feature_importance_doc_native_xgb(hmin: int, horizons: List[int]) -> Dict[str, object]:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
    base = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now,
        "horizon_min": int(hmin),
        "method": "xgboost_native_get_score",
        "rows": [],
        "notes": [],
    }
    uri = _resolve_model_uri(hmin, horizons)
    if not uri:
        return {**base, "method": "missing_model_uri", "notes": [f"MODEL_URI_{hmin} or MODEL_URI/ MODEL_URI_TEMPLATE not set"]}

    try:
        obj = _load_joblib_from_gcs(uri)
        booster = _extract_booster(obj)
    except Exception as e:
        return {**base, "method": "load_error", "notes": [str(e)]}

    try:
        gain   = booster.get_score(importance_type="gain")   or {}
        weight = booster.get_score(importance_type="weight") or {}
        cover  = booster.get_score(importance_type="cover")  or {}
        feats = sorted(set(gain) | set(weight) | set(cover))
        if not feats:
            return {**base, "method": "no_importances", "notes": []}

        sum_gain  = float(sum(gain.values())) if gain else 0.0
        sum_cover = float(sum(cover.values())) if cover else 0.0

        rows = []
        for f in feats:
            g = float(gain.get(f, 0.0))
            w = float(weight.get(f, 0.0))
            c = float(cover.get(f, 0.0))
            rows.append({
                "feature": f,
                "gain": _round_float(g, 6),
                "weight": _round_float(w, 6),
                "cover": _round_float(c, 6),
                "gain_share": (_round_float(g / sum_gain, 6) if sum_gain > 0 else None),
                "cover_share": (_round_float(c / sum_cover, 6) if sum_cover > 0 else None),
            })
        rows.sort(key=lambda r: (r["gain"] if r["gain"] is not None else 0.0), reverse=True)
        if FI_TOPK_TO_PUBLISH > 0:
            rows = rows[:FI_TOPK_TO_PUBLISH]
        return {**base, "rows": rows}
    except Exception as e:
        return {**base, "method": "get_score_error", "notes": [str(e)]}


# ────────────────────────── Publication latest ───────────────────────
def _publish_latest(base_alias: str, horizon_min: int, payloads: Dict[str, dict]) -> None:
    hdir = f"h{int(horizon_min)}"
    for name, obj in payloads.items():
        _upload_json_gs(obj, f"{base_alias}/{hdir}/{name}.json")


# ────────────────────────── Main ─────────────────────────────────────
def main() -> int:
    EXPORTS_PREFIX = os.environ.get("GCS_EXPORTS_PREFIX")
    MON_PREFIX     = os.environ.get("GCS_MONITORING_PREFIX")
    if not (EXPORTS_PREFIX and EXPORTS_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_EXPORTS_PREFIX manquant ou invalide")
    if not (MON_PREFIX and MON_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX manquant ou invalide")

    TZNAME   = os.environ.get("EXPLAIN_TZ", "Europe/Paris")
    HORIZONS = [int(x.strip()) for x in os.environ.get("EXPLAIN_HORIZONS", "15").split(",") if x.strip()]

    now = datetime.now(timezone.utc)

    mon_base = MON_PREFIX.rstrip("/")
    if not mon_base.endswith("/monitoring"):
        mon_base = mon_base + "/monitoring"
    base_alias = f"{mon_base}/model/explainability/latest"

    print(f"[model.explain] reading perf from {EXPORTS_PREFIX}")
    perf, perf_day = _read_latest_perf(EXPORTS_PREFIX)
    if perf.empty:
        print("[model.explain] perf is empty — nothing to do")
        # On publie tout de même un manifest vide pour la page
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": now.isoformat().replace("+00:00","Z"),
            "latest_prefix": base_alias,
            "window_days": (int(WINDOW_DAYS) if WINDOW_DAYS > 0 else None),
            "horizons": [],
            "horizons_info": [],
        }
        _upload_json_gs(manifest, f"{base_alias}/manifest.json")
        return 0

    cols = _pick_cols(perf)
    perf = _normalize_types(perf, cols)

    if WINDOW_DAYS and WINDOW_DAYS > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)
        ts_all = pd.to_datetime(perf[cols["ts"]], utc=True, errors="coerce")
        before = len(perf)
        perf = perf.loc[ts_all >= cutoff].copy()
        after = len(perf)
        print(f"[model.explain] window_days={WINDOW_DAYS} → kept {after:,}/{before:,} rows since {cutoff.isoformat()}")

    perf["__hbin"] = _normalize_hbin_series(perf, cols)

    ts = pd.to_datetime(perf[cols["ts"]], utc=True, errors="coerce")
    ts_min = ts.min(); ts_max = ts.max()
    n_rows = int(len(perf))
    n_stations = int(perf[cols["station"]].nunique())

    available_hbins = sorted([int(x) for x in pd.Series(perf["__hbin"].dropna().unique()).tolist() if pd.notna(x)])
    requested_hbins = sorted(list(set([max(1, int(round(h / BIN_MIN))) for h in HORIZONS])))
    target_hbins = [hb for hb in requested_hbins if hb in available_hbins] if available_hbins else ([] if requested_hbins else [])

    manifests_items: List[Dict[str, object]] = []
    do_segment_by_h = bool(len(available_hbins) > 0)

    if do_segment_by_h and target_hbins:
        print(f"[model.explain] horizons disponibles={available_hbins} | demandés={requested_hbins} | ciblés={target_hbins}")
        for hb in target_hbins:
            hmin = int(hb * BIN_MIN)
            sub = perf.loc[perf["__hbin"] == hb].copy()
            if sub.empty:
                print(f"[model.explain] skip h={hmin} (empty after filter)")
                continue

            ts_h = pd.to_datetime(sub[cols["ts"]], utc=True, errors="coerce")
            overview = {
                "schema_version": SCHEMA_VERSION,
                "generated_at": now.isoformat().replace("+00:00","Z"),
                "tz": TZNAME,
                "anchor_day_perf": perf_day,
                "perf_rows": int(len(sub)),
                "perf_stations": int(sub[cols["station"]].nunique()),
                "ts_min_perf": (ts_h.min().isoformat().replace("+00:00","Z") if pd.notna(ts_h.min()) else None),
                "ts_max_perf": (ts_h.max().isoformat().replace("+00:00","Z") if pd.notna(ts_h.max()) else None),
                "has_y_pred": bool(cols["y_pred"]),
                "has_uncertainty": bool(cols["ylo"] and cols["yhi"] or cols.get("sigma")),
                "horizon_min": hmin,
                "window_days": (int(WINDOW_DAYS) if WINDOW_DAYS > 0 else None),
            }

            residuals_doc   = build_residuals_docs(sub, cols, TZNAME) | {"horizon_min": hmin, "window_days": (int(WINDOW_DAYS) if WINDOW_DAYS > 0 else None)}
            calibration_doc = build_calibration_docs(sub, cols, TZNAME) | {"horizon_min": hmin, "window_days": (int(WINDOW_DAYS) if WINDOW_DAYS > 0 else None)}
            uncertainty_doc = build_uncertainty_doc(sub, cols)         | {"horizon_min": hmin, "window_days": (int(WINDOW_DAYS) if WINDOW_DAYS > 0 else None)}
            fi_doc          = build_feature_importance_doc_native_xgb(hmin, HORIZONS)

            payloads: Dict[str, dict] = {
                "overview": overview,
                "residuals": residuals_doc,
                "calibration": calibration_doc,
                "uncertainty": uncertainty_doc,
                "feature_importance": fi_doc,
            }
            _publish_latest(base_alias, horizon_min=hmin, payloads=payloads)

            manifests_items.append({
                "horizon_min": int(hmin),
                "prefix_latest": f"{base_alias}/h{int(hmin)}",
                "artifacts": list(payloads.keys())
            })

    else:
        fallback_targets = (HORIZONS if HORIZONS else [15])
        print(f"[model.explain] aucun champ d'horizon exploitable — fallback sous h{{H}} pour H={fallback_targets}")

        overview_global = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": now.isoformat().replace("+00:00","Z"),
            "tz": os.environ.get("EXPLAIN_TZ", "Europe/Paris"),
            "anchor_day_perf": perf_day,
            "perf_rows": n_rows,
            "perf_stations": n_stations,
            "ts_min_perf": (ts_min.isoformat().replace("+00:00","Z") if pd.notna(ts_min) else None),
            "ts_max_perf": (ts_max.isoformat().replace("+00:00","Z") if pd.notna(ts_max) else None),
            "has_y_pred": bool(cols["y_pred"]),
            "has_uncertainty": bool(cols["ylo"] and cols["yhi"] or cols.get("sigma")),
            "window_days": (int(WINDOW_DAYS) if WINDOW_DAYS > 0 else None),
        }

        residuals_doc   = build_residuals_docs(perf, cols, TZNAME) | {"window_days": (int(WINDOW_DAYS) if WINDOW_DAYS > 0 else None)}
        calibration_doc = build_calibration_docs(perf, cols, TZNAME) | {"window_days": (int(WINDOW_DAYS) if WINDOW_DAYS > 0 else None)}
        uncertainty_doc = build_uncertainty_doc(perf, cols)         | {"window_days": (int(WINDOW_DAYS) if WINDOW_DAYS > 0 else None)}

        for H in fallback_targets:
            hmin = int(H)
            fi_doc = build_feature_importance_doc_native_xgb(hmin, HORIZONS)
            payloads: Dict[str, dict] = {
                "overview": overview_global | {"horizon_min": hmin},
                "residuals": residuals_doc  | {"horizon_min": hmin},
                "calibration": calibration_doc | {"horizon_min": hmin},
                "uncertainty": uncertainty_doc | {"horizon_min": hmin},
                "feature_importance": fi_doc,
            }
            _publish_latest(base_alias, horizon_min=hmin, payloads=payloads)
            manifests_items.append({
                "horizon_min": int(hmin),
                "prefix_latest": f"{base_alias}/h{int(hmin)}",
                "artifacts": list(payloads.keys())
            })

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now.isoformat().replace("+00:00","Z"),
        "latest_prefix": base_alias,
        "window_days": (int(WINDOW_DAYS) if WINDOW_DAYS > 0 else None),
        "horizons": [int(it["horizon_min"]) for it in manifests_items],
        "horizons_info": manifests_items,
    }
    _upload_json_gs(manifest, f"{base_alias}/manifest.json")
    print("[model.explain] done (latest only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
