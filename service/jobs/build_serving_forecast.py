# service/jobs/build_serving_forecast.py
from __future__ import annotations
import os, sys, shutil, json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict
import importlib.util as _iu

import numpy as np
import pandas as pd
from google.cloud import storage

# ─────────────────────────────────────────────
#  Repo root auto-resolution (service/ or train/)
# ─────────────────────────────────────────────
def _ensure_repo_root():
    if _iu.find_spec("service") is not None or _iu.find_spec("train") is not None:
        return
    here = Path(__file__).resolve()
    for c in [here] + list(here.parents) + [Path("/app"), Path.cwd()]:
        if (c / "service").exists() or (c / "train").exists():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            print(f"[serving_forecast] repo_root={c}")
            return
    if Path("/app").exists() and "/app" not in sys.path:
        sys.path.insert(0, "/app")
        print("[serving_forecast] fallback /app")

_ensure_repo_root()

# ─────────────────────────────────────────────
#  Imports training/forecast (multi-layout safe)
# ─────────────────────────────────────────────
try:
    from service.core.cal_features import add_time_features
    from service.core.features import BASE_COLUMNS as TRAIN_BASE_COLUMNS
    from service.core.forecast import predict_from_features_df
except ModuleNotFoundError:
    try:
        from service.core.cal_features import add_time_features
        from service.core.features import BASE_COLUMNS as TRAIN_BASE_COLUMNS
        from service.core.forecast import predict_from_features_df
    except ModuleNotFoundError:
        from service.core.cal_features import add_time_features
        from service.core.features import BASE_COLUMNS as TRAIN_BASE_COLUMNS
        from service.core.forecast import predict_from_features_df

# ─────────────────────────────────────────────
#  ENV config
# ─────────────────────────────────────────────
RAW_PREFIX = os.environ["GCS_RAW_PREFIX"]
BIN_MINUTES = 5
WINDOW_HOURS = int(os.environ.get("WINDOW_HOURS", "4"))
LAG_MAX_BINS = 48
WINDOW_BINS = max(WINDOW_HOURS * 60 // BIN_MINUTES, LAG_MAX_BINS + 1)

WITH_FORECAST = str(os.environ.get("WITH_FORECAST", "1")).lower() in ("1","true","yes")
SERVING_FORECAST_PREFIX = os.environ.get("SERVING_FORECAST_PREFIX")
FORECAST_HORIZONS = os.environ.get("FORECAST_HORIZONS", "15,60")

GCS_MODEL_URI_T15 = os.environ.get("GCS_MODEL_URI_T15")
GCS_MODEL_URI_T60 = os.environ.get("GCS_MODEL_URI_T60")

# ─────────────────────────────────────────────
#  Time utils
# ─────────────────────────────────────────────
def _floor_5min(dt: datetime) -> datetime:
    m = (dt.minute // 5) * 5
    return dt.replace(minute=m, second=0, microsecond=0)

def _iter_hours(start: datetime, end: datetime):
    cur = start.replace(minute=0, second=0, microsecond=0)
    last = end.replace(minute=0, second=0, microsecond=0)
    while cur <= last:
        yield cur
        cur += timedelta(hours=1)

# ─────────────────────────────────────────────
#  GCS utils
# ─────────────────────────────────────────────
def _parse_gs(uri: str) -> Tuple[str, str]:
    assert uri.startswith("gs://"), f"bad GCS uri: {uri}"
    bkt, key = uri[5:].split("/", 1)
    return bkt, key

def _upload_bytes(cli: storage.Client, data: bytes, dest_uri: str, content_type: str = "application/json") -> None:
    bkt, key = _parse_gs(dest_uri)
    cli.bucket(bkt).blob(key).upload_from_string(data, content_type=content_type)

def _list_raw_files_for_window(cli: storage.Client, start: datetime, end: datetime) -> List[str]:
    bkt, pfx_root = _parse_gs(RAW_PREFIX)
    out: List[str] = []
    for h in _iter_hours(start, end):
        day = h.strftime("%Y-%m-%d"); hh = h.strftime("%H")
        prefix = f"{pfx_root}/date={day}/hour={hh}/"
        for b in cli.list_blobs(bkt, prefix=prefix):
            if b.name.endswith(".parquet"):
                out.append(f"gs://{bkt}/{b.name}")
    return sorted(out)

def _download_gs_files(cli: storage.Client, uris: List[str], dest_dir: Path) -> List[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for uri in uris:
        bkt, key = _parse_gs(uri)
        local = dest_dir / Path(key).name
        cli.bucket(bkt).blob(key).download_to_filename(str(local))
        paths.append(local)
    return paths

# ─────────────────────────────────────────────
#  IO / normalization
# ─────────────────────────────────────────────
BASE_COLS = list(TRAIN_BASE_COLUMNS)  # aligne avec training

def _to_naive_utc(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    return dt.dt.tz_convert("UTC").dt.tz_localize(None)

def _read_concat_parquets(files: List[Path]) -> pd.DataFrame:
    dfs = []
    for p in files:
        try:
            df = pd.read_parquet(p)
            dfs.append(df)
        except Exception as e:
            print(f"[read] skip {p.name}: {e}")
    if not dfs:
        return pd.DataFrame(columns=BASE_COLS)

    df = pd.concat(dfs, ignore_index=True)

    # typer/normaliser temps
    if "ts_utc" in df.columns:
        df["ts_utc"] = _to_naive_utc(df["ts_utc"])
    if "tbin_utc" in df.columns:
        df["tbin_utc"] = _to_naive_utc(df["tbin_utc"])

    # numériques (ne PAS toucher station_id)
    for c in ["bikes","capacity","mechanical","ebike"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["lat","lon","temp_C","precip_mm","wind_mps"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # texte
    if "status" in df.columns:
        df["status"] = df["status"].astype("string")
    if "name" in df.columns:
        df["name"] = df["name"].astype("string")

    # ── station_id robuste ─────────────────────────────────────────────
    if "station_id" not in df.columns:
        if "stationcode" in df.columns:
            df["station_id"] = df["stationcode"].astype("string")
        else:
            df["station_id"] = pd.NA
    df["station_id"] = df["station_id"].astype("string")
    if "stationcode" in df.columns:
        sc = df["stationcode"].astype("string")
        m_empty = df["station_id"].isna() | (df["station_id"].str.strip() == "")
        df.loc[m_empty, "station_id"] = sc

    # dédoublonnage
    if set(["station_id","tbin_utc","ts_utc"]).issubset(df.columns):
        df = df.sort_values(["station_id","tbin_utc","ts_utc"])
        df = df.groupby(["station_id","tbin_utc"], as_index=False, dropna=True).tail(1).reset_index(drop=True)

    # force présence colonnes attendues
    for c in BASE_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    return df[BASE_COLS]

# ─────────────────────────────────────────────
#  Feature building helpers (per-station)
# ─────────────────────────────────────────────
def _build_one_station_full(st_df: pd.DataFrame) -> pd.Series:
    """
    Calcule un set complet "proche training" mais condensé en une seule ligne (dernier bin).
    Hypothèse: st_df contient une fenêtre historique de >= 48 bins (4h).
    """
    st_df = st_df.sort_values("tbin_utc").copy()

    cur = st_df.iloc[-1]
    st_id = cur.get("station_id", np.nan)
    tbin_latest = pd.to_datetime(cur["tbin_utc"], errors="coerce")

    bikes = pd.to_numeric(st_df["bikes"], errors="coerce")
    capacity = pd.to_numeric(st_df["capacity"], errors="coerce")

    # lags
    def lag_last(s: pd.Series, n: int):
        return s.shift(n).iloc[-1] if len(s) >= (n + 1) else np.nan
    lag_set = (1,2,3,6,12,24,48)
    lag_vals = {f"lag_bikes_{L}": lag_last(bikes, L) for L in lag_set}

    # rollings (sur bikes) en évitant le leak (on décale d'1 bin)
    roll_vals: Dict[str, float] = {}
    for W in (3,6,12):
        s_shift = bikes.shift(1)
        roll_vals[f"roll_mean_{W}"] = s_shift.rolling(W, min_periods=max(1, W//2)).mean().iloc[-1]
        roll_vals[f"roll_std_{W}"]  = s_shift.rolling(W, min_periods=max(1, W//2)).std().iloc[-1]

    # tendances 12 bins
    def _slope_per_5m(ts: pd.Series, y: pd.Series) -> float:
        t = pd.to_datetime(ts, errors="coerce")
        m = t.notna() & y.notna()
        if m.sum() < 2: return np.nan
        x = t[m].astype("datetime64[s]").astype("int64").astype(np.float64) / (BIN_MINUTES * 60.0)
        yy = y[m].astype(float).to_numpy()
        vx = np.var(x)
        if vx == 0: return np.nan
        return float(np.cov(x, yy, ddof=0)[0, 1] / vx)

    win_nb  = st_df.tail(13).iloc[:-1]
    win_occ = st_df.tail(13).iloc[:-1]
    trend_nb_12b  = _slope_per_5m(win_nb["tbin_utc"],  win_nb["bikes"]) if "bikes" in win_nb.columns else np.nan
    trend_occ_12b = _slope_per_5m(
        win_occ["tbin_utc"],
        (win_occ["bikes"] / win_occ["capacity"].where(win_occ["capacity"] > 0))
    ) if {"bikes","capacity"}.issubset(win_occ.columns) else np.nan

    # ratios & météo lag
    occ_ratio_bin = (cur.get("bikes", np.nan) / cur.get("capacity", np.nan)) if pd.notna(cur.get("capacity", np.nan)) and cur.get("capacity", 0) > 0 else np.nan
    temp_C    = pd.to_numeric(st_df["temp_C"], errors="coerce")
    precip_mm = pd.to_numeric(st_df["precip_mm"], errors="coerce")
    wind_mps  = pd.to_numeric(st_df["wind_mps"], errors="coerce")
    temp_C_lag1    = lag_last(temp_C, 1)
    precip_mm_lag1 = lag_last(precip_mm, 1)
    wind_mps_lag1  = lag_last(wind_mps, 1)

    # calendaires via cal_features.add_time_features (UTC + Paris)
    cal = pd.DataFrame({"tbin_utc": [tbin_latest]})
    cal = add_time_features(cal, ts_col="tbin_utc", add_paris_derived=True).iloc[0].to_dict()

    status = cur.get("status", None)

    out = {
        "station_id":      st_id,
        "tbin_latest":     tbin_latest,
        "capacity_bin":    cur.get("capacity", np.nan),
        "occ_ratio_bin":   occ_ratio_bin,
        "occ_ratio":       occ_ratio_bin,
        "mechanical":      cur.get("mechanical", np.nan),
        "ebike":           cur.get("ebike", np.nan),
        "lat":             cur.get("lat", np.nan),
        "lon":             cur.get("lon", np.nan),
        "temp_C":          cur.get("temp_C", np.nan),
        "precip_mm":       cur.get("precip_mm", np.nan),
        "wind_mps":        cur.get("wind_mps", np.nan),
        "temp_C_lag1":     temp_C_lag1,
        "precip_mm_lag1":  precip_mm_lag1,
        "wind_mps_lag1":   wind_mps_lag1,
        "trend_nb_12b":    trend_nb_12b,
        "trend_occ_12b":   trend_occ_12b,
        "status":          status,     # encodé en status_code ensuite
        # calendaires
        "hour":        cal.get("hour"),
        "minute":      cal.get("minute"),
        "dow":         cal.get("dow"),
        "month":       cal.get("month"),
        "is_weekend":  cal.get("is_weekend"),
        "hod_sin":     cal.get("hod_sin"),
        "hod_cos":     cal.get("hod_cos"),
        "dow_sin":     cal.get("dow_sin"),
        "dow_cos":     cal.get("dow_cos"),
        "paris_hour":  cal.get("paris_hour"),
        "paris_dow":   cal.get("paris_dow"),
        "paris_is_we": cal.get("paris_is_we"),
    }
    out.update(lag_vals)
    out.update(roll_vals)
    return pd.Series(out)

def _build_features(df: pd.DataFrame, start_tbin: datetime, end_tbin: datetime) -> pd.DataFrame:
    # fenêtre d’historique pour calculer les features
    m = (df["tbin_utc"] >= start_tbin) & (df["tbin_utc"] <= end_tbin)
    cols_needed = [
        "tbin_utc","station_id","bikes","capacity","mechanical","ebike","status",
        "lat","lon","temp_C","precip_mm","wind_mps"
    ]
    dfw = df.loc[m, cols_needed].dropna(subset=["station_id","tbin_utc"]).copy()
    if dfw.empty:
        return pd.DataFrame()

    # ⚠️ on fixe station_id à partir de la clé du groupby (et pas des valeurs de colonnes)
    def _build_with_key(g: pd.DataFrame) -> pd.Series:
        out = _build_one_station_full(g)
        out["station_id"] = str(g.name) if g.name is not None else pd.NA
        return out

    try:
        feats = (
            dfw.groupby("station_id", dropna=True, group_keys=False)
               .apply(_build_with_key, include_groups=False)  # pandas ≥ 2.2
               .reset_index(drop=True)
        )
    except TypeError:
        feats = (
            dfw.groupby("station_id", dropna=True, group_keys=False)
               .apply(_build_with_key)
               .reset_index(drop=True)
        )

    # Typage final
    feats["station_id"]  = feats["station_id"].astype("string")
    feats["tbin_latest"] = pd.to_datetime(feats["tbin_latest"], errors="coerce")

    # Encodage status_code global
    if "status" in feats.columns:
        cats = sorted([s for s in feats["status"].dropna().unique()])
        status_map = {s: i for i, s in enumerate(cats)}
        feats["status_code"] = feats["status"].map(status_map).astype("Int64")
        feats = feats.drop(columns=["status"])

    # (optionnel) petit log pour contrôle
    try:
        print("[features] sample station_id:", feats["station_id"].head(3).tolist())
    except Exception:
        pass

    return feats

# ─────────────────────────────────────────────
#  JSON sanitization
# ─────────────────────────────────────────────
def _to_jsonable(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        dt = v
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    return v

def _records_jsonable(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    df2 = df.copy()
    for c in df2.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        s = pd.to_datetime(df2[c], utc=True, errors="coerce")
        df2[c] = s.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    recs = df2.to_dict(orient="records")
    return [{k: _to_jsonable(v) for k, v in r.items()} for r in recs]

# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main() -> int:
    # 0) now / fenêtre de calcul
    if "NOW_UTC_ISO" in os.environ:
        now_utc = datetime.fromisoformat(os.environ["NOW_UTC_ISO"].replace("Z", "+00:00")).astimezone(timezone.utc)
    else:
        now_utc = datetime.now(timezone.utc)

    end_tbin_aware = _floor_5min(now_utc)
    start_tbin_aware = end_tbin_aware - timedelta(minutes=BIN_MINUTES * (WINDOW_BINS - 1))
    end_naive   = end_tbin_aware.replace(tzinfo=None)
    start_naive = start_tbin_aware.replace(tzinfo=None)

    print(f"[features_4h][cfg] RAW={RAW_PREFIX} WIN_H={WINDOW_HOURS} BINS={WINDOW_BINS}", flush=True)
    print(f"[features_4h] window UTC: {start_tbin_aware.isoformat()} → {end_tbin_aware.isoformat()} (inclusive)", flush=True)

    cli = storage.Client()

    # 1) list + 2) download
    gcs_files = _list_raw_files_for_window(cli, start_tbin_aware, end_tbin_aware)
    print(f"[features_4h] gcs files found = {len(gcs_files)}", flush=True)
    if not gcs_files:
        print("[features_4h] no raw files in window — exit 0", flush=True)
        return 0

    work = Path("/tmp/features_4h_raw"); shutil.rmtree(work, ignore_errors=True)
    local_files = _download_gs_files(cli, gcs_files, work)
    print(f"[features_4h] local files = {len(local_files)}", flush=True)

    # 3) read + features
    df = _read_concat_parquets(local_files)
    feats = _build_features(df, start_naive, end_naive)
    print(f"[features_4h] features rows={len(feats):,}", flush=True)
    if feats.empty:
        print("[features_4h] no features → no forecast", flush=True)
        return 0

    # 4) inference → JSON consolidé
    if not WITH_FORECAST:
        print("[forecast] WITH_FORECAST disabled — nothing to do", flush=True)
        return 0
    if not SERVING_FORECAST_PREFIX:
        raise RuntimeError("SERVING_FORECAST_PREFIX is required to write latest_forecast.json")

    def _model_uri_for(hmin: int) -> str | None:
        if hmin == 15 and GCS_MODEL_URI_T15: return GCS_MODEL_URI_T15
        if hmin == 60 and GCS_MODEL_URI_T60: return GCS_MODEL_URI_T60
        return None  # horizon non configuré → on skip

    horizons_min = [int(x.strip()) for x in FORECAST_HORIZONS.split(",") if x.strip()]
    consolidated: Dict[str, list] = {}
    generated_at = end_tbin_aware.isoformat().replace("+00:00", "Z")

    for hmin in horizons_min:
        uri = _model_uri_for(hmin)
        if not uri:
            print(f"[forecast][skip] no model configured for h={hmin} — skipping")
            consolidated[str(hmin)] = []
            continue

        preds = predict_from_features_df(
            feats_df=feats,
            model_uri=uri,
            horizon_bins=max(1, hmin // 5),
            model_alias=None,
        )

        if preds.empty:
            print(f"[forecast] empty preds for h={hmin}")
            consolidated[str(hmin)] = []
            continue

        # Réindexer pour alignement positionnel sûr
        preds = preds.reset_index(drop=True).copy()
        feats_idx = feats.reset_index(drop=True).copy()

        # station_id : forcer depuis feats (string)
        preds["station_id"] = feats_idx["station_id"].astype("string")

        # tbin_latest & capacity_bin utiles au front
        if "tbin_latest" not in preds.columns:
            preds["tbin_latest"] = feats_idx["tbin_latest"].values
        if "capacity_bin" not in preds.columns and "capacity_bin" in feats_idx.columns:
            preds["capacity_bin"] = feats_idx["capacity_bin"].values

        # horizon + horodatage d'exécution + version du modèle
        preds["horizon_min"] = hmin
        preds["pred_ts_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if "model_version" not in preds.columns:
            try:
                preds["model_version"] = uri.rsplit("/", 1)[-1].replace(".joblib", "")
            except Exception:
                preds["model_version"] = f"model_h{hmin}"

        # prédiction entière : arrondi, borne à ≥ 0
        if "bikes_pred_int" not in preds.columns and "bikes_pred" in preds.columns:
            preds["bikes_pred_int"] = (
                np.rint(pd.to_numeric(preds["bikes_pred"], errors="coerce"))
                .clip(lower=0)
                .astype("Int64")
            )

        try:
            print("[forecast][sample]",
                  preds[["station_id","bikes_pred","bikes_pred_int"]].head(3).to_dict("records"))
        except Exception:
            pass

        recs = _records_jsonable(preds)
        consolidated[str(hmin)] = recs

    bundle = {
        "generated_at": generated_at,
        "horizons": horizons_min,
        "data": consolidated,
    }

    _upload_bytes(
        cli,
        json.dumps(bundle, ensure_ascii=False).encode("utf-8"),
        f"{SERVING_FORECAST_PREFIX.rstrip('/')}/latest_forecast.json"
    )
    print(f"[forecast] uploaded SINGLE JSON: latest_forecast.json", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())