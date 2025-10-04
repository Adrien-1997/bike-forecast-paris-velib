# pipeline/build_features_4h.py
from __future__ import annotations
import os, sys, shutil
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
from google.cloud import storage

# ==== Config via ENV ====
RAW_PREFIX      = os.environ["GCS_RAW_PREFIX"]         # gs://.../velib/bronze
SERVING_PREFIX  = os.environ["GCS_SERVING_PREFIX"]     # gs://.../velib/serving/features_4h
WINDOW_HOURS    = int(os.environ.get("WINDOW_HOURS", "4"))  # 4h par défaut
DIAG            = os.environ.get("DIAG", "0") == "1"

# ==== Utils temps ====
def _floor_5min(dt: datetime) -> datetime:
    m = (dt.minute // 5) * 5
    return dt.replace(minute=m, second=0, microsecond=0)

def _iter_hours(start: datetime, end: datetime):
    cur = start.replace(minute=0, second=0, microsecond=0)
    last = end.replace(minute=0, second=0, microsecond=0)
    while cur <= last:
        yield cur
        cur += timedelta(hours=1)

# ==== GCS utils ====
def _parse_gs(uri: str) -> Tuple[str, str]:
    assert uri.startswith("gs://"), f"bad GCS uri: {uri}"
    bkt, key = uri[5:].split("/", 1)
    return bkt, key

def _list_raw_files_for_window(cli: storage.Client, start: datetime, end: datetime) -> List[str]:
    """Liste tous les parquets bronze pour la fenêtre [start, end] incluse (UTC aware pour le listing)."""
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

def _upload_file(cli: storage.Client, local: Path, dest_uri: str) -> None:
    bkt, key = _parse_gs(dest_uri)
    cli.bucket(bkt).blob(key).upload_from_filename(str(local))

# ==== Lecture/normalisation ====
BASE_COLS = [
    "ts_utc","tbin_utc","station_id","bikes","capacity",
    "mechanical","ebike","status","lat","lon","name",
    "temp_C","precip_mm","wind_mps"
]

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

    # typage
    for c in ["ts_utc","tbin_utc"]:
        df[c] = pd.to_datetime(df.get(c), utc=True, errors="coerce").dt.tz_convert(None)
    for c in ["station_id","bikes","capacity"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    # dédoublonnage (garde dernier ts_utc par (station_id, tbin_utc))
    df = df.sort_values(["station_id","tbin_utc","ts_utc"])
    df = df.groupby(["station_id","tbin_utc"], as_index=False, dropna=True).tail(1).reset_index(drop=True)
    return df

# ==== Calculs de features ====
def _slope_per_5m(ts: pd.Series, y: pd.Series) -> float:
    """Pente (y) par pas de 5 minutes via régression linéaire simple sur le temps."""
    t = pd.to_datetime(ts, errors="coerce")
    m = t.notna() & y.notna()
    if m.sum() < 2:
        return np.nan
    # échelle en "bins 5 min"
    x = t[m].astype("datetime64[s]").astype("int64").astype(np.float64) / (5 * 60.0)
    yy = y[m].astype(float).to_numpy()
    vx = np.var(x)
    if vx == 0:
        return np.nan
    return float(np.cov(x, yy, ddof=0)[0, 1] / vx)

def _build_one_station(st_df: pd.DataFrame) -> pd.Series:
    """Calcule les features pour une station sur la fenêtre 4h."""
    st_df = st_df.sort_values("tbin_utc")
    # occupation
    occ = st_df["bikes"] / st_df["capacity"].where(st_df["capacity"] > 0)

    # bin courant
    cur = st_df.iloc[-1]
    tbin_latest   = cur["tbin_utc"]
    capacity_bin  = cur["capacity"]
    occ_ratio_bin = (cur["bikes"] / cur["capacity"]) if capacity_bin and capacity_bin > 0 else np.nan

    # lags (bikes & occ) sur t-1, t-24, t-48 (bins)
    def lag(series: pd.Series, n: int):
        return series.shift(1).iloc[-n] if len(series) >= (n + 1) else np.nan

    lag_nb_1b  = lag(st_df["bikes"], 1)
    lag_nb_24b = lag(st_df["bikes"], 24)
    lag_nb_48b = lag(st_df["bikes"], 48)

    lag_occ_1b  = lag(occ, 1)
    lag_occ_24b = lag(occ, 24)
    lag_occ_48b = lag(occ, 48)

    # rollings sans fuite sur 12 bins (t-12..t-1)
    roll_nb_12b  = st_df["bikes"].shift(1).rolling(12, min_periods=3).mean().iloc[-1]
    roll_occ_12b = occ.shift(1).rolling(12, min_periods=3).mean().iloc[-1]

    # trends (pente) sur 12 bins précédents
    win_nb  = st_df.tail(13).iloc[:-1]  # 12 bins
    win_occ = st_df.tail(13).iloc[:-1]
    trend_nb_12b  = _slope_per_5m(win_nb["tbin_utc"],  win_nb["bikes"])
    trend_occ_12b = _slope_per_5m(win_occ["tbin_utc"], (win_occ["bikes"] / win_occ["capacity"].where(win_occ["capacity"] > 0)))

    # calendrier (UTC)
    hour = int(pd.to_datetime(tbin_latest).hour)
    hour_sin = float(np.sin(2*np.pi*hour/24.0))
    hour_cos = float(np.cos(2*np.pi*hour/24.0))

    return pd.Series({
        "station_id":    cur["station_id"],
        "tbin_latest":   tbin_latest,
        # snapshot
        "capacity_bin":  capacity_bin,
        "occ_ratio_bin": occ_ratio_bin,
        # lags nb
        "lag_nb_1b":     lag_nb_1b,
        "lag_nb_24b":    lag_nb_24b,
        "lag_nb_48b":    lag_nb_48b,
        # lags occ
        "lag_occ_1b":    lag_occ_1b,
        "lag_occ_24b":   lag_occ_24b,
        "lag_occ_48b":   lag_occ_48b,
        # rollings
        "roll_nb_12b":   roll_nb_12b,
        "roll_occ_12b":  roll_occ_12b,
        # trends
        "trend_nb_12b":  trend_nb_12b,
        "trend_occ_12b": trend_occ_12b,
        # calendar
        "hour":          hour,
        "hour_sin":      hour_sin,
        "hour_cos":      hour_cos,
    })

def _build_features(df: pd.DataFrame, start_tbin: datetime, end_tbin: datetime) -> pd.DataFrame:
    # Filtre fenêtre (UTC naïf)
    m = (df["tbin_utc"] >= start_tbin) & (df["tbin_utc"] <= end_tbin)
    dfw = df.loc[m, ["tbin_utc","station_id","bikes","capacity"]].dropna(subset=["station_id","tbin_utc"]).copy()
    if dfw.empty:
        return pd.DataFrame(columns=[
            "station_id","tbin_latest","capacity_bin","occ_ratio_bin",
            "lag_nb_1b","lag_nb_24b","lag_nb_48b",
            "lag_occ_1b","lag_occ_24b","lag_occ_48b",
            "roll_nb_12b","roll_occ_12b",
            "trend_nb_12b","trend_occ_12b",
            "hour","hour_sin","hour_cos",
        ])

    feats = (
        dfw.groupby("station_id", dropna=True, group_keys=False)
           .apply(_build_one_station)
           .reset_index(drop=True)
    )

    # types clean
    feats["station_id"] = pd.to_numeric(feats["station_id"], errors="coerce").astype("Int64")
    feats["tbin_latest"] = pd.to_datetime(feats["tbin_latest"], errors="coerce")
    return feats

# ==== Main ====
def main() -> int:
    # now / fenêtre (aware UTC)
    if "NOW_UTC_ISO" in os.environ:
        now_utc = datetime.fromisoformat(os.environ["NOW_UTC_ISO"].replace("Z","+00:00")).astimezone(timezone.utc)
    else:
        now_utc = datetime.now(timezone.utc)

    end_tbin = _floor_5min(now_utc)
    start_tbin = end_tbin - timedelta(hours=WINDOW_HOURS) + timedelta(minutes=5)

    # Convertir en NAÏF (UTC sans tzinfo) pour matcher df['tbin_utc']
    end_naive = end_tbin.replace(tzinfo=None)
    start_naive = start_tbin.replace(tzinfo=None)

    print(f"[features_4h][cfg] RAW={RAW_PREFIX} SERVING={SERVING_PREFIX} WIN_H={WINDOW_HOURS}", flush=True)
    print(f"[features_4h] window UTC: {start_tbin.isoformat()} → {end_tbin.isoformat()} (inclusive)", flush=True)

    cli = storage.Client()

    # 1) Lister & 2) Télécharger
    gcs_files = _list_raw_files_for_window(cli, end_tbin - timedelta(hours=WINDOW_HOURS), end_tbin)
    print(f"[features_4h] gcs files found = {len(gcs_files)}", flush=True)
    if not gcs_files:
        print("[features_4h] no raw files in window — exit 0", flush=True)
        return 0

    work = Path("/tmp/features_4h_raw"); shutil.rmtree(work, ignore_errors=True)
    local_files = _download_gs_files(cli, gcs_files, work)
    print(f"[features_4h] local files = {len(local_files)}", flush=True)

    # 3) Read + features
    df = _read_concat_parquets(local_files)

    DIAG = str(os.getenv("DIAG", "0")).lower() not in ("", "0", "false", "no")

    if DIAG:
        try:
            # pick the right time column (raw often uses tbin_utc, later tables tbin_latest)
            time_col = "tbin_utc" if "tbin_utc" in df.columns else ("tbin_latest" if "tbin_latest" in df.columns else None)
            if time_col is None:
                raise KeyError("No time bin column found (expected 'tbin_utc' or 'tbin_latest').")

            # ensure datetime for grouping/uniques (no-op if already datetime)
            if not np.issubdtype(df[time_col].dtype, np.datetime64):
                df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

            bins_unique = df[time_col].nunique(dropna=True)
            stations_unique = df["station_id"].nunique(dropna=True)

            c1 = (df.groupby(time_col)["station_id"].nunique()
                    .rename("stations_per_bin"))
            c2 = (df.groupby("station_id")[time_col].nunique()
                    .rename("bins_per_station")
                    .sort_values(ascending=False))

            print(f"[diag] unique bins in window = {bins_unique}")
            print(f"[diag] unique stations = {stations_unique}")
            print("[diag] stations_per_bin (first 5):")
            print(c1.head(5).to_string())
            print("[diag] bins_per_station (top 10):")
            print(c2.head(10).to_string())

        except Exception as e:
            print(f"[diag] failed: {e}")
    # ---------------------------------------------------------------------------


    feats = _build_features(df, start_naive, end_naive)
    print(f"[features_4h] features rows={len(feats):,}", flush=True)
    if feats.empty:
        return 0

    # 4) Save parquet (stamped + latest) & upload
    out_dir = Path("/tmp/features_4h_out"); out_dir.mkdir(parents=True, exist_ok=True)
    stamped = f"features_4h_{end_naive.strftime('%Y%m%dT%H%M')}.parquet"
    local_stamped = out_dir / stamped
    local_latest  = out_dir / "latest.parquet"

    feats.to_parquet(local_stamped, index=False)
    feats.to_parquet(local_latest, index=False)
    print(f"[features_4h] wrote local: {local_stamped.name} & {local_latest.name}", flush=True)

    bkt, pfx = _parse_gs(SERVING_PREFIX)
    _upload_file(cli, local_stamped, f"gs://{bkt}/{pfx}/{stamped}")
    _upload_file(cli, local_latest,  f"gs://{bkt}/{pfx}/latest.parquet")
    print(f"[features_4h] uploaded → gs://{bkt}/{pfx}/{stamped} & gs://{bkt}/{pfx}/latest.parquet", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
