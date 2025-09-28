# pipeline/build_features_4h.py
from __future__ import annotations
import os, sys, shutil
from pathlib import Path
from datetime import datetime, timedelta, timezone
from google.cloud import storage
import duckdb

RAW_PREFIX = os.environ["GCS_RAW_PREFIX"]              # gs://.../velib/bronze
SERVING_PREFIX = os.environ["GCS_SERVING_PREFIX"]      # gs://.../velib/serving/features_4h
WINDOW_HOURS = int(os.environ.get("WINDOW_HOURS", "4"))
DO_DIAG = os.environ.get("DIAG", "0") == "1"

def _floor_5min(dt: datetime) -> datetime:
    m = (dt.minute // 5) * 5
    return dt.replace(minute=m, second=0, microsecond=0)

def _iter_hours(start: datetime, end: datetime):
    cur = start.replace(minute=0, second=0, microsecond=0)
    last = end.replace(minute=0, second=0, microsecond=0)
    while cur <= last:
        yield cur
        cur += timedelta(hours=1)

def _list_raw_files_for_window(cli: storage.Client, start: datetime, end: datetime) -> list[str]:
    assert RAW_PREFIX.startswith("gs://")
    bkt, pfx = RAW_PREFIX[5:].split("/", 1)
    out = []
    for h in _iter_hours(start, end):
        day = h.strftime("%Y-%m-%d"); hh = h.strftime("%H")
        prefix = f"{pfx}/date={day}/hour={hh}/"
        for b in cli.list_blobs(bkt, prefix=prefix):
            if b.name.endswith(".parquet"):
                out.append(f"gs://{bkt}/{b.name}")
    return out

def _download_gs_files(cli: storage.Client, uris: list[str], dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for uri in uris:
        assert uri.startswith("gs://")
        bkt, key = uri[5:].split("/", 1)
        local = dest_dir / Path(key).name
        cli.bucket(bkt).blob(key).download_to_filename(str(local))
        paths.append(local)
    return paths

def _upload_file(cli: storage.Client, local: Path, dest_uri: str) -> None:
    assert dest_uri.startswith("gs://")
    bkt, key = dest_uri[5:].split("/", 1)
    cli.bucket(bkt).blob(key).upload_from_filename(str(local))

def main() -> int:
    # now / fenêtre
    if "NOW_UTC_ISO" in os.environ:
        now_utc = datetime.fromisoformat(os.environ["NOW_UTC_ISO"].replace("Z","+00:00")).astimezone(timezone.utc)
    else:
        now_utc = datetime.now(timezone.utc)
    end_tbin = _floor_5min(now_utc)
    start_tbin = end_tbin - timedelta(hours=WINDOW_HOURS) + timedelta(minutes=5)  # 48 bins

    print(f"[features_4h][cfg] RAW={RAW_PREFIX} SERVING={SERVING_PREFIX}", flush=True)
    print(f"[features_4h] window UTC: {start_tbin.isoformat()} → {end_tbin.isoformat()} (inclusive)", flush=True)

    cli = storage.Client()

    # 1) Lister + 2) Télécharger
    gcs_files = _list_raw_files_for_window(cli, start_tbin, end_tbin)
    print(f"[features_4h] gcs files found = {len(gcs_files)}", flush=True)
    if not gcs_files:
        print("[features_4h] no raw files in window — exit 0", flush=True)
        return 0

    work = Path("/tmp/features_4h_raw")
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    local_files = _download_gs_files(cli, gcs_files, work)
    print(f"[features_4h] local files = {len(local_files)} in {work}", flush=True)

    # 3) DuckDB (in-memory)
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads=4;")

    start_lit = start_tbin.strftime("%Y-%m-%d %H:%M:%S")
    end_lit   = end_tbin.strftime("%Y-%m-%d %H:%M:%S")
    glob_pat  = (work / "*.parquet").as_posix()

    # Vue brute (union_by_name) et découverte des colonnes
    con.execute(f"CREATE OR REPLACE VIEW raw AS SELECT * FROM read_parquet('{glob_pat}', union_by_name=true);")
    cols = [r[0] for r in con.execute("DESCRIBE SELECT * FROM raw").fetchall()]
    def has(c: str) -> bool: return c in cols
    def pick(*options: str) -> str:
        for c in options:
            if has(c): return f"raw.{c}"
        return "NULL"

    expr_tbin     = pick("tbin_utc", "tbin", "ts_bin", "time_bin")
    expr_ts       = pick("ts_utc", "timestamp_utc", "ts")
    expr_sid      = pick("station_id", "stationId", "id_station")
    expr_bikes    = pick("bikes", "numbikesavailable", "num_bikes_available")
    expr_capacity = pick("capacity", "cap", "dock_capacity", "num_docks_available")
    expr_mech     = pick("mechanical", "mech_bikes", "num_bikes_mechanical")
    expr_ebike    = pick("ebike", "e_bikes", "num_bikes_ebike", "ebikes")
    expr_status   = pick("status", "station_status")
    expr_lat      = pick("lat", "latitude")
    expr_lon      = pick("lon", "longitude")
    expr_temp     = pick("temp_C", "temperature", "temperature_2m")
    expr_precip   = pick("precip_mm", "precipitation", "precip")
    expr_wind     = pick("wind_mps", "wind_speed_10m", "wind")

    # Table base robuste + filtres via paramètres (évite les {start_lit} littéraux)
    con.execute(f"""
        CREATE TABLE base AS
        SELECT
          CAST(TRY_CAST({expr_tbin}     AS TIMESTAMP) AS TIMESTAMP) AS tbin_utc,
          CAST(TRY_CAST({expr_ts}       AS TIMESTAMP) AS TIMESTAMP) AS ts_utc,
          CAST(TRY_CAST({expr_sid}      AS BIGINT)    AS BIGINT)    AS station_id,
          CAST(TRY_CAST({expr_bikes}    AS INTEGER)   AS INTEGER)   AS bikes,
          CAST(TRY_CAST({expr_capacity} AS INTEGER)   AS INTEGER)   AS capacity,
          CAST(TRY_CAST({expr_mech}     AS INTEGER)   AS INTEGER)   AS mechanical,
          CAST(TRY_CAST({expr_ebike}    AS INTEGER)   AS INTEGER)   AS ebike,
          {expr_status} AS status,
          CAST(TRY_CAST({expr_lat}      AS DOUBLE)    AS DOUBLE)    AS lat,
          CAST(TRY_CAST({expr_lon}      AS DOUBLE)    AS DOUBLE)    AS lon,
          CAST(TRY_CAST({expr_temp}     AS DOUBLE)    AS DOUBLE)    AS temp_C,
          CAST(TRY_CAST({expr_precip}   AS DOUBLE)    AS DOUBLE)    AS precip_mm,
          CAST(TRY_CAST({expr_wind}     AS DOUBLE)    AS DOUBLE)    AS wind_mps
        FROM raw
        WHERE CAST(TRY_CAST({expr_tbin} AS TIMESTAMP) AS TIMESTAMP)
              BETWEEN CAST(? AS TIMESTAMP) AND CAST(? AS TIMESTAMP);
    """, [start_lit, end_lit])

    if DO_DIAG:
        n_all  = con.execute("SELECT COUNT(*) FROM base").fetchone()[0]
        n_st   = con.execute("SELECT COUNT(DISTINCT station_id) FROM base").fetchone()[0]
        n_null = con.execute("SELECT COUNT(*) FROM base WHERE station_id IS NULL").fetchone()[0]
        n_last = con.execute("""
            WITH lb AS (SELECT MAX(tbin_utc) tmax FROM base)
            SELECT COUNT(DISTINCT station_id) FROM base, lb WHERE tbin_utc = tmax;
        """).fetchone()[0]
        print(f"[diag] base rows={n_all} distinct_stations={n_st} null_station_id={n_null} last_bin_stations={n_last}", flush=True)

    # Agrégats/features
    con.execute("""
        CREATE TABLE features AS
        WITH agg AS (
          SELECT
            station_id,
            MAX(tbin_utc) AS tbin_latest,
            arg_max(ts_utc, tbin_utc)     AS ts_utc_latest,
            arg_max(bikes, tbin_utc)      AS bikes_latest,
            arg_max(capacity, tbin_utc)   AS capacity_latest,
            AVG(bikes)                    AS bikes_mean_4h,
            median(bikes)                 AS bikes_median_4h,
            MIN(bikes)                    AS bikes_min_4h,
            MAX(bikes)                    AS bikes_max_4h,
            COUNT(*)                      AS bins_present_4h,
            100.0 * COUNT(*) / 48.0       AS completeness_pct_4h,
            regr_slope(
              bikes,
              CAST(date_diff('minute', TIMESTAMP '1970-01-01 00:00:00', tbin_utc) AS DOUBLE) / 5.0
            )                             AS bikes_slope_per_5m,
            arg_max(temp_C, tbin_utc)     AS temp_C,
            arg_max(precip_mm, tbin_utc)  AS precip_mm,
            arg_max(wind_mps, tbin_utc)   AS wind_mps,
            arg_max(lat, tbin_utc)        AS lat,
            arg_max(lon, tbin_utc)        AS lon
          FROM base
          WHERE station_id IS NOT NULL
          GROUP BY 1
        )
        SELECT
          station_id,
          tbin_latest,
          ts_utc_latest,
          bikes_latest,
          capacity_latest,
          bikes_mean_4h, bikes_median_4h, bikes_min_4h, bikes_max_4h,
          bins_present_4h, completeness_pct_4h, bikes_slope_per_5m,
          temp_C, precip_mm, wind_mps, lat, lon,
          CAST(? AS TIMESTAMP) AS window_start_utc,
          CAST(? AS TIMESTAMP) AS window_end_utc
        FROM agg;
    """, [start_lit, end_lit])

    # 4) Écrire Parquet (latest + horodaté)
    out_dir = Path("/tmp/features_4h_out"); out_dir.mkdir(parents=True, exist_ok=True)
    stamped = f"features_4h_{end_tbin.strftime('%Y%m%dT%H%M')}.parquet"
    local_stamped = out_dir / stamped
    local_latest  = out_dir / "latest.parquet"

    con.execute(f"COPY (SELECT * FROM features) TO '{local_stamped.as_posix()}' (FORMAT PARQUET);")
    shutil.copyfile(local_stamped, local_latest)
    print(f"[features_4h] wrote local: {local_stamped.name} & {local_latest.name}", flush=True)

    # 5) Upload GCS
    bkt, pfx = SERVING_PREFIX[5:].split("/", 1)
    _upload_file(cli, local_stamped, f"gs://{bkt}/{pfx}/{stamped}")
    _upload_file(cli, local_latest,  f"gs://{bkt}/{pfx}/latest.parquet")
    print(f"[features_4h] uploaded → gs://{bkt}/{pfx}/{stamped} & gs://{bkt}/{pfx}/latest.parquet", flush=True)

    con.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
