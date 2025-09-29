from __future__ import annotations
import os, tempfile, shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import duckdb
from google.cloud import storage

def _daterange(d0: date, d1: date):
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)

def _download_daily_shard(cli: storage.Client, root: str, day: date, dest_dir: Path) -> Path | None:
    assert root.startswith("gs://")
    bkt, pfx = root[5:].split("/", 1)
    fname = f"velib_{day.strftime('%Y%m%d')}.duckdb"
    key = f"{pfx}/{fname}"
    blob = cli.bucket(bkt).blob(key)
    if not blob.exists(cli):
        return None
    local = dest_dir / fname
    blob.download_to_filename(str(local))
    return local

def main() -> int:
    gcs_daily = os.environ["GCS_DB_DAILY"]
    start_s = os.environ.get("START_DAY")
    end_s   = os.environ.get("END_DAY")
    if not start_s or not end_s:
        end = datetime.utcnow().date()
        start = end - timedelta(days=30)
    else:
        start = datetime.strptime(start_s, "%Y-%m-%d").date()
        end   = datetime.strptime(end_s,   "%Y-%m-%d").date()

    out_dir = Path("exports"); out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "velib.parquet"

    cli = storage.Client()
    work = Path(tempfile.mkdtemp(prefix="shards_"))

    # 1) Télécharger les shards
    shards: list[Path] = []
    for d in _daterange(start, end):
        p = _download_daily_shard(cli, gcs_daily, d, work)
        if p: shards.append(p)
    if not shards:
        print("[export_training] no shards found in range")
        return 0

    # 2) Connexion & ATTACH
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads=4;")
    for i, p in enumerate(shards, 1):
        con.execute(f"ATTACH '{p.as_posix()}' AS sh{i} (READ_ONLY);")

    # DEBUG: stats par shard (utiliser con.execute)
    for i in range(1, len(shards)+1):
        n, tmin, tmax = con.execute(
            f"SELECT COUNT(*), min(tbin_utc), max(tbin_utc) FROM sh{i}.bronze.raw_snapshots_5min"
        ).fetchone()
        print(f"[dbg] sh{i} raw rows={n} span=({tmin}..{tmax})")

    # 3) UNION ALL bronze + météo (optionnelle)
    unions = []
    for i in range(1, len(shards) + 1):
        has_weather = bool(con.execute(f"""
            SELECT 1
            FROM sh{i}.information_schema.tables
            WHERE table_schema = 'bronze' AND table_name = 'weather_5min'
            LIMIT 1
        """).fetchone())

        select_part = f"""
            SELECT
              b.tbin_utc::TIMESTAMP                             AS tbin_utc,
              b.ts_utc::TIMESTAMP                               AS ts_utc,
              CAST(b.station_id AS VARCHAR)                     AS stationcode,
              COALESCE(b.bikes, b.mechanical + b.ebike, 0)      AS nb_velos_bin,
              b.capacity                                        AS capacity_bin,
              CASE WHEN b.capacity IS NOT NULL
                   THEN GREATEST(b.capacity - COALESCE(b.bikes, b.mechanical + b.ebike, 0), 0)
                   ELSE NULL END::INT                           AS nb_bornes_bin,
              CASE WHEN b.capacity IS NOT NULL AND b.capacity > 0
                   THEN LEAST(GREATEST(COALESCE(b.bikes, b.mechanical + b.ebike, 0)::DOUBLE
                                       / b.capacity::DOUBLE, 0.0), 1.0)
                   ELSE NULL END                                 AS occ_ratio_bin,
              {('w.temp_C, w.precip_mm, w.wind_mps,') if has_weather else ('NULL::DOUBLE AS temp_C, NULL::DOUBLE AS precip_mm, NULL::DOUBLE AS wind_mps,')}
              date_trunc('hour', b.tbin_utc)                    AS hour_utc
            FROM sh{i}.bronze.raw_snapshots_5min b
            {f"LEFT JOIN sh{i}.bronze.weather_5min w ON w.tbin_utc = date_trunc('hour', b.tbin_utc)" if has_weather else ""}
        """
        unions.append(select_part)

    sql = " UNION ALL ".join(unions)
    con.execute(f"CREATE OR REPLACE VIEW train_base AS {sql};")

    # DEBUG: taille union & sample (via con.execute)
    n_union = con.execute("SELECT COUNT(*) FROM train_base").fetchone()[0]
    print(f"[dbg] union rows={n_union}")
    try:
        sample = con.execute(
            "SELECT tbin_utc, stationcode, nb_velos_bin FROM train_base ORDER BY tbin_utc DESC LIMIT 5"
        ).fetchdf()
        print(sample)
    except Exception as e:
        print(f"[dbg] sample error: {e}")

    # 4) Filtre souple
    df = con.execute("""
        SELECT *
        FROM train_base
        WHERE stationcode IS NOT NULL
          AND tbin_utc   IS NOT NULL
          AND nb_velos_bin IS NOT NULL
          AND nb_velos_bin >= 0
        ORDER BY stationcode, tbin_utc
    """).fetchdf()
    print(f"[dbg] selected rows={len(df)}")

    # 5) Parquet
    df.to_parquet(out_parquet, index=False)
    print(f"[export_training] wrote {len(df)} rows -> {out_parquet}")

    con.close()
    shutil.rmtree(work, ignore_errors=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
