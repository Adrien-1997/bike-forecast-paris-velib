# tools/check_ingestion.py
from pathlib import Path
import sys
import pandas as pd
import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB   = ROOT / "warehouse.duckdb"
EXP  = ROOT / "exports" / "velib_hourly.parquet"

required_cols = [
    "ts_utc","stationcode","name","lat","lon",
    "numbikesavailable","numdocksavailable","capacity","mechanical","ebike"
]

def ok(msg):  print(f"[OK] {msg}")
def warn(msg): print(f"[WARN] {msg}")
def err(msg):  print(f"[ERR] {msg}")

def main():
    if not DB.exists():
        err("warehouse.duckdb introuvable. Lance d'abord: py -m src.ingest")
        sys.exit(2)
    con = duckdb.connect(str(DB))

    # 1) table présente ?
    tables = con.execute("SHOW TABLES").fetchdf()["name"].str.lower().tolist()
    if "velib_snapshots" not in tables:
        err("Table velib_snapshots absente. Lance py -m src.ingest")
        sys.exit(2)
    ok("Table velib_snapshots trouvée")

    # 2) colonnes attendues + stats globales
    info = con.execute("PRAGMA table_info('velib_snapshots')").fetchdf()
    cols = info["name"].tolist()
    missing = [c for c in required_cols if c not in cols]
    if missing:
        warn(f"Colonnes manquantes vs schéma attendu: {missing}")
    else:
        ok("Schéma minimal OK")

    # 3) fraicheur + volume
    q = """
    SELECT
      COUNT(*) AS n_rows,
      COUNT(DISTINCT stationcode) AS n_stations,
      MIN(ts_utc) AS ts_min,
      MAX(ts_utc) AS ts_max
    FROM velib_snapshots
    """
    g = con.execute(q).fetchdf().iloc[0]
    print(f"- Rows: {g.n_rows:,} | Stations: {g.n_stations:,}")
    print(f"- ts_utc min: {g.ts_min} | ts_utc max: {g.ts_max}")
    if pd.to_datetime(g.ts_max) < pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=6):
        warn("Dernier snapshot a plus de 6h. OK en mode offline, sinon re-lancer l’ingestion.")
    else:
        ok("Fraîcheur du dernier snapshot OK (≤ 6h)")

    # 4) checks d’invariants de base
    checks = [
        ("négatifs", "SELECT COUNT(*) FROM velib_snapshots WHERE numbikesavailable<0 OR numdocksavailable<0 OR capacity<0 OR mechanical<0 OR ebike<0"),
        ("lat/lon NULL", "SELECT COUNT(*) FROM velib_snapshots WHERE lat IS NULL OR lon IS NULL"),
        ("bikes > capacity", "SELECT COUNT(*) FROM velib_snapshots WHERE capacity IS NOT NULL AND numbikesavailable > capacity"),
        ("mech+ebike > bikes", "SELECT COUNT(*) FROM velib_snapshots WHERE mechanical IS NOT NULL AND ebike IS NOT NULL AND numbikesavailable IS NOT NULL AND mechanical+ebike > numbikesavailable"),
    ]
    for label, sql in checks:
        n = con.execute(sql).fetchone()[0]
        if n > 0:
            warn(f"Invariant '{label}' violé pour {n} lignes")
        else:
            ok(f"Invariant '{label}' OK")

    # 5) aperçu des 5 dernières lignes (par station la plus récente)
    print("\nDernières lignes (5):")
    last5 = con.execute("""
        SELECT stationcode, name, ts_utc, numbikesavailable, numdocksavailable, capacity, mechanical, ebike, lat, lon
        FROM velib_snapshots
        ORDER BY ts_utc DESC
        LIMIT 5
    """).fetchdf()
    print(last5.to_string(index=False))

    # 6) export horaire parquet
    if EXP.exists():
        try:
            dfh = pd.read_parquet(EXP)
            if dfh.empty:
                warn("exports/velib_hourly.parquet est vide")
            else:
                # garde dernières 24h pour signalement bref
                dfh["hour_utc"] = pd.to_datetime(dfh["hour_utc"], errors="coerce")
                latest = dfh["hour_utc"].max()
                n_pairs = dfh[ dfh["hour_utc"]==latest ]["stationcode"].nunique()
                ok(f"Parquet horaire OK • dernier hour_utc={latest} • stations agrégées={n_pairs}")
        except Exception as e:
            warn(f"Lecture parquet échouée: {e}")
    else:
        warn("exports/velib_hourly.parquet absent. Lance: py -m src.aggregate")

    print("\nRésumé rapide:")
    print("- Si tu es en offline (FORCE_OFFLINE=1), c’est normal d’avoir peu de stations et une fraîcheur ≈ maintenant.")
    print("- En online, attends-toi à ~1400+ stations et un ts_max très récent.")
    print("- Si des invariants sont violés en masse, recheck src.ingest/schema & mapping.")
    return 0

if __name__ == "_main_":
    raise SystemExit(main())