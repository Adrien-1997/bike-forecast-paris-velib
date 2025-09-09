import duckdb, pandas as pd
from pathlib import Path

DOCS = Path("docs")
OUT = DOCS / "partials" / "kpi_results.md"
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    con = duckdb.connect("warehouse.duckdb")
    q = """
    WITH ranked AS (
      SELECT stationcode, name, lat, lon, capacity, ts_utc,
             numbikesavailable AS nb_velos, numdocksavailable AS nb_bornes,
             ROW_NUMBER() OVER (PARTITION BY stationcode ORDER BY ts_utc DESC) rn
      FROM velib_snapshots
    )
    SELECT * FROM ranked WHERE rn=1 AND lat IS NOT NULL AND lon IS NOT NULL
    """
    df = con.execute(q).fetchdf()
    if df.empty:
        OUT.write_text("> Aucune donnée disponible.", encoding="utf-8"); return

    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
    df["occ_ratio"] = (df["nb_velos"] / df["capacity"]).where(df["capacity"]>0)
    k = {
      "stations": df["stationcode"].nunique(),
      "bikes_total": int(pd.to_numeric(df["nb_velos"]).fillna(0).sum()),
      "docks_total": int(pd.to_numeric(df["nb_bornes"]).fillna(0).sum()),
      "occ_mean": round(100*pd.to_numeric(df["occ_ratio"]).dropna().mean(),1),
      "ts_latest": pd.to_datetime(df["ts_utc"]).max()
    }
    md = f"""
<div style="display:flex;gap:16px;flex-wrap:wrap;margin:8px 0 16px 0">
  <div><b>Stations</b><br/>{k['stations']}</div>
  <div><b>Vélos dispo</b><br/>{k['bikes_total']}</div>
  <div><b>Bornes libres</b><br/>{k['docks_total']}</div>
  <div><b>Occupation moyenne</b><br/>{k['occ_mean']}%</div>
  <div><b>Dernier snapshot (UTC)</b><br/>{k['ts_latest']}</div>
</div>
"""
    OUT.write_text(md.strip(), encoding="utf-8")
    print("[kpi] OK -> docs/partials/kpi_results.md")

if __name__=="__main__":
    main()
