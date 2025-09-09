# tools/make_map.py
import os
import duckdb
import pandas as pd
import folium
from folium.plugins import MarkerCluster

DB = "warehouse.duckdb"
OUT_HTML = "docs/assets/map.html"
os.makedirs("docs/assets", exist_ok=True)

con = duckdb.connect(DB)

# 1) On prend l’instantané le plus récent par station (depuis velib_snapshots)
q = """
WITH ranked AS (
  SELECT
    stationcode, name, lat, lon, capacity,
    ts_utc, numbikesavailable AS nb_velos, numdocksavailable AS nb_bornes,
    ROW_NUMBER() OVER (PARTITION BY stationcode ORDER BY ts_utc DESC) AS rn
  FROM velib_snapshots
)
SELECT stationcode, name, lat, lon, capacity,
       ts_utc, nb_velos, nb_bornes
FROM ranked
WHERE rn = 1 AND lat IS NOT NULL AND lon IS NOT NULL
"""

df = con.execute(q).fetchdf()
if df.empty:
    raise SystemExit("Aucune donnée pour la carte. Lance d'abord: py -m src.ingest")

# 2) KPIs dérivés
df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")
df["occ_ratio"] = (df["nb_velos"] / df["capacity"]).where(df["capacity"] > 0, None)
df["occ_ratio"] = df["occ_ratio"].clip(0, 1)

# 3) Centre carte ≈ centre géographique des stations
center = [df["lat"].mean(), df["lon"].mean()]
m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

# 4) Palette simple selon taux d’occupation
def color_for_ratio(r):
    if pd.isna(r): return "#9e9e9e"   # gris: inconnu
    if r <= 0.20:  return "#f44336"   # rouge: quasi vide
    if r <= 0.50:  return "#ff9800"   # orange
    if r <= 0.80:  return "#ffc107"   # jaune
    return "#4caf50"                  # vert: bien rempli

cluster = MarkerCluster().add_to(m)

for row in df.itertuples():
    cap = int(row.capacity) if pd.notna(row.capacity) else None
    popup = folium.Popup(
        html=f"""
        <b>{row.name}</b><br/>
        Code: {row.stationcode}<br/>
        Vélos: <b>{row.nb_velos}</b>{' / ' + str(cap) if cap else ''}<br/>
        Bornes libres: {row.nb_bornes}<br/>
        Taux: {'' if pd.isna(row.occ_ratio) else f'{row.occ_ratio*100:.0f}%'}<br/>
        TS: {pd.to_datetime(row.ts_utc)}
        """,
        max_width=280
    )
    folium.CircleMarker(
        location=[row.lat, row.lon],
        radius=6,
        fill=True,
        color=color_for_ratio(row.occ_ratio),
        fill_opacity=0.85,
        popup=popup,
        tooltip=f"{row.name} — vélos: {row.nb_velos}"
    ).add_to(cluster)

# 5) Légende minimaliste
legend_html = """
<div style="
 position: fixed; bottom: 20px; left: 20px; z-index: 9999;
 background: white; padding: 10px 12px; border: 1px solid #ccc; border-radius: 8px;
 font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; font-size: 13px;">
<b>Disponibilité vélos</b><br/>
<span style="display:inline-block;width:10px;height:10px;background:#f44336;border-radius:50%;margin-right:6px;"></span> ≤ 20%<br/>
<span style="display:inline-block;width:10px;height:10px;background:#ff9800;border-radius:50%;margin-right:6px;"></span> ≤ 50%<br/>
<span style="display:inline-block;width:10px;height:10px;background:#ffc107;border-radius:50%;margin-right:6px;"></span> ≤ 80%<br/>
<span style="display:inline-block;width:10px;height:10px;background:#4caf50;border-radius:50%;margin-right:6px;"></span> > 80%<br/>
<span style="display:inline-block;width:10px;height:10px;background:#9e9e9e;border-radius:50%;margin-right:6px;"></span> inconnu
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

m.save(OUT_HTML)
print(f"[map] écrit → {OUT_HTML} ({len(df)} stations)")
