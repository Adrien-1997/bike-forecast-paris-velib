# tools/make_map.py
import pathlib, requests, pandas as pd, folium
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


ROOT = pathlib.Path(__file__).resolve().parents[1]
exp  = ROOT / "exports"
docs = ROOT / "docs"
(docs / "assets").mkdir(parents=True, exist_ok=True)

# 1) Dernier état d'occupation par station
hour = pd.read_parquet(exp/"velib_hourly.parquet") if (exp/"velib_hourly.parquet").exists() else pd.read_csv(exp/"velib_hourly.csv")
last = pd.to_datetime(hour["hour_utc"], utc=True, errors="coerce").max()
snap = hour[pd.to_datetime(hour["hour_utc"], utc=True, errors="coerce")==last].copy()
snap["occ_pct"] = (snap["occ_ratio_hour"]*100).clip(0,100).round(1)

# 2) Coords: si pas de colonnes lat/lon -> fetch mapping depuis Opendatasoft
if not {"lat","lon"}.issubset(snap.columns):
    url = ("https://data.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
           "velib-emplacement-des-stations@parisdata/records?limit=10000")
    js  = requests.get(url, timeout=30).json()
    rows = js.get("results", [])
    mapping = {}
    for r in rows:
        code = str(r.get("stationcode"))
        geo  = r.get("geo_point_2d") or {}
        lat, lon = geo.get("lat"), geo.get("lon")
        if code and lat is not None and lon is not None:
            mapping[code] = (float(lat), float(lon))
    snap["lat"] = snap["stationcode"].astype(str).map(lambda c: mapping.get(c, (None,None))[0])
    snap["lon"] = snap["stationcode"].astype(str).map(lambda c: mapping.get(c, (None,None))[1])

snap = snap.dropna(subset=["lat","lon"])

# 3) Carte
m = folium.Map(location=[48.8566, 2.3522], zoom_start=12, tiles="cartodbpositron")
for _, r in snap.iterrows():
    color = "red" if r["occ_pct"]>80 else "orange" if r["occ_pct"]>60 else "green"
    folium.CircleMarker([r["lat"], r["lon"]], radius=5, color=color, fill=True, fill_opacity=0.85,
                        popup=f"{r.get('name','?')} — {r['occ_pct']}%").add_to(m)
out = docs / "assets" / "map.html"
m.save(out)
print("OK map:", out)
