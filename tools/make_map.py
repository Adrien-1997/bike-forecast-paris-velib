import pandas as pd, folium, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
hour = pd.read_parquet(root/"exports/velib_hourly.parquet")
last = hour["hour_utc"].max()
snap = hour[hour["hour_utc"]==last].dropna(subset=["lat","lon"]).copy()
snap["occ_%"] = (snap["occ_ratio_hour"]*100).round(1)
m = folium.Map(location=[48.8566,2.3522], zoom_start=12, tiles="cartodbpositron")
for _,r in snap.iterrows():
    folium.CircleMarker([r["lat"],r["lon"]], radius=5, fill=True,
        popup=f"{r['name']} — {r['occ_%']}%").add_to(m)
out = root/"docs/assets/map.html"; m.save(out); print("OK:", out)
