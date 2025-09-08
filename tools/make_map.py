import pandas as pd, folium, pathlib


root = pathlib.Path(__file__).resolve().parents[1]
hour = pd.read_parquet(root/"exports/velib_hourly.parquet")
last = hour["hour_utc"].max()
snap = hour[hour["hour_utc"]==last].dropna(subset=["lat","lon"]).copy()
snap["occ_pct"] = (snap["occ_ratio_hour"]*100).clip(0,100).round(1)
m = folium.Map(location=[48.8566,2.3522], zoom_start=12, tiles="cartodbpositron")
for _,r in snap.iterrows():
    color = "red" if r["occ_pct"]>80 else "orange" if r["occ_pct"]>60 else "green"
    folium.CircleMarker([r["lat"],r["lon"]], radius=5, color=color, fill=True, fill_opacity=0.85,
        popup=f"{r['name']} — {r['occ_pct']}%").add_to(m)
out = root/"docs/assets/map.html"; out.parent.mkdir(parents=True, exist_ok=True); m.save(out)
print("OK map:", out)
