import pandas as pd, pathlib
p = pathlib.Path("data/weather_hourly.parquet")
print("exists:", p.exists())
if p.exists():
    w = pd.read_parquet(p)
    print("rows:", len(w))
    if len(w):
        print("min/max:", w["hour_utc"].min(), "->", w["hour_utc"].max())
        print(w.tail(5))
