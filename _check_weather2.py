import pandas as pd, pathlib
exp = pathlib.Path("exports")
df = pd.read_parquet(exp/"velib_hourly.parquet") if (exp/"velib_hourly.parquet").exists() else pd.read_csv(exp/"velib_hourly.csv")
df["hour_utc"] = pd.to_datetime(df["hour_utc"], errors="coerce")  # <- NAÏF attendu
print("dtype hour_utc:", df["hour_utc"].dtype)
last = df["hour_utc"].max()
last24 = df[df["hour_utc"] > last - pd.Timedelta("24h")]
for c in ["temp_C","precip_mm","wind_mps"]:
    if c in df.columns:
        nn = last24[c].notna().sum()
        print(f"{c} non-nulls last24:", int(nn), "/", len(last24))
print("\nSample non-null rows:")
sub = df.dropna(subset=[c for c in ["temp_C","precip_mm","wind_mps"] if c in df.columns])
print(sub[["hour_utc","stationcode","temp_C","precip_mm","wind_mps"]].tail(5).to_string(index=False) if len(sub) else "(none)")
