import pandas as pd, pathlib
exp = pathlib.Path("exports")
df = pd.read_parquet(exp/"velib_hourly.parquet") if (exp/"velib_hourly.parquet").exists() else pd.read_csv(exp/"velib_hourly.csv")
df["hour_utc"] = pd.to_datetime(df["hour_utc"], errors="coerce")  # ne PAS remettre utc=True ici
print("dtype hour_utc:", df["hour_utc"].dtype)  # doit être datetime64[ns] (sans tz)
cols = [c for c in ["hour_utc","stationcode","occ_ratio_hour","temp_C","precip_mm","wind_mps"] if c in df.columns]
print("cols:", cols)
last24 = df[df["hour_utc"] > df["hour_utc"].max() - pd.Timedelta("24h")]
for c in ["temp_C","precip_mm","wind_mps"]:
    if c in df.columns:
        print(f"{c} nulls last24:", int(last24[c].isna().sum()), "/", len(last24))
print("\nSample non-null weather rows:")
sub = df.dropna(subset=[c for c in ["temp_C","precip_mm","wind_mps"] if c in df.columns])
print(sub[cols].tail(5).to_string(index=False) if len(sub) else "(none)")
