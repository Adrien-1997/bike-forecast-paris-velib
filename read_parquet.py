import pandas as pd
from pathlib import Path

# === 1) Charger le fichier ===
path = Path(r"H:\Downloads\velib_monthly_compact_202510.parquet")
df = pd.read_parquet(path)

print("=== FILE INFO ===")
print(f"File: {path.name}")
print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
print("Columns:", list(df.columns))
print()

# === 2) Normalisation temps ===
df["ts_utc"]   = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce").dt.tz_localize(None)
df["tbin_utc"] = pd.to_datetime(df["tbin_utc"], utc=True, errors="coerce").dt.tz_localize(None)

# === 3) Aperçu temporel ===
print("=== TIME RANGE ===")
print("ts_utc  :", df["ts_utc"].min(),  "→", df["ts_utc"].max())
print("tbin_utc:", df["tbin_utc"].min(),"→", df["tbin_utc"].max())
print("unique tbin_utc:", df["tbin_utc"].nunique())
print()

# === 4) Couverture par bin (nb de stations présentes) ===
by_bin = df.groupby("tbin_utc")["station_id"].nunique().rename("stations_present")
expected = by_bin.max()  # proxy : max observé = réseau complet
report = pd.DataFrame({
    "stations_present": by_bin,
    "completeness_%": (by_bin / expected * 100).round(2)
}).sort_index()

print("=== COVERAGE BY BIN (first 10) ===")
print(report.head(10).to_string())
print("\n=== COVERAGE BY BIN (last 10) ===")
print(report.tail(10).to_string())
print("\n=== COVERAGE SUMMARY ===")
print(report["stations_present"].describe())
print(f"Expected (max observed): {expected}")
print(f"Average completeness: {report['completeness_%'].mean():.1f}%")
print()

# === 5) Doublons par (station_id, tbin_utc) ===
dups = df.duplicated(subset=["station_id","tbin_utc"]).sum()
print("=== DUPLICATES ===")
print("Duplicate rows on (station_id, tbin_utc):", dups)
if dups:
    print(df[df.duplicated(subset=["station_id","tbin_utc"], keep=False)]
            .sort_values(["station_id","tbin_utc"])
            .head(20)
            .to_string())
print()

# === 6) Stations manquantes dans les bins les plus incomplets ===
if expected:
    worst = report.nsmallest(5, "stations_present").index
    miss_list = []
    for t in worst:
        present = set(df.loc[df["tbin_utc"]==t, "station_id"])
        fullset = set(df.loc[df["tbin_utc"]==report["stations_present"].idxmax(), "station_id"])
        missing = sorted(list(fullset - present))
        miss_list.append((t, len(missing), missing[:10]))  # échantillon de 10 IDs
    print("=== WORST BINS (missing examples) ===")
    for t, n, sample in miss_list:
        print(f"{t}  missing={n}   examples={sample}")
print()

# === 7) Qualité météo (si présent) ===
meteo_cols = [c for c in ["temp_C","precip_mm","wind_mps"] if c in df.columns]
if meteo_cols:
    null_rate = (df[["tbin_utc"]+meteo_cols]
                 .assign(one=1)
                 .groupby("tbin_utc")
                 .apply(lambda g: (g[meteo_cols].isna().sum()/len(g)*100).round(2))
                 .rename(columns=lambda c: f"{c}_null_%"))
    print("=== WEATHER NULLS BY BIN (tail) ===")
    print(null_rate.tail(10).to_string())
