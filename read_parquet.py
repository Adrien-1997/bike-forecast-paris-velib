# read_parquet_adaptive.py
import pandas as pd
from pathlib import Path

# =========================
# Display options (pour voir TOUTES les colonnes proprement)
# =========================
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 200)

# =========================
# 1) Charger le fichier
# =========================
path = Path(r"H:\Downloads\velib_bronze_date=2025-10-06_hour=13_2025-10-06T13-35.parquet")
if not path.exists():
    raise FileNotFoundError(f"Fichier introuvable: {path}")

df = pd.read_parquet(path)

print("=== FILE INFO ===")
print(f"File: {path.name}")
print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
print("Columns:", list(df.columns))
print()

# =========================
# 2) Normalisation du temps (auto-détection des colonnes disponibles)
# =========================
def _to_naive_utc(s: pd.Series) -> pd.Series:
    # to_datetime(utc=True) → tz-aware UTC; on supprime ensuite la tz pour un affichage homogène
    out = pd.to_datetime(s, utc=True, errors="coerce")
    # .dt.tz_localize(None) ne s'applique qu'à des timestamps tz-aware
    return out.dt.tz_convert("UTC").dt.tz_localize(None) if out.dt.tz is not None else out.dt.tz_localize(None)

time_col = None
if "tbin_utc" in df.columns:
    time_col = "tbin_utc"
    df["tbin_utc"] = _to_naive_utc(df["tbin_utc"])
elif "tbin_latest" in df.columns:
    # Serving latest → on standardise vers tbin_utc pour la suite du script
    time_col = "tbin_utc"
    df["tbin_utc"] = _to_naive_utc(df["tbin_latest"])
elif "ts_utc" in df.columns:
    time_col = "ts_utc"
    df["ts_utc"] = _to_naive_utc(df["ts_utc"])

# =========================
# 3) Aperçu temporel
# =========================
if time_col is not None:
    print("=== TIME RANGE ===")
    mn, mx = df[time_col].min(), df[time_col].max()
    nunique = df[time_col].nunique()
    print(f"{time_col}: {mn} → {mx}  | unique={nunique}")
    print()
else:
    print("=== TIME RANGE ===")
    print("Aucune colonne temporelle trouvée (ni tbin_utc, ni tbin_latest, ni ts_utc).")
    print()

# =========================
# 4) Couverture par bin (si plusieurs bins)
# =========================
def show_coverage_on_bins(df: pd.DataFrame, tcol: str):
    by_bin = df.groupby(tcol)["station_id"].nunique().rename("stations_present")
    if by_bin.empty:
        print("Pas de données pour la couverture par bin.")
        return
    expected = by_bin.max()  # proxy du réseau complet
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

    # Stations manquantes dans les bins les plus incomplets
    if expected and report["stations_present"].nunique() > 1:
        worst = report.nsmallest(5, "stations_present").index
        miss_list = []
        # On prend comme "fullset" le bin le plus complet
        fullset_bin = report["stations_present"].idxmax()
        fullset = set(df.loc[df[tcol] == fullset_bin, "station_id"])
        for t in worst:
            present = set(df.loc[df[tcol] == t, "station_id"])
            missing = sorted(list(fullset - present))
            miss_list.append((t, len(missing), missing[:10]))  # échantillon 10 IDs
        print("=== WORST BINS (missing examples) ===")
        for t, n, sample in miss_list:
            print(f"{t}  missing={n}   examples={sample}")
        print()

if time_col == "tbin_utc" and df["tbin_utc"].nunique() > 1:
    show_coverage_on_bins(df, "tbin_utc")
elif time_col == "tbin_utc" and df["tbin_utc"].nunique() == 1:
    # Cas "serving latest": un seul snapshot
    print("=== SNAPSHOT INFO ===")
    print("Latest tbin_utc:", df["tbin_utc"].iloc[0])
    print("Unique stations:", df["station_id"].nunique() if "station_id" in df.columns else "n/a")
    print()

# =========================
# 5) Doublons pertinents
#    - si tbin_utc existe: doublons sur (station_id, tbin_utc)
#    - sinon: doublons sur station_id (rare mais utile à détecter dans serving)
# =========================
print("=== DUPLICATES ===")
if "station_id" in df.columns and "tbin_utc" in df.columns:
    dups = df.duplicated(subset=["station_id", "tbin_utc"]).sum()
    print("Duplicate rows on (station_id, tbin_utc):", dups)
    if dups:
        print(df[df.duplicated(subset=["station_id", "tbin_utc"], keep=False)]
                .sort_values(["station_id", "tbin_utc"])
                .head(20)
                .to_string(index=False))
elif "station_id" in df.columns:
    dups = df.duplicated(subset=["station_id"]).sum()
    print("Duplicate rows on station_id:", dups)
    if dups:
        print(df[df.duplicated(subset=["station_id"], keep=False)]
                .sort_values(["station_id"])
                .head(20)
                .to_string(index=False))
else:
    print("Colonnes nécessaires absentes pour tester les doublons.")
print()

# =========================
# 6) Statistiques des features
# =========================
print("=== FEATURES SUMMARY (numeric describe) ===")
num_cols = df.select_dtypes(include="number").columns.tolist()
if num_cols:
    print(df[num_cols].describe().to_string())
else:
    print("Aucune colonne numérique détectée.")
print()

print("=== NULL RATES (top 30) ===")
nulls = (df.isna().mean() * 100).sort_values(ascending=False)
print(nulls.head(30).round(2).to_string())
print()

# =========================
# 7) Qualité météo (optionnel, si colonnes présentes)
# =========================
meteo_cols = [c for c in ["temp_C", "precip_mm", "wind_mps"] if c in df.columns]
if meteo_cols and "tbin_utc" in df.columns and df["tbin_utc"].nunique() > 1:
    null_rate = (df[["tbin_utc"] + meteo_cols]
                 .assign(_n=1)
                 .groupby("tbin_utc")
                 .apply(lambda g: (g[meteo_cols].isna().sum() / len(g) * 100).round(2))
                 .rename(columns=lambda c: f"{c}_null_%"))
    print("=== WEATHER NULLS BY BIN (tail) ===")
    print(null_rate.tail(10).to_string())
    print()
elif meteo_cols:
    print("=== WEATHER COLUMNS PRESENT ===")
    print(f"Cols: {meteo_cols} — pas de série temporelle multiple, affichage bin global.")
    print(df[meteo_cols].isna().mean().mul(100).round(2).rename("null_%").to_string())
    print()

# =========================
# 8) Aperçu brut (quelques lignes)
# =========================
print("=== HEAD (10 rows) ===")
print(df.head(10).to_string(index=False))
print()

print("=== DONE ===")
