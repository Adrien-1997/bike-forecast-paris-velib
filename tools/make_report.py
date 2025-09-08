import pathlib, pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib.dates as mdates
ROOT = pathlib.Path(__file__).resolve().parents[1]
exp, docs = ROOT/'exports', ROOT/'docs'
(docs/'assets').mkdir(parents=True, exist_ok=True); (docs/'exports').mkdir(parents=True, exist_ok=True)
LOCAL_TZ='Europe/Paris'; HIST=36
hpq, hcsv = exp/'velib_hourly.parquet', exp/'velib_hourly.csv'
hourly = pd.read_parquet(hpq) if hpq.exists() else pd.read_csv(hcsv)
hourly["hour_utc"]=pd.to_datetime(hourly["hour_utc"], utc=True, errors="coerce")
hourly["hour_local"]=hourly["hour_utc"].dt.tz_convert(LOCAL_TZ)
ppq, pcsv = exp/'velib_forecast_24h.parquet', exp/'velib_forecast_24h.csv'
preds = pd.read_parquet(ppq) if ppq.exists() else (pd.read_csv(pcsv) if pcsv.exists() else pd.DataFrame(columns=['stationcode','hour_utc','pred_occ']))
if not preds.empty: 
    preds["hour_utc"]=pd.to_datetime(preds["hour_utc"], utc=True, errors="coerce")
    preds["hour_local"]=preds["hour_utc"].dt.tz_convert(LOCAL_TZ)
n_stations = hourly["stationcode"].nunique()
dmin, dmax = hourly["hour_local"].min(), hourly["hour_local"].max()
recent = hourly[hourly["hour_local"] > (dmax - pd.Timedelta(hours=HIST))]
coverage = recent.groupby("stationcode")["occ_ratio_hour"].count().sort_values(ascending=False)
sc = str(coverage.index[0]) if len(coverage) else str(hourly["stationcode"].iloc[0])
hist = hourly[hourly["stationcode"].astype(str)==sc].sort_values("hour_local").tail(HIST)
fc   = preds[preds.get("stationcode","").astype(str)==sc] if not preds.empty else pd.DataFrame()
png = docs/'assets'/'sample_forecast.png'
plt.figure(figsize=(10,4))
if not hist.empty: plt.plot(hist["hour_local"], hist["occ_ratio_hour"], label="historique")

if not fc.empty:   plt.plot(fc["hour_local"],   fc["pred_occ"],      label="forecast 24h")

plt.title(f"Occupation ratio — station {sc}"); plt.xlabel(f"Heure locale ({LOCAL_TZ})"); plt.ylabel("ratio (0–1)")

if not hist.empty or not fc.empty: plt.legend()

ax=plt.gca(); ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))

if not hist.empty: plt.xlim(hist["hour_local"].min(), (fc["hour_local"].max() if not fc.empty else hist["hour_local"].max()))

plt.tight_layout(); plt.savefig(png, dpi=150); plt.close()

top = (hourly.dropna(subset=["occ_ratio_hour"]).groupby(["stationcode","name"], as_index=False)["occ_ratio_hour"]
        .std().rename(columns={"occ_ratio_hour":"std_occ"}).sort_values("std_occ", ascending=False).head(10))

top["std_occ"]=top["std_occ"].round(3)

# --- Health ingestion ---
import duckdb, pandas as pd
con = duckdb.connect("warehouse.duckdb")
snap_count = con.sql("SELECT COUNT(*) c FROM velib_snapshots").fetchone()[0]
last_ts = con.sql("SELECT max(ts_utc) FROM velib_snapshots").fetchone()[0]
stations = con.sql("SELECT COUNT(DISTINCT stationcode) FROM velib_snapshots").fetchone()[0]
health_md = f"**Snapshots**: {snap_count}  •  **Stations**: {stations}  •  **Last (UTC)**: {last_ts}"

# --- Petit scatter occ vs temp (si dispo) ---
if "temp_C" in hourly.columns and hourly["temp_C"].notna().any():
    import matplotlib.pyplot as plt
    sc_path = docs / "assets" / "occ_vs_temp.png"
    sample = hourly.dropna(subset=["occ_ratio_hour","temp_C"]).sample(min(5000, len(hourly)), random_state=42)
    plt.figure(figsize=(6,4))
    plt.scatter(sample["temp_C"], sample["occ_ratio_hour"], s=6, alpha=0.3)
    plt.xlabel("Temp (°C)"); plt.ylabel("Occ ratio")
    plt.tight_layout(); plt.savefig(sc_path, dpi=120); plt.close()
else:
    sc_path = None

def table_md(df):
    try:
        import tabulate as _t
        return df.to_markdown(index=False)
    except Exception:
        return "<div>"+df.to_html(index=False, border=0)+"</div>"
for src in [exp/'velib_forecast_24h.csv', exp/'velib_hourly.csv']:
    if src.exists(): (docs/'exports'/src.name).write_bytes(src.read_bytes())
# --- Construction de la page Results ---
md = []
md += ["# Results", ""]

# 👉 Ajoute l'état de santé dès le haut
md += [health_md, ""]

# Stats générales (heure locale)
md += [f"**Historique couvert** : {dmin_loc} → {dmax_loc}  "]
md += [f"**Stations** : {n_stations}  "]
md += [f"*(Heure affichée : {LOCAL_TZ})*", ""]

# Graphe exemple
md += ["## Example (historique + forecast 24h)"]
md += ["![sample](assets/sample_forecast.png)", ""]

# (option) Corrélation simple si l'image a été générée
if sc_path:
    md += ["## Corrélation simple", "Relation occ_ratio vs. température (échantillon)", 
           "![occ vs temp](assets/occ_vs_temp.png)", ""]

# Top volatilité
md += ["## Top 10 stations les plus volatiles", table_md(top), ""]

# Exports
md += ["## Exports"]
if pred_csv.exists():
    md += ["- [Prévision 24h (CSV)](exports/velib_forecast_24h.csv)"]
if (docs / "exports" / "velib_hourly.csv").exists():
    md += ["- [Occupations horaires (échantillon CSV)](exports/velib_hourly.csv)"]

# Écriture du fichier
(docs / "results.md").write_text("\n".join(md), encoding="utf-8")
print("OK — docs/results.md mis à jour avec health + corrélation.")

