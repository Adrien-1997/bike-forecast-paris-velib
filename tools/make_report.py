# tools/make_report.py
import pathlib, duckdb, pandas as pd
import matplotlib.pyplot as plt, matplotlib.dates as mdates

ROOT = pathlib.Path(__file__).resolve().parents[1]
exp  = ROOT / "exports"
docs = ROOT / "docs"
(docs / "assets").mkdir(parents=True, exist_ok=True)
(docs / "exports").mkdir(parents=True, exist_ok=True)

LOCAL_TZ = "Europe/Paris"
HIST_WINDOW_H = 36

# --- Charger hourly ---
hour_path_parq = exp / "velib_hourly.parquet"
hour_path_csv  = exp / "velib_hourly.csv"
if hour_path_parq.exists():
    hourly = pd.read_parquet(hour_path_parq)
elif hour_path_csv.exists():
    hourly = pd.read_csv(hour_path_csv)
else:
    raise SystemExit("Aucun fichier hourly trouvé dans exports/")

hourly["hour_utc"] = pd.to_datetime(hourly["hour_utc"], errors="coerce", utc=True)
hourly["hour_local"] = hourly["hour_utc"].dt.tz_convert(LOCAL_TZ)

# --- Charger forecast (si présent) ---
pred_parq = exp / "velib_forecast_24h.parquet"
pred_csv  = exp / "velib_forecast_24h.csv"
if pred_parq.exists():
    preds = pd.read_parquet(pred_parq)
elif pred_csv.exists():
    preds = pd.read_csv(pred_csv)
else:
    preds = pd.DataFrame(columns=["stationcode","hour_utc","pred_occ"])
if not preds.empty:
    preds["hour_utc"] = pd.to_datetime(preds["hour_utc"], errors="coerce", utc=True)
    preds["hour_local"] = preds["hour_utc"].dt.tz_convert(LOCAL_TZ)

# --- KPIs globaux ---
n_stations = hourly["stationcode"].nunique()
dmin_loc, dmax_loc = hourly["hour_local"].min(), hourly["hour_local"].max()

# --- Station exemple avec > points récents ---
recent = hourly[hourly["hour_local"] > (dmax_loc - pd.Timedelta(hours=HIST_WINDOW_H))]
coverage = (recent.groupby("stationcode")["occ_ratio_hour"].count().sort_values(ascending=False))
sc = str(coverage.index[0]) if len(coverage) else str(hourly["stationcode"].astype(str).iloc[0])

hist = (hourly[hourly["stationcode"].astype(str) == sc]
        .sort_values("hour_local").tail(HIST_WINDOW_H))
fc = preds[preds.get("stationcode","").astype(str) == sc] if not preds.empty else pd.DataFrame()

# --- Graphe exemple ---
out_png = docs / "assets" / "sample_forecast.png"
plt.figure(figsize=(10, 4))
if not hist.empty:
    plt.plot(hist["hour_local"], hist["occ_ratio_hour"], label="historique")
if not fc.empty:
    plt.plot(fc["hour_local"], fc["pred_occ"], label="forecast 24h")
plt.title(f"Occupation ratio — station {sc}")
plt.xlabel(f"Heure locale ({LOCAL_TZ})"); plt.ylabel("ratio (0–1)")
if not hist.empty or not fc.empty: plt.legend()
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
if not hist.empty:
    plt.xlim(hist["hour_local"].min(), (fc["hour_local"].max() if not fc.empty else hist["hour_local"].max()))
plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

# --- Health ingestion (DuckDB) ---
con = duckdb.connect(str(ROOT / "warehouse.duckdb"))
snap_count = con.sql("SELECT COUNT(*) c FROM velib_snapshots").fetchone()[0]
last_ts    = con.sql("SELECT max(ts_utc) FROM velib_snapshots").fetchone()[0]
stations   = con.sql("SELECT COUNT(DISTINCT stationcode) FROM velib_snapshots").fetchone()[0]
health_md  = f"**Snapshots**: {snap_count}  •  **Stations**: {stations}  •  **Last (UTC)**: {last_ts}"

# --- Corrélation simple occ vs temp (si dispo) ---
if "temp_C" in hourly.columns and hourly["temp_C"].notna().any():
    sc_path = docs / "assets" / "occ_vs_temp.png"
    sample = hourly.dropna(subset=["occ_ratio_hour","temp_C"]).sample(min(5000, len(hourly)), random_state=42)
    plt.figure(figsize=(6,4))
    plt.scatter(sample["temp_C"], sample["occ_ratio_hour"], s=6, alpha=0.3)
    plt.xlabel("Temp (°C)"); plt.ylabel("Occ ratio")
    plt.tight_layout(); plt.savefig(sc_path, dpi=120); plt.close()
else:
    sc_path = None

# --- Top volatilité ---
def table_md(df: pd.DataFrame) -> str:
    try:
        import tabulate as _t  # noqa
        return df.to_markdown(index=False)
    except Exception:
        return "<div>" + df.to_html(index=False, border=0) + "</div>"

top = (hourly.dropna(subset=["occ_ratio_hour"])
       .groupby(["stationcode","name"], as_index=False)["occ_ratio_hour"]
       .std().rename(columns={"occ_ratio_hour":"std_occ"})
       .sort_values("std_occ", ascending=False).head(10))
top["std_occ"] = top["std_occ"].round(3)

# --- Copier exports (CSV) vers docs/exports ---
for src in [pred_csv, hour_path_csv]:
    if src.exists():
        dst = docs / "exports" / src.name
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())
if not hour_path_csv.exists():
    hourly.head(5000).to_csv(hour_path_csv, index=False)
    (docs / "exports" / "velib_hourly.csv").write_bytes(hour_path_csv.read_bytes())

# --- Construction de la page Results ---
md = []
md += ["# Results", ""]
md += [health_md, ""]
md += [f"**Historique couvert** : {dmin_loc} → {dmax_loc}  "]
md += [f"**Stations** : {n_stations}  "]
md += [f"*(Heure affichée : {LOCAL_TZ})*", ""]
md += ["## Example (historique + forecast 24h)"]
md += ["![sample](assets/sample_forecast.png)", ""]
if sc_path:
    md += ["## Corrélation simple", "Relation occ_ratio vs. température (échantillon)", 
           "![occ vs temp](assets/occ_vs_temp.png)", ""]
md += ["## Top 10 stations les plus volatiles", table_md(top), ""]
md += ["## Exports"]
if pred_csv.exists(): md += ["- [Prévision 24h (CSV)](exports/velib_forecast_24h.csv)"]
if (docs / "exports" / "velib_hourly.csv").exists(): md += ["- [Occupations horaires (échantillon CSV)](exports/velib_hourly.csv)"]

(docs / "results.md").write_text("\n".join(md), encoding="utf-8")
print("OK — docs/results.md mis à jour.")
