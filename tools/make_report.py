# tools/make_report.py  — robuste même sans prédictions
import os, pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
exp  = ROOT / "exports"
docs = ROOT / "docs"
(docs / "assets").mkdir(parents=True, exist_ok=True)
(docs / "exports").mkdir(parents=True, exist_ok=True)

# Charger hourly
hour_path_parq = exp / "velib_hourly.parquet"
hour_path_csv  = exp / "velib_hourly.csv"
if hour_path_parq.exists():
    hourly = pd.read_parquet(hour_path_parq)
elif hour_path_csv.exists():
    hourly = pd.read_csv(hour_path_csv)
else:
    raise SystemExit("Aucun fichier hourly trouvé dans exports/")

# Charger forecast (peut être vide/inexistant)
pred_parq = exp / "velib_forecast_24h.parquet"
pred_csv  = exp / "velib_forecast_24h.csv"
if pred_parq.exists():
    preds = pd.read_parquet(pred_parq)
elif pred_csv.exists():
    preds = pd.read_csv(pred_csv)
else:
    preds = pd.DataFrame(columns=["stationcode","hour_utc","pred_occ"])

# Nettoyage types
hourly["hour_utc"] = pd.to_datetime(hourly["hour_utc"], errors="coerce")
if "hour_utc" in preds.columns:
    preds["hour_utc"] = pd.to_datetime(preds["hour_utc"], errors="coerce")

# KPIs
n_stations = hourly["stationcode"].nunique()
dmin, dmax = hourly["hour_utc"].min(), hourly["hour_utc"].max()

# Top volatilité (écart-type) — peut être vide au début
top = (hourly.dropna(subset=["occ_ratio_hour"])
              .groupby(["stationcode","name"], as_index=False)["occ_ratio_hour"]
              .std()
              .rename(columns={"occ_ratio_hour":"std_occ"})
              .sort_values("std_occ", ascending=False)
              .head(10))
if not len(top):
    # fallback: stations les plus observées
    top = (hourly.groupby(["stationcode","name"], as_index=False)["hour_utc"]
                 .count().rename(columns={"hour_utc":"n_obs"})
                 .sort_values("n_obs", ascending=False).head(10))
    top["std_occ"] = np.nan

# Choisir une station pour le graphe
out_png = None
if n_stations > 0:
    sc = str((top.iloc[0]["stationcode"]))
    hist = hourly[hourly["stationcode"].astype(str) == sc].sort_values("hour_utc").tail(24*7)
    if "stationcode" in preds.columns:
        fc = preds[preds["stationcode"].astype(str) == sc].copy()
    else:
        fc = pd.DataFrame(columns=["hour_utc","pred_occ"])

    # Graphe
    plt.figure(figsize=(10,4))
    if not hist.empty:
        plt.plot(hist["hour_utc"], hist["occ_ratio_hour"], label="historique")
    if not fc.empty:
        plt.plot(fc["hour_utc"], fc["pred_occ"], label="forecast 24h")
    plt.title(f"Occupation ratio — station {sc}")
    plt.xlabel("UTC hour"); plt.ylabel("ratio (0–1)")
    if not fc.empty or not hist.empty:
        plt.legend()
    out_png = docs / "assets" / "sample_forecast.png"
    plt.tight_layout(); plt.savefig(out_png); plt.close()

# Copier exports vers docs/exports (si présents)
for src in [pred_csv, hour_path_csv]:
    if src.exists():
        dst = docs / "exports" / src.name
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())

# Si hourly CSV absent, créer un échantillon léger
if not hour_path_csv.exists():
    hourly.head(5000).to_csv(hour_path_csv, index=False)
    (docs / "exports" / "velib_hourly.csv").write_bytes(hour_path_csv.read_bytes())

# Écrire la page Results
md = []
md += ["# Results", ""]
md += [f"**Historique couvert** : {dmin} → {dmax}  \n**Stations** : {n_stations}", ""]

if out_png and out_png.exists():
    md += ["## Example (historique + forecast 24h)", "![sample](assets/sample_forecast.png)", ""]

md += ["## Top 10 stations les plus volatiles"]
md += [top.to_markdown(index=False)]

md += ["", "## Exports"]
if pred_csv.exists():
    md += ["- [Prévision 24h (CSV)](exports/velib_forecast_24h.csv)"]
if (docs / "exports" / "velib_hourly.csv").exists():
    md += ["- [Occupations horaires (échantillon CSV)](exports/velib_hourly.csv)"]

(docs / "results.md").write_text("\n".join(md), encoding="utf-8")
print("OK — docs/results.md généré.")
