# 🚲 Vélib’ Paris — Forecast & Monitoring
[![CI — ingestion](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/ingest.yml/badge.svg)](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/ingest.yml)
[![CI — training](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/train.yml/badge.svg)](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/train.yml)
[![Docs](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/site.yml/badge.svg?branch=main)](https://adrien-1997.github.io/bike-forecast-paris-velib/)
[![App Streamlit](https://img.shields.io/badge/app-streamlit-green)](https://adrien-1997-bike-forecast-paris-velib-appstreamlit-app-vq1xma.streamlit.app/)
![Version](https://img.shields.io/badge/version-v2.2.0-blue.svg)
![License](https://img.shields.io/badge/License-MIT-black)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB)

![Carte réseau](docs/map.png)

Short-term forecasting (+60 min) and professional monitoring of the Vélib’ bike network in Paris.
Public GBFS snapshots -> normalized 15-min aggregates -> features & model training -> monitoring with auto-retrain.

> Quick links:
> Docs -> https://adrien-1997.github.io/bike-forecast-paris-velib/
> App  -> https://adrien-1997-bike-forecast-paris-velib-appstreamlit-app-vq1xma.streamlit.app/

---

## Table of Contents

- [Key Features](#-key-features)
- [Pipelines — Data → ML → Docs & App](#-pipelines--data--ml--docs--app)
- [CI/CD (GitHub Actions)](#-cicd-github-actions)
- [Run locally](#-run-locally)
- [Streamlit App](#-app-streamlit)
- [Data Contracts (canonical schemas)](#-data-contracts-canonical-schemas)
- [Project Layout](#-project-layout)
- [Release Notes — v2.2.0](#-release-notes--v210)
- [Author & License](#-author--license)

---

## 🔎 Key Features

- Ingestion every 15 min -> DuckDB snapshots.
- 15-min aggregation -> canonical exports (`docs/exports/velib.parquet`, CSV) + weather.
- Real-time monitoring: map + availability/saturation.
- KPIs & history: occupancy, availability, daily/hourly profiles.
- Forecasts: LightGBM predicts bikes at T+60 (1h).
- CI/CD:
  1) velib-ingest (15 min) -> export parquet
  2) velib-train (daily or triggered) -> models + baseline.json
  3) monitoring-site (4x/day) -> build pages, drift/perf checks, deploy gh-pages
- ML artifacts: `models/lgb_nbvelos_T+60min.joblib`.
- Streamlit app: interactive map + forecasts; reads models + parquet.
- Usage analytics: station clustering, heatmaps, variability.
- Forecast performance: MAE/RMSE, lift, obs vs pred, residuals, calibration.
- Monitoring: data health, PSI drift, MAE trend.

> All figures/tables come from `docs/exports/velib.parquet` via `tools/*` scripts.

---

## 🧭 Pipelines — Data → ML → Docs & App

```mermaid
flowchart TD
  A[GBFS ingest] --> B[DuckDB]
  B --> C[Aggregate + weather]
  C --> D[Export velib.parquet]
  D --> E[Normalize datasets]
  E --> EV[events.parquet]
  E --> PF[perf.parquet]

  E --> G[Build features from velib]
  G --> H[Train LGBM h60]
  H --> I[Model bundle]
  I --> M[Apply model -> y_pred]
  EV --> M
  PF --> M

  subgraph Pages
    R1[network overview]
    R2[network stations]
    R3[network dynamics]
    M1[model performance]
    M2[model pipeline]
    M3[model explainability]
    Q1[monitoring data_health]
    Q2[monitoring drift]
    Q3[monitoring model_health]
    D1[data exports]
    D2[data dictionary]
    D3[data methodology]
  end

  EV --> R1 --> X[assets]
  EV --> R2 --> X
  EV --> R3 --> X
  PF --> M1 --> X
  D --> M2 --> X
  PF --> M3 --> X
  EV --> Q1 --> X
  EV --> Q2 --> X
  PF --> Q3 --> X
  D --> D1 --> X
  D --> D2 --> X
  D --> D3 --> X
  X --> K[gh-pages]

  CI1[ingest 15m] --> B
  CI3[site 4/day] --> K
  M1 --> T[check retrain]
  Q3 --> T
  Q2 --> T
  T -->|yes| CI2[train daily]
  CI2 --> H
  CI2 --> K
```

### Core src/* chain

1) Ingestion — `src/ingest.py` (every 15 min)  
   Pull GBFS, normalize, upsert station meta, append one row/station in snapshots.

2) Aggregation — `src/aggregate.py` (15-min + weather)  
   Right-closed 15-min grid; `occ = bikes/capacity` (safe clamp); join weather; write `docs/exports/velib.parquet`.

3) Feature builder — `src/features.py`  
   Hour/DOW cyclics, lags (t-15/30/45/60), rolling mean/std, static station, weather. (horizon=60)

4) Forecast — `src/forecast.py`  
   LightGBM with early-stopping; saves `models/*.joblib` and `docs/exports/baseline.json`.

---

### 🤖 CI/CD (GitHub Actions)

This pipeline keeps data and docs fresh while guarding model quality:

- **Ingestion (`velib-ingest.yml`, every 15 min):** fetch GBFS → update DuckDB snapshots → aggregate to 15‑min bins → export `docs/exports/velib.parquet`.
- **Docs & monitoring (`site.yml`, 4×/day @ 00/06/12/18 UTC):** build MkDocs, compute metrics, check **drift/perf** (PSI / MAE_24h). If thresholds are exceeded, it triggers retraining.
- **Training (`train.yml`, daily fallback or immediate):** train LightGBM T+60, write `models/*.joblib` and `docs/exports/baseline.json`, then rebuild docs.

```mermaid
flowchart LR
  subgraph CI[CI/CD]
    I[velib-ingest.yml\nEvery 15 min] -->|exports| D[docs/exports/velib.parquet]
    S[site.yml\n4x/day] -->|builds| M[Docs -> gh-pages]
    S -->|checks| T[check_retrain\nPSI or MAE_24h]
    T -->|yes| R[train.yml]
    R -->|models + baseline| J[models/*.joblib + baseline.json]
    R -->|rebuild| M
  end
```

#### Jobs at a glance

| Job | What it does | Deps |
|---|---|---|
| **build-monitor** (`site.yml`) | Install `requirements-doc.txt` (headless `MPLBACKEND=Agg`), run `tools/generate_monitoring.py`, decide retrain via `tools/check_retrain.py` (**PSI ≥ 0.20** or **MAE_24h ≥ 1.20× baseline**), build & deploy MkDocs to `gh-pages`. | pandas, numpy, pyarrow, matplotlib, scikit-learn, folium, mkdocs |
| **retrain-and-rebuild** (`train.yml`, conditional) | Install `requirements-train.txt` (scikit-learn + lightgbm), run `src.forecast.train(horizon_minutes=60, lookback_days=30)`, save `models/*.joblib` and `docs/exports/baseline.json`, **force-add** ignored paths, re‑generate monitoring, rebuild docs, deploy. | scikit-learn, lightgbm |

**Operational notes**  
- `tools/check_retrain.py` supports legacy `docs/exports/metrics.json` **and** the new CSV/Parquet tables.  
- The commit step **force‑adds** `baseline.json` (and models if you choose) even if `.gitignore` ignores `docs/exports` / `models`.  
- Cache `~/.cache/pip` to speed up both jobs.

---

## 🚀 Run locally

```bash
# 1) Analytics / docs deps
pip install -r requirements-doc.txt

# 2) Build all figures/pages from canonical parquet
python tools/generate_monitoring.py

# 3) Serve docs locally
mkdocs serve
```

**Use a trained model?** Merge predictions into `docs/exports/perf.parquet` (`ts, station_id, y_pred`) then re‑run monitoring.

---

## 🧪 App (Streamlit)

```bash
pip install streamlit
streamlit run app/streamlit_app.py
```
- Reads: `models/*.joblib` + `docs/exports/velib.parquet`.  
- Live demo: see README badges.

---

## 📐 Data Contracts (canonical schemas)

**A) `warehouse.duckdb::snapshots` (append‑only)**  
`ts, station_id, bikes, capacity, docks_free, is_renting, is_returning, is_installed, name, lat, lon[, ebikes]`  
Index hint: clustered by `(ts, station_id)`.

**B) `docs/exports/velib.parquet` (15‑min canonical)**  
`ts(UTC), station_id, bikes, capacity, occ, lat, lon, name, temp_c, rain_mm, wind_kph, dow, hour`

**C) Training exports (`tools/datasets.py`)**  
- `events.parquet` → `ts, station_id, bikes, capacity, occ, lat, lon, name`  
- `perf.parquet` → `ts, station_id, y_true, y_pred[, y_pred_baseline]` (rebuild `y_true` via **T+60** shift if missing)

---

## 📁 Project Layout

```
bike-forecast-paris-velib/
├─ app/ # Streamlit app
├─ src/ # ingestion, aggregation, features, forecast
│ ├─ ingest.py
│ ├─ aggregate.py
│ ├─ features.py
│ └─ forecast.py
├─ tools/
│ ├─ datasets.py
│ ├─ apply_model.py
│ ├─ build_*.py
│ └─ check_retrain.py
├─ docs/
│ ├─ assets/{figs,tables,maps}
│ ├─ exports/
│ └─ {network,model,monitoring,data}/
├─ models/
├─ .github/workflows/{ingest.yml,train.yml,site.yml}
└─ mkdocs.yml
```
---


## Release Notes — v2.2.0

**MkDocs revamp (site overhaul)**
- Clear 4×3 structure: **Network**, **Model**, **Monitoring**, **Data**.
- *One script = one page*: dedicated `tools/build_*` generators with KPI injection into Markdown.
- Maps & charts: Folium map **with legend**, “Today vs Median” curve, daily KPIs vs **D-7/D-14/D-21**.
- Robust relative paths (`use_directory_urls: true`) and auto-created `docs/assets/{figs,tables,maps}`.
- Windows-friendly: ASCII logs, UTF-8 (no BOM).

**Model & data**
- `apply_model.py`: correct **T+h → T** alignment, stable station mapping.
- `datasets.py`: cleaner output; safer types/encoding.

---

## 👤 Author & License

**Adrien Morel** — Data Scientist (applied math & ML)  
Docs: https://adrien-1997.github.io/bike-forecast-paris-velib/ • App: https://adrien-1997-bike-forecast-paris-velib-appstreamlit-app-vq1xma.streamlit.app/

**License:** MIT
