# 🚲 Vélib’ Paris — Forecast & Monitoring (v3)

[![CI — ingestion](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/ingest.yml/badge.svg)](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/ingest.yml)
[![CI — training](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/train.yml/badge.svg)](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/train.yml)
[![Docs](https://github.com/Adrien-1997/bike-forecast-paris-velib/actions/workflows/site.yml/badge.svg?branch=main)](https://adrien-1997.github.io/bike-forecast-paris-velib/)
[![App Streamlit](https://img.shields.io/badge/app-streamlit-green)](https://velib-forecast.streamlit.app/)
![Version](https://img.shields.io/badge/version-v3.0.0-blue.svg)
![License](https://img.shields.io/badge/License-MIT-black)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB)

![Carte réseau](docs/map.png)

Short‑term forecasting (**T+15 min**) and professional monitoring of the Vélib’ bike network in Paris.  
Public GBFS snapshots → normalized **5‑min** aggregates → features & model training → monitoring with auto‑retrain.

> Quick links:  
> Docs → https://adrien-1997.github.io/bike-forecast-paris-velib/  
> App  → https://velib-forecast.streamlit.app/

---

## 🔎 Key Features

- **15‑min forecasting (T+15)**  
  LightGBM predicts bike availability 15 minutes ahead with strict feature parity (lags/rollings + calendar/weather).

- **Robust data pipeline (5‑min cadence)**  
  Ingestion every **5 min** → DuckDB snapshots → **5‑min** canonical export with weather join (right‑closed grid).

- **Clean targets & alignment**  
  `y_true = bikes@T+15` (per station via shift), baseline = persistence@T, predictions aligned on **T** and evaluated at **T+15**.

- **Hugging Face integration**  
  - **Datasets**: parquet exports on HF Datasets.  
  - **Models**: `.joblib` bundles (model + feature contract + horizon=15) on HF Model Hub.

- **Monitoring & quality**  
  - **Data health**: freshness, completeness, schema checks.  
  - **Drift**: PSI across key features.  
  - **Model health**: MAE/RMSE, lift vs baseline, coverage, calibration.  
  - **Parquet freshness badge** in the app UI.

- **Documentation pages (MkDocs)**  
  - **Data**: Exports, Dictionary, Methodology.  
  - **Monitoring**: Data health, Drift, Model health.  
  - **Model**: Pipeline, Performance, Explainability.  
  - **Network**: Overview, Stations, Dynamics (profiles computed on **15‑min** quarters; live KPIs on **5‑min** cadence).

- **Streamlit app**  
  Interactive map, real‑time availability/saturation, **T+15** forecasts.

---

## 🧭 Pipelines — Data → ML → Docs & App (+ Hugging Face)

```mermaid
flowchart TD

  %% SOURCES & AGG
  subgraph Sources
    GBFS[GBFS snapshots (5m)];
    WX[Weather hourly];
  end

  GBFS --> AGG[Aggregate (5m) + weather join];
  WX   --> AGG;
  AGG  --> VELIB[docs/exports/velib.parquet];

  %% NORMALISATION
  VELIB --> NORM[Normalize datasets];
  NORM  --> EV[events.parquet (5m)];
  NORM  --> PF[perf.parquet (T & T+15 align)];

  %% MODEL PIPELINE
  CAL[Calendar features];
  VELIB --> FEAT[Build features];
  CAL   --> FEAT;
  FEAT  --> TRAIN[Train LGBM T+15];
  TRAIN --> MODEL[Model bundle (.joblib, horizon=15)];

  %% PREDICTION
  MODEL --> APPLY[Apply model → y_pred];
  EV    --> APPLY;
  PF    --> APPLY;

  %% PAGES
  subgraph Pages
    P1[Data pages];
    P2[Monitoring pages];
    P3[Model pages];
    P4[Network pages];
  end
  EV    --> P4;
  EV    --> P1;
  PF    --> P2;
  PF    --> P3;

  %% HUGGING FACE HUB
  subgraph HF[Hugging Face]
    HFDS[Datasets];
    HFM[Models];
  end
  VELIB --> HFDS;
  MODEL --> HFM;

  %% CI/CD
  subgraph CI[CI/CD]
    CI1[ingest every 5m];
    CI3[site build 4x/day];
    CI2[train daily or on-trigger];
    CHECK[check_retrain (MAE/PSI)];
  end

  CI1 --> GBFS;
  CI1 --> WX;
  CI1 --> HFDS;

  CI3 --> NORM;
  CI3 --> APPLY;
  CI3 --> P1; CI3 --> P2; CI3 --> P3; CI3 --> P4;

  P2 -.-> CHECK;
  P3 -.-> CHECK;
  CHECK --> CI2;
  CI2 --> TRAIN;
  CI2 --> HFM;
```

### Core `src/*` chain (v3)

1) **Ingestion — `src/ingest.py` (every 5 min)**  
   Pull GBFS, normalize, upsert station meta, append one row/station in snapshots.

2) **Aggregation — `src/aggregate.py` (5‑min + weather)**  
   Right‑closed **5‑min** grid; `occ = bikes/capacity` (safe clamp); weather join; write `docs/exports/velib.parquet`.

3) **Feature builder — `src/features.py` (horizon=15)**  
   Calendar cyclics, lags (t‑5/10/15), rollings, static station, weather.

4) **Forecast — `src/forecast.py`**  
   LightGBM with early‑stopping; saves `models/*.joblib` and `docs/exports/baseline.json` (baseline=Persistence@T).

---

## 🤖 CI/CD (GitHub Actions)

- **Ingestion (`ingest.yml`, every 5 min):** fetch GBFS → update DuckDB snapshots → aggregate to **5‑min** bins → export `docs/exports/velib.parquet`.
- **Docs & monitoring (`site.yml`, 4×/day @ 00/06/12/18 UTC):** build MkDocs, compute metrics, check **drift/perf** (PSI / MAE). If thresholds are exceeded, it triggers retraining.
- **Training (`train.yml`, daily or on‑demand):** train LightGBM **T+15**, write `models/*.joblib` and `docs/exports/baseline.json`, then rebuild docs.

```mermaid
flowchart LR
  subgraph CI[CI/CD]
    I[ingest.yml — 5m] -->|exports| D[docs/exports/velib.parquet]
    S[site.yml — 4x/day] -->|builds| M[Docs → gh-pages]
    S -->|checks| T[check_retrain (PSI or MAE lift)]
    T -->|yes| R[train.yml]
    R -->|models + baseline| J[models/*.joblib + baseline.json]
    R -->|rebuild| M
  end
```

---

## 🚀 Run locally

```bash
# 1) Analytics / docs deps
pip install -r requirements-doc.txt

# 2) Build all figures/pages from canonical parquet
python tools/build_monitoring.py

# 3) Serve docs locally
mkdocs serve
```

**Use a trained model?**  
Merge predictions into `docs/exports/perf.parquet` (`ts, station_id, y_pred`) then re‑run monitoring.

---

## 🧪 App (Streamlit)

```bash
pip install streamlit
streamlit run app/streamlit_app.py
```
- Reads: `models/*.joblib` + `docs/exports/velib.parquet`.  
- Live demo: see README badges.

---

## 📐 Data Contracts (canonical schemas, v3)

**A) `warehouse.duckdb::snapshots` (append‑only, 5m)**  
`ts, station_id, bikes, capacity, docks_free, is_renting, is_returning, is_installed, name, lat, lon[, ebikes]`

**B) `docs/exports/velib.parquet` (5‑min canonical)**  
`ts(UTC), station_id, bikes, capacity, occ, lat, lon, name, temp_c, rain_mm, wind_kph, dow, hour`

**C) Training exports (`tools/datasets.py`)**  
- `events.parquet` → **5‑min** `ts, station_id, bikes, capacity, occ, lat, lon, name`  
- `perf.parquet`   → `ts, station_id, y_true, y_pred[, y_pred_baseline], horizon_min=15, ts_decision=T, ts_target=T+15`

> Note: Network analytics pages still display **24h profiles on 15‑min quarters (96 bins)** for stability and readability, while live KPIs operate at **5‑min cadence**.

---

## 📁 Project Layout

```
bike-forecast-paris-velib/
├─ app/                       # Streamlit app
├─ src/                       # ingestion, aggregation, features, forecast
├─ tools/                     # build scripts (pages, datasets, checks)
├─ docs/                      # assets, exports, site
├─ models/                    # trained bundles
└─ .github/workflows/         # ingest.yml, train.yml, site.yml
```

---

## Release Notes — v3.0.0

- **Cadence 5 min** end‑to‑end (ingestion → canonical parquet).  
- **Horizon T+15**: training & evaluation; `ts_decision` / `ts_target` added to `perf.parquet`.  
- **Docs refresh**: wording and figures updated to reflect 5‑min cadence; profiles remain quarter‑hour for readability.  
- **Tooling**: `check_retrain.py` thresholds documented; robust plots & Folium legends.
- **Safety**: deterministic KMeans (random_state=42), consistent relative paths for MkDocs.

---

## 👤 Author & License

**Adrien Morel** — Data Scientist (applied math & ML)  
Docs: https://adrien-1997.github.io/bike-forecast-paris-velib/ • App: https://velib-forecast.streamlit.app/

**License:** MIT
