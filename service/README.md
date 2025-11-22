# `service/`

Backend Python package for **Vélib’ Forecast**.

It contains:

- the **core logic** for feature engineering, model training and inference,
- the **batch jobs** used in production (Cloud Run / GCS),
- all the bricks reused by the API and the UI monitoring layer.

The UI (Next.js / `ui/`) and infra (Netlify, Cloud Run, Cloud Build…) live
outside this package and simply **consume the artefacts** produced here
(Parquet datasets, models, JSON monitoring, batch forecasts).

---

## 1. Package structure

High-level layout:

```text
service/
├── core/      # reusable logic (features, training, monitoring orchestration, GCS launcher)
└── jobs/      # batch jobs (ingest, compact, exports, training, monitoring, serving)
```

### 1.1. `service/core`

Core modules, fully decoupled from scheduling / infra:

- `forecast.py`  
  XGBoost training + inference utilities. Integrates:
  - temporal features (lags, rolling stats, weather),
  - spatial features (distance to centre, spatial grid, KMeans cluster, TE),
  - KNN dynamics (neighbours’ bikes at t-L),
  - model pack format: `{"model", "feat_cols", "horizon_bins"}`,
  - model versioning + JSON metadata + GCS publish,
  - offline prediction on latest snapshots.

- `time_features.py`  
  Purely temporal feature stack:
  - strict schema from 5-min snapshots (`ts_utc`, `tbin_utc`, `bikes`, `capacity`, weather…),
  - per-bin dedupe,
  - target `y_nb` at +`horizon_bins`,
  - lags / rollings / weather lags,
  - occupancy ratio,
  - calendar + sinusoidal features,
  - returns `(full_df, X, y, feat_cols)` for training.

- `spatial_features.py`  
  Temporal stack + spatial extras:
  - station-level static features (`dist_center_km`, `grid_x`, `grid_y`,
    `kmeans_cluster`, `te_cap_mean`),
  - KNN neighbour maps and dynamic neighbour lags,
  - training frame `build_training_frame_spatial(...)`.

- `cal_features.py`  
  Calendar features from `tbin_utc`:
  - hour / minute / dow / month / weekend,
  - sin/cos encodings over 24h and 7 days,
  - Paris-derived proxies if needed.

- `monitoring_pipeline.py`  
  Monitoring orchestrator:
  - defines a set of steps (`health`, `drift`, `perf`, `explain`,
    `network`, `intro`, …),
  - enables them via `STEPS` env (CSV or pattern),
  - manages shared time windows / horizons / GCS roots,
  - calls the corresponding jobs in `service.jobs`.

- `utils_gcs.py`  
  Generic launcher for Cloud Run / GCS:
  - dispatches to jobs via `JOB` env,
  - computes `DAY` automatically for compact_* if needed,
  - optional GCS lock to avoid concurrent runs,
  - `DRY_RUN` mode for debugging.

A more detailed README for each submodule lives in **`service_core_README.txt`**.

---

### 1.2. `service/jobs`

Batch jobs, each one focused on a single responsibility, typically run as
Cloud Run jobs or one-off CLI scripts.

Key jobs:

#### Ingestion & compaction

- `ingest.py`  
  5-minute ingestion from:
  - Vélib’ GBFS (status + info → joined to get station name),
  - Open-Meteo (optional, can be disabled via env).  
  Writes raw snapshots to local and/or GCS (`bronze`).

- `compact_daily.py`  
  Daily compaction:
  - all 5-min snapshots for a UTC day → one `compact_YYYY-MM-DD.parquet`
    in `daily/` (with optional deletion of raw snapshots).

- `compact_monthly.py`  
  Monthly compaction:
  - all dailies for a month → one `compact_YYYY-MM.parquet`
    in `monthly/`.

#### Exports for training / evaluation

- `build_datasets.py`  
  From a `compact_YYYY-MM-DD.parquet`:
  - builds an **events view** (per-bin events),
  - builds a **perf view** with model predictions (`y_pred_int`),
  - writes dated & rolling versions under `exports/`.

- `export_training_base.py`  
  Aggregates multiple days/months into a single **training base** parquet
  (optional pre-step to accelerate training).

#### Training

- `train_model.py`  
  Cloud Run entrypoint for XGBoost training:
  - pulls dailies / base from GCS,
  - calls `service.core.forecast.train_model`,
  - saves model artefacts (local + GCS, with semver bump + `latest.*`).

#### Serving

- `build_serving_forecast.py`  
  Builds a sliding 4h window of features from recent raw data and publishes
  **batch forecasts** per horizon as JSON under
  `serving/forecast/h{H}/latest.json`.

#### Monitoring

- `build_data_health.py`  
  Data health KPIs: freshness, duplication, flat sequences, coverage…

- `build_data_drift.py`  
  Drift metrics (PSI, etc.) comparing a “current” window to a “reference”
  window.

- `build_model_performance.py`  
  Performance KPIs by day / hour / dow / horizon (MAE, RMSE, lift…).

- `build_model_explainability.py`  
  Explainability artefacts:
  - feature importance,
  - residual distributions,
  - top episodes,
  - station / time profiles.

- `build_network_overview.py`  
  Network-level overview:
  - usage volumes,
  - mechanical vs e-bike distribution,
  - trends on recent weeks.

- `build_network_stations.py`  
  Station-level profiles:
  - usage intensity,
  - tension,
  - metadata and quantiles consumed by the UI.

- `build_network_dynamics.py`  
  Network dynamics:
  - tension / penury / saturation episodes,
  - hourly “heatmap” profiles,
  - regularity of the current day vs history.

- `build_monitoring_intro.py`  
  Aggregates the key KPIs from all previous artefacts into a single
  `intro/latest/intro.json` consumed by the Monitoring landing page.

A more detailed README for these jobs lives in **`service_jobs_README.txt`**.

---

## 2. Data model & GCS layout

The package assumes a **parquet-first then JSON** pipeline with the
following GCS layout (under a common root, e.g. `gs://…/velib`):

```text
bronze/           # raw 5-min snapshots (GBFS + météo)
daily/            # compact_YYYY-MM-DD.parquet
monthly/          # compact_YYYY-MM.parquet
exports/          # events_YYYY-MM-DD.parquet, perf_YYYY-MM-DD.parquet
models/           # h15/h60/... model.joblib + version.json + latest.*
serving/forecast/ # batch forecasts JSON (h15/latest.json, h60/latest.json, ...)
monitoring/       # all monitoring JSON artefacts (data, model, network, intro)
```

All JSON folders used by the UI are **LATEST-only**:
the frontend never manipulates dates directly, it only reads from
`.../latest/*`.

---

## 3. Conventions

- **Time binning** :  
  - 5-minute bins (`tbin_utc`), always UTC naive internally.
  - 1 bin = 5 minutes → horizon in minutes = `horizon_bins * 5`.

- **Targets & forecasts** :  
  - target: `y_nb` = bikes at `tbin_utc + horizon_bins * 5 min`,
  - model outputs are clipped to `[0, capacity]` then rounded to integers.

- **TZ handling** :  
  - storage & model alignment in UTC,
  - “business logic” and monitoring aggregates expressed in `Europe/Paris`,
    via the `MON_TZ` / `TZ_APP` envs.

- **Idempotent jobs** :  
  All jobs are idempotent on a given window/day:
  re-running them for the same period simply rewrites the same artefacts.

---

## 4. How everything connects

- `service/jobs` is the **batch layer**:
  - launched via Cloud Run / core.utils_gcs,
  - reads/writes Parquet and JSON to GCS.

- `service/core` is the **ML / monitoring core**:
  - provides training, inference and monitoring building blocks,
  - used both by jobs and potentially by the API.

- The **API** (FastAPI or similar, outside this package) loads:
  - model packs (`model.joblib`) via `forecast`,
  - monitoring JSON artefacts under `monitoring/`,
  - serving forecasts under `serving/forecast/`.

- The **UI** (Next.js, outside this package) hits the API and only sees
  the **latest JSON** artefacts generated here.

This keeps the concerns clean:
- `service/` cares about data & models,
- infra handles scheduling & credentials,
- UI/API consume the outputs.