# Service — Velib Monitoring & Forecast

Ce dossier contient les **jobs de pipeline**, la **logique core** (features, forecast, orchestrateur), ainsi que la configuration **Docker** pour exécuter le projet en local ou en production (GCS).

## Arborescence

```
service/
├─ core/
│  ├─ cal_features.py          # Features calendaires (UTC & Paris)
│  ├─ features.py              # Colonnes de base & helpers training/serving
│  ├─ forecast.py              # Chargement modèles & prédiction
│  ├─ monitoring_pipeline.py   # Orchestrateur "monitoring" (multi-steps + lock)
│  └─ utils_gcs.py             # Runner/lock GCS pour un job unique (JOB=…)
│
├─ jobs/
│  ├─ ingest.py                # Ingestion snapshots (bronze) ← Velib API
│  ├─ compact_daily.py         # Compact snapshots/jour → daily/compact_YYYY-MM-DD.parquet
│  ├─ compact_monthly.py       # Compact dailies/mois → monthly/compact_YYYYMM.parquet
│  ├─ build_serving_forecast.py# Features 4h + inference (T+15,T+60) → serving/latest_forecast.json
│  ├─ export_training_base.py  # Export base d’entraînement depuis dailies
│  ├─ export_data_health.py    # Inventaire dailies → monitoring/health/daily_inventory*.json/csv
│  ├─ build_datasets.py        # Prépare exports/*.parquet (events, perf, …) pour le monitoring
│  ├─ build_monitoring.py      # Génère manifest.json (indexe monitoring/*)
│  ├─ build_monitoring_drift.py# Surveillances de drift (features) → monitoring/drift/*.json
│  ├─ build_monitoring_model_health.py  # Perf modèle (h15/h60) → monitoring/model/perf/*.json
│  ├─ build_network_dynamics.py# Dynamiques réseau (agrégats, heatmaps) → monitoring/network/dynamics.json
│  └─ build_network_stations.py# Santé réseau & stations → monitoring/network/stations*.json + health/summary*.json
│
├─ Dockerfile
├─ requirements.txt
└─ (autres fichiers .ignore, etc.)
```

## Conventions GCS

Racine (exemple) :
```
gs://<bucket>/velib/
  ├─ bronze/     # snapshots 5 min (date=YYYY-MM-DD/hour=HH/*.parquet)
  ├─ daily/      # compact_YYYY-MM-DD.parquet
  ├─ monthly/    # compact_YYYYMM.parquet
  ├─ exports/    # events.parquet, perf.parquet, ...
  ├─ serving/    # latest_forecast.json
  └─ monitoring/
       ├─ model/perf/      # daily_h15.json, daily_h60.json, segments_*.json
       ├─ network/         # dynamics.json, stations.json (+ versionnés)
       ├─ health/          # daily_inventory*.json, summary*.json
       ├─ drift/           # drift_*.json
       ├─ docs/            # dictionary.json, methodology.json, exports.json
       └─ manifest.json    # index des artefacts (catégories, dernières versions)
```

## Variables d’environnement (principales)

- **Entrées / sorties GCS**
  - `GCS_RAW_PREFIX` → `gs://.../velib/bronze`
  - `GCS_DAILY_PREFIX` → `gs://.../velib/daily`
  - `GCS_MONTHLY_PREFIX` → `gs://.../velib/monthly`
  - `GCS_EXPORTS_PREFIX` → `gs://.../velib/exports`
  - `GCS_MONITORING_PREFIX` → `gs://.../velib/monitoring`
  - `SERVING_FORECAST_PREFIX` → `gs://.../velib/serving`

- **Forecast / modèles**
  - `FORECAST_HORIZONS` (ex: `15,60`)
  - `GCS_MODEL_URI_T15` (ex: `gs://.../models/model_T15.joblib`)
  - `GCS_MODEL_URI_T60` (ex: `gs://.../models/model_T60.joblib`)

- **Fenêtres**
  - `NETWORK_WINDOW_DAYS` (par défaut 14)
  - `EVENTS_WINDOW_DAYS`, `PERF_WINDOW_DAYS` (selon jobs)
  - `DAY` (YYYY-MM-DD) pour `compact_daily` (si absent → aujourd’hui UTC)
  - `MONTH` (YYYY-MM) pour `compact_monthly`

- **Divers**
  - `CSV_ENABLE` (`1|0`) pour `export_data_health`
  - `LEGACY_COMPAT` (`1|0`) pour maintenir des noms historiques de fichiers
  - `GCS_LOCK` pour verrouillage (ex: `gs://.../locks/monitoring.lock`)

## Orchestrateur monitoring

L’orchestrateur **enchaîne** les steps et gère un **lock GCS** optionnel.

- Fichier : `service/core/monitoring_pipeline.py`
- Steps par défaut :
  1. `build_datasets`
  2. `data_health`
  3. `model_perf`
  4. `network_dynamics`
  5. `network_stations`
  6. `drift`
  7. `docs_dictionary`
  8. `docs_methodology`
  9. `docs_exports`
  10. `manifest`

### Lancer (tous les steps)
**PowerShell**
```powershell
$env:GCS_RAW_PREFIX          = "gs://<bucket>/velib/bronze"
$env:GCS_DAILY_PREFIX        = "gs://<bucket>/velib/daily"
$env:GCS_MONTHLY_PREFIX      = "gs://<bucket>/velib/monthly"
$env:GCS_EXPORTS_PREFIX      = "gs://<bucket>/velib/exports"
$env:GCS_MONITORING_PREFIX   = "gs://<bucket>/velib/monitoring"
$env:SERVING_FORECAST_PREFIX = "gs://<bucket>/velib/serving"

python -m service.core.monitoring_pipeline
```

**bash**
```bash
export GCS_RAW_PREFIX="gs://<bucket>/velib/bronze"
export GCS_DAILY_PREFIX="gs://<bucket>/velib/daily"
export GCS_MONTHLY_PREFIX="gs://<bucket>/velib/monthly"
export GCS_EXPORTS_PREFIX="gs://<bucket>/velib/exports"
export GCS_MONITORING_PREFIX="gs://<bucket>/velib/monitoring"
export SERVING_FORECAST_PREFIX="gs://<bucket>/velib/serving"

python -m service.core.monitoring_pipeline
```

### Options utiles
- **Limiter des steps** : `STEPS="network_dynamics,network_stations"`
- **Dry-run** (affiche l’ordre sans exécuter) : `DRY_RUN=1`
- **Continuer sur erreur** : `CONTINUE_ON_ERROR=1`
- **Lock** : `GCS_LOCK="gs://.../locks/monitoring.lock"`

## Exemples de jobs unitaires

- **Compacter la veille (UTC)**
```powershell
$env:GCS_RAW_PREFIX="gs://<bucket>/velib/bronze"
$env:GCS_DAILY_PREFIX="gs://<bucket>/velib/daily"
$env:DAY="2025-10-08"
python -m service.jobs.compact_daily
```

- **Compacter le mois**
```powershell
$env:GCS_DAILY_PREFIX="gs://<bucket>/velib/daily"
$env:GCS_MONTHLY_PREFIX="gs://<bucket>/velib/monthly"
$env:MONTH="2025-10"
python -m service.jobs.compact_monthly
```

- **Forecast serving (T+15 / T+60)**
```powershell
$env:GCS_RAW_PREFIX="gs://<bucket>/velib/bronze"
$env:SERVING_FORECAST_PREFIX="gs://<bucket>/velib/serving"
$env:GCS_MODEL_URI_T15="gs://<bucket>/velib/models/model_T15.joblib"
$env:GCS_MODEL_URI_T60="gs://<bucket>/velib/models/model_T60.joblib"
$env:FORECAST_HORIZONS="15,60"
python -m service.jobs.build_serving_forecast
```

- **Inventaire dailies (JSON + CSV)**
```powershell
$env:GCS_DAILY_PREFIX="gs://<bucket>/velib/daily"
$env:GCS_MONITORING_PREFIX="gs://<bucket>/velib/monitoring"
$env:CSV_ENABLE="1"
python -m service.jobs.export_data_health
```

## Schémas & contrats (résumé)

- **Daily parquet (strict)**
```
ts_utc, tbin_utc, station_id, bikes, capacity, mechanical, ebike, status,
lat, lon, name, temp_C, precip_mm, wind_mps
```
- **serving/latest_forecast.json**
```
{
  generated_at, horizons: [15,60],
  data: { "15": [ {...preds} ], "60": [ {...preds} ] }
}
```
- **monitoring/**
  - `model/perf/*` : courbes daily & segments (par horizon)
  - `network/dynamics.json` : agrégats réseau (heatmap-ready)
  - `network/stations*.json` : stats station + profils horaires
  - `health/daily_inventory*.json` : inventaire dailies (par jour)
  - `health/summary*.json` : complétude réseau globale (fenêtre stations)
  - `docs/*` : dictionnaire, méthodologie, exports
  - `manifest.json` : indexe les fichiers par catégorie (model_perf, network, drift, health, docs)

## Déploiement & planning

- **Cloud Scheduler** → **Cloud Build** (ou Cloud Run Job)
  - 05:00 UTC : `compact_daily` (jour J-1)
  - 05:20 UTC : `service.core.monitoring_pipeline` (tous les steps)
- **IAM/CORS**
  - Soit : UI lit via **API proxy** (FastAPI) qui stream les JSON GCS.
  - Soit : lecture publique du bucket *monitoring* + CORS.

## Dev local rapide

- Python 3.10+ recommandé.
- `pip install -r service/requirements.txt`
- Auth GCP : `gcloud auth application-default login` (ou variables SA).

## Troubleshooting

- **404 modèle** : vérifier `GCS_MODEL_URI_T15/T60` exacts + droits IAM.
- **CRC32C lent** (warning) : facultatif, installer le wheel natif `google-crc32c`.
- **JSON volumineux** (stations) : régler `STATIONS_MAX` ou réduire `WINDOW_DAYS`.
- **Manifest incomplet** : relancer `build_monitoring.py` (indexe `monitoring/*`).
