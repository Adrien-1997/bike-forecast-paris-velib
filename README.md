# Vélib' Forecast Paris — Plateforme de prévision et de monitoring

Ce dépôt regroupe **l’ensemble du projet Vélib' Forecast Paris / velo-paris.fr** :  
pipeline de données, entraînement des modèles, API, interface web et monitoring de la qualité / performance.

L’objectif : fournir une **plateforme de prévision de disponibilité Vélib’** (stations, vélos, places disponibles)
avec un **pipeline reproductible**, un **suivi de la qualité des données** et un **monitoring temps quasi-réel** du réseau et des modèles.

---

## 1. Vue d’ensemble

### 1.1. Fonctionnalités principales

- Ingestion régulière des snapshots Vélib' (format GBFS).
- Compactage et préparation des jeux de données quotidiens / mensuels.
- Construction de jeux de données d’entraînement et d’évaluation.
- Entraînement et déploiement de modèles de prévision (XGBoost, horizons 15 / 60 min, etc.).
- Exposition d’une **API FastAPI** pour :
  - la carte temps réel,
  - les prévisions batch par station / horizon,
  - les artefacts de monitoring (réseau, data, modèles).
- Interface web **Next.js / React** pour :
  - l’application carte utilisateur (`/app`),
  - le monitoring des données / modèles / réseau (`/monitoring`).
- Jobs de monitoring pour :
  - Santé des données (complétude, fraîcheur, valeurs aberrantes),
  - Dérive des données (PSI, statistiques par feature / zone),
  - Performance des modèles (MAE, lift vs baseline, résidus),
  - Dynamiques réseau (pénurie / saturation, cartes, heatmaps 7×24),
  - Explicabilité modèle (importance des features, résidus, calibration).

### 1.2. Architecture globale (haut niveau)

```text
            +--------------------+         +------------------------+
            |  API Vélib' GBFS   |         |  API météo (Open-Meteo |
            +---------+----------+         +-----------+------------+
                      |                                |
                      v                                v
              [ Jobs ingestion ]                [ Helper météo ]
                      |                                |
                      +----------->  GCS Buckets  <-----+
                                (bronze / daily / exports / models / serving / monitoring)
                                           |
                                           v
                                [ Jobs de feature engineering ]
                                           |
                                           v
                                [ Entraînement modèles ML ]
                                           |
                                           v
                                 GCS models/ (h15, h60, ...)
                                           |
                         +-----------------+-----------------+
                         |                                   |
                         v                                   v
                 API FastAPI (Cloud Run)              Jobs Monitoring
             (stations, prévisions, monitoring)  (health, drift, perf, network)
                         |                                   |
                         +-----------------+-----------------+
                                           v
                                  UI Next.js / React
                            (velo-paris.fr + /monitoring)
```

---

## 2. Structure du dépôt

```text
.
├── service/                # Backend data / ML : core + jobs batch
│   ├── core/               # Feature engineering, training, forecast, monitoring orchestrator
│   └── jobs/               # Ingestion, compactage, exports, training, serving, monitoring
│
├── api/                    # API FastAPI (endpoints produit + monitoring)
│   ├── core/               # Settings, GCS helpers, forecast reader, snapshot & météo live
│   ├── routes/             # Routes FastAPI (snapshot, stations, forecast, monitoring, etc.)
│   ├── schemas/            # Schémas Pydantic
│   ├── requirements.txt    # Dépendances API
│   ├── api-env.yaml        # Exemple d’ENV (sans secrets)
│   └── app.py              # Entrypoint FastAPI
│
├── ui/                     # Interface web Next.js (app carte + monitoring)
│   ├── pages/              # Pages Next.js (/, /app, /monitoring/...)
│   ├── components/         # Composants UI (GlobalHeader, MonitoringNav, KpiBar, ...)
│   ├── lib/                # Services HTTP, helpers Plotly, types métiers
│   ├── public/             # Assets statiques (favicon, stations.index.json, images...)
│   ├── styles/             # CSS globaux + CSS contextuels (landing, app, monitoring)
│   ├── netlify/            # Fonctions serverless / proxies (si utilisé)
│   ├── scripts/            # Scripts utilitaires (ex: buildStationsIndex.ts)
│   ├── package.json        # Dépendances front
│   ├── tsconfig.json       # Configuration TypeScript
│   └── next.config.js      # Configuration Next.js
│
├── docs/                   # Documentation supplémentaire (optionnel)
├── scripts/                # Scripts globaux (maintenance, migrations, etc.)
├── infra/                  # Infra as code / notes GCP (optionnel)
│
├── .env                    # Configuration globale (backend/API/UI) – non commitée
├── .gitignore
├── .gcloudignore
└── README.md               # Ce fichier
```

---

## 3. Backend data / ML — `service/`

### 3.1. `service/core` — logique cœur

Brique réutilisable par les jobs **et** par l’API : feature engineering, entraînement, inférence, orchestration du monitoring.

Principaux modules :

- **`forecast.py`**  
  - empile les features complètes (calendaires, temporelles, spatiales, KNN),  
  - entraîne des modèles XGBoost par horizon,  
  - gère le versioning et la publication des artefacts modèle (local + GCS),  
  - fournit les helpers d’inférence (`predict_from_features_df`, `predict_latest_offline`).

- **`time_features.py`**  
  - construit la base d’entraînement à partir des snapshots 5 minutes,  
  - ajoute lags, rollings, météo, encodages sin/cos,  
  - définit la cible `y_nb` à horizon `horizon_bins * 5 min`.

- **`spatial_features.py`**  
  - calcule des features géographiques (distance au centre, grille, clusters, target encoding),  
  - ajoute des KNN entre stations + lags dynamiques.

- **`cal_features.py`**  
  - crée des features calendaires à partir de `tbin_utc` (heure, jour de semaine, mois, week-end, sin/cos, etc.).

- **`monitoring_pipeline.py`**  
  - orchestrateur Monitoring : exécute en séquence les jobs dédiés :  
    `build_data_health`, `build_data_drift`,  
    `build_model_performance`, `build_model_explainability`,  
    `build_network_overview`, `build_network_stations`, `build_network_dynamics`,  
    `build_monitoring_intro`,  
  - gère les fenêtres temporelles (courante/référence), les horizons, les options de dry-run.

- **`utils_gcs.py`**  
  - **launcher Cloud Run / CLI** générique, utilisé comme unique entrypoint dans Cloud Run Jobs :  
    - lit `JOB` (nom du job à exécuter : `ingest`, `compact_daily`, `build_datasets`, `monitoring_pipeline`…),  
    - gère `DAY` / `DAY_OFFSET`,  
    - applique un verrouillage GCS (`GCS_LOCK`) pour éviter les exécutions concurrentes,  
    - construit et exécute la commande Python finale.

Chaque module core est fortement documenté (docstrings) pour clarifier inputs, outputs et hypothèses.

### 3.2. `service/jobs` — jobs batch

Les jobs batch sont pensés pour être exécutés via **Cloud Run Jobs** ou en CLI locale.  
Ils orchestrent uniquement l’I/O et s’appuient sur `service/core` pour la logique métier.

#### 3.2.1. Ingestion 5 minutes

- **`ingest.py`**  
  - appelle les endpoints Vélib (status + info) + météo (selon configuration),  
  - construit un snapshot 5 min (colonnes `ts_utc`, `tbin_utc`, `station_id`, `bikes`, `capacity`, etc.),  
  - écrit en Parquet en local et/ou sur GCS (`bronze/`),  
  - publie des JSON de fraîcheur (optionnel) pour le monitoring de latence.

#### 3.2.2. Compaction quotidienne / mensuelle

- **`compact_daily.py`**  
  - prend tous les snapshots 5 min d’un jour donné (depuis `GCS_RAW_PREFIX`),  
  - produit un unique `compact_YYYY-MM-DD.parquet` dans `GCS_DAILY_PREFIX`.

- **`compact_monthly.py`**  
  - agrège tous les `compact_YYYY-MM-DD.parquet` d’un mois,  
  - produit `compact_YYYY-MM.parquet` dans `GCS_MONTHLY_PREFIX`.

#### 3.2.3. Exports “events / perf”

- **`build_datasets.py`**  
  - entrée : un `compact_YYYY-MM-DD.parquet`,  
  - sorties :  
    - `events.parquet` + `events_YYYY-MM-DD.parquet` (événements alignés),  
    - `perf.parquet` + `perf_YYYY-MM-DD.parquet` (performance modèle vs baseline),  
  - appelle les modèles (ex: horizons 15/60 min) pour remplir les colonnes de prédiction,  
  - garantit un strict alignement temporel entre features, `y_true` et `y_pred`.

#### 3.2.4. Base d’entraînement & modèles

- **`export_training_base.py`**  
  - agrège plusieurs jours d’events/perf en une base d’entraînement globale,  
  - applique les filtres / découpes nécessaires.

- **`train_model.py`**  
  - lit la base d’entraînement (ou les dailies),  
  - entraîne les modèles par horizon (XGBoost),  
  - sauvegarde les modèles sous forme de pack (`model.joblib` + `version.json`, etc.),  
  - publie les artefacts en local et sur GCS (`models/h{H}/`),  
  - gère un alias `latest.*` pour faciliter le déploiement.

#### 3.2.5. Prévisions “serving”

- **`build_serving_forecast.py`**  
  - lit quelques heures de snapshots récents (depuis `GCS_RAW_PREFIX`),  
  - reconstruit la **même pile de features** que le training,  
  - applique les modèles par horizon,  
  - écrit des prévisions batch sous forme de JSON :  

    ```text
    gs://.../serving/forecast/h{H}/latest.json
    ```

  - optionnellement un bundle unique `latest_forecast.json` (multi-horizons).

#### 3.2.6. Jobs de monitoring (JSON pour l’UI)

- **`build_data_health.py`**  
  - KPIs de santé des données : complétude, frais de mise à jour, schéma, duplications, séquences “plates”, etc.  

- **`build_data_drift.py`**  
  - dérive des features : PSI / KS / deltas de moyennes et variances, par feature et par zone,  
  - calcul de séries temporelles de PSI global, alertes de drift.

- **`build_model_performance.py`**  
  - KPIs modèle : MAE, RMSE, lift vs baseline,  
  - découpes par jour, heure, jour de semaine, station, cluster,  
  - histogrammes de résidus, séries temporelles par station.

- **`build_model_explainability.py`**  
  - importance des features (gain, split count, etc.),  
  - analyses des résidus (QQ plots, distributions, autocorrélation),  
  - calibration et incertitude,  
  - épisodes spécifiques (périodes où le modèle se comporte différemment).

- **`build_network_overview.py`**  
  - KPIs globaux réseau (taux de remplissage, tension, pénuries / saturations),  
  - snapshot carte réseau, distributions de dispo.

- **`build_network_stations.py`**  
  - profils station (usage, tension, quantiles, clusters),  
  - données pour les vues PCA / clusters / listes station.

- **`build_network_dynamics.py`**  
  - dynamiques horaires (heatmaps 7×24, épisodes, profils par jour),  
  - agrégations par zone/station.

- **`build_monitoring_intro.py`**  
  - agrège les signaux clés (réseau, data, modèles, forecasts) dans un JSON compact,  
  - alimente la page d’intro du Monitoring (`/monitoring`).

Tous ces jobs écrivent sous `GCS_MONITORING_PREFIX/.../latest/` (pattern **LATEST only** : pas de dates dans les paths consommés par l’UI).

---

## 4. Layout GCS & conventions temporelles

### 4.1. Layout GCS

Le pipeline est **Parquet-first**, puis JSON pour l’API / UI.

```text
gs://<project>_cloudbuild/velib
├── bronze/           # raw 5-min snapshots (GBFS + météo)
├── daily/            # compact_YYYY-MM-DD.parquet
├── monthly/          # compact_YYYY-MM.parquet
├── exports/          # events_YYYY-MM-DD.parquet, perf_YYYY-MM-DD.parquet
├── models/           # h15/h60/... model.joblib + version.json + latest.*
├── serving/
│   └── forecast/     # batch forecasts JSON (h15/latest.json, h60/latest.json, latest_forecast.json...)
└── monitoring/       # tous les artefacts JSON de monitoring (data, model, network, intro)
```

### 4.2. Conventions temporelles

- 1 bin = **5 minutes** → `horizon_bins * 5 = horizon en minutes`.  
- `tbin_utc` est toujours en **UTC naïf** (datetime sans tz-info).  
- La cible de régression `y_nb` est définie à `tbin_utc + horizon_bins * 5 min`.  
- Les prédictions sont **clippées** dans `[0, capacity]` puis **arrondies** en entiers.

---

## 5. API FastAPI — `api/`

### 5.1. Structure

```text
api/
├─ app.py                     # Entrypoint FastAPI (montage des routers, CORS, token global…)
├─ Dockerfile                 # Image Docker de l’API
├─ requirements.txt           # Dépendances de l’API
├─ api-env.yaml               # Exemple de configuration d’ENV (sans secrets)
├─ core/
│   ├─ settings.py            # Settings Pydantic (GCS, météo, monitoring, CORS…)
│   ├─ gcs.py                 # Utilitaires GCS (download bytes/json, cache, headers)
│   ├─ snapshot_live.py       # Helper GBFS → snapshot live (DataFrame)
│   ├─ weather_live.py        # Helper Open-Meteo → dict météo live
│   └─ forecast_reader.py     # Lecture & normalisation des bundles de forecast JSON
├─ routes/
│   ├─ snapshot.py            # /snapshot (snapshot live Vélib’)
│   ├─ stations.py            # /stations (métadonnées & KPIs stations)
│   ├─ weather.py             # /weather/live (météo live)
│   ├─ forecast.py            # /forecast, /forecast/latest (prévisions par horizon)
│   ├─ badges.py              # /badges (badges header UI)
│   ├─ health.py              # /health (healthcheck étendu API + GCS)
│   ├─ history.py             # /history (désactivé, compatibilité)
│   ├─ monitoring/
│   │   ├─ network_overview.py
│   │   ├─ network_dynamics.py
│   │   ├─ network_stations.py
│   │   ├─ model_performance.py
│   │   ├─ model_explainability.py
│   │   ├─ data_health.py
│   │   ├─ data_drift.py
│   │   ├─ data_freshness.py
│   │   └─ intro.py
│   └─ ...
└─ schemas/
    └─ forecast.py            # Schémas Pydantic pour les réponses de forecast
```

### 5.2. Helpers principaux

- **`core/settings.py`**  
  - Lit toutes les variables d’environnement (GCS, monitoring, forecast, CORS, token global, etc.) via Pydantic.  
  - Expose un objet `settings` utilisé partout.

- **`core/gcs.py`**  
  - Helpers GCS (parsing `gs://`, client `google-cloud-storage`, lecture binaire / texte / JSON),  
  - Gestion de cache mémoire simple avec TTL + validation ETag,  
  - Construction de headers HTTP (`Cache-Control`, `ETag`, `Last-Modified`).

- **`core/snapshot_live.py`**  
  - Récupère les flux GBFS Vélib’ (status + info),  
  - merge sur `station_id`,  
  - construit un `DataFrame` typé avec `ts_utc`, `tbin_utc`, `bikes`, `capacity`, `mechanical`, `ebike`, `status`, `lat`, `lon`, `name`.

- **`core/weather_live.py`**  
  - Récupère la météo live via Open-Meteo (`temp_C`, `precip_mm`, `wind_mps`, `ts_utc`),  
  - renvoie un dict ; en cas d’erreur, retourne `{}` sans faire échouer les routes.

- **`core/forecast_reader.py`**  
  - lit les bundles `latest_forecast.json` et/ou `latest_h{h}.json` depuis GCS (`SERVING_FORECAST_PREFIX`),  
  - normalise le `DataFrame` (datetimes, station_id, types numériques),  
  - gère un cache mémoire par horizon (`FORECAST_CACHE_TTL_SECONDS`),  
  - permet un filtrage par liste de stations.

### 5.3. Endpoints “produit”

- **`GET /snapshot`**  
  Snapshot live du réseau Vélib’ (DataFrame → JSON records).  
  Utilise `fetch_live_snapshot()`, renvoie un tableau d’objets (stations).

- **`GET /stations`**  
  Liste les stations et quelques KPI (lat/lon, capacité, bikes, docks).  
  Source principale : snapshot live ; fallback : données de forecast si besoin.

- **`GET /weather/live`**  
  Renvoie la météo actuelle (via Open-Meteo).

- **`GET /forecast/available`**  
  Liste des horizons disponibles, préfixes GCS, exemples d’URL.

- **`GET /forecast/latest?h=15`**  
  Lit `serving/forecast/h{H}/latest.json`, sanitise le JSON, renvoie les prévisions pour l’horizon demandé.

- **`GET /badges`**  
  Aggrège météo + fraîcheur des données de forecast pour les badges du header UI :  
  - température / météo,  
  - `age_minutes` depuis la dernière génération de forecast.

- **`GET /health`**  
  Healthcheck enrichi :  
  - statut global,  
  - métadonnées sur les artefacts GCS (forecast bundle, modèle),  
  - horizons supportés,  
  - erreurs éventuelles (infos textuelles, pas d’exception).

- **`GET /history`**  
  Endpoint historique aujourd’hui désactivé (retourne 204), gardé pour compatibilité.

### 5.4. Endpoints de monitoring

Tous les endpoints `/monitoring/...` :  

- lisent **uniquement** des JSON déjà produits par les jobs dans `service/jobs`,  
- ne font aucun calcul lourd,  
- appliquent une sanitisation (NaN/Inf → `null`),  
- exposent un JSON propre pour l’UI.

Groupes principaux :

- `/monitoring/network/overview`  
- `/monitoring/network/dynamics`  
- `/monitoring/network/stations`  
- `/monitoring/model/performance`  
- `/monitoring/model/explainability`  
- `/monitoring/data/health`  
- `/monitoring/data/drift`  
- `/monitoring/data/freshness`  
- `/monitoring/intro`

### 5.5. Sécurité & CORS

- **Token global** (optionnel) :  
  - variable : `API_GLOBAL_TOKEN`,  
  - si définie, toutes les routes (sauf `/` & `/health`) exigent `Authorization: Bearer <token>`.

- **CORS** :  
  - configuré via `CORS_ORIGINS` (liste d’origines autorisées),  
  - si vide, comportement permissif (`*`) possible, selon le déploiement.

---

## 6. Interface Utilisateur — `ui/` (Next.js / React)

L’UI sert à la fois d’**application grand public** et de **vitre de monitoring avancé**.

### 6.1. Objectifs

1. Application carte Vélib’ en temps (quasi) réel.  
2. Module Monitoring (réseau, data, modèle) pour montrer les capacités Data/MLOps.  
3. Démonstration de :
   - React / Next.js,
   - intégration Plotly (graphes avancés),
   - Leaflet (cartographie),
   - architecture front propre (services typés, ETag, etc.).

### 6.2. Architecture

```text
ui/
 ├─ components/          # Composants réutilisables (UI, cartes, KPI, nav…)
 ├─ lib/                 # Services HTTP, loaders, helpers, typage
 │   └─ services/        # Services Monitoring + pages /app
 ├─ pages/               # Pages Next.js (landing, app, monitoring)
 ├─ public/              # Assets statiques (data, images, favicon…)
 ├─ styles/              # CSS globaux + CSS contextuels (monitoring, app)
 ├─ netlify/             # Fonctions serverless proxy API (optionnel)
 ├─ scripts/             # Scripts utilitaires (build index stations, etc.)
 ├─ next.config.js       # Config Next.js
 ├─ tsconfig.json        # Config TS
 └─ package.json         # Dépendances & scripts
```

### 6.3. Routage & contextes

- 3 contextes principaux : `landing`, `app`, `monitoring`.  
- `_app.tsx` choisit dynamiquement les CSS / layouts selon le contexte.  
- support d’un mode `embed` / `nochrome` (pas de header/footer) via query `?embed=1` ou intégration iframe.

### 6.4. Services HTTP & ETag

- Tous les appels aux endpoints API passent par des helpers comme `fetchJsonWithEtag`.  
- Gestion transparente des ETags, revalidation, réduction de bande passante.  
- Services typés pour chaque page (ex: `lib/services/monitoring/model_performance.ts`).

### 6.5. Visualisations (Plotly)

- usage de `react-plotly.js` en dynamic import (pas de SSR),  
- graphiques :
  - performance modèle (MAE, lift, biais),
  - séries temporelles,
  - heatmaps 7×24,
  - barplots comparatifs (jour vs historique),
- thème Plotly dédié (couleurs, police, fonds).

### 6.6. Cartographie (Leaflet)

- cartes pour :
  - état réseau (clusters, tension, pénuries/saturations),
  - dynamiques temporelles,
  - stations top/bottom lift.  
- fallback automatique de Carto → OSM.  
- chaque carte est encapsulée dans un composant React autonome.

### 6.7. Pages Monitoring

- `/monitoring` (intro)  
  - KPIs réseau, statuts systèmes, liens rapides.

- `/monitoring/network/overview`  
  - snapshot global + courbes J/J−1, carte, stations en tension.

- `/monitoring/network/stations`  
  - clusters, PCA, distributions par station/cluster.

- `/monitoring/network/dynamics`  
  - heatmaps 7×24, épisodes de tension, profils horaires.

- `/monitoring/model/performance`  
  - MAE, lift vs baseline, coupes par heure/jour/station.

- `/monitoring/model/explainability`  
  - importance des features, résidus, calibration.

- `/monitoring/data/health`  
  - complétude, fraîcheur, anomalies.

- `/monitoring/data/drift`  
  - PSI global & par feature / zone.

Toutes ces pages consomment exclusivement les JSON exposés par l’API `/monitoring/...`.

---

## 7. Variables d’environnement (synthèse)

> Les valeurs exactes sont décrites dans les fichiers d’exemple (`.env`, `api-env.yaml`, etc.).

### 7.1. Commun / GCS

- `GCP_PROJECT`  
- `GCS_BUCKET_ROOT` (ex: `gs://velib-forecast-472820_cloudbuild/velib`)  
  Dérivés :
  - `GCS_RAW_PREFIX` (`${GCS_BUCKET_ROOT}/bronze`)
  - `GCS_DAILY_PREFIX` (`${GCS_BUCKET_ROOT}/daily`)
  - `GCS_MONTHLY_PREFIX` (`${GCS_BUCKET_ROOT}/monthly`)
  - `GCS_EXPORTS_PREFIX` (`${GCS_BUCKET_ROOT}/exports`)
  - `GCS_MODELS_PREFIX` (`${GCS_BUCKET_ROOT}/models`)
  - `SERVING_FORECAST_PREFIX` (`${GCS_BUCKET_ROOT}/serving/forecast`)
  - `GCS_MONITORING_PREFIX` (`${GCS_BUCKET_ROOT}/monitoring`)

### 7.2. Live / météo

- `VELIB_STATUS_URL`, `VELIB_INFO_URL`  
- `OPENMETEO_URL`, `OPENMETEO_LAT`, `OPENMETEO_LON`, `OPENMETEO_TIMEOUT`  
- `METEO_DISABLE` (pour couper la météo si nécessaire)

### 7.3. Forecast / modèles

- `FORECAST_HORIZONS`, `FORECAST_SUPPORTED` (ex. `"15,60"`)  
- `MODEL_URI_15`, `MODEL_URI_60`, etc.  
- `GCS_MODEL_URI` (legacy)  
- `FORECAST_CACHE_TTL_SECONDS` (TTL cache API pour les prévisions)  
- `FORECAST_HTTP_FIRST` (préférer HTTP public GCS avant le client GCS)

### 7.4. Monitoring

- `MON_TZ` (ex. `Europe/Paris`)  
- `MON_LAST_DAYS` (fenêtre courante, ex. 14)  
- `MON_REF_DAYS` (fenêtre de référence, ex. 28)  
- `MON_PENURY_THRESH`, `MON_SATURATION_THRESH`  
- préfixes / knobs spécifiques : `OVERVIEW_*`, `DYNAMICS_*`, `NETWORK_*`, `PERF_*`, `DATA_HEALTH_*`, etc.

### 7.5. API FastAPI

- `API_GLOBAL_TOKEN` (token global optionnel)  
- `CORS_ORIGINS` (origines autorisées)  
- `API_SELF_BASE` / `LIVE_SNAPSHOT_URL` (auto-appels internes pour certains badges)  
- `TZ_APP`, `IMAGE_TAG` (métadonnées runtime)

### 7.6. UI Next.js

- `NEXT_PUBLIC_API_BASE_URL` (URL de base de l’API)  
- variables `NEXT_PUBLIC_*` pour les options d’affichage / tracking éventuels.

---

## 8. Développement local

### 8.1. Backend jobs

```bash
# Depuis la racine
cd service
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt  # ou requirements backend

# Charger les variables d'environnement
export $(grep -v '^#' ../.env | xargs)

# Ingestion ponctuelle (snapshot 5 min)
INGEST_SAVE_PARQUET=1 LOCAL_RAW_DIR=data_local/raw \
INGEST_TO_GCS=1 GCS_RAW_PREFIX=${GCS_RAW_PREFIX} \
python -m service.jobs.ingest

# Compaction daily pour un jour donné
GCS_RAW_PREFIX=${GCS_RAW_PREFIX} \
GCS_DAILY_PREFIX=${GCS_DAILY_PREFIX} \
DAY=2025-11-01 \
python -m service.jobs.compact_daily

# Build datasets (events + perf)
GCS_DAILY_PREFIX=${GCS_DAILY_PREFIX} \
GCS_EXPORTS_PREFIX=${GCS_EXPORTS_PREFIX} \
DAY=2025-11-01 \
MODEL_URI_15=${MODEL_URI_15} \
MODEL_URI_60=${MODEL_URI_60} \
python -m service.jobs.build_datasets
```

### 8.2. Entraînement & serving

```bash
# Entraînement modèle (ex. h15)
GCS_DAILY_PREFIX=${GCS_DAILY_PREFIX} \
MODEL_GCS_BUCKET=<bucket_name> \
MODEL_GCS_PREFIX=velib/models/h15 \
python -m service.jobs.train_model

# Prévisions serving
GCS_RAW_PREFIX=${GCS_RAW_PREFIX} \
SERVING_FORECAST_PREFIX=${SERVING_FORECAST_PREFIX} \
FORECAST_HORIZONS="15,60" \
GCS_MODEL_URI_T15=${MODEL_URI_15} \
GCS_MODEL_URI_T60=${MODEL_URI_60} \
python -m service.jobs.build_serving_forecast

# Monitoring complet
GCS_EXPORTS_PREFIX=${GCS_EXPORTS_PREFIX} \
GCS_MONITORING_PREFIX=${GCS_MONITORING_PREFIX} \
MON_LAST_DAYS=14 MON_REF_DAYS=28 \
python -m service.core.monitoring_pipeline
```

### 8.3. API FastAPI

```bash
cd api
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Copier/adapter api-env.yaml vers .env (ou définir les variables à la main)
cp api-env.yaml .env

# Lancer l’API
uvicorn app:app --reload --port 5050
# Swagger UI : http://localhost:5050/docs
```

### 8.4. UI Next.js

```bash
cd ui
npm install

# Dev
npm run dev
# -> http://localhost:3000

# Build production
npm run build
npm run start
```

---

## 9. Déploiement

### 9.1. Jobs & API sur Cloud Run

Exemple (à adapter) :

```bash
# Build image API
cd api
gcloud builds submit --tag gcr.io/$GCP_PROJECT/velib-api

# Déploiement Cloud Run API
gcloud run deploy velib-api \
  --image gcr.io/$GCP_PROJECT/velib-api \
  --region=europe-west1 \
  --platform=managed \
  --allow-unauthenticated \
  --set-env-vars-file=api-env.yaml
```

Les jobs batch peuvent être packagés de la même manière (image unique + `JOB` différent via ENV, exécutée par `service.core.utils_gcs`) et déclenchés via **Cloud Run Jobs + Cloud Scheduler**.

### 9.2. UI (Netlify / autre)

- Netlify :  
  - dir de build : `ui/`,  
  - commande : `npm install && npm run build`,  
  - variables d’environnement : `NEXT_PUBLIC_API_BASE_URL`, etc.  
- Alternative : déploiement de l’UI également sur Cloud Run (image Next.js).

---

## 10. Tests & qualité

*(Adapter selon les outils réellement utilisés dans le projet.)*

- Tests Python : `pytest`  
- Lint Python : `ruff` / `flake8`  
- Lint TS/JS : `npm run lint` dans `ui/`  
- Formatage : `black` / `isort` côté Python, `prettier` côté JS/TS.

---

## 11. Roadmap indicative

- Extension à d’autres horizons (ex. 5, 30 min).  
- Amélioration de l’explicabilité (SHAP complet, scénarios “what-if”).  
- Ajout de métriques de robustesse (stress tests météo, phénomènes extrêmes).  
- API publique minimaliste pour démonstration externe (lecture seule).  

---

## 12. Licence & crédits

- Données Vélib' : se référer aux CGU de **Vélib' Métropole / Île-de-France Mobilités**.  
- Données météo : se référer aux CGU du fournisseur (Open-Meteo, etc.).  
- Code & modèles : `MIT`

Ce projet est utilisé comme **démonstration technique / portfolio** et n’est pas destiné, en l’état, à un usage commercial tiers.
