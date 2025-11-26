Package backend Python pour **Vélib’ Forecast**.

Il contient :

- la **logique cœur** pour le feature engineering, l’entraînement et l’inférence,
- les **jobs batch** utilisés en production (Cloud Run / GCS),
- toutes les briques réutilisées par l’API et la couche de monitoring UI.

L’UI (Next.js / `ui/`) et l’infra (Netlify, Cloud Run, Cloud Build…) vivent
à l’extérieur de ce package et se contentent de **consommer les artefacts**
produits ici (datasets Parquet, modèles, JSON de monitoring, prévisions batch).

---

## 1. Structure du package

Structure haut niveau :

```text
service/
├── core/      # logique réutilisable (features, training, orchestration monitoring, launcher GCS)
└── jobs/      # jobs batch (ingest, compact, exports, training, monitoring, serving)
```

### 1.1. `service/core`

Modules cœur, entièrement découplés de la planification / infra :

- `forecast.py`  
  Entraînement XGBoost + utilitaires d’inférence. Intègre :
  - features temporelles (lags, rolling stats, météo),
  - features spatiales (distance au centre, grille spatiale, cluster KMeans, TE),
  - dynamiques KNN (vélos des voisins à t-L),
  - format de “model pack” : `{"model", "feat_cols", "horizon_bins"}`,
  - versioning + métadonnées JSON + publication GCS,
  - prédiction offline sur les derniers snapshots.

- `time_features.py`  
  Pile de features purement temporelles :
  - schéma strict issu des snapshots 5 min (`ts_utc`, `tbin_utc`, `bikes`, `capacity`, météo…),
  - déduplication par bin,
  - target `y_nb` à +`horizon_bins`,
  - lags / rollings / weather lags,
  - occupancy ratio,
  - features calendaires + sinusoïdales,
  - retourne `(full_df, X, y, feat_cols)`.

- `spatial_features.py`  
  Pile temporelle + extras spatiaux :
  - features station statiques (`dist_center_km`, `grid_x`, `grid_y`,
    `kmeans_cluster`, `te_cap_mean`),
  - cartes KNN + lags dynamiques,
  - frame d'entraînement `build_training_frame_spatial(...)`.

- `cal_features.py`  
  Features calendaires à partir de `tbin_utc` :
  - heure / minute / jour semaine / mois / week-end,
  - encodages sin/cos 24h & 7 jours,
  - proxys dérivés Paris si nécessaire.

- `monitoring_pipeline.py`  
  Orchestrateur de monitoring :
  - définit les étapes (`health`, `drift`, `perf`, `explain`,
    `network`, `intro`, …),
  - activées via l’env `STEPS`,
  - gère les fenêtres temporelles / horizons / racines GCS partagées,
  - appelle les jobs correspondants dans `service.jobs`.

- `utils_gcs.py`  
  Launcher générique Cloud Run / GCS :
  - dispatch via env `JOB`,
  - calcule `DAY` automatiquement pour compact_*,
  - lock GCS optionnel,
  - mode `DRY_RUN` pour debug.

Un README détaillé par sous-module est dans **`service_core_README.txt`**.

---

### 1.2. `service/jobs`

Jobs batch, chacun focalisé sur une seule responsabilité.

#### Ingestion & compactage

- `ingest.py`  
  Ingestion 5 minutes depuis :
  - GBFS Vélib’ (status + info),
  - Open-Meteo (optionnel).  
  Écrit les snapshots bruts en local et/ou GCS (`bronze`).

- `compact_daily.py`  
  Compactage quotidien :
  - tous les snapshots d’un jour UTC → `compact_YYYY-MM-DD.parquet`.

- `compact_monthly.py`  
  Compactage mensuel :
  - toutes les daily → `compact_YYYY-MM.parquet`.

#### Exports entraînement / évaluation

- `build_datasets.py`  
  Depuis `compact_YYYY-MM-DD.parquet` :
  - construit la **events view**,
  - construit la **perf view** avec `y_pred_int`,
  - écrit les versions datées & rolling sous `exports/`.

- `export_training_base.py`  
  Agrège plusieurs jours/mois → un parquet d'entraînement unique.

#### Entraînement

- `train_model.py`  
  Entrypoint Cloud Run XGBoost :
  - charge data depuis GCS,
  - appelle `forecast.train_model`,
  - sauvegarde artefacts (local + GCS, semver + `latest.*`).

#### Serving

- `build_serving_forecast.py`  
  Construit une fenêtre glissante 4h et publie les prévisions batch par horizon
  dans `serving/forecast/h{H}/latest.json`.

#### Monitoring

- `build_data_health.py` : fraîcheur, duplications, séquences plates, couverture.
- `build_data_drift.py` : PSI & dérive entre fenêtre “courante” et “référence”.
- `build_model_performance.py` : MAE/RMSE par jour/heure/jour semaine/horizon.
- `build_model_explainability.py` : importance, résiduels, épisodes.
- `build_network_overview.py` : volumes, mécaniques/électriques, tendances.
- `build_network_stations.py` : profils station, tension, quantiles UI.
- `build_network_dynamics.py` : pénurie/saturation, heatmaps horaires, régularité.
- `build_monitoring_intro.py` : agrège KPIs → `intro/latest/intro.json`.

README dédié dans **`service_jobs_README.txt`**.

---

## 2. Modèle de données & Layout GCS

Pipeline **Parquet → JSON** avec layout :

```text
bronze/           # snapshots bruts
daily/            # compact_YYYY-MM-DD.parquet
monthly/          # compact_YYYY-MM.parquet
exports/          # events/perf par jour
models/           # h15/h60/... + version.json + latest.*
serving/forecast/ # batch forecasts JSON
monitoring/       # artefacts monitoring (data, model, network, intro)
```

Les dossiers JSON consommés par l’UI sont **LATEST-only**.

---

## 3. Conventions

- **Time binning** :
  - bins 5 minutes (`tbin_utc`), UTC naïf.
  - horizon minutes = `horizon_bins * 5`.

- **Target & prévisions** :
  - `y_nb` = vélos à `tbin_utc + horizon`,
  - outputs bornés `[0, capacity]` puis arrondis.

- **Fuseaux horaires** :
  - stockage & modèles en UTC,
  - logique métier & agrégats monitoring en `Europe/Paris`
    via `MON_TZ` / `TZ_APP`.

- **Idempotence** :  
  Rejouer un job pour la même fenêtre réécrit les mêmes artefacts.

---

## 4. Connexions globales

- `service/jobs` = couche batch :
  - lancée via Cloud Run / utils_gcs,
  - lit/écrit Parquet & JSON.

- `service/core` = cœur ML / monitoring :
  - brique d’entraînement, d’inférence et de monitoring,
  - utilisée par les jobs et potentiellement l’API.

- **API** (FastAPI externe) :
  - charge les model packs,
  - charge les JSON de monitoring,
  - charge les prévisions batch.

- **UI** (Next.js externe) :
  - lit uniquement les JSON **latest**.

Séparation nette :
- `service/` gère data & modèles,
- l’infra gère planif & credentials,
- UI/API consomment les outputs.