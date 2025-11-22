# service/jobs

Batch jobs for **Vélib’ Forecast** (Cloud Run / GCS):

- ingestion 5 minutes (GBFS + météo),
- compaction journalière / mensuelle,
- préparation des bases d’entraînement / perf,
- génération de prévisions “serving”,
- production des artefacts JSON pour la page **Monitoring** (overview, health, drift, perf, explainability, network, intro).

Ces jobs sont lancés soit :

- **directement** : `python -m service.jobs.<job_name>`,
- soit via le **launcher générique** `service/core/utils_gcs.py` avec `JOB=<job_name>` (recommandé en Cloud Run).

---

## 1. Vue d’ensemble

### 1.1. Pipeline logique

1. **Ingestion 5 minutes**
   - `ingest.py`  
   → écrit des snapshots 5 min côté local et/ou GCS (`bronze`).

2. **Compaction**
   - `compact_daily.py` : tous les snapshots d’un jour → `compact_YYYY-MM-DD.parquet` (`daily`).  
   - `compact_monthly.py` : tous les dailies d’un mois → `compact_YYYY-MM.parquet` (`monthly`).

3. **Exports “events / perf”**
   - `build_datasets.py` : à partir d’un `compact_YYYY-MM-DD.parquet` :
     - `events.parquet` (+ version datée)
     - `perf.parquet` (+ version datée, avec `y_pred_int`).

4. **Base d’entraînement / modèles**
   - `export_training_base.py` : construit une base d’entraînement agrégée (multi-jours) à partir des exports/daily.  
   - `train_model.py` : télécharge les daily / base d’entraînement, appelle `service.core.forecast.train_model()` et publie les modèles `.joblib` + `version.json` (local + GCS).

5. **Prévisions “serving”**
   - `build_serving_forecast.py` : prend les dernières données raw + modèles GCS, construit 4h de features et publie des **prévisions batch** par horizon (`h15/latest.json`, `h60/latest.json`, …).

6. **Monitoring (JSON artefacts)**
   - `build_data_health.py`       : KPI de santé des données (fraîcheur, duplication, plats, etc.).
   - `build_data_drift.py`        : PSI / drift sur les features.
   - `build_model_performance.py` : MAE / lift / métriques par jour, heure, DOW…  
   - `build_model_explainability.py` : FI XGBoost + résidus / épisodes / profils.
   - `build_network_overview.py`  : vision macro du réseau (volumes, distribution, tendances).
   - `build_network_stations.py`  : profil par station (capacités, usage, tension).
   - `build_network_dynamics.py`  : tension, pénuries / saturations, épisodes, heatmaps.
   - `build_monitoring_intro.py`  : agrège les KPI clés en un intro JSON pour la page Monitoring.

Tous les jobs Monitoring écrivent sous `GCS_MONITORING_PREFIX` dans des dossiers `.../latest/` (pas de versionnement daté côté web, la version étant implicitement “latest”).

---

## 2. Variables d’environnement (communes)

Les variables ci-dessous ne sont pas toutes utilisées par chaque job, mais reviennent souvent :

### 2.1. Racines GCS

- `GCS_RAW_PREFIX`  
  `gs://<bucket>/velib/bronze` – snapshots 5 min.
- `GCS_DAILY_PREFIX`  
  `gs://<bucket>/velib/daily` – compacts journaliers.
- `GCS_MONTHLY_PREFIX`  
  `gs://<bucket>/velib/monthly` – compacts mensuels.
- `GCS_EXPORTS_PREFIX`  
  `gs://<bucket>/velib/exports` – `events_*.parquet`, `perf_*.parquet`.
- `GCS_MONITORING_PREFIX`  
  `gs://<bucket>/velib` ou `gs://<bucket>/velib/monitoring` – base des artefacts Monitoring.
- `SERVING_FORECAST_PREFIX`  
  `gs://<bucket>/velib/serving/forecast` – JSON de prévision batch (h15, h60, …).

### 2.2. Modèles

- `GCS_MODEL_URI_T15`, `GCS_MODEL_URI_T60`, …  
  URIs GCS des modèles par horizon (fichier `.joblib` ou préfixe résolu en `latest.joblib`).
- `MODEL_ARTEFACTS_DIR`  
  Répertoire local où `train_model.py` écrit les modèles datés (`artifacts/h15/<stamp>/...`).
- `MODEL_GCS_BUCKET`, `MODEL_GCS_PREFIX`  
  Bucket + préfixe GCS où `train_model.py` publie `model.joblib` + `version.json` + alias `latest.*`.

### 2.3. Monitoring

- `MON_TZ`                (ex : `Europe/Paris`)
- `MON_LAST_DAYS`         (fenêtre “courante” : 7, 10, 14 jours…)
- `MON_REF_DAYS`          (fenêtre de référence pour le drift, ex : 28)
- `MON_HORIZONS`          (ex : `"15,60"`)
- `OVERVIEW_LAST_DAYS`, `NETWORK_WINDOW_DAYS`, `PERF_LAST_DAYS`  
  → knobs spécifiques, mais toujours avec un alias `MON_*` possible.

---

## 3. Détail par job

### 3.1. `ingest.py`

**Rôle**  
Ingestion toutes les 5 minutes des données :

- **Vélib’ GBFS** (status + info → enrichi en `name`),
- **Météo** via Open-Meteo (ou désactivable).

**Schéma écrit (timestamps UTC naïfs)**  

ts_utc, tbin_utc, station_id, bikes, capacity, mechanical, ebike, status,
lat, lon, name, temp_C, precip_mm, wind_mps

**Sorties**

- Local : `data_local/raw/date=YYYY-MM-DD/hour=HH/YYYY-MM-DDT%H-%M.parquet`
- GCS   : `${GCS_RAW_PREFIX}/date=YYYY-MM-DD/hour=HH/YYYY-MM-DDT%H-%M.parquet`

**ENV clés**

- `INGEST_SAVE_PARQUET` = `1|0` (def. `1`)
- `LOCAL_RAW_DIR`       = dossier local (def. `data_local/raw`)
- `INGEST_TO_GCS`       = `1|0` (def. `0`)
- `GCS_RAW_PREFIX`      = `gs://.../bronze`
- `OPENMETEO_LAT`, `OPENMETEO_LON`
- `METEO_DISABLE`       = `1|0` (désactive l’appel météo)

### 3.2. `compact_daily.py`

**Rôle**  
Compacter **tous les snapshots 5min d’un jour UTC** → **1 Parquet daily** + purge optionnelle des snapshots.

**Entrée**

- `${GCS_RAW_PREFIX}/date=YYYY-MM-DD/hour=HH/*.parquet`

**Sortie**

- `${GCS_DAILY_PREFIX}/compact_YYYY-MM-DD.parquet`

**ENV**

- `GCS_RAW_PREFIX`      (requis)
- `GCS_DAILY_PREFIX`    (requis)
- `DAY` = `YYYY-MM-DD`  (requis ou calculé par `utils_gcs` via `DAY_OFFSET`)
- `DELETE_AFTER_COMPACT` = `1|0` (def. `1`, et seulement si `DAY < today_UTC`)

### 3.3. `compact_monthly.py`

**Rôle**  
Compacter **tous les compacts journaliers d’un mois** → **1 Parquet mensuel**.

**Entrées**

- `${GCS_DAILY_PREFIX}/compact_YYYY-MM-DD.parquet` (pour tout le mois)

**Sortie**

- `${GCS_MONTHLY_PREFIX}/compact_YYYY-MM.parquet`

**ENV**

- `GCS_DAILY_PREFIX`   (requis)
- `GCS_MONTHLY_PREFIX` (requis)
- `MONTH` = `YYYY-MM`  (requis)
- `DRY_RUN` = `"1"`    → affiche ce qui serait fait, sans écrire ni supprimer.

### 3.4. `build_datasets.py`

**Rôle**  
À partir d’un `compact_YYYY-MM-DD.parquet`, construit la “vue évènementielle” et la “vue perf” :

- `events.parquet` / `events_YYYY-MM-DD.parquet`
- `perf.parquet` / `perf_YYYY-MM-DD.parquet` (avec prédictions modèles)

**Entrées**

- `${GCS_DAILY_PREFIX}/compact_YYYY-MM-DD.parquet`
- Modèles par horizon (fichier ou préfixe) :
  - `GCS_MODEL_URI_T15`, `GCS_MODEL_URI_T60`, …

**Sorties**

- `${GCS_EXPORTS_PREFIX}/events.parquet`
- `${GCS_EXPORTS_PREFIX}/events_YYYY-MM-DD.parquet`
- `${GCS_EXPORTS_PREFIX}/perf.parquet`
- `${GCS_EXPORTS_PREFIX}/perf_YYYY-MM-DD.parquet`

### 3.5. `export_training_base.py`

**Rôle**  
Préparer une **base d’entraînement** consolidée (multi-jours) à partir des dailies / exports, et l’écrire en **Parquet unique** (local ou GCS).

Typiquement utilisé avant `train_model.py` pour éviter de recharger N fois les mêmes shards.

### 3.6. `train_model.py`

**Rôle**  
Entrypoint de training **Cloud Run** (harmonisé) pour les modèles XGBoost.

**Fonctions clés**

- Télécharge et concatène les shards daily depuis GCS (ou lit un parquet unique local),
- Résout dynamiquement le module d’entraînement (ici `service.core.forecast.train_model`),
- Gère la version du modèle (lecture du `latest.json` précédent + bump semver),
- Sauvegarde le modèle packé (`{"model", "feat_cols", "horizon_bins"}`) localement et sur GCS.

### 3.7. `build_serving_forecast.py`

**Rôle**  
Construire les features sur un **fenêtre glissante de 4h** (bins de 5 minutes) à partir des données brutes, appliquer les modèles de production et publier des **prévisions batch** pour plusieurs horizons.

### 3.8. `build_data_health.py`

**Rôle**  
Calculer l’**état de santé des données** sur les derniers `MON_LAST_DAYS` jours.

### 3.9. `build_data_drift.py`

**Rôle**  
Mesurer la **dérive des distributions** (PSI, etc.) sur les features clefs.

### 3.10. `build_model_performance.py`

**Rôle**  
Construire les artefacts de **performance modèle** (MAE, RMSE, lift, etc.).

### 3.11. `build_model_explainability.py`

**Rôle**  
Produire les artefacts d’**explicabilité modèle** (FI, résidus, épisodes, etc.).

### 3.12. `build_network_overview.py`

**Rôle**  
Vue **macro réseau** sur les derniers jours (tendances, distribution, volumes).

### 3.13. `build_network_stations.py`

**Rôle**  
Profil détaillé par station sur la fenêtre réseau.

### 3.14. `build_network_dynamics.py`

**Rôle**  
Analyser la **dynamique** du réseau (pénuries, saturations, profils horaires, etc.).

### 3.15. `build_monitoring_intro.py`

**Rôle**  
Construire l’**intro unique** de la page Monitoring à partir des artefacts déjà produits.

---

## 4. Lancement via `utils_gcs.py`

Pour les jobs destinés à Cloud Run, on passe par :

JOB=compact_daily GCS_RAW_PREFIX=gs://.../velib/bronze GCS_DAILY_PREFIX=gs://.../velib/daily DAY_OFFSET=-1 python -m service.core.utils_gcs

Les jobs Monitoring se lancent de la même façon :

JOB=build_data_health GCS_EXPORTS_PREFIX=gs://.../velib/exports GCS_MONITORING_PREFIX=gs://.../velib/monitoring MON_LAST_DAYS=14 python -m service.core.utils_gcs

En local, pour debug, on peut appeler directement :

python -m service.jobs.build_data_health
python -m service.jobs.build_model_performance
python -m service.jobs.build_network_overview

---

Ce README est volontairement descriptif, sans rentrer dans la logique métier fine.
Pour la mécanique complète, se référer directement aux docstrings en tête de chaque fichier de `service/jobs` et aux modules core (`service/core/*.py`).
