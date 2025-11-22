# service/core

Core logic for **Vélib’ Forecast**:

- feature engineering (temps + spatial) à partir des snapshots 5 minutes,
- entraînement et inférence des modèles XGBoost,
- construction des frames de prédiction offline / serving,
- orchestration du pipeline de monitoring,
- launcher générique (Cloud Run / GCS) avec verrouillage.

Ce répertoire regroupe la logique “coeur” de l’application : tout ce qui est
réutilisable par plusieurs jobs (`service/jobs`) et par l’API.

---

## 1. Vue d’ensemble des modules

### 1.1. `forecast.py`

**Rôle**  
Module central de **training + inference** pour le forecasting Vélib’.

Il contient :

- toute la pile de features :
  - calendaires (`add_time_features` via `cal_features.py`),
  - temporelles (lags, rollings, météo),
  - spatiales (distance au centre, grille, KMeans, target encoding),
  - dynamiques KNN (moyenne des voisins à t-L),
- la logique d’entraînement XGBoost (avec hyperparamètres Optuna),
- le packaging du modèle (`{"model", "feat_cols", "horizon_bins"}`),
- la logique de versioning (lecture du `latest.json` + bump de semver),
- la publication GCS (répertoires datés + alias `latest.*`),
- les fonctions d’inférence :

  - `predict_from_features_df(feats_df, model_uri, horizon_bins=None, model_alias=None)`  
    → prend un DataFrame de features déjà prêts, résout le modèle (local ou GCS),
      applique le modèle, clippe / arrondit en nb de vélos, et renvoie un DataFrame
      standardisé avec `station_id`, `tbin_latest`, `horizon_min`, `bikes_pred`,
      `bikes_pred_int`, `capacity_bin`, `pred_ts_utc`, `target_ts_utc`, `model_version`.

  - `predict_latest_offline(src, model_path_or_prefix)`  
    → lit les snapshots 5 min, construit un historique court (60 min),
      dérive les features **spatiales** cohérentes avec le training, et renvoie
      les prédictions pour la dernière observation par station.

  - `train_model(src, horizon_bins=3, out_path="model.joblib", start_date=None, end_date=None, lookback_days=None)`  
    → construit la frame d’entraînement (pile spatiale forcée), split temporel
      train/valid, entraîne XGBoost, logge MAE/RMSE, sauvegarde modèle + JSON
      de version et publie éventuellement sur GCS.

**CLI intégré**

```bash
# Entraînement
python -m service.core.forecast train   --src path/to/daily_or_base   --horizon 3   --start 2025-01-01 --end 2025-01-31   --lookback-days 30

# Prédiction offline
python -m service.core.forecast predict   --src path/to/raw_or_daily   --horizon 3   --model gs://bucket/velib/models/h15
```

---

### 1.2. `time_features.py` (pile temporelle pure)

**Rôle**  
Construire la frame d’entraînement avec la **pile de features temporelles uniquement**.

Fonction principale :

- `build_training_frame(src, start_date=None, end_date=None, horizon_bins=3)`  
  → renvoie `(full_df, X, y, feat_cols)` où :

  - `full_df` : DataFrame enrichi (raw + `y_nb` + toutes les features),
  - `X` : features en `float32`,
  - `y` : cible `y_nb` (float32),
  - `feat_cols` : liste ordonnée des colonnes de X.

Pipeline interne :

1. `_read_many_parquets`  
   → lit un fichier, un dossier ou un glob de Parquet, et garantit la présence
     de toutes les `BASE_COLUMNS` (schéma strict).

2. `_coerce_types`  
   → normalise les types (timestamps UTC naïfs, numerics, strings).

3. `_dedupe_per_bin`  
   → garde le **dernier** enregistrement par `(station_id, tbin_utc)`.

4. `_add_target_and_lags`  
   → ajoute :
   - `y_nb` à +`horizon_bins` 5 minutes,
   - lags `lag_bikes_{L}` pour L dans (1,2,3,6,12,24,36,48),
   - rollings `roll_mean_{W}`, `roll_std_{W}` sur (3,6,12,24,36,48),
   - `occ_ratio`, lags météo (L=1),
   - features calendaires (via `cal_features.add_time_features`),
   - `status_code` (encodage ordinal de `status`).

**Utilisation typique**

Le training via `forecast.train_model` utilise une version “fusionnée” des
features temporelles et spatiales (voir `spatial_features.py`), mais
`time_features.py` reste réutilisable pour des modèles “simplement temporels”
ou des analyses exploratoires.

---

### 1.3. `spatial_features.py` (pile temporelle + spatiale)

**Rôle**  
Étendre la pile temporelle avec des **features spatiales** :

- statiques :
  - distance au centre de Paris (`dist_center_km`),
  - grille spatiale (`grid_x`, `grid_y` ~300m),
  - cluster KMeans sur (`lat`, `lon`, `capacity`),
  - target encoding `te_cap_mean` (capacité moyenne par cluster),
- dynamiques KNN :
  - pour chaque station, k voisins (par défaut 5),
  - moyennes de `bikes` des voisins à t-L (L=1,3,12 bins → 5,15,60 minutes),
    agrégées dans `knn_mean_bikes_lag{L}`.

Fonctions principales :

- `build_station_static(df, center=(48.853, 2.349), grid_m=300, k_clusters=30, random_state=42)`  
  → 1 ligne par station avec toutes les features spatiales statiques.

- `compute_knn_map(stations_df, k=5)`  
  → DataFrame “station → voisins” :
    `station_id`, `rank`, `neighbor_id`, `dist_km`.

- `_add_features_spatial(df, horizon_bins)`  
  → assemble les features temporelles + spatiales (internes à ce module).

- `build_training_frame_spatial(src, start_date=None, end_date=None, horizon_bins=3)`  
  → analogue à `build_training_frame` de `time_features`, mais avec la pile
    **complète** (temporal + spatial + KNN).

Ce module est utilisé par `forecast.train_model` pour construire la frame
d’entraînement “production”.

---

### 1.4. `cal_features.py`

**Rôle**  
Centraliser les features calendaires dérivées du timestamp `tbin_utc` :

- heure, minute, jour de semaine, mois,
- week-end ou non,
- encodage sin/cos sur 24h et 7j (`hod_sin`, `hod_cos`, `dow_sin`, `dow_cos`),
- proxy “Paris” (heure locale, week-end) si nécessaire.

Fonction principale :

- `add_time_features(df, ts_col="tbin_utc", add_paris_derived=True)`  
  → ajoute toutes les colonnes calendaires à un DataFrame.

Ce module est importé par `time_features.py`, `spatial_features.py` et
`forecast.py`.

---

### 1.5. `monitoring_pipeline.py`

**Rôle**  
Orchestrateur “meta” de la génération des artefacts Monitoring.

Ce module :

- définit une liste de **steps** (build_data_health, build_data_drift, etc.),
- sait lesquelles activer à partir de la variable d’environnement `STEPS`
  (liste CSV ou pattern),
- gère les erreurs partielles (continuer ou non selon `CONTINUE_ON_ERROR`),
- prépare le contexte commun (fenêtres temporelles, horizons, chemins GCS),
- appelle les jobs de `service/jobs` correspondants, typiquement via
  `subprocess` ou via des fonctions Python directes selon la version.

**ENV typiques**

- `GCS_EXPORTS_PREFIX`, `GCS_MONITORING_PREFIX`
- `MON_LAST_DAYS`, `MON_REF_DAYS`, `MON_HORIZONS`
- `STEPS` (ex : `"health,drift,perf,explain,network,intro"`)
- `CONTINUE_ON_ERROR` = `1|0`
- `DRY_RUN` = `1|0` (n’exécute pas, loggue seulement ce qui serait fait)

Ce module est généralement appelé via **Cloud Run** :

```bash
JOB=monitoring_pipeline GCS_EXPORTS_PREFIX=gs://.../velib/exports GCS_MONITORING_PREFIX=gs://.../velib/monitoring MON_LAST_DAYS=14 MON_REF_DAYS=28 python -m service.core.utils_gcs
```

---

### 1.6. `utils_gcs.py`

**Rôle**  
Launcher générique pour les jobs Vélib’ côté Cloud Run / GCS, avec :

- résolution du bon module à lancer selon `JOB`,
- calcul automatique de `DAY` pour les jobs de compaction,
- gestion d’un **verrou GCS optionnel** pour éviter les exécutions concurrentes,
- mode `DRY_RUN` pour inspecter les commandes sans exécution,
- echo d’un sous-ensemble de variables d’environnement utiles pour le debug.

Fonctions / concepts clés :

- `parse_gs(uri)`  
  → découpe `gs://bucket/path` en `(bucket, path)`.

- `lock_blob(client, lock_uri)`  
  → créé un blob vide avec `if_generation_match=0` pour simuler un verrou;
    si l’objet existe déjà, le lock est considéré comme occupé.

- `_dispatch_command(job)`  
  → mappe `JOB` à un module Python :
    - `ingest`                       → `service.jobs.ingest`
    - `compact_daily`                → `service.jobs.compact_daily`
    - `compact_monthly`              → `service.jobs.compact_monthly`
    - `build_serving_forecast`       → `service.jobs.build_serving_forecast`
    - `train_model`                  → `service.jobs.train_model`
    - `export_training_base`         → `service.jobs.export_training_base`
    - `export_training_base_to_gcs`  → `service.jobs.export_training_base_to_gcs`
    - `monitoring` / `monitoring_pipeline`
                                     → `service.core.monitoring_pipeline`

- `_maybe_set_day_env(job)`  
  → si `JOB` est `compact_daily` ou `compact_monthly` et que `DAY` n’est
    pas défini, calcule automatiquement :
    - soit via `DAY_OFFSET` (ex : `-1` → veille),
    - soit `DAY = today_utc - 1` par défaut.

- `main()`  
  → point d’entrée :
    1. crée le client GCS
    2. calcule `DAY` si besoin
    3. tente d’acquérir le lock (si `GCS_LOCK` défini)
    4. construit la commande
    5. affiche quelques ENV
    6. exécute ou non selon `DRY_RUN`
    7. libère le lock si acquis.

**ENV principaux**

- `JOB`               : nom du job à lancer.
- `PYTHON_BIN`        : binaire Python (def. `python`).
- `GCS_LOCK`          : URI GCS du fichier de verrou.
- `DRY_RUN`           : `"1"` → n’exécute pas la commande.

---

## 2. Contrat avec `service/jobs` et l’API

- Les **jobs** (`service/jobs/*.py`) ne re-implémentent pas la logique métier :
  ils s’appuient sur les fonctions de `service/core` pour :
  - construire les features (time + spatial),
  - entraîner et versionner les modèles,
  - générer les artefacts de monitoring.
- L’**API** (non décrite ici) consomme les mêmes briques :
  - chargement des modèles packés (`forecast._load_joblib_from_uri`),
  - appel à `predict_from_features_df` ou `predict_latest_offline`,
  - lecture des JSON “latest” du monitoring.

En pratique :

- `service/core` = brique “coeur” (logique pure, sans scheduling ni infra),
- `service/jobs` = brique “batch” (Cloud Run, GCS, Cron),
- `ui/` (Next.js) = brique “UI” qui consomme les outputs JSON.

---

## 3. Bonnes pratiques pour évoluer le core

- **Stabilité des feature names** :  
  Toute modification de `feat_cols` (time/spatial) doit être réfléchie
  car elle impacte :
  - le training,
  - l’inférence,
  - le monitoring (distributions, drift, perf).

- **Versioning modèles** :  
  Utiliser les helpers de `forecast.py` (`_compute_next_version`, `_write_version_json`,
  `_maybe_publish_to_gcs`) pour conserver un historique propre et un alias
  `latest.*` exploitable côté serving.

- **Isolation infra vs métier** :  
  Garder tout ce qui est lié à GCP / Cloud Run / GCS spécifique dans
  `utils_gcs.py` ou dans `service/jobs`, pas dans le coeur des features.

- **Tests / notebooks** :  
  Les notebooks de R&D peuvent appeler directement `build_training_frame_*`
  ou `forecast.train_model` en local, en pointant sur un dossier Parquet
  ou un répertoire monté depuis GCS.
