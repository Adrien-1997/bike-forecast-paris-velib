# Vélib' Forecast Paris — Plateforme de prévision et de monitoring

Ce dépôt regroupe **l’ensemble du projet Vélib' Forecast Paris / velo-paris.fr** :  
pipeline de données, entraînement des modèles, API, interface web et monitoring de la qualité / performance.

L’objectif : fournir une **plateforme de prévision de disponibilité Vélib’** (stations, vélos, places disponibles)
avec un **pipeline reproductible**, un **suivi de la qualité des données** et un **monitoring temps quasi-réel** du réseau et des modèles.

---

## 1. Vue d’ensemble

### 1.1. Fonctionnalités principales

- Ingestion régulière des snapshots Vélib' (format GBFS).
- Compactage et préparation des jeux de données quotidiens.
- Construction de jeux de données d’entraînement et d’évaluation.
- Entraînement et déploiement de modèles de prévision (XGBoost, horizons 15/60 min, etc.).
- Exposition d’une **API FastAPI** pour la carte temps réel et le monitoring.
- Interface web **Next.js / React** pour :
  - la carte utilisateur (`/app`),
  - le monitoring des données / modèles / réseau (`/monitoring`).
- Jobs de monitoring pour :
  - Santé des données (complétude, fraîcheur, schéma),
  - Dérive des données (PSI, statistiques),
  - Performance des modèles (MAE, lift vs baseline),
  - Dynamiques réseau (pénurie / saturation, cartes, heatmaps).

### 1.2. Architecture globale (haut niveau)

```text
+--------------------+         +------------------------+
|  API Vélib' GBFS   |         |  API météo (ex. Open-  |
|  + historiques     |         |  Meteo / autre)        |
+---------+----------+         +-----------+------------+
          |                                |
          v                                v
  [ Jobs ingestion ]                [ Jobs météo ]
          |                                |
          +----------->  GCS Buckets  <-----+
                        (raw / daily / exports / monitoring)
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
                 +-------------+-------------+
                 |                           |
                 v                           v
          API FastAPI (Cloud Run)      Jobs Monitoring
         (stations, prévisions,      (health, drift,
          endpoints monitoring)       performance, réseau)
                 |                           |
                 +-------------+-------------+
                               v
                      UI Next.js / React
                  (velo-paris.fr / monitoring)
```

---

## 2. Structure du dépôt

> Les chemins exacts peuvent évoluer, mais l’esprit général du découpage est le suivant.

```text
.
├── apps/
│   └── api/                  # API FastAPI (endpoints cartes & monitoring)
│
├── service/
│   └── jobs/                 # Jobs batch (ingest, compact_daily, build_datasets,
│                             #             monitoring/data, monitoring/model, etc.)
│
├── ui/                       # Interface web Next.js (site public + monitoring)
│   ├── pages/                # Pages Next.js (/, /app, /monitoring/...)
│   ├── components/           # Composants UI (GlobalHeader, MonitoringNav, KpiBar, ...)
│   ├── lib/                  # Services HTTP, helpers Plotly, index stations, etc.
│   ├── public/               # Assets statiques (favicon, data/stations.index.json, ...)
│   ├── styles/               # CSS globaux + CSS contextuels (landing, app, monitoring)
│   └── ...                   # Config Next, types, scripts, etc.
│
├── scripts/                  # Scripts utilitaires (ex: buildStationsIndex.ts, etc.)
│
├── infra/                    # Scripts & fichiers d’infrastructure (GCP, Terraform, ...)
│                             # → Dossier aujourd’hui ignoré par .gitignore
│
├── docs/                     # Documentation complémentaire / assets (optionnel)
│
├── Dockerfile                # Image principale (API / jobs ou multi-stage selon usage)
├── netlify.toml              # Configuration Netlify pour le build / déploiement UI
├── bucket-lifecycle.json     # (optionnel) Gestion du cycle de vie GCS
├── .gcloudignore             # Fichiers ignorés lors des déploiements gcloud
├── .gitignore                # Ignorés Git (données lourdes, env, node_modules, ...)
├── package.json              # Dépendances JS/TS (UI + scripts)
├── tsconfig.json             # Config TypeScript
└── README.md                 # Ce fichier
```

---

## 3. Prérequis

### 3.1. Outils

- **Python** ≥ 3.11 (pour les jobs et l'API)
- **Node.js** ≥ 20 (UI Next.js)
- **npm** ou **pnpm** (gestion des dépendances front)
- **Docker** (build images, exécution Cloud Run en local si besoin)
- **gcloud CLI** (déploiements GCP)
- **Make** (si des `Makefile` sont présents, optionnel mais pratique)

### 3.2. Comptes & services externes

- Projet **Google Cloud Platform** avec :
  - Cloud Storage (buckets raw / daily / exports / monitoring),
  - Cloud Run (API + jobs),
  - Cloud Scheduler (+ éventuellement Pub/Sub),
  - Cloud Build / Artifact Registry (optionnel).  
- Compte pour l’hébergement de l’UI :
  - **Netlify** (config via `netlify.toml`) ou
  - **Cloud Run / autre** si l’UI est déployée côté GCP.

---

## 4. Variables d’environnement (aperçu)

> Les fichiers `.env`, `.env.*` et `.env.local` sont **ignorés** par Git.  
> Les noms ci-dessous sont un exemple représentatif de la configuration réelle.

### 4.1. Backend / jobs (Python)

| Variable                         | Rôle                                                                        |
| -------------------------------- | ---------------------------------------------------------------------------- |
| `GCP_PROJECT`                    | ID du projet GCP                                                            |
| `GCS_BUCKET`                     | Bucket principal (si préfixes dérivés)                                      |
| `GCS_RAW_PREFIX`                 | Préfixe GCS des snapshots bruts Vélib'                                     |
| `GCS_DAILY_PREFIX`               | Préfixe GCS des fichiers compactés jour (`compact_YYYY-MM-DD.parquet`)     |
| `GCS_EXPORTS_PREFIX`             | Préfixe GCS des exports (events/perf, etc.)                                 |
| `MONITORING_BASE`                | Préfixe GCS racine pour les JSON de monitoring                             |
| `FORECAST_HORIZONS`              | Liste d’horizons de prévision (ex: `15,60`)                                 |
| `MODEL_URI_15`, `MODEL_URI_60`   | URI GCS vers les modèles (fichiers `.joblib` ou préfixes)                   |
| `VELIB_API_URL`                  | URL base de l’API Vélib' (GBFS)                                            |
| `WEATHER_API_URL`                | URL base de la source météo                                                |
| `WEATHER_API_KEY`                | Clé API météo (si nécessaire)                                              |

### 4.2. API FastAPI

| Variable                 | Rôle                              |
| ------------------------ | --------------------------------- |
| `API_ENV`               | Environnement (`local`, `prod`, …) |
| `API_ROOT_PATH`         | Root path éventuel derrière un proxy |
| `GCS_MONITORING_BASE`   | Préfixe GCS lu par l’API pour servir les JSON de monitoring |
| `CORS_ORIGINS`          | Origines autorisées (UI, etc.)    |

### 4.3. UI Next.js

| Variable                             | Rôle                                                   |
| ------------------------------------ | ------------------------------------------------------ |
| `NEXT_PUBLIC_API_BASE_URL`           | Base URL de l’API (utilisée côté navigateur)          |
| `NEXT_PUBLIC_MONITORING_BASE_PATH`   | Préfixe des routes monitoring (si proxifiées)         |
| `NEXT_PUBLIC_MAPBOX_TOKEN` (ex.)     | Token éventuel pour un fond de carte tiers            |

> Adapter ces noms à la configuration réelle du projet.  
> Les `.env.local` et `.env.production` ont priorité dans la config Next.js.

---

## 5. Démarrage rapide (local)

### 5.1. Cloner le dépôt

```bash
git clone https://github.com/<user>/velib-forecast.git
cd velib-forecast
```

### 5.2. Configurer les environnements

1. Créer vos fichiers d’environnement à partir des exemples (si présents) :

   ```bash
   cp .env.example .env
   cp ui/.env.example ui/.env.local
   ```

   Puis éditer les valeurs (GCP, API, etc.).

2. Vérifier que les chemins GCS utilisés dans les jobs pointent vers vos buckets.

### 5.3. Installer et lancer l’UI

```bash
cd ui
npm install
npm run dev
```

L’interface est disponible sur `http://localhost:3000` :

- `/` : landing / présentation
- `/app` : carte utilisateur (réseau Vélib’)
- `/monitoring` : hub Monitoring (network / data / model)

### 5.4. Lancer l’API (mode développement)

Exemple avec Uvicorn (adapter au chemin réel du module FastAPI) :

```bash
cd apps/api
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

L’UI doit pointer vers `http://localhost:8000` via `NEXT_PUBLIC_API_BASE_URL`.

---

## 6. Pipeline de données (jobs)

> Les noms exacts des fichiers peuvent varier (ex: `service/jobs/*.py`), mais la logique reste identique.

### 6.1. Ingestion & compactage

- **`ingest_snapshots.py`**  
  - Appelé toutes les X minutes (Cloud Scheduler).  
  - Récupère les flux GBFS Vélib’ (stations status, stations information).  
  - Écrit des fichiers bruts horodatés dans `GCS_RAW_PREFIX`.

- **`compact_daily.py`**  
  - Prend les snapshots bruts d’un jour donné.  
  - Produit `compact_YYYY-MM-DD.parquet` dans `GCS_DAILY_PREFIX`.  
  - Nettoie, convertit en schéma standard (stations, bikes, docks, etc.).

### 6.2. Construction des jeux d’entraînement

- **`build_datasets.py`**  
  - Entrée : `GCS_DAILY_PREFIX/compact_YYYY-MM-DD.parquet`.  
  - Sorties :  
    - `exports/events.parquet` (+ version datée)  
    - `exports/perf.parquet` (+ version datée)  
  - Ajoute les features (temporalité, occupation, contexte spatial, etc.).  
  - Définit les target `y_true` pour différents horizons.

### 6.3. Entraînement & modèles

- **`train_model_*` / `build_model.py`** (selon la structure du repo) :  
  - Utilise les données dans `exports/`.  
  - Entraîne un modèle par horizon (ex: 15, 60 min).  
  - Sauvegarde les modèles dans `gs://.../models/h{H}/latest.joblib`.

### 6.4. Monitoring (santé, dérive, performance, réseau)

- **`build_data_health.py`**  
  - Analyse complétude, fraîcheur, schéma, valeurs aberrantes.  
  - Écrit des JSON sous `MONITORING_BASE/data/health/...`.

- **`build_data_drift.py`**  
  - Calcule le PSI global et par variable, KS, deltas de moyennes/variances.  
  - Écrit sous `MONITORING_BASE/data/drift/...`.

- **`build_model_performance.py`**  
  - Lit `perf_YYYY-MM-DD.parquet`.  
  - Calcule MAE, RMSE, lift vs baseline, distributions, coupes (heure, jour, station).  
  - Écrit sous `MONITORING_BASE/model/performance/...`.

- **`build_network_overview.py` / `build_network_dynamics.py`**  
  - Agrège les états de pénurie/saturation.  
  - Produit les cartes, heatmaps 7×24, profils, épisodes et tensions stations.  
  - Écrit sous `MONITORING_BASE/network/...`.

Tout ce contenu est consommé par la section `/monitoring` de l’UI.

---

## 7. Monitoring (UI)

L’interface de monitoring (Next.js) expose plusieurs pages, toutes **en lecture seule** :

- `/monitoring` : vue d’ensemble (intro, statuts systèmes, KPIs globaux).
- `/monitoring/network/overview` : snapshot réseau, carte, courbes de dispo, stations en tension.
- `/monitoring/network/stations` : liste complète des stations, clusters, filtres (non cliquable vers la carte publique).
- `/monitoring/network/dynamics` : heatmaps 7×24, profils par jour, épisodes et tension par station.
- `/monitoring/model/performance` : lift vs baseline, MAE, découpe par heure/jour, stations top/bottom lift.
- `/monitoring/model/explainability` : importance des features, SHAP / dérivés (selon implémentation).  
- `/monitoring/data/health` : complétude, fraîcheur, anomalies structurales.  
- `/monitoring/data/drift` : PSI global & par variable, zones de dérive.

L’UI consomme les JSON exposés par l’API ou directement par un endpoint proxy, en gérant les ETags.
Les styles spécifiques sont chargés via `monitoring.css`, `monitoringnav.css`, `kpibar.css`, etc.

---

## 8. Déploiement

### 8.1. API & Jobs (GCP Cloud Run)

1. Construire l’image Docker (exemple) :

   ```bash
   gcloud builds submit --tag gcr.io/$GCP_PROJECT/velib-forecast-api
   ```

2. Déployer sur Cloud Run :

   ```bash
   gcloud run deploy velib-forecast-api          --image gcr.io/$GCP_PROJECT/velib-forecast-api          --region europe-west1          --platform managed          --allow-unauthenticated
   ```

3. Déployer les jobs batch comme services Cloud Run séparés ou via Cloud Run Jobs,  
   puis les orchestrer avec Cloud Scheduler.

### 8.2. UI (Netlify / autre)

- Netlify lit la configuration dans `netlify.toml` (répertoire `ui/`, commande `npm run build`, etc.).  
- La sortie Next.js (`.next` / `out`) est servie derrière le domaine public (ex: `velo-paris.fr`).  
- L’UI communique avec l’API via `NEXT_PUBLIC_API_BASE_URL`.

---

## 9. Tests & qualité

(Adapter selon ce qui est réellement présent dans le dépôt.)

- Tests Python :

  ```bash
  pytest
  ```

- Linting Python :

  ```bash
  ruff check .
  ```

- Linting TypeScript / Next.js :

  ```bash
  cd ui
  npm run lint
  ```

- Build UI :

  ```bash
  cd ui
  npm run build
  ```

---

## 10. Roadmap (indicative)

- Ajout d’autres horizons de prévision.
- Monitoring des temps de réponse API & erreurs.
- Version publique minimale du dashboard Monitoring (mode démo).
- Tutoriaux / notebooks associés publiés sur Kaggle / docs.

---

## 11. Licence & crédits

- Données Vélib' : voir les conditions d’utilisation de **Vélib' Métropole / Île-de-France Mobilités**.
- Données météo : se référer aux CGU du fournisseur choisi (Open-Meteo, OpenWeather, etc.).
- Code et modèles : licence à préciser (`MIT`, `Apache-2.0`, ou autre, selon ton choix).

N’oublie pas de mettre à jour cette section avec la licence réelle lorsque tu publies le dépôt.
