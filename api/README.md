# Vélib’ Forecast — API (FastAPI)

Backend HTTP de l’application Vélib’ Forecast, exposant :
- les endpoints “produit” (prévisions, stations, météo…),
- les endpoints de monitoring (overview réseau, stations, dynamique, performance modèle, explicabilité, data health, data drift…),
- et quelques routes techniques (health, badges, snapshot live).

Tout le code de l’API vit dans ce répertoire `api`.

---

## 1. Structure du dossier

```text
api/
├─ app.py                     # Entrypoint FastAPI (montage des routers, CORS, token global…)
├─ Dockerfile                 # Image Docker de l’API (python:3.11-slim + Uvicorn)
├─ requirements.txt           # Dépendances de l’API
├─ api-env.yaml               # Exemple de configuration d’ENV (sans secrets)
├─ core/
│   ├─ settings.py            # Settings Pydantic (GCS, météo, monitoring, CORS…)
│   ├─ gcs.py                 # Utilitaires GCS (download JSON/parquet, etc.)
│   └─ forecast_reader.py     # Lecture des bundles de forecast JSON pour les routes
├─ routes/
│   ├─ health.py              # /healthz (liveness / readiness basique)
│   ├─ stations.py            # /stations (métadonnées et KPI stations)
│   ├─ forecast.py            # /forecast, /forecast/latest (prévisions par station/horizon)
│   ├─ history.py             # /history (historique de prévisions)
│   ├─ badges.py              # /badges (infos pour badges dans l’UI)
│   ├─ snapshot.py            # /snapshot (état réseau “presque live” côté GCS)
│   ├─ snapshot_live.py       # /snapshot/live (appel direct Vélib’ + météo)
│   ├─ weather.py             # /weather (météo côté monitoring)
│   ├─ weather_live.py        # /weather/live (météo live Open-Meteo)
│   ├─ data_health.py         # /monitoring/data/health (JSON produits par build_data_health.py)
│   ├─ data_drift.py          # /monitoring/data/drift (JSON produits par build_data_drift.py)
│   ├─ data_freshness.py      # /monitoring/data/freshness (fraîcheur des données)
│   ├─ intro.py               # /monitoring/intro (document de synthèse pour la page Monitoring)
│   ├─ model_performance.py   # /monitoring/model/performance (KPIs, lift, résidus, etc.)
│   ├─ model_explainability.py# /monitoring/model/explainability (SHAP / résidus / calibration…)
│   ├─ network_overview.py    # /monitoring/network/overview (KPIs globaux, distributions, cartes…)
│   ├─ network_dynamics.py    # /monitoring/network/dynamics (épisodes de tension, profils horaires…)
│   ├─ network_stations.py    # /monitoring/network/stations (PCA, clusters, stats stations)
│   └─ badges.py              # Badges pour l’UI (réutilisé dans plusieurs pages)
└─ schemas/
    └─ forecast.py            # Schémas Pydantic pour les réponses de forecast
```

Les routes de monitoring sont toutes **JSON-only** : l’API ne recalcule rien,
elle proxifie simplement les artefacts produits par les jobs `service/jobs/*`
et stockés sur GCS.

---

## 2. Configuration & variables d’environnement

L’API est entièrement paramétrée par des variables d’environnement Pydantic
(voir `core/settings.py`). La configuration de référence est décrite dans :

- `api/api-env.yaml` : fichier **documenté** sans secrets, utilisable comme
  base pour :
  - un `.env` local,
  - ou un bloc “Variables d’environnement” dans Cloud Run.

Points clés :

- **GCS_MONITORING_PREFIX**  
  Racine des JSON de monitoring, ex.  
  `gs://velib-forecast-472820_cloudbuild/velib/monitoring`

- **GCS_SERVING_PREFIX / SERVING_FORECAST_PREFIX**  
  Racine des bundles de prévision (`latest_forecast.json`, `latest_h{h}.json`).

- **GCS_MODEL_URI**  
  URI par défaut du modèle de forecast (utilisé pour certaines routes legacy).

- **OPENMETEO_* / METEO_DISABLE**  
  Configuration des appels Open-Meteo (météo live, timeouts, désactivation).

- **CORS_ORIGINS**  
  Liste (JSON ou CSV) des origines autorisées pour les appels front (Netlify,
  domaine custom, localhost…).

- **API_GLOBAL_TOKEN**  
  Si présent, toutes les routes (sauf `/` et `/healthz`) exigent un header  
  `Authorization: Bearer <token>`.

---

## 3. Lancer l’API en local

### 3.1. Avec Python directement

Depuis la racine du repo :

```bash
cd api
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Copier/adapter api-env.yaml vers .env (ou définir les variables à la main)
cp api-env.yaml .env  # puis éditer si besoin

# Lancer l’API
python -m uvicorn app:app --reload --port 5050
```

L’API sera accessible sur `http://localhost:5050`  
La documentation interactive sera disponible sur `http://localhost:5050/docs`.

### 3.2. Avec Docker

```bash
cd api
docker build -t velib-api .
docker run --rm -p 8080:8080 --env-file api-env.yaml velib-api
```

> Note : adapter l’option `--env-file` selon votre manière d’injecter les
> variables (env-file Docker, variables directes, secrets manager…).

---

## 4. Endpoints principaux

Quelques endpoints utiles en phase de dev :

- **Health / status**
  - `GET /healthz` : ping simple de l’API (liveness).
- **Stations**
  - `GET /stations` : stations + métadonnées + quelques KPI.
- **Forecast**
  - `GET /forecast/latest` : dernière prévision disponible (bundle JSON).
  - `GET /forecast` : accès plus fin par station/horizon selon l’implémentation.
- **Monitoring (JSON-only, consommé par l’UI)**
  - `GET /monitoring/network/overview/...`
  - `GET /monitoring/network/dynamics/...`
  - `GET /monitoring/network/stations/...`
  - `GET /monitoring/model/performance/...`
  - `GET /monitoring/model/explainability/...`
  - `GET /monitoring/data/health/...`
  - `GET /monitoring/data/drift/...`
  - `GET /monitoring/data/freshness/...`
  - `GET /monitoring/intro` (document de synthèse).

Tous ces endpoints lisent les artefacts JSON générés par les jobs `service/jobs`
(et décrits dans le README du dossier `service/`).

---

## 5. Sécurité & CORS

- **Token global**  
  - Env : `API_GLOBAL_TOKEN`
  - Si défini, tout appel (hors `/` et `/healthz`) doit inclure :  
    `Authorization: Bearer <token>`

- **CORS**  
  - Configuré par `settings.cors_list` (variable `CORS_ORIGINS`).
  - Si vide, l’API autorise `*` en origines (raisonnable si le token global
    protège l’API).

---

## 6. Lien avec le backend jobs (`service/`)

Cette API ne fait **aucun calcul lourd** :

- ingestion, feature engineering, entraînement du modèle,
- construction des artefacts de monitoring (overview, dynamics, stations, perf, explainability…),
- génération des bundles de forecast JSON,

sont tous gérés par les jobs Python du dossier `service/` (déployés sur Cloud Run jobs).

L’API se contente de :
- lire les JSON / Parquet dans GCS,
- les adapter (sanitisation des NaN/Inf),
- les exposer au frontend via des routes propres, versionnées et documentées.

---

Ce README est en cours de rédaction et pourra être complété au fur et à mesure
(ajout d’exemples de payloads, de diagrammes d’architecture, etc.).
