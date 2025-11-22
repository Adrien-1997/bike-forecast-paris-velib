# =============================================================================
# Vélib’ Forecast — Jobs / Tools image
# =============================================================================
# Image générique pour tous les jobs batch Vélib’ Forecast :
#   - ingestion 5 minutes (GBFS + météo),
#   - compaction journalière / mensuelle,
#   - export des bases d’entraînement,
#   - entraînement des modèles,
#   - génération des artefacts de monitoring,
#   - batch de prévisions “serving”.
#
# Le choix du job concret se fait via la variable d’environnement JOB et
# le launcher Python `core.utils_gcs` (voir ENTRYPOINT / CMD en bas).
# =============================================================================
FROM python:3.11-slim

# -----------------------------------------------------------------------------
# Base environment
# -----------------------------------------------------------------------------
# - PYTHONDONTWRITEBYTECODE :
#     évite la génération de .pyc (moins de bruit sur le disque).
# - PYTHONUNBUFFERED :
#     logs non bufferisés (stdout/stderr visibles immédiatement dans Cloud Run).
# - PIP_* :
#     désactive le cache et les checks inutiles pour des builds plus propres.
# - TZ :
#     timezone de l’OS ; ici Europe/Paris pour cohérence avec les analyses.
# - PYTHONPATH :
#     on met /app pour que `service/` soit importable partout.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Europe/Paris \
    PYTHONPATH=/app

# -----------------------------------------------------------------------------
# System dependencies (minimal)
# -----------------------------------------------------------------------------
# - tzdata, ca-certificates, curl, openssl :
#     nécessaires pour HTTP/TLS (APIs, GCS, etc.).
# - libssl3, libstdc++6, libgomp1 :
#     runtime pour les libs C++ (pyarrow, xgboost, google-cloud-*…).
# - nettoyage de /var/lib/apt/lists pour limiter la taille de l’image.
RUN apt-get update && apt-get install -y --no-install-recommends \
      tzdata ca-certificates curl openssl libssl3 libstdc++6 libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Non-root user + working dir
# -----------------------------------------------------------------------------
# On crée un utilisateur dédié `appuser` et on travaille sous /app.
RUN useradd -ms /bin/bash appuser
WORKDIR /app

# -----------------------------------------------------------------------------
# Python dependencies
# -----------------------------------------------------------------------------
# On copie uniquement requirements.txt d’abord pour profiter du cache Docker :
# tant que ce fichier ne change pas, la couche pip install est réutilisée.
# NB : requirements.txt est versionné dans service/.
COPY service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Project code
# -----------------------------------------------------------------------------
# On copie tout le repo dans /app (service/, jobs, core, éventuelle API, etc.).
COPY . .

# -----------------------------------------------------------------------------
# Writable dirs
# -----------------------------------------------------------------------------
# /app/exports : répertoire “générique” pour des dumps temporaires / exports.
# On s’assure que appuser possède l’ensemble de /app.
RUN mkdir -p /app/exports && chown -R appuser:appuser /app

# -----------------------------------------------------------------------------
# Defaults (override at deploy)
# -----------------------------------------------------------------------------
# Les ARG définissent des valeurs par défaut au build. Elles sont ensuite
# recopiées dans les ENV ci-dessous, qui peuvent être overridés au runtime.
#
# - GCP_REGION / GCP_PROJECT :
#     infos GCP utilisées par les jobs (pour logs / context).
# - GCS_BUCKET_ROOT :
#     racine GCS unique du projet (bronze, daily, exports, models, monitoring…).
#
# - FORECAST_HORIZONS :
#     horizons de prévision (en minutes) pour les jobs serving/monitoring.
# - MODEL_URI_15 / MODEL_URI_60 :
#     URIs GCS des modèles par horizon (fichier .joblib ou préfixe).
#
# - MON_* :
#     configuration par défaut du monitoring (fenêtres temporelles, seuils).
ARG GCP_REGION="europe-west1"
ARG GCP_PROJECT="velib-forecast-472820"
ARG GCS_BUCKET_ROOT="gs://velib-forecast-472820_cloudbuild/velib"

ARG FORECAST_HORIZONS="15,60"
ARG MODEL_URI_15="${GCS_BUCKET_ROOT}/models/h15/latest.joblib"
ARG MODEL_URI_60="${GCS_BUCKET_ROOT}/models/h60/latest.joblib"

ARG MON_TZ="Europe/Paris"
ARG MON_LAST_DAYS="14"
ARG MON_REF_DAYS="28"
ARG MON_PENURY_THRESH="2"
ARG MON_SATURATION_THRESH="2"

# -----------------------------------------------------------------------------
# Runtime ENV (initialisés depuis les ARG ci-dessus)
# -----------------------------------------------------------------------------
# Ces variables sont celles réellement utilisées par les jobs :
# on peut les overrider facilement dans Cloud Run / Cloud Build.
#
# Racines GCS
#   - GCS_RAW_PREFIX       : snapshots 5 minutes (bronze).
#   - GCS_DAILY_PREFIX     : compacts journaliers.
#   - GCS_SERVING_ROOT     : outputs serving.
#   - GCS_EXPORTS_PREFIX   : events_*.parquet, perf_*.parquet.
#   - GCS_MONITORING_PREFIX: artefacts JSON du monitoring.
#   - GCS_MODELS_PREFIX    : modèles entraînés.
#
# Serving
#   - SERVING_FORECAST_PREFIX :
#       JSON de prévision batch consommés par le front (h15, h60, ...).
#
# Modèles / horizons
#   - FORECAST_HORIZONS, MODEL_URI_15, MODEL_URI_60 :
#       paramètres par défaut pour les jobs serving/monitoring.
#
# Monitoring
#   - MON_TZ, MON_LAST_DAYS, MON_REF_DAYS :
#       fenêtres courante / référence pour les agrégations et le drift.
#   - MON_PENURY_THRESH / MON_SATURATION_THRESH :
#       seuils de pénurie / saturation en nb de vélos.
#
# Divers
#   - TZ_APP :
#       timezone logique côté application (pandas, etc.).
#   - TMP_DIR :
#       répertoire temporaire (utilisé par certains jobs).
ENV GCP_REGION=${GCP_REGION} \
    GCP_PROJECT=${GCP_PROJECT} \
    GCS_BUCKET_ROOT=${GCS_BUCKET_ROOT} \
    GCS_RAW_PREFIX=${GCS_BUCKET_ROOT}/bronze \
    GCS_DAILY_PREFIX=${GCS_BUCKET_ROOT}/daily \
    GCS_SERVING_ROOT=${GCS_BUCKET_ROOT}/serving \
    GCS_EXPORTS_PREFIX=${GCS_BUCKET_ROOT}/exports \
    GCS_MONITORING_PREFIX=${GCS_BUCKET_ROOT}/monitoring \
    GCS_MODELS_PREFIX=${GCS_BUCKET_ROOT}/models \
    SERVING_FORECAST_PREFIX=${GCS_BUCKET_ROOT}/serving/forecast \
    FORECAST_HORIZONS=${FORECAST_HORIZONS} \
    MODEL_URI_15=${MODEL_URI_15} \
    MODEL_URI_60=${MODEL_URI_60} \
    MON_TZ=${MON_TZ} \
    MON_LAST_DAYS=${MON_LAST_DAYS} \
    MON_REF_DAYS=${MON_REF_DAYS} \
    MON_PENURY_THRESH=${MON_PENURY_THRESH} \
    MON_SATURATION_THRESH=${MON_SATURATION_THRESH} \
    TZ_APP=Europe/Paris \
    TMP_DIR=/tmp

# -----------------------------------------------------------------------------
# Healthcheck
# -----------------------------------------------------------------------------
# Healthcheck très simple : on vérifie juste que Python démarre et que les
# libs critiques sont importables (GCS, pyarrow, xgboost). Si l’import échoue,
# le conteneur est marqué unhealthy par Cloud Run.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import json,google.cloud.storage,pyarrow,xgboost; print(json.dumps({'ok':True}))" || exit 1

# -----------------------------------------------------------------------------
# Security: non-root user
# -----------------------------------------------------------------------------
# On repasse explicitement en `appuser` pour toute la phase d’exécution
# (bonne pratique : éviter de tourner en root).
USER appuser

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
# Entrypoint = “python”, la commande par défaut lance le launcher générique
# `core.utils_gcs` qui lira la variable d’environnement JOB pour dispatch
# vers le bon module (ingest, compact_daily, monitoring_pipeline, etc.).
ENTRYPOINT ["python"]

CMD ["-m", "core.utils_gcs"]
