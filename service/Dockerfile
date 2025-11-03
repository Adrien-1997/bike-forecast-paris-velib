# =============================================================================
# Vélib’ Forecast — Jobs / Tools image (clean final)
# =============================================================================
FROM python:3.11-slim

# -----------------------------------------------------------------------------
# Base environment
# -----------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Europe/Paris \
    PYTHONPATH=/app

# -----------------------------------------------------------------------------
# System dependencies (minimal)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      tzdata ca-certificates curl openssl libssl3 libstdc++6 libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Non-root user + working dir
# -----------------------------------------------------------------------------
RUN useradd -ms /bin/bash appuser
WORKDIR /app

# -----------------------------------------------------------------------------
# Python dependencies
# -----------------------------------------------------------------------------
# build context = racine, requirements.txt est dans service/
COPY service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Project code
# -----------------------------------------------------------------------------
COPY . .

# -----------------------------------------------------------------------------
# Writable dirs
# -----------------------------------------------------------------------------
RUN mkdir -p /app/exports && chown -R appuser:appuser /app

# -----------------------------------------------------------------------------
# Defaults (override at deploy)
# -----------------------------------------------------------------------------
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
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import json,google.cloud.storage,pyarrow,xgboost; print(json.dumps({'ok':True}))" || exit 1

# -----------------------------------------------------------------------------
# Security: non-root user
# -----------------------------------------------------------------------------
USER appuser

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
ENTRYPOINT ["python"]

CMD ["-m", "core.utils_gcs"]
