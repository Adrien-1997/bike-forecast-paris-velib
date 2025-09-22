FROM python:3.11-slim

# Deps système
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata curl \
 && rm -rf /var/lib/apt/lists/*

# Env
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Paris \
    WEATHER_TIMEOUT=25 \
    NO_SSL_VERIFY=0 \
    PYTHONPATH=/app

WORKDIR /app

# Dépendances Python
COPY requirements-pipeline.txt /app/requirements-pipeline.txt
RUN python -m pip install -U pip \
 && pip install --no-cache-dir -r requirements-pipeline.txt \
 && pip install --no-cache-dir huggingface_hub

# Copie explicite (évite un .dockerignore trop large)
COPY src/ /app/src/
COPY tools/ /app/tools/

# S'assure que src est un package
RUN [ -f /app/src/__init__.py ] || printf "" > /app/src/__init__.py

# Petit script de check import (au build, pour le run)
RUN printf "import importlib\nm=importlib.import_module('src.weather')\nprint('[check] src.weather:', m.__file__)\n" > /app/tools/check_import.py

# Non-root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Commande job — sans here-doc, enchaînement simple
CMD bash -lc '\
  echo "[job] start $(date -u +%FT%TZ)"; \
  echo "[job] check import"; python tools/check_import.py; \
  echo "[job] run src.ingest"; python -m src.ingest; \
  echo "[job] run src.aggregate"; python -m src.aggregate; \
  echo "[job] run tools/push_hf.py"; python tools/push_hf.py \
'
