FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata curl \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Paris \
    WEATHER_TIMEOUT=25 \
    NO_SSL_VERIFY=0 \
    PYTHONPATH=/app

WORKDIR /app

# Déps
COPY requirements-pipeline.txt /app/requirements-pipeline.txt
RUN python -m pip install -U pip \
 && pip install --no-cache-dir -r requirements-pipeline.txt \
 && pip install --no-cache-dir huggingface_hub

# 🔒 Copie explicite (évite les effets du .dockerignore)
COPY src/ /app/src/
COPY tools/ /app/tools/

# S'assure que src est un package
RUN [ -f /app/src/__init__.py ] || printf "" > /app/src/__init__.py

# (Optionnel) petit check build-time pour tracer le contenu
RUN echo "[build] tree /app/src:" && ls -la /app/src || true

# Non-root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Run: modules sous src + script tools
CMD bash -lc '\
  echo "[job] start $(date -u +%FT%TZ)" && \
  echo "[job] check import src.weather" && python - << "PY"\nimport importlib; import sys\nm=importlib.import_module("src.weather"); print("[check] found:", m.__file__)\nPY\n && \
  echo "[job] run src.ingest" && python -m src.ingest && \
  echo "[job] run src.aggregate" && python -m src.aggregate && \
  echo "[job] run tools/push_hf.py" && python tools/push_hf.py \
'
