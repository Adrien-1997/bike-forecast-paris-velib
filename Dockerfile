# ---- Base ----
FROM python:3.11-slim

# ---- System deps (SSL, TZ, debug utils) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata curl \
 && rm -rf /var/lib/apt/lists/*

# ---- Env ----
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Paris \
    WEATHER_TIMEOUT=25 \
    NO_SSL_VERIFY=0

WORKDIR /app

# ---- Python deps ----
COPY requirements-pipeline.txt /app/requirements-pipeline.txt
RUN python -m pip install -U pip \
 && pip install --no-cache-dir -r requirements-pipeline.txt \
 && pip install --no-cache-dir huggingface_hub

# ---- Code ----
COPY . /app

# ---- User non-root ----
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# ---- Command (Cloud Run Job) ----
# Si besoin, ajoute `sleep 60 &&` juste après bash -lc pour laisser la source publier.
CMD bash -lc '\
  export PYTHONPATH=/app && \
  echo "[job] start $(date -u +%FT%TZ)" && \
  echo "[job] run src.ingest" && python -m src.ingest && \
  echo "[job] run src.aggregate" && python -m src.aggregate && \
  echo "[job] run tools/push_hf.py" && python tools/push_hf.py \
'
