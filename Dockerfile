# ---- Base image ----
FROM python:3.11-slim

# ---- System deps (SSL, TZ, curl pour debug) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata curl \
 && rm -rf /var/lib/apt/lists/*

# ---- Runtime env ----
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Paris \
    # Variables météo par défaut (overridables dans Cloud Run)
    WEATHER_TIMEOUT=25 \
    NO_SSL_VERIFY=0

# ---- Workdir ----
WORKDIR /app

# ---- Python deps ----
# Si tu utilises requirements-pipeline.txt, on l'installe. Sinon, passe à requirements.txt.
COPY requirements-pipeline.txt /app/requirements-pipeline.txt
RUN python -m pip install -U pip \
 && pip install --no-cache-dir -r requirements-pipeline.txt \
 && pip install --no-cache-dir huggingface_hub

# ---- App code ----
COPY . /app

# ---- Non-root user (sécurité) ----
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# ---- Commande (Job) ----
# Optionnel: ajoute 'sleep 60' si tu veux laisser la source publier avant l’ingest.
# Ici on log bien chaque étape, et on force PYTHONPATH=/app pour les imports internes.
CMD bash -lc '\
  echo "[job] start at $(date -u +"%Y-%m-%dT%H:%M:%SZ")" && \
  export PYTHONPATH=/app && \
  echo "[job] running ingest.py" && python ingest.py && \
  echo "[job] running aggregate.py" && python aggregate.py && \
  echo "[job] running push_hf.py" && python push_hf.py \
'
