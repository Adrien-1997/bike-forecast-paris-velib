FROM python:3.11-slim

# 1) System deps (certifs pour HTTPS)
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Déps Python (couche cacheable)
COPY requirements-pipeline.txt .
RUN python -m pip install -U pip \
 && pip install --no-cache-dir -r requirements-pipeline.txt

# 3) Code
COPY . .

# 4) Runtime envs par défaut
#    - WITH_WEATHER=0 : plus rapide; mets 1 dans Cloud Run si tu veux la météo
#    - PYTHONUNBUFFERED pour logs immédiats
ENV WITH_WEATHER=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 5) Entrypoint unique : ingest -> aggregate -> export shard -> push HF
CMD ["python", "-m", "src.aggregate"]
