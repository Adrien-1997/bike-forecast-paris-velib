FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata curl \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Paris \
    PYTHONPATH=/app \
    PUSH_SRC=exports/velib.parquet \
    PUSH_DEST=exports/velib.parquet

WORKDIR /app

COPY requirements-pipeline.txt .
RUN python -m pip install -U pip \
 && pip install --no-cache-dir -r requirements-pipeline.txt \
 && pip install --no-cache-dir huggingface_hub

COPY src/ /app/src/
COPY tools/ /app/tools/
RUN [ -f /app/src/__init__.py ] || printf "" > /app/src/__init__.py

USER nobody

CMD bash -lc '\
  echo "[job] start $(date -u +%FT%TZ)"; \
  mkdir -p exports; \
  python -m src.ingest; \
  python -m src.aggregate; \
  # on force la sortie dans exports/velib.parquet
  if [ -f docs/exports/velib.parquet ]; then cp docs/exports/velib.parquet exports/velib.parquet; fi; \
  ls -lh exports; \
  python tools/push_hf.py \
'
