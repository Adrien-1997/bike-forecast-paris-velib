FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-pipeline.txt .
RUN python -m pip install -U pip && pip install --no-cache-dir -r requirements-pipeline.txt && pip install --no-cache-dir huggingface_hub
COPY . .
# Tampon 60s → ingest → aggregate
CMD bash -lc "sleep 60 && PYTHONPATH=/app python -m src.ingest && PYTHONPATH=/app python -m src.aggregate && PUSH_SRC=docs/exports/velib.parquet PUSH_DEST=exports/velib.parquet python tools/push_hf.py"
