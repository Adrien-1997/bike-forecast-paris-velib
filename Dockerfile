FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Paris \
    PYTHONPATH=/app

# httpfs de DuckDB + TLS ont besoin de certifs et libcurl
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps (inclure pyarrow + google-cloud-storage)
COPY requirements-pipeline.txt .
RUN pip install --no-cache-dir -r requirements-pipeline.txt

# code
COPY . .

# entrypoint générique (JOB via env : ingest / compact_daily / build_daily / etc.)
ENTRYPOINT ["python", "tools/gcs_job.py"]
