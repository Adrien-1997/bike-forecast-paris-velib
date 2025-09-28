FROM python:3.11-slim

# ---- Environment ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Paris \
    PYTHONPATH=/app

# ---- System deps ----
# httpfs (DuckDB) + TLS + pyarrow ont besoin de certifs + libcurl + libssl + libstdc++
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata ca-certificates curl openssl libssl3 libstdc++6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python deps ----
COPY requirements-pipeline.txt .
RUN pip install --no-cache-dir -r requirements-pipeline.txt

# ---- Project code ----
COPY . .

# ---- Security: non-root user ----
RUN useradd -ms /bin/bash appuser
USER appuser

# ---- Entrypoint ----
# JOB est fourni en variable d'env (ex: ingest, build_daily, etc.)
ENTRYPOINT ["python", "-m", "tools.gcs_job"]
