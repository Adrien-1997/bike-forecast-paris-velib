# apps/api/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Europe/Paris \
    PYTHONPATH=/app

WORKDIR /app

# deps
COPY apps/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# code
COPY apps/api/ /app/api/

# Cloud Run listens on $PORT; default 8080
ENV PORT=8080
CMD exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT}
