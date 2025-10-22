# apps/api/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Europe/Paris

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends tzdata && rm -rf /var/lib/apt/lists/*

# deps
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# code (flat: app.py + routes/)
COPY . /app

ENV PORT=8080
EXPOSE 8080

CMD ["sh","-c","python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
