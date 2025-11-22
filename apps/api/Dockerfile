# apps/api/Dockerfile

# =============================================================================
# Image Docker pour l'API Vélib' Forecast (FastAPI + Uvicorn).
#
# Rôle :
# - Fournir une image légère basée sur python:3.11-slim.
# - Installer les dépendances via requirements.txt.
# - Copier le code de l'API (app.py + routes, core, etc.).
# - Lancer Uvicorn sur le port 8080 (configurable via $PORT).
#
# Ce Dockerfile est prévu pour :
# - un déploiement Cloud Run (ou équivalent),
# - un usage local via `docker run -p 8080:8080 ...`.
# =============================================================================

# Base légère Python 3.11 (Debian slim).
FROM python:3.11-slim

# Variables d'environnement Python :
# - PYTHONDONTWRITEBYTECODE : pas de .pyc sur le disque (inutile en container).
# - PYTHONUNBUFFERED       : logs non bufferisés → visibles en temps réel.
# - PIP_NO_CACHE_DIR       : pas de cache pip pour limiter la taille de l'image.
# - PIP_DISABLE_PIP_VERSION_CHECK : évite le warning pip à chaque install.
# - TZ                      : fuseau horaire (aligné sur Europe/Paris).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Europe/Paris

# Répertoire de travail dans le container.
WORKDIR /app

# Installation minimale de tzdata pour respecter TZ=Europe/Paris.
# On nettoie ensuite les listes apt pour réduire la taille de l'image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

# ───────── Dépendances Python ─────────
# On copie uniquement requirements.txt pour profiter du cache Docker :
# si requirements.txt ne change pas, la couche pip est réutilisée.
COPY requirements.txt ./requirements.txt

# Mise à jour de pip + installation des dépendances.
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# ───────── Code applicatif ─────────
# On copie le reste du code (app.py, core/, routes/, etc.) dans /app.
COPY . /app

# ───────── Exposition HTTP ─────────
# Port par défaut utilisé par l'API (Cloud Run lit aussi $PORT).
ENV PORT=8080
EXPOSE 8080

# ───────── Commande de démarrage ─────────
# On lance Uvicorn avec :
# - module 'app:app' (fichier app.py, variable FastAPI `app`),
# - host 0.0.0.0 pour accepter les connexions externes,
# - port fourni par $PORT (fallback 8080).
CMD ["sh","-c","python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
