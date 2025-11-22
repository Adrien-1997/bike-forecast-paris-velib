# apps/api/app.py

"""Vélib' Forecast — Entrypoint FastAPI.

Ce module construit et configure l’application FastAPI utilisée pour exposer
l’API publique de Vélib' Forecast :

- Chargement des variables d’environnement depuis `.env` (local + racine repo).
- Résolution souple du module `core` pour partager des utilitaires (settings…).
- Chargement des `Settings` Pydantic (chemins GCS, CORS, météo, monitoring…).
- Montage des routers :
    * routes "legacy" : health, stations, forecast, history, badges, snapshot, weather,
    * routes "monitoring" : network overview/dynamics/stations, model perf/explain,
      data health/drift/freshness, intro.
- Middleware global :
    * enforcement d’un token d’accès global (API_GLOBAL_TOKEN),
    * compression GZip des réponses,
    * CORS configuré via les origines autorisées définies dans les settings.
- Hook startup :
    * affichage de toutes les routes montées (path + méthodes) dans les logs.

Ce fichier correspond donc au "main" applicatif côté backend, exploité à la fois
en local (uvicorn) et dans Cloud Run.
"""

from __future__ import annotations
import sys, importlib, os
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

# ─────────── .env ───────────
# On tente d'abord de charger un .env dans le cwd (utile en local),
# puis un .env à la racine du repo (utile en prod/Cloud Run si monté).
try:
    from dotenv import load_dotenv, find_dotenv
    found = find_dotenv(filename=".env", usecwd=True)
    if found:
        load_dotenv(found, override=False)
    repo_root = Path(__file__).resolve().parents[2]
    env_file = repo_root / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)
    print(f"[app] .env chargé ✓ (found='{found}', repo_root='{repo_root}')")
except Exception as e:
    # L’API reste fonctionnelle même si .env n’est pas trouvé : on se repose
    # alors uniquement sur les variables d’environnement déjà présentes.
    print(f"[app] .env non chargé ({e})")

# ─────────── Alias 'core' ───────────
# Permet d'importer `core` de façon homogène dans tout le projet, que l’on soit
# dans un contexte `apps.api.*`, `api.*` ou directement `core.*`.
for mod in ("apps.api.core", "api.core", "core"):
    try:
        sys.modules.setdefault("core", importlib.import_module(mod))
        break
    except ModuleNotFoundError:
        continue

# ─────────── Settings ───────────
# On tente plusieurs chemins pour récupérer l'instance `settings` (Pydantic),
# afin que le fichier reste robuste aux variations de layout/imports.
try:
    from apps.api.core.settings import settings
except Exception:
    try:
        from .core.settings import settings
    except Exception:
        try:
            from core.settings import settings
        except Exception:
            # En dernier recours, `settings` vaut None : l’API reste utilisable
            # mais certaines fonctionnalités (CORS, etc.) seront dégradées.
            settings = None

# ─────────── Routes legacy ───────────
# Endpoints principaux (avant monitoring) :
# - /healthz, /stations, /forecast, /history, /badges, /snapshot, /weather, …
try:
    from apps.api.routes import health, stations, forecast, history, badges, snapshot, weather
except Exception:
    try:
        from .routes import health, stations, forecast, history, badges, snapshot, weather
    except Exception:
        from routes import health, stations, forecast, history, badges, snapshot, weather

# ─────────── Routes monitoring ───────────
# Endpoints de monitoring structurés par sous-pages :
# - network_overview, network_dynamics, network_stations
# - model_performance, model_explainability
# - data_health, data_drift, data_freshness
# - intro (bloc de synthèse / landing monitoring)
try:
    from apps.api.routes.monitoring import (
        network_overview,
        network_dynamics,
        network_stations,
        model_performance,
        model_explainability,
        data_health,
        data_drift,
        data_freshness,   # ✅ ajout ici
        intro,
    )
except Exception:
    try:
        from .routes.monitoring import (
            network_overview,
            network_dynamics,
            network_stations,
            model_performance,
            model_explainability,
            data_health,
            data_drift,
            data_freshness,   # ✅ ajout ici
            intro,
        )
    except Exception:
        from routes.monitoring import (
            network_overview,
            network_dynamics,
            network_stations,
            model_performance,
            model_explainability,
            data_health,
            data_drift,
            data_freshness,   # ✅ ajout ici
            intro,
        )

# ─────────── App ───────────
# Application FastAPI principale (titre/version utilisés aussi pour OpenAPI).
app = FastAPI(title="velib-api", version="0.2.0")

# ─────────── Token global (middleware) ───────────
# API_GLOBAL_TOKEN (env) permet de protéger l'ensemble des endpoints avec un
# simple Bearer token "global" (pas de gestion fine utilisateur / scopes).
GLOBAL_TOKEN = os.getenv("API_GLOBAL_TOKEN", "").strip()

@app.middleware("http")
async def enforce_global_token(request: Request, call_next):
    """Middleware d'enforcement du token global.

    Règles :
    - OPTIONS (preflight CORS) toujours autorisées,
    - si GLOBAL_TOKEN n'est pas configuré → API entièrement ouverte (utile en dev),
    - routes racine "/" et "/healthz" toujours publiques,
    - pour le reste :
        * Authorization: Bearer <token> requis,
        * <token> doit correspondre exactement à API_GLOBAL_TOKEN sinon 403.
    """
    # Laisser passer les preflights CORS
    if request.method == "OPTIONS":
        return await call_next(request)

    # Pas de token configuré → API ouverte (utile dev)
    if not GLOBAL_TOKEN:
        return await call_next(request)

    # (optionnel) routes publiques
    if request.url.path in ("/", "/healthz"):
        return await call_next(request)

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if auth.removeprefix("Bearer ").strip() != GLOBAL_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    return await call_next(request)

# ─────────── GZip ───────────
# Compression des réponses HTTP à partir de 1 Ko, pour limiter la bande passante
# (utile notamment pour les gros JSON de monitoring).
app.add_middleware(GZipMiddleware, minimum_size=1024)

# ─────────── CORS (à AJOUTER EN DERNIER) ───────────
# On autorise soit la liste définie par `settings.cors_list`, soit `*`
# (ce qui reste acceptable car la protection principale est le token global).
allow_origins = (
    settings.cors_list
    if (settings and getattr(settings, "cors_list", None))
    else ["*"]  # possible puisque token global protège déjà
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],  # inclut Authorization
)

# ─────────── Montage des routers ───────────
# Legacy
app.include_router(health.router)
app.include_router(stations.router)
app.include_router(forecast.router)
app.include_router(history.router)
app.include_router(badges.router)
app.include_router(snapshot.router)
app.include_router(weather.router)

# Monitoring
app.include_router(network_overview.router)
app.include_router(network_dynamics.router)
app.include_router(network_stations.router)
app.include_router(model_performance.router)
app.include_router(model_explainability.router)
app.include_router(data_health.router)
app.include_router(data_drift.router)
app.include_router(data_freshness.router)   # ✅ nouveau
app.include_router(intro.router)

# ─────────── Debug ───────────
@app.on_event("startup")
async def _print_routes() -> None:
    """Log de toutes les routes montées au démarrage.

    Utile en environnement de dev / staging pour vérifier rapidement
    que tous les routers ont bien été inclus et que les méthodes HTTP
    exposées correspondent aux attentes.
    """
    print("=== ROUTES MONTÉES ===")
    for r in app.router.routes:
        try:
            methods = ",".join(sorted(r.methods)) if hasattr(r, "methods") else ""
            print(f"{r.path}  {methods}")
        except Exception:
            # En cas de route “exotique” sans attribut .methods, on ignore l’erreur
            pass
    print("=======================")
