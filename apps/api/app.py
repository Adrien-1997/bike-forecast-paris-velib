# apps/api/app.py
from __future__ import annotations
import sys, importlib, os
from pathlib import Path

# ─────────── Charger le .env automatiquement ───────────
try:
    from dotenv import load_dotenv, find_dotenv  # pip install python-dotenv
    # 1) cherche .env dans le cwd ou ses parents
    found = find_dotenv(filename=".env", usecwd=True)
    if found:
        load_dotenv(found, override=False)
    # 2) fallback : racine du dépôt (2 niveaux au-dessus de ce fichier)
    repo_root = Path(__file__).resolve().parents[2]
    env_file = repo_root / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)
    print(f"[app] .env chargé ✓ (found='{found}', repo_root='{repo_root}')")
except Exception as e:
    print(f"[app] .env non chargé ({e})")

# ───────────────────────── Alias module 'core' → 'api.core' ─────────────────────────
sys.modules.setdefault("core", importlib.import_module("api.core"))

# ───────────────────────── FastAPI & middlewares ─────────────────────────
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

# ───────────────────────── Settings (optionnel) ─────────────────────────
try:
    from .core.settings import settings
except Exception:
    settings = None  # on continue en mode défaut

# ───────────────────────── Routes legacy ─────────────────────────
from .routes import health, stations, forecast, history, badges, snapshot, weather  # type: ignore

# ───────────────────────── Routes monitoring ─────────────────────────
from .routes.monitoring import (
    network_overview,
    network_dynamics,
    network_stations,
    model_performance,
    model_explainability,
    data_health,
    data_drift,
)  # type: ignore

# ───────────────────────── App ─────────────────────────
app = FastAPI(title="velib-api", version="0.2.0")

# CORS
allow_origins = (
    settings.cors_list
    if (settings and getattr(settings, "cors_list", None))
    else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compression des réponses (JSON volumineux)
app.add_middleware(GZipMiddleware, minimum_size=1024)

# ───────────────────────── Montage des routers ─────────────────────────
# Legacy
app.include_router(health.router)
app.include_router(stations.router)
app.include_router(forecast.router)
app.include_router(history.router)
app.include_router(badges.router)
app.include_router(snapshot.router)
app.include_router(weather.router)

# Monitoring
app.include_router(network_overview.router)       # /monitoring/network/overview/*
app.include_router(network_dynamics.router)       # /monitoring/network/dynamics/*
app.include_router(network_stations.router)       # /monitoring/network/stations/*
app.include_router(model_performance.router)      # /monitoring/model/*
app.include_router(model_explainability.router)   # /monitoring/model/*
app.include_router(data_health.router)            # /monitoring/data/*
app.include_router(data_drift.router)             # /monitoring/data/*

# ───────────────────────── Debug : listing des routes au démarrage ─────────────────────────
@app.on_event("startup")
async def _print_routes() -> None:
    print("=== ROUTES MONTÉES ===")
    for r in app.router.routes:
        try:
            methods = ",".join(sorted(r.methods)) if hasattr(r, "methods") else ""
            print(f"{r.path}  {methods}")
        except Exception:
            pass
    print("=======================")