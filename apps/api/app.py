# apps/api/app.py
from __future__ import annotations
import sys, importlib, os
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

# ─────────── .env ───────────
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
    print(f"[app] .env non chargé ({e})")

# ─────────── Alias 'core' ───────────
for mod in ("apps.api.core", "api.core", "core"):
    try:
        sys.modules.setdefault("core", importlib.import_module(mod))
        break
    except ModuleNotFoundError:
        continue

# ─────────── Settings ───────────
try:
    from apps.api.core.settings import settings
except Exception:
    try:
        from .core.settings import settings
    except Exception:
        try:
            from core.settings import settings
        except Exception:
            settings = None

# ─────────── Routes ───────────
try:
    from apps.api.routes import health, stations, forecast, history, badges, snapshot, weather
except Exception:
    try:
        from .routes import health, stations, forecast, history, badges, snapshot, weather
    except Exception:
        from routes import health, stations, forecast, history, badges, snapshot, weather

try:
    from apps.api.routes.monitoring import (
        network_overview,
        network_dynamics,
        network_stations,
        model_performance,
        model_explainability,
        data_health,
        data_drift,
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
        )

# ─────────── App ───────────
app = FastAPI(title="velib-api", version="0.2.0")

# ─────────── Token global (middleware) ───────────
GLOBAL_TOKEN = os.getenv("API_GLOBAL_TOKEN", "").strip()

@app.middleware("http")
async def enforce_global_token(request: Request, call_next):
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
app.add_middleware(GZipMiddleware, minimum_size=1024)

# ─────────── CORS (à AJOUTER EN DERNIER) ───────────
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

# ─────────── Debug ───────────
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
 