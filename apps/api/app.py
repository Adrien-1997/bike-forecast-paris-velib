# apps/api/app.py
from __future__ import annotations

# ───────────────────────── Alias module 'core' → 'api.core' (doit être AVANT les imports) ─────────────────────────
import sys, importlib
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

# ───────────────────────── Routes monitoring (import tolérant + log erreur) ─────────────────────────
_monitoring_import_ok = False
try:
    from .routes.monitoring import manifest as mon_manifest  # type: ignore
    from .routes.monitoring import perf as mon_perf          # type: ignore
    from .routes.monitoring import network as mon_network    # type: ignore
    from .routes.monitoring import drift as mon_drift        # type: ignore
    from .routes.monitoring import docs as mon_docs          # type: ignore
    _monitoring_import_ok = True
except Exception as e:
    import traceback
    print("[monitoring import error]", e)
    traceback.print_exc()
    _monitoring_import_ok = False

# ───────────────────────── App ─────────────────────────
app = FastAPI(title="velib-api", version="0.2.0")

# CORS
allow_origins = settings.cors_list if (settings and getattr(settings, "cors_list", None)) else ["*"]
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

# Monitoring (seulement si les imports ont réussi)
if _monitoring_import_ok:
    app.include_router(mon_manifest.router)
    app.include_router(mon_perf.router)
    app.include_router(mon_network.router)
    app.include_router(mon_drift.router)
    app.include_router(mon_docs.router)

# ───────────────────────── Debug: listing des routes au démarrage ─────────────────────────
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

