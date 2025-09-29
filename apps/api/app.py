from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.settings import settings
from .routes import health, model, stations, forecast, history, badges

app = FastAPI(title="velib-api", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(model.router)
app.include_router(stations.router)
app.include_router(forecast.router)
app.include_router(history.router)
app.include_router(badges.router)
