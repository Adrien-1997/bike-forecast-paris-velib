from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ⬇️ flat imports (no "api.")
from routes import health, stations, forecast, history, badges, snapshot, weather

app = FastAPI(title="velib-api", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router)
app.include_router(stations.router)
app.include_router(forecast.router)
app.include_router(history.router)
app.include_router(badges.router)
app.include_router(snapshot.router)
app.include_router(weather.router)
