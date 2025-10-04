# apps/api/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.core.settings import settings
from api.routes import health, model, stations, forecast, history, badges, snapshot

app = FastAPI(title="velib-api", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # ← tout autoriser
    allow_credentials=False,    # ← OBLIGATOIRE si "*" (sinon FastAPI ne mettra pas l’en-tête)
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(model.router)
app.include_router(stations.router)
app.include_router(forecast.router)
app.include_router(history.router)
app.include_router(badges.router)
app.include_router(snapshot.router)

if __name__ == "__main__":
    import os, uvicorn
    uvicorn.run("apps.api.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), workers=1)
