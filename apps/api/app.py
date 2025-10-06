from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.core.settings import settings
from api.routes import health, stations, forecast, history, badges, snapshot, weather  # ← add this

app = FastAPI(title="velib-api", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # ← allow all origins
    allow_credentials=False,    # ← required if using "*"
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
app.include_router(weather.router)   # ← new line

if __name__ == "__main__":
    import os, uvicorn
    uvicorn.run("apps.api.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), workers=1)
