from pydantic import BaseSettings
from typing import List

class Settings(BaseSettings):
    models_prefix: str  = ""  # gs://.../velib/models
    gcs_db_daily: str   = ""  # gs://.../velib/db/daily
    tz_app: str         = "Europe/Paris"
    cors_origins: List[str] = []
    tmp_dir: str        = "/tmp"

    class Config:
        env_prefix = ""  # lit directement MODELS_PREFIX, GCS_DB_DAILY, etc.

settings = Settings()
