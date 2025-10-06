from __future__ import annotations
import json
import re
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------- Helpers ----------
def _parse_cors_any(v: Optional[str]) -> List[str]:
    if not v:
        return []
    s = v.strip()
    if s.startswith("["):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    return [x.strip() for x in s.split(",") if x.strip()]


def _strip_serving_prefix(p: str) -> str:
    """
    Normalize a possibly full path like:
      gs://.../serving/features_*/latest.parquet
      gs://.../serving/forecast/latest_forecast.json
    to its base '.../serving'.

    This keeps old envs working while we serve JSON bundles now.
    """
    if not p:
        return p
    # e.g. gs://bucket/velib/serving[/something]/(latest.parquet|latest_forecast.json)?
    m = re.match(r"^(gs://.+?/serving)(?:/[^/]+)?(?:/(?:latest\.parquet|latest_forecast\.json))?$", p)
    if m:
        return m.group(1)
    if p.endswith("/latest.parquet"):
        return p[: -len("/latest.parquet")]
    if p.endswith("/latest_forecast.json"):
        return p[: -len("/latest_forecast.json")]
    return p


def _strip_latest(p: str) -> str:
    """Remove trailing '/latest.parquet' or '/latest_forecast.json' if present."""
    if not p:
        return p
    for suf in ("/latest.parquet", "/latest_forecast.json"):
        if p.endswith(suf):
            return p[: -len(suf)]
    return p


# ---------- Settings ----------
class Settings(BaseSettings):
    # ---------- CORS ----------
    cors_origins: Optional[str] = None

    # ---------- Legacy compat (kept so old env vars won't break) ----------
    # NOTE: We serve JSON now. These fields remain for backward compatibility only.
    models_prefix: Optional[str] = None          # ex: gs://.../models/lgb_*.joblib (or a single .joblib)
    gcs_serving_prefix: Optional[str] = None     # ex: gs://.../serving/[...]/latest.parquet (legacy)

    # ---------- GCS / Forecast (JSON-first) ----------
    # Default model URI (still used by server-side inference if any)
    GCS_MODEL_URI: str = "gs://velib-forecast-472820_cloudbuild/velib/models/lgb_nbvelos_T+15min.joblib"

    # Parent "serving" folder (generic). We keep this for structure, but JSON is under SERVING_FORECAST_PREFIX.
    GCS_SERVING_PREFIX: str = "gs://velib-forecast-472820_cloudbuild/velib/serving"

    # JSON forecast bundle location (writer drops latest_h{h}.json and latest_forecast.json here)
    SERVING_FORECAST_PREFIX: str = "gs://velib-forecast-472820_cloudbuild/velib/serving/forecast"

    # Horizons exposed by the API (CSV)
    FORECAST_SUPPORTED: str = "15"  # set to "15,60" when ready

    # In-memory cache TTL for reading latest_* bundles (seconds)
    FORECAST_CACHE_TTL_SECONDS: int = 120

    # ---------- Weather (live) ----------
    OPENMETEO_URL: str = "https://api.open-meteo.com/v1/forecast"
    OPENMETEO_LAT: float = 48.8566
    OPENMETEO_LON: float = 2.3522
    OPENMETEO_TIMEOUT: float = 5.0
    METEO_DISABLE: bool = False

    # ---------- Misc ----------
    IMAGE_TAG: str = ""
    tz_app: str = "Europe/Paris"
    tmp_dir: str = "/tmp"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_list(self) -> List[str]:
        return _parse_cors_any(self.cors_origins)

    def __init__(self, **data):
        super().__init__(**data)

        # 1) Map legacy fields if provided (do not break old envs)
        # If models_prefix points to a single .joblib, treat it as GCS_MODEL_URI
        if self.models_prefix and self.models_prefix.startswith("gs://") and self.models_prefix.endswith(".joblib"):
            self.GCS_MODEL_URI = self.models_prefix

        # If gcs_serving_prefix was given (possibly pointing to a legacy latest.parquet),
        # normalize to .../serving base so JSON code paths still work.
        if self.gcs_serving_prefix and self.gcs_serving_prefix.startswith("gs://"):
            self.GCS_SERVING_PREFIX = _strip_serving_prefix(self.gcs_serving_prefix)

        # 2) Normalize GCS_SERVING_PREFIX in case someone passes a full path
        self.GCS_SERVING_PREFIX = _strip_serving_prefix(self.GCS_SERVING_PREFIX)

        # 3) Clean SERVING_FORECAST_PREFIX from accidental trailing files
        self.SERVING_FORECAST_PREFIX = _strip_latest(self.SERVING_FORECAST_PREFIX)

        # 4) Create legacy aliases so old code paths won't crash
        if not getattr(self, "gcs_serving_prefix", None):
            object.__setattr__(self, "gcs_serving_prefix", self.GCS_SERVING_PREFIX)
        if not getattr(self, "models_prefix", None):
            object.__setattr__(self, "models_prefix", self.GCS_MODEL_URI)


# ---------- Singleton ----------
settings = Settings()
