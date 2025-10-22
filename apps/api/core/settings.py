from __future__ import annotations
import json
import re
from typing import Optional, List
from pydantic import Field, field_validator
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
    cors_origins: Optional[str] = Field(default=None, env="CORS_ORIGINS")

    # ---------- Legacy compat (kept so old env vars won't break) ----------
    # NOTE: We serve JSON now. These fields remain for backward compatibility only.
    models_prefix: Optional[str] = Field(default=None, env="MODELS_PREFIX")         # ex: gs://.../models/*.joblib OR a single .joblib
    gcs_serving_prefix: Optional[str] = Field(default=None, env="GCS_SERVING_PREFIX")  # ex: gs://.../serving/[...]/latest.parquet (legacy)

    # ---------- GCS / Forecast (JSON-first) ----------
    # Default model URI (still used by server-side inference if any)
    GCS_MODEL_URI: str = Field(
        default="gs://velib-forecast-472820_cloudbuild/velib/models/lgb_nbvelos_T+15min.joblib",
        env="GCS_MODEL_URI",
    )

    # Parent "serving" folder (generic). We keep this for structure, but JSON is under SERVING_FORECAST_PREFIX.
    GCS_SERVING_PREFIX: str = Field(
        default="gs://velib-forecast-472820_cloudbuild/velib/serving",
        env="GCS_SERVING_PREFIX",
    )

    # JSON forecast bundle location (writer drops latest_h{h}.json and latest_forecast.json here)
    SERVING_FORECAST_PREFIX: str = Field(
        default="gs://velib-forecast-472820_cloudbuild/velib/serving/forecast",
        env="SERVING_FORECAST_PREFIX",
    )

    # Horizons exposed by the API (CSV)
    FORECAST_SUPPORTED: str = Field(default="15", env="FORECAST_SUPPORTED")  # set to "15,60" when ready

    # In-memory cache TTL for reading latest_* bundles (seconds)
    FORECAST_CACHE_TTL_SECONDS: int = Field(default=120, env="FORECAST_CACHE_TTL_SECONDS")

    # ---------- Monitoring JSON root (NEW) ----------
    # Root used by monitoring endpoints (manifest, perf, network, drift, docs)
    GCS_MONITORING_PREFIX: str = Field(
        default="gs://velib-forecast-472820_cloudbuild/velib/monitoring",
        env="GCS_MONITORING_PREFIX",
    )

    # ---------- Weather (live) ----------
    OPENMETEO_URL: str = Field(default="https://api.open-meteo.com/v1/forecast", env="OPENMETEO_URL")
    OPENMETEO_LAT: float = Field(default=48.8566, env="OPENMETEO_LAT")
    OPENMETEO_LON: float = Field(default=2.3522, env="OPENMETEO_LON")
    OPENMETEO_TIMEOUT: float = Field(default=5.0, env="OPENMETEO_TIMEOUT")
    METEO_DISABLE: bool = Field(default=False, env="METEO_DISABLE")  # accepts 1/true/True

    # ---------- Misc ----------
    IMAGE_TAG: str = Field(default="", env="IMAGE_TAG")
    tz_app: str = Field(default="Europe/Paris", env="TZ_APP")
    tmp_dir: str = Field(default="/tmp", env="TMP_DIR")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,  # explicit for deterministic ENV mapping
    )

    # ---------- Derived / helpers ----------
    @property
    def cors_list(self) -> List[str]:
        return _parse_cors_any(self.cors_origins)

    # ---------- Validators ----------
    @field_validator("GCS_MODEL_URI", "GCS_SERVING_PREFIX", "SERVING_FORECAST_PREFIX", "GCS_MONITORING_PREFIX")
    @classmethod
    def _must_be_gs_uri(cls, v: str) -> str:
        # Allow empty only for optional fields (these are required defaults, so non-empty expected)
        if not v:
            return v
        if not v.startswith("gs://"):
            raise ValueError("GCS/SERVING prefixes must start with 'gs://'")
        return v.rstrip("/")

    @field_validator("METEO_DISABLE", mode="before")
    @classmethod
    def _bool_compat(cls, v):
        # Accept "1", "true", "True", "yes"
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes")
        return v

    def __init__(self, **data):
        super().__init__(**data)

        # 1) Map legacy fields if provided (do not break old envs)
        # If models_prefix points to a single .joblib, treat it as GCS_MODEL_URI
        if self.models_prefix and self.models_prefix.startswith("gs://") and self.models_prefix.endswith(".joblib"):
            object.__setattr__(self, "GCS_MODEL_URI", self.models_prefix)

        # If gcs_serving_prefix was given (possibly pointing to a legacy latest.parquet),
        # normalize to .../serving base so JSON code paths still work.
        if self.gcs_serving_prefix and self.gcs_serving_prefix.startswith("gs://"):
            object.__setattr__(self, "GCS_SERVING_PREFIX", _strip_serving_prefix(self.gcs_serving_prefix))

        # 2) Normalize GCS_SERVING_PREFIX in case someone passes a full path
        object.__setattr__(self, "GCS_SERVING_PREFIX", _strip_serving_prefix(self.GCS_SERVING_PREFIX))

        # 3) Clean SERVING_FORECAST_PREFIX from accidental trailing files
        object.__setattr__(self, "SERVING_FORECAST_PREFIX", _strip_latest(self.SERVING_FORECAST_PREFIX))

        # 4) Create legacy aliases so old code paths won't crash
        if not getattr(self, "gcs_serving_prefix", None):
            object.__setattr__(self, "gcs_serving_prefix", self.GCS_SERVING_PREFIX)
        if not getattr(self, "models_prefix", None):
            object.__setattr__(self, "models_prefix", self.GCS_MODEL_URI)

        # 5) Normalize monitoring prefix (strip trailing slash)
        if getattr(self, "GCS_MONITORING_PREFIX", None):
            object.__setattr__(self, "GCS_MONITORING_PREFIX", self.GCS_MONITORING_PREFIX.rstrip("/"))


# ---------- Singleton ----------
settings = Settings()
