# apps/api/core/settings.py
from __future__ import annotations
import json
import re
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict

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

def _strip_trailing_latest(p: str) -> str:
    if not p:
        return p
    m = re.match(r"^(gs://.+?/serving)(?:/features_[^/]+)?(?:/latest\.parquet)?$", p)
    if m:
        return m.group(1)
    if p.endswith("/latest.parquet"):
        return p[: -len("/latest.parquet")]
    return p

class Settings(BaseSettings):
    # ---------- CORS ----------
    cors_origins: Optional[str] = None

    # ---------- Compat "legacy" ----------
    # Tu avais ces clés (souvent remplies avec des *URIs* directement)
    models_prefix: Optional[str] = None          # ex: gs://.../models/lgb_*.joblib
    gcs_serving_prefix: Optional[str] = None     # ex: gs://.../serving/features_4h/latest.parquet

    # ---------- Clés "officielles" lues par les routes ----------
    GCS_MODEL_URI: str = "gs://velib-forecast-472820_cloudbuild/velib/models/lgb_nbvelos_T+15min.joblib"
    GCS_SERVING_PREFIX: str = "gs://velib-forecast-472820_cloudbuild/velib/serving"

    # ---------- Divers ----------
    IMAGE_TAG: str = ""
    tz_app: str = "Europe/Paris"
    tmp_dir: str = "/tmp"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_list(self) -> List[str]:
        return _parse_cors_any(self.cors_origins)

    def __init__(self, **data):
        super().__init__(**data)

        # 1) Mapper les champs "legacy" si fournis
        if self.models_prefix and self.models_prefix.startswith("gs://"):
            # si c'est bien un .joblib, on considère que c'est le modèle
            if self.models_prefix.endswith(".joblib"):
                self.GCS_MODEL_URI = self.models_prefix

        if self.gcs_serving_prefix and self.gcs_serving_prefix.startswith("gs://"):
            # l'utilisateur peut avoir passé le chemin complet jusqu'à latest.parquet
            self.GCS_SERVING_PREFIX = _strip_trailing_latest(self.gcs_serving_prefix)

        # 2) Normaliser GCS_SERVING_PREFIX (au cas où latest.parquet a été mis directement)
        self.GCS_SERVING_PREFIX = _strip_trailing_latest(self.GCS_SERVING_PREFIX)

        # 3) Créer les **alias descendents** pour le code existant
        #    (ainsi features_live.py qui lit "settings.gcs_serving_prefix" ne casse plus)
        if not getattr(self, "gcs_serving_prefix", None):
            object.__setattr__(self, "gcs_serving_prefix", self.GCS_SERVING_PREFIX)
        if not getattr(self, "models_prefix", None):
            object.__setattr__(self, "models_prefix", self.GCS_MODEL_URI)

settings = Settings()
