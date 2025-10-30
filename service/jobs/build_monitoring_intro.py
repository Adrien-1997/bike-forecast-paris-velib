# service/jobs/build_monitoring_intro.py
from __future__ import annotations
import os, json
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from google.cloud import storage

SCHEMA_VERSION = "1.0"
TZ = "Europe/Paris"

# ─────────────── URIs fixes (sources) ───────────────
def _gs(path: str) -> str:
    return path if path.startswith("gs://") else f"gs://{path}"

OVERVIEW_KPIS_URI    = _gs("velib-forecast-472820_cloudbuild/velib/monitoring/network/overview/latest/kpis.json")
HEALTH_KPIS_URI      = _gs("velib-forecast-472820_cloudbuild/velib/monitoring/data/health/latest/kpis.json")
DRIFT_SUMMARY_URI    = _gs("velib-forecast-472820_cloudbuild/velib/monitoring/data/drift/latest/summary.json")
MODEL_LATEST_H15_URI = _gs("velib-forecast-472820_cloudbuild/velib/models/h15/latest.json")
MODEL_LATEST_H60_URI = _gs("velib-forecast-472820_cloudbuild/velib/models/h60/latest.json")
FORECAST_LATEST_URI  = _gs("velib-forecast-472820_cloudbuild/velib/serving/forecast/latest_forecast.json")

# ─────────────── Destination ───────────────
INTRO_OUT_PREFIX = _gs("velib-forecast-472820_cloudbuild/velib/monitoring/intro/latest/")

# ─────────────── Helpers GCS ───────────────
def _split_gs(gs: str):
    assert gs.startswith("gs://"), f"bad GCS URI: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k

def _read_json_gcs(client: storage.Client, gs: Optional[str]) -> Optional[Dict[str, Any]]:
    if not gs:
        return None
    try:
        b, k = _split_gs(gs)
        data = client.bucket(b).blob(k).download_as_bytes()
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None

def _write_json_gcs(client: storage.Client, gs: str, doc: Dict[str, Any]) -> None:
    b, k = _split_gs(gs)
    blob = client.bucket(b).blob(k)
    blob.cache_control = "no-store"
    blob.content_type = "application/json; charset=utf-8"
    blob.upload_from_string(
        json.dumps(doc, ensure_ascii=False, separators=(",", ":")),
        content_type=blob.content_type,
    )

def _minutes_since(iso: Optional[str]) -> Optional[float]:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 60.0)
    except Exception:
        return None

def _led_from_value(x: Optional[float], ok: float, warn: float, reverse: bool = False) -> str:
    if x is None:
        return "down"
    v = -x if reverse else x
    if v <= ok:
        return "ok"
    if v <= warn:
        return "warn"
    return "down"

def _led_from_psi(psi: Optional[float]) -> str:
    """Seuils drift (PSI): <0.10 ok, <0.20 warn, sinon down."""
    if psi is None or not isinstance(psi, (int, float)):
        return "down"
    if psi < 0.10:
        return "ok"
    if psi < 0.20:
        return "warn"
    return "down"

# ─────────────── Main ───────────────
def main() -> int:
    client = storage.Client()

    overview = _read_json_gcs(client, OVERVIEW_KPIS_URI) or {}
    health   = _read_json_gcs(client, HEALTH_KPIS_URI)   or {}
    drift    = _read_json_gcs(client, DRIFT_SUMMARY_URI) or {}
    model15  = _read_json_gcs(client, MODEL_LATEST_H15_URI) or {}
    model60  = _read_json_gcs(client, MODEL_LATEST_H60_URI) or {}
    forecast = _read_json_gcs(client, FORECAST_LATEST_URI)  or {}

    # modèles
    model_version_15 = model15.get("version") or model15.get("model_version") or "h15"
    model_version_60 = model60.get("version") or model60.get("model_version") or "h60"
    model_versions = f"{model_version_15} / {model_version_60}"

    # forecast info
    ts_forecast      = forecast.get("generated_at")
    forecast_rows    = forecast.get("n_rows") or forecast.get("rows") or None
    forecast_age_min = _minutes_since(ts_forecast)

    # autres KPI
    stations_active = overview.get("stations_active")
    freshness_p95   = health.get("freshness_age_p95_min")
    coverage_7d     = health.get("coverage_global_pct")

    psi_global      = drift.get("psi_global")
    drift_top_feat  = drift.get("top_feature")
    drift_top_psi   = drift.get("top_feature_psi")
    ts_drift        = drift.get("generated_at")

    ts_overview = overview.get("generated_at")
    ts_health   = health.get("generated_at")

    # LED statuses
    api_led  = "ok" if (stations_active and stations_active > 0) else "down"
    bat_led  = _led_from_value(forecast_age_min, ok=7, warn=20)
    met_led  = _led_from_value(freshness_p95, ok=5, warn=12)
    drift_led = _led_from_psi(psi_global if isinstance(psi_global, (int, float)) else None)

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # ─────────────── Build document ───────────────
    doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso,
        "tz": TZ,
        "kpis": {
            "stations_active": stations_active,
            "freshness_p95_min": freshness_p95,
            "coverage_7d_pct": coverage_7d,
            "psi_global": psi_global,
            "model_versions": model_versions,
        },
        "statuses": {
            "api_stations": {
                "led": api_led,
                "stations_active": stations_active,
            },
            "batch_forecast": {
                "led": bat_led,
                "age_min": forecast_age_min,
                "source_generated_at": ts_forecast,
                "rows": forecast_rows,
            },
            "weather_provider": {
                "led": met_led,
                "freshness_p95_min": freshness_p95,
                "source_generated_at": ts_health,
            },
            "data_drift": {
                "led": drift_led,                 # ← NEW: ok | warn | down
                "psi_global": psi_global,         # valeur brute
                "top_feature": drift_top_feat,    # si dispo
                "top_feature_psi": drift_top_psi, # si dispo
                "source_generated_at": ts_drift,
            },
        },
        "activity": [
            {"label": "Versions modèle (h15/h60)", "value": model_versions, "generated_at": None},
            {"label": "Couverture (7 jours)", "value": coverage_7d, "generated_at": ts_health},
            {"label": "Stations actives", "value": stations_active, "generated_at": ts_overview},
            {"label": "Prévisions générées", "value": forecast_rows, "generated_at": ts_forecast},
            {"label": "PSI global (drift)", "value": psi_global, "generated_at": ts_drift},
        ],
        "sources": {
            "overview_kpis": OVERVIEW_KPIS_URI,
            "health_kpis": HEALTH_KPIS_URI,
            "drift_summary": DRIFT_SUMMARY_URI,
            "model_h15": MODEL_LATEST_H15_URI,
            "model_h60": MODEL_LATEST_H60_URI,
            "forecast_latest": FORECAST_LATEST_URI,
        },
    }

    # sorties
    latest_uri = os.path.join(INTRO_OUT_PREFIX, "intro.json")
    dated_uri  = os.path.join(INTRO_OUT_PREFIX, f"{now_iso.replace(':','-')}.json")
    _write_json_gcs(client, latest_uri, doc)
    _write_json_gcs(client, dated_uri, doc)
    print(f"[intro] wrote {latest_uri} and {dated_uri}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
