# =============================================================================
# service/jobs/build_monitoring_intro.py
# =============================================================================
# Vélib’ Forecast — Monitoring Intro (v1.2)
#
# Rôle
# ----
# Agrège des indicateurs clés pour l’entête/intro du site Monitoring à partir
# des JSON déjà produits :
#   - network/overview/latest/kpis.json
#   - data/health/latest/kpis.json
#   - data/drift/latest/summary.json
#   - models h15/h60 (facultatif, si présents)
#   - forecasts h15/h60 (latest, un fichier par horizon)
#
# Sorties (LATEST only)
# ---------------------
#   {GCS_MONITORING_PREFIX}/monitoring/intro/latest/intro.json
#   {GCS_MONITORING_PREFIX}/monitoring/intro/latest/{ISO}.json
#
# ENV requis
# ----------
#   GCS_MONITORING_PREFIX   gs://<bucket>/velib   (l’alias /monitoring sera ajouté si absent)
#
# ENV optionnels (URIs sources — sinon valeurs par défaut relatives au MONITORING)
# ------------------------------------------------------------------------------
#   MON_OVERVIEW_KPIS_URI     gs://.../monitoring/network/overview/latest/kpis.json
#   MON_HEALTH_KPIS_URI       gs://.../monitoring/data/health/latest/kpis.json
#   MON_DRIFT_SUMMARY_URI     gs://.../monitoring/data/drift/latest/summary.json
#   MON_MODEL_LATEST_H15_URI  gs://.../velib/models/h15/latest.json
#   MON_MODEL_LATEST_H60_URI  gs://.../velib/models/h60/latest.json
#   MON_FORECAST_H15_URI      gs://.../velib/serving/forecast/h15/latest.json
#   MON_FORECAST_H60_URI      gs://.../velib/serving/forecast/h60/latest.json
#
# ENV optionnels (seuils LED)
# ---------------------------
#   INTRO_LED_FC_OK_MIN       défaut "7"   (âge minutes OK des batch forecasts)
#   INTRO_LED_FC_WARN_MIN     défaut "20"  (âge minutes WARN des batch forecasts)
#   INTRO_LED_FRESH_OK_MIN    défaut "5"   (p95 fraicheur OK)
#   INTRO_LED_FRESH_WARN_MIN  défaut "12"  (p95 fraicheur WARN)
#
# Notes
# -----
# - Document JSON “safe” (NaN → null).
# - Tolérant aux fichiers manquants.
# - H15/H60 distincts + bloc "global" backward-compat (h15 sinon h60).
# =============================================================================

from __future__ import annotations
import os, json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from google.cloud import storage
import numpy as np

SCHEMA_VERSION = "1.2"
TZ = "Europe/Paris"

# ─────────────────────────── Helpers ENV / GCS ───────────────────────────

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v if (v is not None and v != "") else default

def _ensure_mon_base(mon_prefix: str) -> str:
    base = mon_prefix.rstrip("/")
    if not base.endswith("/monitoring"):
        base = base + "/monitoring"
    return base

def _split_gs(gs: str):
    assert gs.startswith("gs://"), f"bad GCS URI: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k

def _read_json_gcs(client: storage.Client, gs: Optional[str]) -> Optional[Dict[str, Any]]:
    if not gs or not gs.startswith("gs://"):
        return None
    try:
        b, k = _split_gs(gs)
        data = client.bucket(b).blob(k).download_as_bytes()
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None

def _san(o):
    if isinstance(o, dict):        return {k: _san(v) for k, v in o.items()}
    if isinstance(o, list):        return [_san(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if (not isinstance(o, float) or np.isfinite(o)) else None
    if isinstance(o, (np.integer, int)):  return int(o)
    if isinstance(o, (np.bool_, bool)):   return bool(o)
    if o is None:                 return None
    try:
        import pandas as pd  # type: ignore
        if pd.isna(o): return None
    except Exception:
        pass
    return o

def _write_json_gcs(client: storage.Client, gs: str, doc: Dict[str, Any]) -> None:
    b, k = _split_gs(gs)
    blob = client.bucket(b).blob(k)
    blob.cache_control = "no-store"
    blob.content_type = "application/json; charset=utf-8"
    safe = _san(doc)
    blob.upload_from_string(
        json.dumps(safe, ensure_ascii=False, separators=(",", ":")),
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
    if x is None or not isinstance(x, (int, float)):
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

# ─────────────────────────── Forecast parsing ───────────────────────────

def _parse_forecast_doc(doc: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[int]]:
    """
    Formes acceptées :
      A) bundle: {"generated_at": "...Z", "horizon_min": 15, "data": [ ... ] }
      B) variante: {"generated_at": "...Z", "predictions": [ ... ] }
      C) legacy: [ {...}, {...} ]
    Retourne (generated_at_iso, rows_count)
    """
    if not doc:
        return None, None
    if isinstance(doc, dict):
        gen = doc.get("generated_at")
        if isinstance(doc.get("data"), list):
            return gen, len(doc["data"])
        if isinstance(doc.get("predictions"), list):
            return gen, len(doc["predictions"])
        return gen, None
    if isinstance(doc, list):
        return None, len(doc)
    return None, None

# ─────────────────────────── Main ───────────────────────────

def main() -> int:
    MON_PREFIX = _env("GCS_MONITORING_PREFIX")
    if not (MON_PREFIX and MON_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX manquant ou invalide")
    mon_base = _ensure_mon_base(MON_PREFIX)

    # Seuils LED configurables
    LED_FC_OK   = float(_env("INTRO_LED_FC_OK_MIN", "7"))
    LED_FC_WARN = float(_env("INTRO_LED_FC_WARN_MIN", "20"))
    LED_FR_OK   = float(_env("INTRO_LED_FRESH_OK_MIN", "5"))
    LED_FR_WARN = float(_env("INTRO_LED_FRESH_WARN_MIN", "12"))

    # URIs par défaut (peuvent être surchargées par ENV)
    overview_kpis_uri = _env("MON_OVERVIEW_KPIS_URI", f"{mon_base}/network/overview/latest/kpis.json")
    health_kpis_uri   = _env("MON_HEALTH_KPIS_URI",   f"{mon_base}/data/health/latest/kpis.json")
    drift_summary_uri = _env("MON_DRIFT_SUMMARY_URI", f"{mon_base}/data/drift/latest/summary.json")

    # Modèles / Forecasts (hors /monitoring car rangés côté /velib/serving & /velib/models)
    # On construit à partir du préfixe projet (première partie de mon_base)
    # Exemple mon_base: gs://bucket/velib/monitoring → proj_base: gs://bucket/velib
    proj_base = mon_base.rsplit("/monitoring", 1)[0]

    model_h15_uri = _env("MON_MODEL_LATEST_H15_URI", f"{proj_base}/models/h15/latest.json")
    model_h60_uri = _env("MON_MODEL_LATEST_H60_URI", f"{proj_base}/models/h60/latest.json")

    fc_h15_uri = _env("MON_FORECAST_H15_URI", f"{proj_base}/serving/forecast/h15/latest.json")
    fc_h60_uri = _env("MON_FORECAST_H60_URI", f"{proj_base}/serving/forecast/h60/latest.json")

    client = storage.Client()

    overview = _read_json_gcs(client, overview_kpis_uri) or {}
    health   = _read_json_gcs(client, health_kpis_uri)   or {}
    drift    = _read_json_gcs(client, drift_summary_uri) or {}

    model15  = _read_json_gcs(client, model_h15_uri) or {}
    model60  = _read_json_gcs(client, model_h60_uri) or {}

    fc15_doc = _read_json_gcs(client, fc_h15_uri) or {}
    fc60_doc = _read_json_gcs(client, fc_h60_uri) or {}

    # Modèles (fallbacks lisibles)
    model_version_15 = model15.get("version") or model15.get("model_version") or "h15"
    model_version_60 = model60.get("version") or model60.get("model_version") or "h60"
    model_versions = f"{model_version_15} / {model_version_60}"

    # Forecasts
    ts_fc15, rows_fc15 = _parse_forecast_doc(fc15_doc)
    ts_fc60, rows_fc60 = _parse_forecast_doc(fc60_doc)
    age_fc15 = _minutes_since(ts_fc15)
    age_fc60 = _minutes_since(ts_fc60)

    # Global compat (h15 prioritaire)
    ts_forecast_global   = ts_fc15 or ts_fc60
    rows_forecast_global = rows_fc15 or rows_fc60
    age_forecast_global  = age_fc15 if age_fc15 is not None else age_fc60

    # KPIs secondaires
    stations_active = overview.get("stations_active")
    freshness_p95   = health.get("freshness_age_p95_min")
    coverage_7d     = health.get("coverage_global_pct")

    psi_global      = drift.get("psi_global")
    drift_top_feat  = drift.get("top_feature")
    drift_top_psi   = drift.get("top_feature_psi")
    ts_drift        = drift.get("generated_at")

    ts_overview = overview.get("generated_at")
    ts_health   = health.get("generated_at")

    # LEDs
    api_led         = "ok" if (isinstance(stations_active, (int, float)) and stations_active > 0) else "down"
    bat_led15       = _led_from_value(age_fc15, LED_FC_OK, LED_FC_WARN)
    bat_led60       = _led_from_value(age_fc60, LED_FC_OK, LED_FC_WARN)
    bat_led_global  = _led_from_value(age_forecast_global, LED_FC_OK, LED_FC_WARN)
    met_led         = _led_from_value(freshness_p95, LED_FR_OK, LED_FR_WARN)
    drift_led       = _led_from_psi(psi_global if isinstance(psi_global, (int, float)) else None)

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

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
            "batch_forecast_h15": {
                "led": bat_led15,
                "age_min": age_fc15,
                "source_generated_at": ts_fc15,
                "rows": rows_fc15,
                "horizon_min": 15,
            },
            "batch_forecast_h60": {
                "led": bat_led60,
                "age_min": age_fc60,
                "source_generated_at": ts_fc60,
                "rows": rows_fc60,
                "horizon_min": 60,
            },
            "batch_forecast": {
                "led": bat_led_global,
                "age_min": age_forecast_global,
                "source_generated_at": ts_forecast_global,
                "rows": rows_forecast_global,
            },
            "weather_provider": {
                "led": met_led,
                "freshness_p95_min": freshness_p95,
                "source_generated_at": ts_health,
            },
            "data_drift": {
                "led": drift_led,
                "psi_global": psi_global,
                "top_feature": drift_top_feat,
                "top_feature_psi": drift_top_psi,
                "source_generated_at": ts_drift,
            },
        },
        "activity": [
            {"label": "Versions modèle (h15/h60)", "value": model_versions, "generated_at": None},
            {"label": "Couverture (7 jours)", "value": coverage_7d, "generated_at": ts_health},
            {"label": "Stations actives", "value": stations_active, "generated_at": ts_overview},
            {"label": "Prévisions générées (h15)", "value": rows_fc15, "generated_at": ts_fc15},
            {"label": "Prévisions générées (h60)", "value": rows_fc60, "generated_at": ts_fc60},
            {"label": "PSI global (drift)", "value": psi_global, "generated_at": ts_drift},
        ],
        "sources": {
            "overview_kpis": overview_kpis_uri,
            "health_kpis": health_kpis_uri,
            "drift_summary": drift_summary_uri,
            "model_h15": model_h15_uri,
            "model_h60": model_h60_uri,
            "forecast_h15_latest": fc_h15_uri,
            "forecast_h60_latest": fc_h60_uri,
        },
    }

    out_prefix = f"{mon_base}/intro/latest"
    latest_uri = f"{out_prefix}/intro.json"
    dated_uri  = f"{out_prefix}/{now_iso.replace(':','-')}.json"

    _write_json_gcs(client, latest_uri, doc)
    _write_json_gcs(client, dated_uri,  doc)
    print(f"[intro] wrote {latest_uri} and {dated_uri}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
