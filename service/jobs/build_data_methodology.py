# service/jobs/build_data_methodology.py
from __future__ import annotations
import os, sys, json
from datetime import datetime, timezone

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage requis") from e

SCHEMA_VERSION = "1.0"

def _split(gs: str):
    assert gs.startswith("gs://"), f"bad GCS uri: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k.rstrip("/")

def _upload_json_gs(obj: dict, gs_uri: str):
    bkt, key = _split(gs_uri)
    storage.Client().bucket(bkt).blob(key).upload_from_string(
        json.dumps(obj, ensure_ascii=False),
        content_type="application/json"
    )
    print(f"[methodology] wrote → {gs_uri}")

def main() -> int:
    MON_PREFIX = os.environ.get("GCS_MONITORING_PREFIX")
    if not (MON_PREFIX and MON_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX manquant ou invalide")

    # Quelques params que l’on expose pour transparence (s’ils existent)
    params = {
        "bin_minutes": int(os.environ.get("BIN_MINUTES","5")),
        "window_hours_serving": int(os.environ.get("WINDOW_HOURS","4")),
        "horizons_min": os.environ.get("FORECAST_HORIZONS","15,60"),
        "penury_threshold": int(os.environ.get("PENURY_THRESH","2")),
        "saturation_threshold": int(os.environ.get("SATURATION_THRESH","2")),
        "events_window_days": int(os.environ.get("EVENTS_WINDOW_DAYS","30")),
        "perf_window_days": int(os.environ.get("PERF_WINDOW_DAYS","30")),
        "network_window_days": int(os.environ.get("NETWORK_WINDOW_DAYS","14")),
        "drift_features": os.environ.get("DRIFT_FEATURES","occ_ratio,temp_C,precip_mm,wind_mps,capacity"),
        "drift_bins": int(os.environ.get("DRIFT_BINS","30")),
    }

    sections = [
        {
            "title": "Ingestion & Bronze",
            "body": (
                "Snapshots 5-min collectés depuis l’API Vélib’ + météo. "
                "Stockage partitionné GCS: bronze/date=YYYY-MM-DD/hour=HH/*.parquet."
            )
        },
        {
            "title": "Compaction Daily & Monthly",
            "body": (
                "Daily: déduplication (dernier ts_utc par (station_id, tbin_utc)), "
                "schéma strict, filtrage sur le jour UTC → compact_YYYY-MM-DD.parquet. "
                "Monthly: concat des daily du mois → compact_YYYYMM.parquet. "
                "Suppression des partitions sources si période close."
            )
        },
        {
            "title": "Exports de Référence",
            "body": (
                "events.parquet (bikes/capacity/occ_ratio, flags pénurie/saturation) sur fenêtre roulante; "
                "perf.parquet (paires (t, t+h) par horizon)."
            )
        },
        {
            "title": "Features & Serving Forecast",
            "body": (
                "Fenêtre 4h (48 bins) par station → lags/rolling/trends + features calendaires Paris. "
                "Inférence modèles (15 et 60 min par défaut) → latest_forecast.json unique."
            )
        },
        {
            "title": "Monitoring JSON",
            "body": (
                "Model perf: daily & segments; Network: dynamics/stations; Drift: PSI/KS distributions; "
                "Docs: data dictionary, methodology, data exports. Un manifest global indexe tous les JSON."
            )
        },
        {
            "title": "Métriques & Baselines",
            "body": (
                "Évaluation par horizon: MAE/RMSE modèle si y_pred, baseline persistance y(t). "
                "Lift = 1 - (MAE_model / MAE_baseline)."
            )
        },
        {
            "title": "Qualité des Données",
            "body": (
                "Health: volumes, bins, nulls. Network/stations: coverage ~288 bins/jour, "
                "volatilité (écart-type de occ_ratio), profils horaires moyens."
            )
        },
        {
            "title": "Drift",
            "body": (
                "Comparaison référence vs actuel sur distributions (bins p1–p99), PSI/KS + deltas moments. "
                "Seuils simples PSI (0.2/0.3) & KS (0.2/0.3) pour flags warn/alert."
            )
        }
    ]

    out = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "params": params,
        "sections": sections
    }

    base = f"{MON_PREFIX.rstrip('/')}/docs"
    tag  = datetime.now(timezone.utc).strftime("%Y%m%d")
    _upload_json_gs(out, f"{base}/data_methodology.json")
    _upload_json_gs(out, f"{base}/data_methodology_{tag}.json")
    print("[methodology] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
    