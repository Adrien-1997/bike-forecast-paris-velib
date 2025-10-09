# service/jobs/build_data_exports.py
from __future__ import annotations
import os, sys, json
from datetime import datetime, timezone
from typing import Dict, List, Any

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage requis") from e

SCHEMA_VERSION = "1.0"

def _split(gs: str):
    assert gs.startswith("gs://"), f"bad GCS uri: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k.rstrip("/")

def _list_objects(gs_prefix: str, suffixes: List[str] | None = None) -> List[Dict[str, Any]]:
    bkt, pfx = _split(gs_prefix)
    cli = storage.Client()
    out: List[Dict[str, Any]] = []
    for b in cli.list_blobs(bkt, prefix=pfx + "/"):
        if suffixes and not any(b.name.endswith(suf) for suf in suffixes):
            continue
        updated = b.updated or b.time_created
        updated = updated.replace(tzinfo=timezone.utc) if updated and updated.tzinfo is None else updated
        out.append({
            "path": f"gs://{bkt}/{b.name}",
            "name": b.name.rsplit("/", 1)[-1],
            "size_bytes": int(b.size or 0),
            "updated_iso": (updated.astimezone(timezone.utc).isoformat().replace("+00:00","Z") if updated else None),
        })
    return sorted(out, key=lambda r: r["name"])

def _upload_json_gs(obj: dict, gs_uri: str):
    bkt, key = _split(gs_uri)
    storage.Client().bucket(bkt).blob(key).upload_from_string(
        json.dumps(obj, ensure_ascii=False),
        content_type="application/json"
    )
    print(f"[data_exports] wrote → {gs_uri}")

def main() -> int:
    EXPORTS_PREFIX = os.environ.get("GCS_EXPORTS_PREFIX")
    MON_PREFIX     = os.environ.get("GCS_MONITORING_PREFIX")
    SERVING_PREFIX = os.environ.get("SERVING_FORECAST_PREFIX")  # optionnel
    DAILY_PREFIX   = os.environ.get("GCS_DAILY_PREFIX")         # optionnel
    MONTHLY_PREFIX = os.environ.get("GCS_MONTHLY_PREFIX")       # optionnel

    if not (EXPORTS_PREFIX and EXPORTS_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_EXPORTS_PREFIX manquant ou invalide")
    if not (MON_PREFIX and MON_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX manquant ou invalide")

    # Exports (canoniques)
    exports_items = _list_objects(EXPORTS_PREFIX, suffixes=[".parquet", ".json"])
    # Serving (dernier forecast)
    serving_items = _list_objects(SERVING_PREFIX, suffixes=[".json"]) if (SERVING_PREFIX and SERVING_PREFIX.startswith("gs://")) else []
    # Dailies / Monthly (inventaire rapide)
    daily_items   = _list_objects(DAILY_PREFIX, suffixes=[".parquet"]) if (DAILY_PREFIX and DAILY_PREFIX.startswith("gs://")) else []
    monthly_items = _list_objects(MONTHLY_PREFIX, suffixes=[".parquet"]) if (MONTHLY_PREFIX and MONTHLY_PREFIX.startswith("gs://")) else []

    def _pick(names: List[str], items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        index = {i["name"]: i for i in items}
        out = []
        for n in names:
            if n in index:
                out.append(index[n])
        return out

    highlight_exports = _pick(["events.parquet", "perf.parquet"], exports_items)
    highlight_serving = _pick(["latest_forecast.json"], serving_items)

    out = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "paths": {
            "exports_prefix": EXPORTS_PREFIX.rstrip("/"),
            "serving_prefix": SERVING_PREFIX.rstrip("/") if SERVING_PREFIX else None,
            "daily_prefix": DAILY_PREFIX.rstrip("/") if DAILY_PREFIX else None,
            "monthly_prefix": MONTHLY_PREFIX.rstrip("/") if MONTHLY_PREFIX else None,
        },
        "exports": {
            "highlight": highlight_exports,
            "all": exports_items,
        },
        "serving": {
            "highlight": highlight_serving,
            "all": serving_items,
        },
        "inventory": {
            "daily_count": len(daily_items),
            "monthly_count": len(monthly_items),
        }
    }

    base = f"{MON_PREFIX.rstrip('/')}/docs"
    tag  = datetime.now(timezone.utc).strftime("%Y%m%d")
    _upload_json_gs(out, f"{base}/data_exports.json")
    _upload_json_gs(out, f"{base}/data_exports_{tag}.json")
    print("[data_exports] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
