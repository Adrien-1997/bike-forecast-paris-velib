# service/jobs/build_monitoring.py
from __future__ import annotations
import os, sys, json
from datetime import datetime, timezone
from typing import Dict, List, Any

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage requis") from e

SCHEMA_VERSION = "1.0"

CATEGORIES = {
    "model/perf/": "model_perf",
    "network/": "network",
    "drift/": "drift",
    "docs/": "docs",
}

def _split(gs: str):
    assert gs.startswith("gs://"), f"bad GCS uri: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k.rstrip("/")

def _category_for(key: str) -> str:
    for pfx, cat in CATEGORIES.items():
        if f"/{pfx}" in key:
            return cat
    return "other"

def _summarize_versions(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Regroupe par 'base' (fichier alias sans suffixe date)
    # ex: dynamics.json et dynamics_YYYYMMDD.json → base=dynamics.json
    by_base: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        name = it["name"]
        if "_" in name and name.rsplit("_", 1)[-1].startswith("20") and name.endswith(".json"):
            base = name[: name.rfind("_")] + ".json"
        else:
            base = name
        by_base.setdefault(base, []).append(it)

    out = []
    for base, files in by_base.items():
        files_sorted = sorted(files, key=lambda r: r["updated"], reverse=True)
        out.append({
            "base": base,
            "count": len(files_sorted),
            "latest": files_sorted[0]["path"] if files_sorted else None,
            "latest_updated": files_sorted[0]["updated_iso"] if files_sorted else None,
        })
    out.sort(key=lambda r: r["base"])
    return {"resources": out}

def main() -> int:
    MON_PREFIX = os.environ.get("GCS_MONITORING_PREFIX")
    if not (MON_PREFIX and MON_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX manquant ou invalide")

    bkt, pfx = _split(MON_PREFIX)
    cli = storage.Client()

    items: List[Dict[str, Any]] = []
    for b in cli.list_blobs(bkt, prefix=pfx + "/"):
        if not b.name.endswith(".json"):
            continue
        cat = _category_for(b.name)
        updated = b.updated or b.time_created
        updated = updated.replace(tzinfo=timezone.utc) if updated.tzinfo is None else updated.astimezone(timezone.utc)
        items.append({
            "path": f"gs://{bkt}/{b.name}",
            "bucket": bkt,
            "key": b.name,
            "name": b.name.rsplit("/", 1)[-1],
            "size": int(b.size or 0),
            "category": cat,
            "updated": updated.timestamp(),
            "updated_iso": updated.isoformat().replace("+00:00","Z"),
        })

    items.sort(key=lambda r: (r["category"], r["name"], -r["updated"]))

    summary_by_cat: Dict[str, Dict[str, Any]] = {}
    for cat in sorted(set(i["category"] for i in items)):
        summary_by_cat[cat] = _summarize_versions([i for i in items if i["category"] == cat])

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "root": MON_PREFIX.rstrip("/"),
        "total_items": len(items),
        "items": items,
        "summary": summary_by_cat,
    }

    out_key = f"{pfx.rstrip('/')}/manifest.json"
    storage.Client().bucket(bkt).blob(out_key).upload_from_string(
        json.dumps(manifest, ensure_ascii=False), content_type="application/json"
    )
    print(f"[manifest] wrote → gs://{bkt}/{out_key} (items={len(items)})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
