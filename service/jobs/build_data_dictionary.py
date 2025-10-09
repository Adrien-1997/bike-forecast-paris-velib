# service/jobs/build_data_dictionary.py
from __future__ import annotations
import os, sys, json
from io import BytesIO
from typing import Dict, List, Any
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception as e:
    raise RuntimeError("pyarrow requis") from e

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    raise RuntimeError("google-cloud-storage requis") from e

SCHEMA_VERSION = "1.0"

# ─────────────── GCS helpers ───────────────
def _split(gs: str):
    assert gs.startswith("gs://"), f"bad GCS uri: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k.rstrip("/")

def _read_parquet_gs(gs_uri: str) -> pd.DataFrame:
    bkt, key = _split(gs_uri)
    buf = BytesIO()
    storage.Client().bucket(bkt).blob(key).download_to_file(buf)
    buf.seek(0)
    tbl = pq.read_table(buf)
    return tbl.to_pandas()

def _upload_json_gs(obj: dict, gs_uri: str):
    bkt, key = _split(gs_uri)
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    storage.Client().bucket(bkt).blob(key).upload_from_string(data, content_type="application/json")
    print(f"[data_dict] wrote → {gs_uri} ({len(data):,} bytes)")

# ─────────────── Domain knowledge ───────────────
_DESCR: Dict[str, Dict[str, str]] = {
    # communes aux deux datasets
    "tbin_utc": {"unit": "UTC datetime (5-min bin)", "desc": "Horodatage arrondi à 5 minutes (UTC)."},
    "station_id": {"unit": "string", "desc": "Identifiant de station (stable)."},
    "bikes": {"unit": "count", "desc": "Nombre de vélos disponibles."},
    "capacity": {"unit": "count", "desc": "Capacité (nombre d’emplacements)."},
    "mechanical": {"unit": "count", "desc": "Vélos mécaniques disponibles."},
    "ebike": {"unit": "count", "desc": "Vélos électriques disponibles."},
    "status": {"unit": "string", "desc": "Statut opérationnel brut (source API)."},
    "status_code": {"unit": "int", "desc": "Encodage numérique du statut."},
    "lat": {"unit": "deg", "desc": "Latitude WGS84."},
    "lon": {"unit": "deg", "desc": "Longitude WGS84."},
    "name": {"unit": "string", "desc": "Nom court de la station."},
    "temp_C": {"unit": "°C", "desc": "Température (météo) au bin."},
    "precip_mm": {"unit": "mm", "desc": "Précipitations (météo) au bin."},
    "wind_mps": {"unit": "m/s", "desc": "Vent (météo) au bin."},
    "occ_ratio": {"unit": "ratio [0,1]", "desc": "Taux d’occupation = bikes / capacity."},
    "is_penury": {"unit": "0/1", "desc": "Flag pénurie (bikes <= seuil)."},
    "is_saturation": {"unit": "0/1", "desc": "Flag saturation (capacity - bikes <= seuil)."},
    # perf-specific
    "horizon_bins": {"unit": "bins (5-min)", "desc": "Horizon de prédiction en pas de 5 minutes."},
    "y_true": {"unit": "count", "desc": "Vérité terrain (bikes à t+h)."},
    "y_baseline_persist": {"unit": "count", "desc": "Baseline (persistance): bikes(t)."},
    "y_pred": {"unit": "count", "desc": "Prédiction modèle (si disponible)."},
}

# colonnes dans chaque dataset
_EVENTS_COL_ORDER = [
    "tbin_utc","station_id","bikes","capacity","mechanical","ebike","status","status_code",
    "lat","lon","name","temp_C","precip_mm","wind_mps","occ_ratio","is_penury","is_saturation"
]
_PERF_COL_ORDER = [
    "tbin_utc","station_id","horizon_bins","y_true","y_baseline_persist","y_pred",
    "bikes","capacity","occ_ratio"
]

# ─────────────── Inference helpers ───────────────
def _logical_type(s: pd.Series) -> str:
    dt = str(s.dtype)
    if "datetime64" in dt:
        return "datetime"
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        return "float"
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    return "string"

def _examples(s: pd.Series, k: int = 3) -> List[Any]:
    vals = s.dropna().unique()
    if len(vals) == 0:
        return []
    # échantillon stable (trié)
    try:
        sample = sorted(vals.tolist(), key=lambda x: str(x))[:k]
    except Exception:
        sample = vals[:k]
    # sérialisation propre pour datetimes
    out = []
    for v in sample:
        if isinstance(v, (pd.Timestamp, np.datetime64)):
            vv = pd.to_datetime(v, errors="coerce", utc=True)
            out.append(vv.isoformat().replace("+00:00","Z"))
        else:
            out.append(v if (isinstance(v, (int,float)) or v is None) else str(v))
    return out

def _null_rate(s: pd.Series) -> float:
    return float(100.0 * s.isna().mean())

def _cardinality(s: pd.Series, max_compute: int = 1_000_000) -> int | None:
    if len(s) > max_compute:
        return None
    try:
        return int(s.nunique(dropna=True))
    except Exception:
        return None

def _describe_frame(df: pd.DataFrame, ordered_cols: List[str]) -> List[Dict[str, Any]]:
    # assure présence des colonnes (même si absentes → null)
    for c in ordered_cols:
        if c not in df.columns:
            df[c] = pd.NA
    # restreint & typage
    d = df[ordered_cols].copy()
    # coercions utiles
    if "tbin_utc" in d.columns:
        d["tbin_utc"] = pd.to_datetime(d["tbin_utc"], errors="coerce")
    if "station_id" in d.columns:
        d["station_id"] = d["station_id"].astype("string")

    out: List[Dict[str, Any]] = []
    for c in ordered_cols:
        s = d[c]
        info = {
            "name": c,
            "type": _logical_type(s),
            "unit": _DESCR.get(c, {}).get("unit"),
            "description": _DESCR.get(c, {}).get("desc"),
            "null_rate_pct": round(_null_rate(s), 2),
            "examples": _examples(s),
        }
        # cardinalité pour colonnes "catégorielles" (string, int petits)
        if info["type"] in ("string", "int"):
            info["cardinality"] = _cardinality(s)
        out.append(info)
    return out

# ─────────────── Main ───────────────
def main() -> int:
    EXPORTS_PREFIX = os.environ.get("GCS_EXPORTS_PREFIX")
    MON_PREFIX     = os.environ.get("GCS_MONITORING_PREFIX")
    if not (EXPORTS_PREFIX and EXPORTS_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_EXPORTS_PREFIX manquant ou invalide")
    if not (MON_PREFIX and MON_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX manquant ou invalide")

    events_uri = f"{EXPORTS_PREFIX.rstrip('/')}/events.parquet"
    perf_uri   = f"{EXPORTS_PREFIX.rstrip('/')}/perf.parquet"

    print(f"[data_dict] read: {events_uri}")
    try:
        ev = _read_parquet_gs(events_uri)
    except Exception as e:
        print(f"[data_dict][warn] events read failed: {e}")
        ev = pd.DataFrame()

    print(f"[data_dict] read: {perf_uri}")
    try:
        pf = _read_parquet_gs(perf_uri)
    except Exception as e:
        print(f"[data_dict][warn] perf read failed: {e}")
        pf = pd.DataFrame()

    # échantillonnage léger si énorme
    def _sample(df: pd.DataFrame, n: int = 1_000_000) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        return df.sample(n=min(len(df), n), random_state=42) if len(df) > n else df

    ev_s = _sample(ev, 1_000_000)
    pf_s = _sample(pf, 1_000_000)

    items: List[Dict[str, Any]] = []
    if not ev_s.empty:
        items.append({
            "dataset": "events",
            "path": events_uri,
            "row_count_est": int(len(ev)),
            "columns": _describe_frame(ev_s, _EVENTS_COL_ORDER),
        })
    if not pf_s.empty:
        items.append({
            "dataset": "perf",
            "path": perf_uri,
            "row_count_est": int(len(pf)),
            "columns": _describe_frame(pf_s, _PERF_COL_ORDER),
        })

    out = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "datasets": items,
    }

    anchor_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    base = f"{MON_PREFIX.rstrip('/')}/docs"
    _upload_json_gs(out, f"{base}/data_dictionary.json")
    _upload_json_gs(out, f"{base}/data_dictionary_{anchor_tag}.json")
    print("[data_dict] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
