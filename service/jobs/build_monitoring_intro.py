# service/jobs/build_monitoring_intro.py

# Vélib’ Forecast — Monitoring Intro (v1.3)
#
# Rôle
# ----
# Agrège des indicateurs clés pour l’entête/intro du site Monitoring à partir
# des JSON déjà produits :
#   - network/overview/latest/kpis.json
#   - data/health/latest/kpis.json       (pour coverage uniquement — fallback)
#   - data/freshness/latest.json         (NOUVEAU — stations & météo)
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
# ENV optionnels (URIs sources)
# -----------------------------
#   MON_OVERVIEW_KPIS_URI     gs://.../monitoring/network/overview/latest/kpis.json
#   MON_HEALTH_KPIS_URI       gs://.../monitoring/data/health/latest/kpis.json
#   MON_FRESHNESS_URI         gs://.../monitoring/data/freshness/latest.json   ← NEW
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
#   INTRO_LED_FRESH_OK_MIN    défaut "5"   (fraîcheur OK - utilisé pour la météo)
#   INTRO_LED_FRESH_WARN_MIN  défaut "12"  (fraîcheur WARN - utilisé pour la météo)
#
# Notes
# -----
# - Document JSON “safe” (NaN → null).
# - Tolérant aux fichiers manquants.
# - H15/H60 distincts + bloc "global" backward-compat (h15 sinon h60).
# - Compat UI: on expose la fraîcheur météo sous « freshness_p95_min » dans
#   statuses.weather_provider, même si c’est une valeur unique (pas un p95).
# =============================================================================

"""
Job Monitoring Intro pour Vélib’ Forecast.

Ce job agrège un **ensemble compact de statuts et de signaux KPI** pour
l’en-tête de la page d’accueil Monitoring, en lisant les artefacts de
monitoring déjà produits :

- KPIs overview réseau (stations actives…)
- KPIs de santé des données (coverage sur 7 jours)
- Fraîcheur des données (stations & météo)
- Résumé de data drift (PSI)
- Métadonnées des modèles (versions h15 / h60)
- Derniers fichiers de batch forecasts pour h15 et h60

Il produit deux documents JSON sous :

- `<GCS_MONITORING_PREFIX>/monitoring/intro/latest/intro.json`
- `<GCS_MONITORING_PREFIX>/monitoring/intro/latest/{ISO}.json`  (snapshot daté)

Le JSON est :
- résilient aux entrées manquantes (n’importe quelle source peut être absente),
- nettoyé des NaN/Inf (convertis en null),
- structuré pour être consommé directement par l’UI Monitoring.
"""

from __future__ import annotations
import os, json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from google.cloud import storage
import numpy as np

SCHEMA_VERSION = "1.3"
TZ = "Europe/Paris"

# ─────────────────────────── Helpers ENV / GCS ───────────────────────────

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Lire une variable d’environnement avec valeur par défaut.

    Paramètres
    ----------
    name : str
        Nom de la variable d’environnement.
    default : str | None
        Valeur de repli retournée si la variable est absente ou vide.

    Retour
    ------
    str | None
        Valeur brute de l’environnement ou valeur par défaut.
    """
    v = os.environ.get(name)
    return v if (v is not None and v != "") else default


def _ensure_mon_base(mon_prefix: str) -> str:
    """
    S’assurer qu’un préfixe GCS se termine par `/monitoring`.

    Exemples
    --------
    - `gs://bucket/velib`      → `gs://bucket/velib/monitoring`
    - `gs://bucket/velib/monitoring` (inchangé)

    Paramètres
    ----------
    mon_prefix : str
        Préfixe racine de monitoring (GCS_MONITORING_PREFIX).

    Retour
    ------
    str
        Préfixe garanti de se terminer par `/monitoring`.
    """
    base = mon_prefix.rstrip("/")
    if not base.endswith("/monitoring"):
        base = base + "/monitoring"
    return base


def _split_gs(gs: str):
    """
    Découper une URI GCS `gs://bucket/path` en (bucket, key).

    Paramètres
    ----------
    gs : str
        URI GCS.

    Retour
    ------
    (str, str)
        Nom du bucket, clé objet (sans slash final).

    Lève
    ----
    AssertionError
        Si l’URI ne commence pas par `gs://`.
    """
    assert gs.startswith("gs://"), f"bad GCS URI: {gs}"
    b, k = gs[5:].split("/", 1)
    return b, k


def _read_json_gcs(client: storage.Client, gs: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Lire et parser un objet JSON depuis GCS.

    Paramètres
    ----------
    client : google.cloud.storage.Client
        Instance de client GCS.
    gs : str | None
        URI GCS du fichier JSON.

    Retour
    ------
    dict | None
        Document JSON parsé, ou None si l’URI est absente/invalide ou si
        une erreur d’I/O / parsing survient (comportement best-effort).
    """
    if not gs or not gs.startswith("gs://"):
        return None
    try:
        b, k = _split_gs(gs)
        data = client.bucket(b).blob(k).download_as_bytes()
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def _san(o):
    """
    Nettoyer récursivement les valeurs pour la sérialisation JSON.

    - scalaires numpy → types natifs Python,
    - NaN / Inf       → None,
    - pandas NA       → None,
    - autres types    → laissés intacts.

    Cela garantit que le JSON final est "safe" et ne contient pas de valeurs
    numériques invalides pour l’UI.
    """
    if isinstance(o, dict):
        return {k: _san(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_san(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if (not isinstance(o, float) or np.isfinite(o)) else None
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if o is None:
        return None
    try:
        import pandas as pd  # type: ignore
        if pd.isna(o):
            return None
    except Exception:
        # Si pandas n’est pas dispo ou si isna échoue, on renvoie la valeur brute.
        pass
    return o


def _write_json_gcs(client: storage.Client, gs: str, doc: Dict[str, Any]) -> None:
    """
    Sérialiser un document en JSON et l’uploader sur GCS avec des options sûres.

    - `cache_control` est positionné à "no-store" (toujours frais pour l’UI).
    - Le content-type est `application/json; charset=utf-8`.
    - Le document est nettoyé pour retirer NaN/Inf.

    Paramètres
    ----------
    client : google.cloud.storage.Client
        Instance de client GCS.
    gs : str
        URI GCS de destination.
    doc : dict
        Payload JSON-sérialisable (sera nettoyé).
    """
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
    """
    Calculer l’âge (en minutes) depuis un timestamp ISO8601 donné.

    Paramètres
    ----------
    iso : str | None
        Timestamp ISO, éventuellement suffixé par 'Z'.

    Retour
    ------
    float | None
        Âge en minutes (>= 0) ou None si le parsing échoue.
    """
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 60.0)
    except Exception:
        return None


def _led_from_value(x: Optional[float], ok: float, warn: float, reverse: bool = False) -> str:
    """
    Mapper une valeur numérique vers un statut LED simple ("ok", "warn", "down").

    Comportement
    ------------
    - Si x est None/non-numérique → "down".
    - Si reverse est False :
        * x <= ok   → "ok"
        * x <= warn → "warn"
        * sinon     → "down"
    - Si reverse est True, la comparaison se fait sur -x.

    Paramètres
    ----------
    x : float | None
        Valeur à évaluer (ex. âge en minutes).
    ok : float
        Seuil en dessous duquel le statut est "ok".
    warn : float
        Seuil en dessous duquel le statut est "warn".
    reverse : bool, défaut False
        Si True, on raisonne sur -x (utile pour des scores où "plus grand" = mieux).

    Retour
    ------
    str
        L’un de "ok", "warn", "down".
    """
    if x is None or not isinstance(x, (int, float)):
        return "down"
    v = -x if reverse else x
    if v <= ok:
        return "ok"
    if v <= warn:
        return "warn"
    return "down"


def _led_from_psi(psi: Optional[float]) -> str:
    """
    Logique LED pour le statut de data drift basé sur le PSI.

    Seuils
    ------
    - PSI < 0.10 → "ok"
    - PSI < 0.20 → "warn"
    - sinon      → "down"

    Paramètres
    ----------
    psi : float | None
        Valeur du Population Stability Index.

    Retour
    ------
    str
        "ok", "warn" ou "down".
    """
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
    Parser un document JSON de forecast et extraire les métadonnées de base.

    Formes acceptées
    ----------------
    A) Document "bundle" :
       {
         "generated_at": "...Z",
         "horizon_min": 15,
         "data": [ ... ]
       }

    B) Variante :
       {
         "generated_at": "...Z",
         "predictions": [ ... ]
       }

    C) Legacy :
       [
         {...},
         {...}
       ]

    Retour
    ------
    (str | None, int | None)
        Tuple (generated_at_iso, rows_count).
        Les valeurs manquantes sont renvoyées à None.
    """
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
    """
    Entrypoint CLI pour le job Monitoring Intro.

    Pipeline high-level
    -------------------
    1. Lire `GCS_MONITORING_PREFIX` et résoudre le préfixe de base monitoring.
    2. Configurer les seuils LED (âge forecast, fraîcheur météo).
    3. Résoudre les URIs de toutes les sources amont :
       - KPIs overview réseau
       - KPIs de santé des données
       - Fraîcheur des données (stations + météo)
       - Résumé de drift
       - Modèles h15/h60
       - Derniers bundles de forecast h15/h60
    4. Lire ces documents JSON GCS en mode **best-effort** :
       fichiers manquants → dicts vides.
    5. Agréger les KPIs clés :
       - stations actives
       - p95 de fraîcheur pour les stations
       - fraîcheur météo (minutes)
       - couverture sur 7 jours
       - PSI global et top feature de drift
       - versions de modèles h15/h60
       - âge et volume des forecasts
    6. Calculer les statuts LED :
       - `api_stations` (actives > 0)
       - `batch_forecast_h15`, `batch_forecast_h60`, `batch_forecast` global
       - `weather_provider`
       - `data_drift` (via PSI)
    7. Construire un document `intro.json` compact avec :
       - `kpis` (chiffres pour les cartes top-level),
       - `statuses` (blocs LED détaillés),
       - `activity` (liste de petits résumés),
       - `sources` (URIs utilisées).
    8. Écrire :
       - `intro/latest/intro.json` (alias mouvant),
       - `intro/latest/{ISO}.json` (snapshot daté).

    Retour
    ------
    int
        Code de sortie (0 en cas de succès).
    """
    MON_PREFIX = _env("GCS_MONITORING_PREFIX")
    if not (MON_PREFIX and MON_PREFIX.startswith("gs://")):
        raise RuntimeError("GCS_MONITORING_PREFIX manquant ou invalide")
    mon_base = _ensure_mon_base(MON_PREFIX)

    # Seuils LED configurables
    LED_FC_OK   = float(_env("INTRO_LED_FC_OK_MIN", "7"))
    LED_FC_WARN = float(_env("INTRO_LED_FC_WARN_MIN", "20"))
    LED_FR_OK   = float(_env("INTRO_LED_FRESH_OK_MIN", "5"))    # utilisé pour MÉTÉO
    LED_FR_WARN = float(_env("INTRO_LED_FRESH_WARN_MIN", "12"))

    # URIs par défaut (surchageables via ENV)
    overview_kpis_uri = _env("MON_OVERVIEW_KPIS_URI", f"{mon_base}/network/overview/latest/kpis.json")
    health_kpis_uri   = _env("MON_HEALTH_KPIS_URI",   f"{mon_base}/data/health/latest/kpis.json")
    freshness_uri     = _env("MON_FRESHNESS_URI",     f"{mon_base}/data/freshness/latest.json")  # NEW
    drift_summary_uri = _env("MON_DRIFT_SUMMARY_URI", f"{mon_base}/data/drift/latest/summary.json")

    # Modèles / Forecasts
    proj_base = mon_base.rsplit("/monitoring", 1)[0]
    model_h15_uri = _env("MON_MODEL_LATEST_H15_URI", f"{proj_base}/models/h15/latest.json")
    model_h60_uri = _env("MON_MODEL_LATEST_H60_URI", f"{proj_base}/models/h60/latest.json")
    fc_h15_uri    = _env("MON_FORECAST_H15_URI",     f"{proj_base}/serving/forecast/h15/latest.json")
    fc_h60_uri    = _env("MON_FORECAST_H60_URI",     f"{proj_base}/serving/forecast/h60/latest.json")

    client = storage.Client()

    overview   = _read_json_gcs(client, overview_kpis_uri) or {}
    health     = _read_json_gcs(client, health_kpis_uri)   or {}
    freshness  = _read_json_gcs(client, freshness_uri)     or {}  # ← NEW
    drift      = _read_json_gcs(client, drift_summary_uri) or {}

    model15    = _read_json_gcs(client, model_h15_uri) or {}
    model60    = _read_json_gcs(client, model_h60_uri) or {}

    fc15_doc   = _read_json_gcs(client, fc_h15_uri) or {}
    fc60_doc   = _read_json_gcs(client, fc_h60_uri) or {}

    # Modèles (fallback lisible)
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

    # NEW: fraîcheur stations & météo depuis data/freshness/latest.json
    st_fresh_p95 = None
    met_fresh    = None
    try:
        st = (freshness.get("stations") or {}).get("freshness") or {}
        st_fresh_p95 = st.get("p95_min")
    except Exception:
        st_fresh_p95 = None
    try:
        met = (freshness.get("weather") or {})
        met_fresh = met.get("freshness_min")
    except Exception:
        met_fresh = None

    # Couverture sur 7 jours — on s’appuie sur health KPI comme source (si dispo)
    coverage_7d = health.get("coverage_global_pct")

    psi_global      = drift.get("psi_global")
    drift_top_feat  = drift.get("top_feature")
    drift_top_psi   = drift.get("top_feature_psi")
    ts_drift        = drift.get("generated_at")

    ts_overview = overview.get("generated_at")
    ts_health   = health.get("generated_at")       # seulement pour "source_generated_at" de coverage
    ts_fresh    = freshness.get("now_utc") or freshness.get("generated_at")

    # LEDs
    api_led         = "ok" if (isinstance(stations_active, (int, float)) and stations_active > 0) else "down"
    bat_led15       = _led_from_value(age_fc15, LED_FC_OK, LED_FC_WARN)
    bat_led60       = _led_from_value(age_fc60, LED_FC_OK, LED_FC_WARN)
    bat_led_global  = _led_from_value(age_forecast_global, LED_FC_OK, LED_FC_WARN)

    # IMPORTANT: la LED météo se base désormais sur la fraîcheur météo réelle (met_fresh)
    met_led         = _led_from_value(met_fresh, LED_FR_OK, LED_FR_WARN)

    drift_led       = _led_from_psi(psi_global if isinstance(psi_global, (int, float)) else None)

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso,
        "tz": TZ,
        "kpis": {
            "stations_active": stations_active,
            # NEW: fraîcheur = p95 stations (depuis data/freshness)
            "freshness_p95_min": st_fresh_p95,
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
            # Compat UI: expose la fraîcheur météo sous freshness_p95_min
            # (mais c'est bien une "freshness_min" unique côté collecte météo)
            "weather_provider": {
                "led": met_led,
                "freshness_p95_min": met_fresh,
                "source_generated_at": ts_fresh,
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
            "freshness": freshness_uri,             # ← NEW
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