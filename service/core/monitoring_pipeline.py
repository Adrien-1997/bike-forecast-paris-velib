# service/core/monitoring_pipeline.py

# =============================================================================
#  Vélib’ Forecast — Pipeline de Monitoring (v2)
# =============================================================================
#  Fichier : service/core/monitoring_pipeline.py
#  Rôle :
#     Orchestrateur central de la brique Monitoring. Exécute séquentiellement
#     les jobs d’analyse qui produisent tous les artefacts JSON consommés par
#     l’UI de monitoring :
#
#       ├── data_health           → couverture des données, fraîcheur, manquants
#       ├── data_drift            → analyse de dérive des features / de la cible
#       ├── model_performance     → performance quotidienne (MAE/RMSE/…)
#       ├── model_explainability  → importance des features, calibration, résiduels
#       ├── network_overview      → KPIs réseau globaux et heatmaps
#       ├── network_dynamics      → évolution horaire, tension, volatilité
#       ├── network_stations      → profils et résumés par station
#       └── intro                 → métadonnées (dates, versions, liens)
#
#  Structure de sortie (GCS) :
#       gs://<project>_cloudbuild/velib/monitoring/
#           ├── data/health/latest/*.json
#           ├── data/drift/latest/*.json
#           ├── model/performance/latest/h15/*.json
#           ├── model/explainability/latest/h60/*.json
#           ├── network/overview/latest/*.json
#           ├── network/dynamics/latest/*.json
#           ├── network/stations/latest/*.json
#           └── intro/latest/intro.json
#
#  Chaque job écrit un bundle JSON atomique suivant le schéma :
#      {
#          "schema_version": "1.x",
#          "generated_at": "2025-11-02T08:15:00Z",
#          "meta": {...},
#          "data": [...]
#      }
#
# =============================================================================
#  UTILISATION
# -----------------------------------------------------------------------------
#   python -m service.core.monitoring_pipeline
#
#   Flags CLI optionnels :
#     --steps=STEP1,STEP2,...   → surcharge de la liste de steps
#                                 (par défaut = DEFAULT_STEPS)
#
#   Exemple :
#     python -m service.core.monitoring_pipeline --steps=data_health,model_performance
#
# =============================================================================
#  VARIABLES D’ENVIRONNEMENT
# -----------------------------------------------------------------------------
#  ─── Requises ────────────────────────────────────────────────────────────────
#   GCS_EXPORTS_PREFIX     gs://.../velib/exports      (lecture events_*.parquet, perf_*.parquet)
#   GCS_MONITORING_PREFIX  gs://.../velib/monitoring   (écriture des JSON de monitoring)
#
#  ─── Recommandées ───────────────────────────────────────────────────────────
#   FORECAST_HORIZONS      15,60
#   MODEL_URI_15           gs://.../models/h15/latest.joblib
#   MODEL_URI_60           gs://.../models/h60/latest.joblib
#   TZ_APP                 Europe/Paris
#
#  ─── Optionnelles (pilotage de la pipeline) ─────────────────────────────────
#   STEPS                  Surcharge de la liste de steps (séparées par des virgules)
#   DAY                    Jour cible (YYYY-MM-DD) propagé aux jobs qui l’utilisent
#   CONTINUE_ON_ERROR      1|0  → continuer même si un step échoue (défaut=0)
#   DRY_RUN                1|0  → afficher les steps sans les exécuter
#   PYTHON_BIN             Binaire Python custom (défaut = sys.executable)
#   GCS_LOCK               gs://.../velib/locks/monitoring.lock (prévenir la concurrence)
#
#  ─── Optionnelles (fenêtres temporelles / seuils par job) ───────────────────
#   # Network overview
#   OVERVIEW_TZ, OVERVIEW_LAST_DAYS, OVERVIEW_REF_DAYS
#
#   # Network dynamics
#   DYNAMICS_TZ, DYNAMICS_LAST_DAYS, DYNAMICS_PENURY_THRESH, DYNAMICS_SATURATION_THRESH
#   (fallback : PENURY_THRESH, SATURATION_THRESH)
#
#   # Network stations (clustering/options)
#   NETWORK_WINDOW_DAYS, NETWORK_MIN_BINS_KEEP, NETWORK_K, NETWORK_K_MIN, NETWORK_K_MAX
#
#   # Model performance
#   PERF_TZ, PERF_LAST_DAYS, PERF_HORIZONS, PERF_RESID_BINS, PERF_TOP_STATIONS,
#   PERF_CLUSTERS_CSV, PERF_TS_MIN_POINTS
#
#   # Data health
#   DATA_HEALTH_TZ, DATA_HEALTH_LAST_DAYS, DATA_HEALTH_DAY
#
#   # Data drift
#   DRIFT_TZ, DRIFT_WINDOW_DAYS, DRIFT_FEATURES, DRIFT_BINS
#
#   # Intro (URIs si besoin de surcharger les defaults)
#   INTRO_OVERVIEW_KPIS_URI, INTRO_HEALTH_KPIS_URI, INTRO_DRIFT_SUMMARY_URI,
#   INTRO_MODEL_LATEST_H15_URI, INTRO_MODEL_LATEST_H60_URI,
#   INTRO_FORECAST_H15_URI, INTRO_FORECAST_H60_URI, INTRO_OUT_PREFIX
#
# =============================================================================
#  NOTES
# -----------------------------------------------------------------------------
#   • Les steps sont exécutés séquentiellement ; si l’un échoue et que
#     CONTINUE_ON_ERROR=0, la pipeline s’arrête immédiatement.
#   • Les jobs réécrivent les “latest” et peuvent aussi publier des bundles datés.
#   • Aucun job ne modifie les datasets ou les modèles — uniquement les sorties
#     de monitoring.
#   • Conçu pour tourner quotidiennement via Cloud Run Job ou Cloud Scheduler.
# =============================================================================

"""
Orchestrateur de pipeline de monitoring pour Vélib' Forecast.

Ce module ne fait **qu'une chose** : lancer, dans le bon ordre, tous les
jobs de monitoring (data, modèle, réseau, intro) en appliquant quelques
règles de configuration :

- résolution de la liste de steps (CLI / env / DEFAULT_STEPS),
- application de valeurs par défaut pour les fenêtres temporelles (14 / 28 jours),
- gestion optionnelle d'un verrou GCS (GCS_LOCK) pour éviter les exécutions concurrentes,
- propagation minimale des variables d'environnement (fenêtres, horizons, URIs).

Important :
-----------
- Le pipeline n'écrit **rien** dans les datasets ni les modèles : il ne
  fabrique que des artefacts JSON consommés par l'UI de monitoring.
- Le comportement en cas d'erreur est contrôlé par CONTINUE_ON_ERROR.
"""

from __future__ import annotations
import os, sys, shlex, contextlib
from dataclasses import dataclass
from subprocess import run, CalledProcessError
from typing import List, Dict, Optional

try:
    from google.cloud import storage  # type: ignore
except Exception:
    storage = None  # lock GCS optionnel

# ──────────────────────────────────────────────────────────────────────────────
# Chaîne MONITORING unifiée (datasets/compact exclus par défaut)
# ──────────────────────────────────────────────────────────────────────────────

#: Steps par défaut pour la pipeline de monitoring (ordre logique).
DEFAULT_STEPS = [
    "data_health",
    "data_drift",
    "model_performance",
    "model_explainability",
    "network_overview",
    "network_dynamics",
    "network_stations",
    "intro",
]

# Mapping step -> module exécuté
MODULES: Dict[str, str] = {
    # (Hors monitoring – disponibles mais non exécutés par défaut)
    "compact_daily":        "service.jobs.compact_daily",
    "datasets":             "service.jobs.build_datasets",

    # Monitoring — data
    "data_health":          "service.jobs.build_data_health",
    "data_drift":           "service.jobs.build_data_drift",

    # Monitoring — model
    "model_performance":    "service.jobs.build_model_performance",
    "model_explainability": "service.jobs.build_model_explainability",

    # Monitoring — network
    "network_overview":     "service.jobs.build_network_overview",
    "network_dynamics":     "service.jobs.build_network_dynamics",
    "network_stations":     "service.jobs.build_network_stations",

    # Monitoring — meta
    "intro":                "service.jobs.build_monitoring_intro",
}


@dataclass
class Cfg:
    """
    Configuration runtime de la pipeline de monitoring.

    Attributs
    ---------
    steps : list[str]
        Steps à exécuter (data_health, model_performance, ...).
    dry_run : bool
        Si True, affiche les commandes sans les exécuter.
    continue_on_error : bool
        Si False, arrêt au premier step en erreur.
    python_bin : str
        Binaire Python utilisé pour lancer les modules (par défaut sys.executable).
    gcs_lock : str | None
        URI GCS d'un lock optionnel pour éviter les exécutions concurrentes.
    """
    steps: List[str]
    dry_run: bool
    continue_on_error: bool
    python_bin: str
    gcs_lock: Optional[str]


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Lire une variable d'environnement avec valeur par défaut.

    Paramètres
    ----------
    name : str
        Nom de la variable.
    default : str | None
        Valeur de repli si la variable est absente ou vide.

    Retourne
    --------
    str | None
        Valeur résolue ou `default`.
    """
    v = os.environ.get(name)
    return v if (v is not None and v != "") else default


def _parse_steps(arg_line: Optional[str]) -> List[str]:
    """
    Résoudre la liste de steps à partir du CLI et de l'env.

    Priorité :
    ----------
    1) CLI : --steps=STEP1,STEP2,...
    2) ENV : STEPS=...
    3) DEFAULT_STEPS

    Un step inconnu provoque une ValueError avec la liste des steps valides.
    """
    cli = None
    if arg_line:
        for a in arg_line.split():
            if a.startswith("--steps="):
                cli = a.split("=", 1)[1]
    env = _env("STEPS")
    raw = cli or env
    if not raw:
        return list(DEFAULT_STEPS)
    parts = [s.strip() for s in raw.split(",") if s.strip()]

    # Validation
    out: List[str] = []
    for s in parts:
        if s not in MODULES:
            valid = ", ".join(MODULES.keys())
            raise ValueError(f"unknown step '{s}'. Valid: {valid}")
        out.append(s)
    return out


def _acquire_lock(uri: Optional[str]):
    """
    Tenter d'acquérir un verrou dans GCS pour éviter les exécutions concurrentes.

    Stratégie :
    -----------
    - Si `uri` est None → lock désactivé (retourne None).
    - Si google-cloud-storage n'est pas dispo → avertissement + pas de lock.
    - Si l’upload avec `if_generation_match=0` échoue → le lock existe déjà
      ou erreur de création → on log et on signale l'absence de lock.

    Paramètres
    ----------
    uri : str | None
        URI GCS du lock (ex : gs://bucket/velib/locks/monitoring.lock).

    Retourne
    --------
    google.cloud.storage.blob.Blob | None
        Blob GCS représentant le lock, ou None si non acquis.
    """
    if not uri:
        print("[lock] disabled (no GCS_LOCK)")
        return None
    if storage is None:
        print("[lock][warn] google-cloud-storage not installed → skipping lock")
        return None
    assert uri.startswith("gs://"), f"invalid GCS_LOCK: {uri}"
    bkt, key = uri[5:].split("/", 1)
    blob = storage.Client().bucket(bkt).blob(key)
    try:
        blob.upload_from_string(b"", if_generation_match=0)
        print(f"[lock] acquired → {uri}")
        return blob
    except Exception:
        print("[lock] busy (or create error) → exit 0")
        return None


def _release_lock(blob):
    """
    Libérer le verrou GCS si présent.

    Toute exception lors du delete est silencieusement ignorée.
    """
    if not blob:
        return
    with contextlib.suppress(Exception):
        blob.delete()
        print("[lock] released")


def _run_module(python_bin: str, module: str) -> int:
    """
    Exécuter un module Python (`python -m <module>`) et retourner le code de sortie.

    Paramètres
    ----------
    python_bin : str
        Binaire Python à utiliser.
    module : str
        Nom du module (ex : 'service.jobs.build_data_health').

    Retourne
    --------
    int
        Code de retour du sous-processus (0 = succès).
    """
    cmd = [python_bin, "-m", module]
    print("[run]", " ".join(shlex.quote(x) for x in cmd))
    try:
        run(cmd, check=True)
        return 0
    except CalledProcessError as e:
        print(f"[run][error] {module} → returncode={e.returncode}", file=sys.stderr)
        return e.returncode
    except Exception as e:
        print(f"[run][error] {module} → {e}", file=sys.stderr)
        return 1


def _echo_env(keys: List[str], title: str = "[env]") -> None:
    """
    Loguer une sélection de variables d'environnement (sans secrets).

    Paramètres
    ----------
    keys : list[str]
        Variables à afficher si elles sont présentes.
    title : str
        Préfixe de la ligne de log.
    """
    pairs = []
    for k in keys:
        v = os.environ.get(k)
        if v is not None:
            pairs.append(f"{k}={v}")
    if pairs:
        print(f"{title} " + " ".join(pairs))


# ──────────────────────────────────────────────────────────────────────────────
# Par défaut : fenêtres “jours” → 14 j partout, OVERVIEW à 28 j
# (ne force rien si l’utilisateur a déjà posé la variable)
# ──────────────────────────────────────────────────────────────────────────────

def _set_default(name: str, value: str) -> None:
    """
    Poser une variable d'environnement si elle n'existe pas déjà.

    Paramètres
    ----------
    name : str
        Nom de la variable.
    value : str
        Valeur à définir si absente.
    """
    if not os.environ.get(name):
        os.environ[name] = value


def _apply_default_windows() -> None:
    """
    Appliquer les valeurs par défaut des fenêtres temporelles de monitoring.

    Convention :
    ------------
    - 14 jours partout (santé, drift, perf, dynamiques, stations, explainability),
    - 28 jours pour l'overview réseau (fenêtre plus longue pour les KPIs globaux).

    Important : cette fonction **n'écrase pas** les variables déjà définies.
    Elle ne fait que fixer des valeurs par défaut.
    """
    # Data Health
    _set_default("DATA_HEALTH_LAST_DAYS", "14")     # si le job lit LAST_DAYS
    _set_default("DATA_HEALTH_CURRENT_DAYS", "14")  # compat avec l’implémentation actuelle

    # Data Drift (v1.4 : MON_* pris en priorité)
    _set_default("MON_CURRENT_DAYS", "14")
    _set_default("MON_REFERENCE_DAYS", "14")
    # Compat alternative
    _set_default("DRIFT_CURRENT_DAYS", "14")
    _set_default("DRIFT_REFERENCE_DAYS", "14")

    # Model Performance
    _set_default("PERF_LAST_DAYS", "14")

    # Model Explainability (fenêtre lookback)
    _set_default("LOOKBACK_DAYS", "14")

    # Network Dynamics
    _set_default("DYNAMICS_LAST_DAYS", "14")

    # Network Stations (fenêtre clustering / profils)
    _set_default("NETWORK_WINDOW_DAYS", "14")

    # Network Overview → EXCEPTION : 28 jours
    _set_default("OVERVIEW_LAST_DAYS", "28")
    # (facultatif) référence utilisée dans l’overview si besoin
    _set_default("OVERVIEW_REF_DAYS", "28")


def main(argv: List[str] | None = None) -> int:
    """
    Point d'entrée principal de la pipeline de monitoring.

    Rôle :
    ------
    - Résoudre la configuration (`Cfg`) à partir des arguments + env.
    - Appliquer les valeurs par défaut des fenêtres (14/28 jours).
    - Afficher les variables d'environnement pertinentes (sanity check).
    - Acquérir éventuellement un verrou GCS (GCS_LOCK).
    - Exécuter séquentiellement les steps demandés via `_run_module`.
    - Respecter CONTINUE_ON_ERROR pour décider de s'arrêter ou non.

    Paramètres
    ----------
    argv : list[str] | None
        Arguments CLI (par défaut `sys.argv`).

    Retourne
    --------
    int
        Code de retour global (0 = succès, sinon dernier code d'erreur rencontré).
    """
    argv = sys.argv if argv is None else argv
    arg_line = " ".join(argv[1:]) if len(argv) > 1 else None
    try:
        steps = _parse_steps(arg_line)
    except Exception as e:
        print(f"[cfg][error] {e}", file=sys.stderr)
        return 2

    cfg = Cfg(
        steps=steps,
        dry_run=(_env("DRY_RUN", "0") in ("1","true","True")),
        continue_on_error=(_env("CONTINUE_ON_ERROR", "0") in ("1","true","True")),
        python_bin=_env("PYTHON_BIN", sys.executable) or sys.executable,
        gcs_lock=_env("GCS_LOCK"),  # ex : gs://bucket/velib/locks/monitoring.lock
    )

    # Applique nos défauts 14 j partout / 28 j pour overview (sans écraser l'existant)
    _apply_default_windows()

    # Echo utile (sans secrets) — tronc commun
    _echo_env([
        "GCS_EXPORTS_PREFIX",
        "GCS_MONITORING_PREFIX",
        "FORECAST_HORIZONS",
        "MODEL_URI_15", "MODEL_URI_60",
        "TZ_APP",
        "DAY", "STEPS",
    ])

    # Echo détaillé par domaine (fenêtres, seuils, TZ, options)
    _echo_env([
        # Overview
        "OVERVIEW_TZ", "OVERVIEW_LAST_DAYS", "OVERVIEW_REF_DAYS",
        # Dynamics
        "DYNAMICS_TZ", "DYNAMICS_LAST_DAYS",
        "DYNAMICS_PENURY_THRESH", "DYNAMICS_SATURATION_THRESH",
        "PENURY_THRESH", "SATURATION_THRESH",
        # Stations / clustering
        "NETWORK_WINDOW_DAYS", "NETWORK_MIN_BINS_KEEP",
        "NETWORK_K", "NETWORK_K_MIN", "NETWORK_K_MAX",
        # Performance
        "PERF_TZ", "PERF_LAST_DAYS", "PERF_HORIZONS",
        "PERF_RESID_BINS", "PERF_TOP_STATIONS",
        "PERF_CLUSTERS_CSV", "PERF_TS_MIN_POINTS",
        # Data health
        "DATA_HEALTH_TZ", "DATA_HEALTH_LAST_DAYS", "DATA_HEALTH_CURRENT_DAYS", "DATA_HEALTH_DAY",
        # Drift
        "MON_CURRENT_DAYS", "MON_REFERENCE_DAYS",
        "DRIFT_TZ", "DRIFT_WINDOW_DAYS", "DRIFT_FEATURES", "DRIFT_BINS",
        # Intro links (surcharges optionnelles)
        "INTRO_OVERVIEW_KPIS_URI", "INTRO_HEALTH_KPIS_URI", "INTRO_DRIFT_SUMMARY_URI",
        "INTRO_MODEL_LATEST_H15_URI", "INTRO_MODEL_LATEST_H60_URI",
        "INTRO_FORECAST_H15_URI", "INTRO_FORECAST_H60_URI", "INTRO_OUT_PREFIX",
    ], title="[env:jobs]")

    # lock
    lock = _acquire_lock(cfg.gcs_lock)
    if cfg.gcs_lock and lock is None:
        # lock demandé mais pas obtenu → on sort silencieusement (exit 0)
        return 0

    try:
        print("[orchestrator] steps:", ", ".join(cfg.steps))
        if cfg.dry_run:
            print("[orchestrator] DRY_RUN=1 → no execution")
            return 0

        exit_code = 0
        for s in cfg.steps:
            module = MODULES[s]
            rc = _run_module(cfg.python_bin, module)
            if rc != 0:
                exit_code = rc if exit_code == 0 else exit_code
                if not cfg.continue_on_error:
                    print("[orchestrator] abort on first error (CONTINUE_ON_ERROR=0)")
                    return exit_code
        return exit_code
    finally:
        _release_lock(lock)


if __name__ == "__main__":
    sys.exit(main())
