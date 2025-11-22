# service/core/utils_gcs.py

# =============================================================================
#  Vélib’ Forecast — Cloud Run / GCS job launcher with locking
# =============================================================================
#  Rôle :
#    - Servir de **wrapper unique** pour lancer les différents jobs Python
#      (ingest, compact_daily, training, monitoring, …) depuis Cloud Run /
#      Cloud Build, en se basant sur la variable d’environnement JOB.
#    - Gérer un verrou GCS optionnel (GCS_LOCK) pour éviter les exécutions
#      concurrentes d’un même job.
#    - Calculer automatiquement la date cible (DAY) pour les jobs de
#      compaction daily / monthly si non fournie.
#    - Exposer un mode DRY_RUN pour visualiser la commande sans l’exécuter.
#
#  Usage typique (Cloud Run job) :
#    - Env:
#        JOB=ingest | compact_daily | compact_monthly | build_serving_forecast
#            | train_model | export_training_base | export_training_base_to_gcs
#            | monitoring | monitoring_pipeline
#        GCS_LOCK=gs://.../velib/locks/<job>.lock   (optionnel)
#        PYTHON_BIN=python                          (optionnel)
#        DRY_RUN=1                                   (optionnel, debug)
#
#    - Commande :
#        python -m service.core.utils_gcs
#
#  Ce module ne contient **aucune** logique métier data/ML : il se contente
#  d’orchestrer l’appel au bon module Python en CLI.
# =============================================================================

from __future__ import annotations
import os, sys, contextlib
from subprocess import run, CalledProcessError
from typing import List, Tuple, Dict
from datetime import datetime, timezone, timedelta

try:
    from google.cloud import storage  # type: ignore
except Exception as e:
    # On échoue tôt si la lib n'est pas disponible, car elle est nécessaire
    # pour construire le client GCS et gérer le verrou.
    raise RuntimeError("google-cloud-storage is required for job locking") from e

# ============================================================================
# ENV attendu
# ----------------------------------------------------------------------------
# - JOB :
#     Ingestion & parquet-first
#       • ingest
#       • compact_daily
#       • compact_monthly
#       • build_serving_forecast        (alias: build_features_4h)
#
#     Training / exports
#       • train_model
#       • export_training_base
#
#     Monitoring (JSON artifacts)
#       • monitoring_pipeline           (alias: monitoring)
#
# - GCS_LOCK            : gs://.../velib/locks/job.lock  (optionnel)
# - PYTHON_BIN          : binaire python (optionnel, défaut: "python")
# - DRY_RUN             : 1|0 (affiche la commande sans l'exécuter)
#
# Aides au calcul de date (pour compact_daily / compact_monthly) :
# - DAY                 : YYYY-MM-DD (prioritaire si présent)
# - DAY_OFFSET          : entier (ex: -1 → veille UTC ; +1 → demain UTC). Ignoré si DAY est défini.
# ============================================================================

#: Nom du job à lancer (ingest, compact_daily, train_model, monitoring, …).
JOB        = (os.environ.get("JOB", "ingest") or "ingest").strip()
#: Binaire Python utilisé pour lancer les modules (par défaut "python").
PYTHON_BIN = os.environ.get("PYTHON_BIN") or "python"
#: URI GCS du verrou optionnel (gs://bucket/path/to/lock).
LOCK_URI   = os.environ.get("GCS_LOCK")
#: Mode dry-run : si vrai, on affiche la commande sans l’exécuter.
DRY_RUN    = os.environ.get("DRY_RUN") in ("1", "true", "True")


def parse_gs(uri: str) -> Tuple[str, str]:
    """
    Parser une URI GCS 'gs://bucket/path/to/object' en (bucket, path).

    Parameters
    ----------
    uri : str
        URI de type "gs://…".

    Returns
    -------
    (bucket, path) : tuple[str, str]

    Raises
    ------
    AssertionError
        Si l’URI ne commence pas par "gs://".
    """
    assert uri.startswith("gs://"), f"invalid GCS URI: {uri}"
    bkt, path = uri[5:].split("/", 1)
    return bkt, path


def lock_blob(client: storage.Client, lock_uri: str | None):
    """Best-effort lock via objet GCS vide (if_generation_match=0).

    Principe
    --------
    - Si `lock_uri` est None → aucun verrou, retourne None.
    - Sinon, on tente de créer un objet vide avec `if_generation_match=0`.
      * si la création réussit → le verrou est acquis, on retourne le blob.
      * si elle échoue → l’objet existe déjà ou erreur → on retourne None.

    Parameters
    ----------
    client : google.cloud.storage.Client
        Client GCS utilisé pour manipuler les blobs.
    lock_uri : str | None
        URI GCS du fichier de verrou (peut être None).

    Returns
    -------
    google.cloud.storage.blob.Blob | None
        L’objet de verrou si acquis, sinon None.
    """
    if not lock_uri:
        return None
    bkt, key = parse_gs(lock_uri)
    blob = client.bucket(bkt).blob(key)
    try:
        blob.upload_from_string(b"", if_generation_match=0)
        print(f"[lock] acquired → {lock_uri}")
        return blob
    except Exception:
        # busy or cannot create (verrou déjà présent ou erreur réseau/droits)
        return None


def _dispatch_command(job: str) -> List[str]:
    """
    Construire la commande Python **par module** (-m), arbo service.jobs.* ou service.core.*.

    Modules supportés
    -----------------
      - service.jobs.ingest
      - service.jobs.compact_daily
      - service.jobs.compact_monthly
      - service.jobs.build_serving_forecast
      - service.jobs.train_model
      - service.jobs.export_training_base
      - service.jobs.export_training_base_to_gcs
      - service.core.monitoring_pipeline            ← orchestration monitoring

    Alias compat
    ------------
      - "monitoring" → "monitoring_pipeline"

    Parameters
    ----------
    job : str
        Valeur de JOB (après normalisation d’alias).

    Returns
    -------
    list[str]
        La commande sous forme de liste d’arguments, ex:
        [PYTHON_BIN, "-m", "service.jobs.ingest"].

    Raises
    ------
    ValueError
        Si JOB ne correspond à aucun module connu.
    """
    alias: Dict[str, str] = {
        "monitoring": "monitoring_pipeline",
    }
    normalized = alias.get(job, job)

    modules = {
        # parquet-first
        "ingest":                       "service.jobs.ingest",
        "compact_daily":                "service.jobs.compact_daily",
        "compact_monthly":              "service.jobs.compact_monthly",
        "build_serving_forecast":       "service.jobs.build_serving_forecast",

        # training / exports
        "train_model":                  "service.jobs.train_model",
        "export_training_base":         "service.jobs.export_training_base",
        "export_training_base_to_gcs":  "service.jobs.export_training_base_to_gcs",

        # monitoring
        "monitoring_pipeline":          "service.core.monitoring_pipeline",
    }
    if normalized not in modules:
        raise ValueError(f"unknown JOB={job}")
    return [PYTHON_BIN, "-m", modules[normalized]]


def _maybe_set_day_env(job: str):
    """
    Calculer DAY (YYYY-MM-DD) pour les jobs compact_* si nécessaire.

    Règle
    -----
    - Ne s’applique qu’à JOB in {"compact_daily", "compact_monthly"}.
    - Si DAY est déjà défini dans l’environnement → ne rien faire.
    - Sinon :
        1) Si DAY_OFFSET est défini et convertible en int :
              DAY = today_utc + DAY_OFFSET
        2) Sinon (défaut) :
              DAY = today_utc - 1  (veille UTC)

    Parameters
    ----------
    job : str
        Nom du job courant (valeur de JOB).
    """
    if job not in ("compact_daily", "compact_monthly"):
        return

    if os.environ.get("DAY"):
        return

    offset_env = os.environ.get("DAY_OFFSET")
    try:
        if offset_env is not None and offset_env.strip() != "":
            offset = int(offset_env)
        else:
            offset = -1  # veille UTC par défaut
    except Exception:
        offset = -1

    day = (datetime.now(timezone.utc) + timedelta(days=offset)).strftime("%Y-%m-%d")
    os.environ["DAY"] = day
    print(f"[job] DAY not provided → computed from offset={offset}: DAY={day} (UTC)")


def main() -> int:
    """
    Point d’entrée principal du lanceur de jobs GCS / Cloud Run.

    Étapes
    ------
    0. Création du client GCS.
    1. Calcul automatique de DAY pour les jobs compact_* (si pertinent).
    2. Acquisition optionnelle d’un verrou GCS (LOCK_URI).
       - Si LOCK_URI est défini et que le lock ne peut pas être acquis :
           on loggue et on sort avec code 0 (exécution déjà en cours).
    3. Construction de la commande Python via `_dispatch_command(JOB)`.
    4. Echo de quelques variables d’environnement utiles pour le debug
       (sans secrets).
    5. Si DRY_RUN=1 :
         - on affiche la commande et on ne l’exécute pas (return 0).
       Sinon :
         - on exécute `run(cmd, check=True)` et on propage le return code.
    6. Libération du verrou GCS si acquis.

    Returns
    -------
    int
        Code de sortie global :
        - 0 si tout s’est bien passé ou si DRY_RUN,
        - code de retour du sous-processus si erreur d’exécution,
        - 2 en cas d’erreur de construction de commande,
        - 1 en cas d’autre exception non gérée.
    """
    client = storage.Client()

    # 0) Calcul auto de DAY si pertinent
    _maybe_set_day_env(JOB)

    # 1) Lock optionnel
    lock = lock_blob(client, LOCK_URI)
    if LOCK_URI and lock is None:
        # Lock demandé mais non obtenu → job déjà en cours ou erreur de création.
        # On sort en "succès" pour ne pas compter cela comme un échec récurrent.
        print(f"[lock] busy or create error → {LOCK_URI} → exit 0")
        return 0
    elif not LOCK_URI:
        print("[lock] disabled (no GCS_LOCK)")

    try:
        # 2) Construire la commande
        try:
            cmd = _dispatch_command(JOB)
        except Exception as e:
            print(f"[job] error while building command: {e}", file=sys.stderr)
            return 2

        # Echo utile debug (sans secrets)
        keys_to_echo = [
            "JOB", "PYTHONPATH",
            # parquet-first
            "GCS_RAW_PREFIX", "GCS_DAILY_PREFIX", "GCS_MONTHLY_PREFIX",
            "GCS_SERVING_PREFIX", "SERVING_FORECAST_PREFIX", "GCS_EXPORTS_PREFIX",
            # models / forecast
            "GCS_MODEL_URI_T15", "GCS_MODEL_URI_T60",
            "FORECAST_HORIZONS", "WINDOW_HOURS", "WITH_FORECAST", "NOW_UTC_ISO",
            # monitoring roots
            "GCS_MONITORING_PREFIX",
            # time helpers
            "DAY", "DAY_OFFSET",
            # training (si utilisé)
            "MODEL_TYPE", "HORIZON_BINS",
            # monitoring pipeline knobs
            "STEPS", "CONTINUE_ON_ERROR", "DRY_RUN",
            "MODEL_URI_15", "MODEL_URI_60", "TZ_APP",
            "MON_LAST_DAYS", "MON_REF_DAYS",
            "OVERVIEW_LAST_DAYS", "DYNAMICS_LAST_DAYS", "NETWORK_WINDOW_DAYS",
            "PERF_LAST_DAYS", "LOOKBACK_DAYS",
        ]
        echo = " ".join(
            f"{k}={os.environ.get(k)}"
            for k in keys_to_echo
            if os.environ.get(k) is not None
        )
        if echo:
            print(f"[env] {echo}")

        # 3) Exécution (ou dry-run)
        print("[job] run:", " ".join(cmd))
        if DRY_RUN:
            print("[job] DRY_RUN=1 → skipping execution")
            return 0

        run(cmd, check=True)
        return 0

    except CalledProcessError as e:
        print(f"[job] failed: {e}", file=sys.stderr)
        return e.returncode
    except Exception as e:
        print(f"[job] error: {e}", file=sys.stderr)
        return 1
    finally:
        if LOCK_URI and lock is not None:
            with contextlib.suppress(Exception):
                lock.delete()
                print("[lock] released")


if __name__ == "__main__":
    sys.exit(main())
