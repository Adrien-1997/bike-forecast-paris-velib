# service/jobs/train_model.py

# =============================================================================
# Entrypoint d'entraînement (Cloud Run) harmonisé.
#
# Rôle principal :
# - Télécharger et concaténer des shards quotidiens depuis GCS (ou récupérer
#   directement un unique Parquet utilisé comme base d’entraînement).
# - Localiser automatiquement le module de training (layouts core/train/tools/flat).
# - Gérer le versionnement du modèle :
#       lecture de latest.json → calcul de la prochaine version → export en ENV.
# - Appeler la fonction d’entraînement (côté forecast : XGBoost uniquement).
# - Uploader l’artefact .joblib entraîné sur GCS comme filet de sécurité
#   lorsque le script d’entraînement ne publie pas lui-même le modèle.
# =============================================================================

from __future__ import annotations
import os, sys, tempfile, re, importlib, inspect, json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Tuple, List, Callable, Dict, Any, Optional

import pandas as pd
from google.cloud import storage
import importlib.util as _iu

# ─────────────────────────────────────────────
# Helpers GCS
# ─────────────────────────────────────────────

RE_COMPACT = re.compile(r".*/compact_(\d{4}-\d{2}-\d{2})\.parquet$")


def _parse_gs(uri: str) -> Tuple[str, str]:
    """
    Découper une URI GCS (gs://bucket/path) en nom de bucket et object key.

    Paramètres
    ----------
    uri : str
        URI GCS commençant par "gs://".

    Retours
    -------
    (str, str)
        Tuple (bucket, key) où key ne possède pas de slash final.

    Lève
    ----
    AssertionError
        Si l’URI ne commence pas par "gs://".
    """
    assert uri.startswith("gs://"), f"Bad GCS uri: {uri}"
    bkt, key = uri[5:].split("/", 1)
    return bkt, key.rstrip("/")


def _download(cli: storage.Client, src_uri: str, dst_path: Path) -> None:
    """
    Télécharger un objet GCS vers un fichier local.

    Paramètres
    ----------
    cli : google.cloud.storage.Client
        Instance du client Storage.
    src_uri : str
        URI GCS source (gs://bucket/path/to/file).
    dst_path : pathlib.Path
        Chemin local de destination. Les dossiers parents sont créés si besoin.
    """
    bkt, key = _parse_gs(src_uri)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cli.bucket(bkt).blob(key).download_to_filename(str(dst_path))


def _upload_to_gcs(local_path: Path, dst_uri: str) -> str:
    """
    Uploader un fichier local vers un objet GCS.

    Paramètres
    ----------
    local_path : pathlib.Path
        Chemin local du fichier à uploader.
    dst_uri : str
        URI GCS de destination (gs://bucket/path/to/file.joblib).

    Retours
    -------
    str
        URI GCS de l’objet uploadé.

    Lève
    ----
    FileNotFoundError
        Si le fichier local n’existe pas.
    """
    if not local_path.exists():
        raise FileNotFoundError(f"Local model not found: {local_path}")
    bkt, key = _parse_gs(dst_uri)
    storage.Client().bucket(bkt).blob(key).upload_from_filename(str(local_path))
    return f"gs://{bkt}/{key}"


def _list_daily(cli: storage.Client, gcs_prefix: str, lookback_days: int) -> List[str]:
    """
    Lister les shards quotidiens compactés (compact_YYYY-MM-DD.parquet) à concaténer.

    La fonction :
    - liste les blobs sous le préfixe GCS donné ;
    - garde uniquement ceux qui matchent RE_COMPACT (compact_YYYY-MM-DD.parquet) ;
    - filtre les shards dont le jour est >= (today_utc - lookback_days) ;
    - trie les URIs par date encodée dans le nom de fichier.

    Paramètres
    ----------
    cli : google.cloud.storage.Client
        Instance du client Storage.
    gcs_prefix : str
        Préfixe GCS où sont stockés les shards daily compactés,
        par ex. "gs://bucket/velib/daily".
    lookback_days : int
        Nombre de jours d’historique à inclure pour l’entraînement.

    Retours
    -------
    list of str
        Liste d’URIs GCS pour les shards sélectionnés.

    Lève
    ----
    RuntimeError
        Si aucun shard n’est trouvé dans la fenêtre de lookback.
    """
    bkt, prefix = _parse_gs(gcs_prefix)
    now = datetime.now(timezone.utc).date()
    min_day = now - timedelta(days=lookback_days)
    uris: List[str] = []

    for blob in cli.list_blobs(bkt, prefix=prefix + "/"):
        m = RE_COMPACT.match(blob.name)
        if not m:
            continue
        day = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        if day >= min_day:
            uris.append(f"gs://{bkt}/{blob.name}")

    # Tri par date extraite dans le nom de fichier
    uris.sort(key=lambda u: RE_COMPACT.match(u).group(1) if RE_COMPACT.match(u) else u)
    if not uris:
        raise RuntimeError(f"Aucun shard daily trouvé sous {gcs_prefix} sur {lookback_days} jours.")
    return uris


def _concat_locally(cli: storage.Client, shard_uris: List[str], out_path: Path) -> Path:
    """
    Télécharger et concaténer des shards quotidiens dans un unique Parquet local.

    Étapes
    ------
    - Télécharger chaque shard dans un dossier temporaire.
    - Charger chaque shard en DataFrame pandas.
    - Concaténer tous les shards ligne à ligne.
    - Écrire le résultat dans `out_path`.

    Paramètres
    ----------
    cli : google.cloud.storage.Client
        Instance du client Storage.
    shard_uris : list of str
        URIs GCS des shards daily compactés à télécharger et fusionner.
    out_path : pathlib.Path
        Chemin du fichier Parquet local pour le dataset concaténé.

    Retours
    -------
    pathlib.Path
        Le `out_path` fourni (pour chaînage).
    """
    tmp_dir = out_path.parent / "daily_shards"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dfs: List[pd.DataFrame] = []

    for i, uri in enumerate(shard_uris, 1):
        local_part = tmp_dir / Path(uri).name
        print(f"[train_job] download shard {i}/{len(shard_uris)}: {uri} → {local_part}", flush=True)
        _download(cli, uri, local_part)
        dfs.append(pd.read_parquet(local_part))

    print(f"[train_job] concat {len(dfs)} shards → {out_path}", flush=True)
    full = pd.concat(dfs, axis=0, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)
    return out_path

# ─────────────────────────────────────────────
# Résolution de la racine du dépôt & sys.path
# ─────────────────────────────────────────────

def _ensure_repo_on_path() -> Path:
    """
    S’assurer que la racine du dépôt (contenant 'service/' ou 'train/')
    est bien présente dans sys.path.

    Ordre de recherche
    ------------------
    1. Si les modules 'service' ou 'train' sont déjà importables,
       renvoyer le cwd.
    2. Si la variable d’environnement REPO_ROOT est définie et pointe vers un
       dossier contenant 'service/' ou 'train/', l’ajouter à sys.path et l’utiliser.
    3. Remonter depuis le chemin du fichier courant, en testant ses parents
       (ainsi que /app et cwd) et prendre le premier dossier contenant
       'service/' ou 'train/'.
    4. En dernier recours, renvoyer cwd et émettre un warning.

    Retours
    -------
    pathlib.Path
        Dossier racine du dépôt tel que résolu.
    """
    if _iu.find_spec("service") is not None or _iu.find_spec("train") is not None:
        return Path.cwd()

    env_root = os.getenv("REPO_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if (p / "service").exists() or (p / "train").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            print(f"[train_job] repo_root from REPO_ROOT: {p}", flush=True)
            return p

    here = Path(__file__).resolve()
    candidates = [here] + list(here.parents) + [Path("/app"), Path.cwd()]
    for c in candidates:
        try:
            if (c / "service").exists() or (c / "train").exists():
                if str(c) not in sys.path:
                    sys.path.insert(0, str(c))
                print(f"[train_job] repo_root resolved: {c}", flush=True)
                return c
        except Exception:
            pass

    print("[train_job][warn] unable to locate repo root containing 'service/' or 'train/'", file=sys.stderr)
    return Path.cwd()

# ─────────────────────────────────────────────
# Helpers de versionning (latest.json → bump)
# ─────────────────────────────────────────────

def _read_latest_version(cli: storage.Client, bucket: str, prefix: str) -> Optional[str]:
    """
    Lire gs://bucket/prefix/latest.json et renvoyer le champ 'version' si présent.

    Paramètres
    ----------
    cli : google.cloud.storage.Client
        Client Storage.
    bucket : str
        Nom du bucket.
    prefix : str
        Préfixe sous lequel latest.json est stocké.

    Retours
    -------
    str ou None
        La chaîne de version si trouvée et non vide, sinon None.
    """
    try:
        blob = cli.bucket(bucket).blob(f"{prefix.rstrip('/')}/latest.json")
        if not blob.exists():
            return None
        data = blob.download_as_bytes()
        obj = json.loads(data.decode("utf-8"))
        v = obj.get("version")
        if isinstance(v, str) and v.strip():
            return v.strip()
        return None
    except Exception as e:
        print(f"[version][warn] unable to read latest.json: {e}")
        return None


def _parse_semver(s: Optional[str]) -> Tuple[int,int,int]:
    """
    Parser 'MAJOR.MINOR[.PATCH]' en (major, minor, patch) entiers.

    Paramètres
    ----------
    s : str ou None
        Version sémantique, ex. '2.3.1'. Si None ou invalide,
        retourne (2,0,0).

    Retours
    -------
    (int, int, int)
        Tuple (major, minor, patch).
    """
    if not s:
        return (2,0,0)
    m = re.match(r"^\s*(\d+)\.(\d+)(?:\.(\d+))?\s*$", s)
    if not m:
        return (2,0,0)
    maj = int(m.group(1)); minr = int(m.group(2)); pat = int(m.group(3) or 0)
    return (maj,minr,pat)


def _bump_version(prev: Optional[str], mode: str = "minor") -> Tuple[str,str]:
    """
    Calculer la version suivante à partir d’une version sémantique précédente.

    Paramètres
    ----------
    prev : str ou None
        Version précédente (ex. '2.0.0'). Si None ou invalide,
        on utilise '2.0.0' comme base.
    mode : {"major", "minor", "patch"}, défaut "minor"
        Type de bump à appliquer.

    Retours
    -------
    (str, str)
        Tuple (base, next) où :
        - base est la version courante résolue (ex. '2.0.0') ;
        - next est la version bumpée (ex. '2.1.0').
    """
    base_tuple = _parse_semver(prev)
    maj,minr,pat = base_tuple
    mode = (mode or "minor").lower()
    if mode == "major":
        next_v = f"{maj+1}.0.0"
    elif mode == "patch":
        next_v = f"{maj}.{minr}.{pat+1}"
    else:
        next_v = f"{maj}.{minr+1}.0"
    base_v = f"{maj}.{minr}.{pat}"
    return base_v, next_v


def _resolve_and_export_version(cli: storage.Client) -> None:
    """
    Résoudre la version courante depuis latest.json et exporter les vars ENV.

    Comportement
    ------------
    - Si MODEL_GCS_BUCKET et MODEL_GCS_PREFIX sont définis :
        - lire latest.json sous ce préfixe ;
        - parser son champ 'version' comme base ;
        - calculer MODEL_NEXT_VERSION en bumpant la base selon VERSION_BUMP.
    - Sinon :
        - retomber sur une base par défaut '2.0.0' et la bumper.

    Variables d’environnement exportées
    -----------------------------------
    MODEL_BASE_VERSION : str
        Version courante résolue (ou '2.0.0' si aucune trouvée).
    MODEL_NEXT_VERSION : str
        Version bumpée selon VERSION_BUMP (minor par défaut).
    """
    bucket = os.environ.get("MODEL_GCS_BUCKET")
    prefix = os.environ.get("MODEL_GCS_PREFIX")
    bump = os.environ.get("VERSION_BUMP", "minor").lower()

    if bucket and prefix:
        prev = _read_latest_version(cli, bucket, prefix)
        base_v, next_v = _bump_version(prev, bump)
    else:
        print("[version][note] MODEL_GCS_BUCKET/PREFIX not set → default versioning used (2.0.0 → 2.1.0)")
        base_v, next_v = _bump_version("2.0.0", bump)

    os.environ["MODEL_BASE_VERSION"] = base_v
    os.environ["MODEL_NEXT_VERSION"] = next_v
    print(f"[version] base='{base_v}'  next='{next_v}'  (bump={bump})")

# ─────────────────────────────────────────────
# Import dynamique du callable de training
# ─────────────────────────────────────────────

def _find_training_callable(module_name: str) -> Callable[..., Any]:
    """
    Découvrir un point d’entrée d’entraînement valide dans un module donné.

    Ordre de recherche
    ------------------
    1. Chercher une fonction callable parmi :
       ['train', 'train_model', 'train_forecast', 'fit', 'run', 'main'].
    2. Fallback : rechercher une classe avec une méthode .fit() et créer
       un wrapper qui instancie la classe puis appelle .fit(**kwargs).

    Paramètres
    ----------
    module_name : str
        Nom de module Python fully-qualified (ex. 'service.core.forecast').

    Retours
    -------
    callable
        Objet appelable pouvant être invoqué avec des kwargs pour entraîner.

    Lève
    ----
    ImportError
        Si aucune fonction ou classe d’entraînement n’est trouvée.
    """
    mod = importlib.import_module(module_name)
    candidates = ["train", "train_model", "train_forecast", "fit", "run", "main"]
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            print(f"[train_job] using training entrypoint: {module_name}.{name}", flush=True)
            return fn
    # fallback : classe avec .fit()
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and callable(getattr(obj, "fit", None)):
            def _callable_wrapper(**kwargs):
                inst = obj()
                return inst.fit(**kwargs) if kwargs else inst.fit()
            print(f"[train_job] using class entrypoint: {module_name}.{attr}.fit", flush=True)
            return _callable_wrapper
    raise ImportError(f"Aucune fonction d'entraînement trouvée dans {module_name}.")


def _adapt_kwargs_for_signature(fn: Callable[..., Any], base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapter les kwargs de base à la signature de la fonction cible.

    Cette aide :
    - inspecte la signature du callable cible ;
    - renomme quelques paramètres "bien connus" si nécessaire, par ex. :
        - horizon_minutes → horizon_bins / horizon_min / horizon /
          minutes_ahead / target_horizon_min
        - lookback_days  → lookback / window_days / days
    - supprime les clés non supportées si la fonction ne prend pas de **kwargs.

    Paramètres
    ----------
    fn : callable
        Callable d’entraînement dont on inspecte la signature.
    base_kwargs : dict
        Kwargs de base à adapter (ex. src, horizon_bins, horizon_minutes,
        lookback_days, out_path, ...).

    Retours
    -------
    dict
        Dictionnaire de kwargs adapté, safe pour l’appel de `fn`.
    """
    sig = inspect.signature(fn)
    params = sig.parameters
    km = dict(base_kwargs)

    # map horizon_minutes -> horizon_bins/etc si besoin
    if "horizon_minutes" in km and "horizon_minutes" not in params:
        for alias in ["horizon_bins", "horizon_min", "horizon", "minutes_ahead", "target_horizon_min"]:
            if alias in params:
                km[alias] = km.pop("horizon_minutes")
                break
        else:
            km.pop("horizon_minutes", None)

    if "lookback_days" in km and "lookback_days" not in params:
        for alias in ["lookback", "window_days", "days"]:
            if alias in params:
                km[alias] = km.pop("lookback_days")
                break
        else:
            km.pop("lookback_days", None)

    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_kwargs:
        print(f"[train_job] call with kwargs (var-kwargs accepted): {km}", flush=True)
        return km
    filtered = {k: v for k, v in km.items() if k in params}
    print(f"[train_job] call with kwargs: {filtered}", flush=True)
    return filtered

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> int:
    """
    Entrypoint d’un job Cloud Run pour l’entraînement du modèle.

    Flow global
    -----------
    1. Lire la configuration depuis les variables d’environnement :
       - TRAIN_EXPORT_GCS : URI GCS de :
           - soit un Parquet unique de training,
           - soit d’un préfixe daily contenant des compact_YYYY-MM-DD.parquet.
       - HORIZON_MIN      : horizon de prévision en minutes (défaut : 15).
       - LOOKBACK_DAYS    : nombre de jours d’historique à concaténer
                            (défaut : 30).
    2. S’assurer que MODEL_TYPE est défini (par défaut 'xgb').
    3. Calculer HORIZON_BINS depuis HORIZON_MIN (bins de 5 minutes).
    4. Résoudre la version du modèle (latest.json) et exporter
       MODEL_BASE_VERSION et MODEL_NEXT_VERSION.
    5. Selon TRAIN_EXPORT_GCS :
       - si se termine par '.parquet' → téléchargement direct vers un Parquet local ;
       - sinon → préfixe daily : listage + concaténation des shards.
    6. S’assurer que la racine du repo est dans sys.path et résoudre
       le module d’entraînement (service.core.forecast en priorité).
    7. Trouver un callable d’entraînement valide dans le module
       et adapter les kwargs à sa signature.
    8. Lancer l’entraînement et logger d’éventuelles métriques retournées.
    9. Si le script d’entraînement n’a pas autopublié le modèle
       (MODEL_GCS_BUCKET/PREFIX non définis),
       uploader l’artefact model.joblib vers MODEL_GCS (ou une valeur par défaut).

    Variables d’environnement
    -------------------------
    TRAIN_EXPORT_GCS : str (requis)
        URI GCS pointant soit vers un Parquet de training, soit vers un préfixe daily.
    HORIZON_MIN : str, défaut "15"
        Horizon de prévision en minutes.
    LOOKBACK_DAYS : str, défaut "30"
        Fenêtre d’historique en jours pour les shards daily.
    MODEL_TYPE : str, défaut "xgb"
        Famille de modèle utilisée par le script d’entraînement.
    MODEL_GCS_BUCKET, MODEL_GCS_PREFIX : str (optionnels)
        Si tous deux sont définis, le trainer est censé autopublier le modèle
        sous ce préfixe ; le job n’effectue alors pas d’upload manuel.
    MODEL_GCS : str, optionnel
        URI GCS fallback où stocker l’artefact modèle si l’autopublish
        n’est pas configuré. Par défaut, un chemin interne gs:// est utilisé.

    Retours
    -------
    int
        Code de sortie du process (0 en cas de succès).
    """
    # Inputs
    TRAIN_EXPORT_GCS = os.environ["TRAIN_EXPORT_GCS"]  # gs://.../training/*.parquet OU gs://.../daily (préfixe)
    HORIZON_MIN = int(os.environ.get("HORIZON_MIN", "15"))
    LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "30"))

    # ✅ Forcer XGB par défaut si non défini (aligné sur forecast.py XGB-only)
    os.environ.setdefault("MODEL_TYPE", "xgb")

    # Horizon en bins 5 minutes (horizon_bins utilisé côté trainer)
    HORIZON_BINS = max(1, HORIZON_MIN // 5)

    tmp_root = Path(tempfile.gettempdir()) / "velib_train"
    local_file = tmp_root / "exports" / "velib_concat.parquet"
    cli = storage.Client()

    # 0) Résolution de version (latest.json -> bump) → export ENV
    _resolve_and_export_version(cli)

    # 1) Dataset d’entraînement : téléchargement d’un Parquet unique
    #    vs concaténation de shards daily récents
    if TRAIN_EXPORT_GCS.endswith(".parquet"):
        print(f"[train_job] download {TRAIN_EXPORT_GCS} → {local_file}", flush=True)
        _download(cli, TRAIN_EXPORT_GCS, local_file)
    else:
        print(f"[train_job] list & concat daily shards from prefix={TRAIN_EXPORT_GCS}", flush=True)
        shard_uris = _list_daily(cli, TRAIN_EXPORT_GCS, LOOKBACK_DAYS)
        _concat_locally(cli, shard_uris, local_file)

    # 2) S’assurer que la racine du repo est dans sys.path avant d’importer le module de training
    _ensure_repo_on_path()

    # 3) Déterminer le module de training (fix des layouts hétérogènes)
    module_name = os.getenv("TRAIN_MODULE")
    if not module_name:
        # ordre : core → tools → train → flat (compat)
        import service.core.forecast  # recommandé
        module_name = "service.core.forecast"

    # 4) Chemin local pour la sortie modèle
    model_out = tmp_root / "model.joblib"
    model_out.parent.mkdir(parents=True, exist_ok=True)

    # 5) Appel du training
    train_fn = _find_training_callable(module_name)
    base_kwargs = {
        "src": str(local_file),
        # les deux sont passés pour robustesse ; côté XGB, le trainer utilise horizon_bins
        "horizon_bins": HORIZON_BINS,
        "horizon_minutes": HORIZON_MIN,
        "lookback_days": LOOKBACK_DAYS,
        "out_path": str(model_out),
    }
    call_kwargs = _adapt_kwargs_for_signature(train_fn, base_kwargs)

    print("[train_job] start training", flush=True)
    metrics = train_fn(**call_kwargs)
    if metrics is not None:
        print("[train_job] metrics:", metrics, flush=True)

    # 6) Publication
    # Cas 1 : forecast.py autopublie si MODEL_GCS_BUCKET/PREFIX présents
    has_self_publish = bool(os.environ.get("MODEL_GCS_BUCKET") and os.environ.get("MODEL_GCS_PREFIX"))

    if has_self_publish:
        print("[train_job] trainer self-publishing enabled (MODEL_GCS_BUCKET/MODEL_GCS_PREFIX set) → skip manual upload", flush=True)
    else:
        # Fallback : uploader un unique artefact “compat” (latest.joblib par défaut)
        MODEL_GCS = os.environ.get(
            "MODEL_GCS",
            f"gs://velib-forecast-472820_cloudbuild/velib/models/h15/latest.joblib"
        )
        local_model = model_out if model_out.exists() else Path("model.joblib")
        gcs_uri = _upload_to_gcs(local_model, MODEL_GCS)
        print(f"[train_job] model uploaded → {gcs_uri}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())