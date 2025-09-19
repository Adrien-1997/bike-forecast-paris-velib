# tools/utils_io.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download

# --------- Configuration via variables d'environnement (surchargeables) ----------
HF_REPO_ID: str = os.getenv("HF_REPO_ID", "adrien-morel/bike-forecast-paris-velib")
HF_REPO_TYPE: str = os.getenv("HF_REPO_TYPE", "dataset")  # dataset | model | space
HF_REVISION: Optional[str] = os.getenv("HF_REVISION")     # ex. "main", "v2025-09-18", ou un SHA
HF_FORCE_HF: bool = os.getenv("HF_FORCE_HF", "0") == "1"  # si "1", ignore le local
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")           # si repo privé

# Miroir simple de l'arbo : local "docs/exports/*" <-> Hub "exports/*"
LOCAL_EXPORTS_DIR = Path("docs/exports")
HUB_EXPORTS_PREFIX = "exports"  # côté Hub

def _hub_filename_for_exports(filename: str) -> str:
    # "perf.parquet" -> "exports/perf.parquet"
    return f"{HUB_EXPORTS_PREFIX.rstrip('/')}/{filename.lstrip('/')}"

def get_export_path(filename: str) -> Path:
    """
    Retourne un chemin de fichier lisible par pandas :
      - Si présent en local (docs/exports/...), renvoie ce chemin (sauf si HF_FORCE_HF=1).
      - Sinon télécharge depuis Hugging Face Hub (dataset) et renvoie le chemin du cache HF.
    """
    local_path = LOCAL_EXPORTS_DIR / filename
    if not HF_FORCE_HF and local_path.exists():
        return local_path

    hub_filename = _hub_filename_for_exports(filename)
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=hub_filename,
        repo_type=HF_REPO_TYPE,
        revision=HF_REVISION,
        token=HF_TOKEN,           # None si public
    )
    return Path(path)

def resolve_path(cli_path: Optional[str], default_filename: str) -> Path:
    """
    Si l'utilisateur a fourni un chemin en CLI et qu'il existe, on l'utilise.
    Sinon, on applique la logique local->Hub sur default_filename.
    """
    if cli_path:
        p = Path(cli_path)
        if p.exists():
            return p
    return get_export_path(default_filename)
