# tools/push_hf.py
from __future__ import annotations
import os, sys, time, pathlib
from huggingface_hub import HfApi, hf_hub_url

def upload(path_local: str, path_repo: str, repo_id: str, repo_type: str = "dataset") -> None:
    api = HfApi()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[push_hf] HF_TOKEN manquant (secret). Abandon.")
        sys.exit(1)

    p = pathlib.Path(path_local)
    if not p.exists():
        print(f"[push_hf] Fichier introuvable: {p}")
        sys.exit(0)

    ts = time.strftime("%Y-%m-%d %H:%M")
    msg = f"Update {path_repo} from Cloud Run Job ({ts})"
    print(f"[push_hf] Upload {p} → {repo_id}:{path_repo}")

    api.upload_file(
        path_or_fileobj=str(p),
        path_in_repo=path_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=msg,
    )
    url = hf_hub_url(repo_id=repo_id, repo_type=repo_type, filename=path_repo)
    print(f"[push_hf] OK → {url}")

def main():
    repo_id   = os.environ.get("HF_REPO_ID", "").strip()
    repo_type = os.environ.get("HF_REPO_TYPE", "dataset").strip() or "dataset"
    if not repo_id:
        print("[push_hf] HF_REPO_ID manquant.")
        sys.exit(1)

    # Fichiers à pousser si présents
    candidates = [
        ("docs/exports/velib.parquet", "exports/velib.parquet"),
        ("docs/exports/velib.csv",     "exports/velib.csv"),
    ]
    pushed = 0
    for src, dest in candidates:
        if pathlib.Path(src).exists():
            upload(src, dest, repo_id, repo_type)
            pushed += 1
    if pushed == 0:
        print("[push_hf] Aucun export à pousser. Fin ok.")

if __name__ == "__main__":
    main()
