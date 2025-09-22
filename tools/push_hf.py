# tools/push_hf.py
from __future__ import annotations
import os, sys, time, hashlib, pathlib, typing as t
from huggingface_hub import HfApi, CommitOperationAdd, HfHubHTTPError

REPO_ID   = os.environ.get("HF_DATASET_ID", "Adrien97/velib-monitoring-historical")
REPO_TYPE = "dataset"
SRC       = os.environ.get("PUSH_SRC", "docs/exports/velib.parquet")
DEST      = os.environ.get("PUSH_DEST", "exports/velib.parquet")
HASH_DEST = os.environ.get("PUSH_HASH_DEST", "exports/velib.sha256")   # petit fichier texte
TOKEN     = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download_text(api: HfApi, repo_id: str, path_in_repo: str) -> str | None:
    try:
        return api.hf_hub_download(repo_id=repo_id, filename=path_in_repo, repo_type=REPO_TYPE, local_dir=os.getenv("HF_HOME", None), local_dir_use_symlinks=False, force_download=False, token=TOKEN, timeout=20)
    except Exception:
        return None

def read_remote_hash(api: HfApi) -> str | None:
    # récupère le contenu du fichier hash (optionnel)
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=REPO_ID, filename=HASH_DEST, repo_type=REPO_TYPE, token=TOKEN, timeout=20)
        with open(p, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except Exception:
        return None

def create_commit_with_backoff(api: HfApi, operations: list, msg: str):
    tries = 0
    delay = 1.5
    while True:
        try:
            return api.create_commit(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                operations=operations,
                commit_message=msg,
                token=TOKEN,
                max_workers=1,   # pas d’uploads parallèles => moins de pression API
            )
        except HfHubHTTPError as e:
            code = getattr(e.response, "status_code", None)
            retry_after = None
            if hasattr(e.response, "headers"):
                retry_after = e.response.headers.get("Retry-After")

            if code == 429 and tries < 6:
                # Respecter Retry-After si fourni, sinon backoff expo + jitter
                if retry_after:
                    try:
                        sleep_s = float(retry_after)
                    except Exception:
                        sleep_s = delay
                else:
                    import random
                    sleep_s = delay + random.uniform(0, 0.6)

                print(f"[push_hf] 429 rate-limited, retrying in {sleep_s:.1f}s (try {tries+1}/6)...", flush=True)
                time.sleep(sleep_s)
                tries += 1
                delay *= 2
                continue
            raise

def main():
    src = pathlib.Path(SRC)
    if not src.exists():
        print(f"[push_hf] Source not found: {src}", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=TOKEN)

    # 1) éviter un push si le contenu n’a pas changé
    local_hash = sha256_of_file(str(src))
    remote_hash = read_remote_hash(api)
    if remote_hash == local_hash:
        print("[push_hf] No change detected (sha256 match). Skip upload.")
        sys.exit(0)

    print(f"[push_hf] Upload {src}  →  {REPO_ID}:{DEST}", flush=True)

    # 2) préparer les opérations (un seul commit)
    ops = [
        CommitOperationAdd(path_in_repo=DEST, path_or_fileobj=str(src)),
        CommitOperationAdd(path_in_repo=HASH_DEST, path_or_fileobj=bytes(local_hash, "utf-8")),
    ]

    # 3) commit avec backoff 429
    info = create_commit_with_backoff(api, ops, msg=f"update {DEST} (rows parquet, hash={local_hash[:8]}...)")
    print(f"[push_hf] Done: {info.commit_url}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
