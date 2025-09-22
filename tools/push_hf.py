# tools/push_hf.py
from __future__ import annotations
import os, sys, time, hashlib, pathlib
from huggingface_hub import HfApi, CommitOperationAdd

try:
    from huggingface_hub.errors import HfHubHTTPError
except Exception:
    try:
        from huggingface_hub.utils._errors import HfHubHTTPError
    except Exception:
        class HfHubHTTPError(Exception):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.response = getattr(self, "response", None)

REPO_ID   = os.environ.get("HF_DATASET_ID", "Adrien97/velib-monitoring-historical")
REPO_TYPE = "dataset"
SRC       = os.environ.get("PUSH_SRC", "exports/velib.parquet")
DEST      = os.environ.get("PUSH_DEST", "exports/velib.parquet")
HASH_DEST = os.environ.get("PUSH_HASH_DEST", "exports/velib.sha256")
TOKEN     = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_remote_hash(api: HfApi) -> str | None:
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=REPO_ID, filename=HASH_DEST, repo_type=REPO_TYPE, token=TOKEN, timeout=20)
        with open(p, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except Exception:
        return None

def _is_rate_limited(e: Exception) -> tuple[bool, float | None]:
    resp = getattr(e, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        try:
            ra_hdr = resp.headers.get("Retry-After")
            return True, float(ra_hdr) if ra_hdr else None
        except Exception:
            return True, None
    return False, None

def _create_commit_compat(api: HfApi, **kwargs):
    try:
        return api.create_commit(max_workers=1, **kwargs)
    except TypeError:
        kwargs.pop("max_workers", None)
        return api.create_commit(**kwargs)

def create_commit_with_backoff(api: HfApi, operations: list, msg: str):
    tries, delay = 0, 1.5
    while True:
        try:
            return _create_commit_compat(
                api,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                operations=operations,
                commit_message=msg,
                token=TOKEN,
            )
        except HfHubHTTPError as e:
            rate, retry_after = _is_rate_limited(e)
            if rate and tries < 6:
                import random
                sleep_s = retry_after if retry_after is not None else delay + random.uniform(0, 0.6)
                print(f"[push_hf] 429 rate-limited, retry in {sleep_s:.1f}s (try {tries+1}/6)...", flush=True)
                time.sleep(sleep_s)
                tries += 1
                delay *= 2
                continue
            # 429 final ou autre erreur HF -> skip proprement (exit 0)
            if rate:
                print("[push_hf] final 429: skip this run without failing the job.")
                return None
            raise
        except Exception as e:
            if tries < 3:
                import random
                sleep_s = delay + random.uniform(0, 0.6)
                print(f"[push_hf] transient error: {e}. retry in {sleep_s:.1f}s...", flush=True)
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

    local_hash = sha256_of_file(str(src))
    remote_hash = read_remote_hash(api)
    if remote_hash == local_hash:
        print("[push_hf] No change detected (sha256 match). Skip upload.")
        return 0

    print(f"[push_hf] Upload {src} -> {REPO_ID}:{DEST}", flush=True)

    ops = [
        CommitOperationAdd(path_in_repo=DEST, path_or_fileobj=str(src)),
        CommitOperationAdd(path_in_repo=HASH_DEST, path_or_fileobj=bytes(local_hash, "utf-8")),
    ]

    info = create_commit_with_backoff(api, ops, msg=f"update {DEST} (hash={local_hash[:8]}...)")
    if info is None:
        return 0  # skip silencieux si 429 final
    print(f"[push_hf] Done: {getattr(info, 'commit_url', '(no url)')}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
