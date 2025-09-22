# tools/push_hf.py
from __future__ import annotations
import os, sys, pathlib
from huggingface_hub import HfApi, CommitOperationAdd

# Compat HfHubHTTPError (multi versions)
try:
    from huggingface_hub.errors import HfHubHTTPError
except Exception:
    try:
        from huggingface_hub.utils._errors import HfHubHTTPError
    except Exception:
        class HfHubHTTPError(Exception): pass

REPO_ID   = os.environ.get("HF_DATASET_ID", "Adrien97/velib-monitoring-historical")
REPO_TYPE = "dataset"
SRC       = os.environ.get("PUSH_SRC") or "exports/velib.parquet"
DEST      = os.environ.get("PUSH_DEST") or "exports/velib.parquet"
TOKEN     = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

def _create_commit_compat(api: HfApi, **kwargs):
    try:
        return api.create_commit(max_workers=1, **kwargs)
    except TypeError:
        kwargs.pop("max_workers", None)
        return api.create_commit(**kwargs)

def main() -> int:
    print(f"[push_hf] cwd={os.getcwd()}", flush=True)
    print(f"[push_hf] repo={REPO_ID} type={REPO_TYPE}", flush=True)
    print(f"[push_hf] src={SRC}  dest={DEST}", flush=True)

    src = pathlib.Path(SRC)
    if not src.exists():
        print(f"[push_hf] Source not found: {src}", file=sys.stderr, flush=True)
        # Petit diagnostic de répertoire
        try:
            if pathlib.Path("exports").exists():
                print(f"[push_hf] ls exports/: {os.listdir('exports')}", flush=True)
            else:
                print("[push_hf] 'exports/' directory does not exist.", flush=True)
        except Exception as e:
            print(f"[push_hf] ls exports/ failed: {e}", flush=True)
        return 1

    try:
        size_mb = src.stat().st_size / (1024 * 1024)
        print(f"[push_hf] file size ≈ {size_mb:.2f} MB", flush=True)
    except Exception:
        pass

    api = HfApi(token=TOKEN)
    print(f"[push_hf] Upload {src} -> {REPO_ID}:{DEST}", flush=True)

    ops = [CommitOperationAdd(path_in_repo=DEST, path_or_fileobj=str(src))]

    tries, delay = 0, 1.5
    while True:
        try:
            info = _create_commit_compat(
                api,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                operations=ops,
                commit_message=f"append shard {DEST}",
                token=TOKEN,
            )
            print(f"[push_hf] Done: {getattr(info, 'commit_url', '(no url)')}", flush=True)
            return 0

        except HfHubHTTPError as e:
            resp = getattr(e, "response", None)
            code = getattr(resp, "status_code", None)
            if code == 429 and tries < 6:
                import time, random
                sleep_s = delay + random.uniform(0, 0.6)
                print(f"[push_hf] 429 rate-limited, retry in {sleep_s:.1f}s (try {tries+1}/6)...", flush=True)
                time.sleep(sleep_s)
                tries += 1
                delay *= 2
                continue
            if code == 429:
                print("[push_hf] final 429: skip this run (append shard).", flush=True)
                return 0
            # Autres erreurs HF : on les remonte après quelques retries génériques
            if tries < 3:
                import time, random
                sleep_s = delay + random.uniform(0, 0.6)
                print(f"[push_hf] HfHubHTTPError ({code}). retry in {sleep_s:.1f}s...", flush=True)
                time.sleep(sleep_s)
                tries += 1
                delay *= 2
                continue
            raise

        except Exception as e:
            if tries < 3:
                import time, random
                sleep_s = delay + random.uniform(0, 0.6)
                print(f"[push_hf] transient error: {e}. retry in {sleep_s:.1f}s...", flush=True)
                time.sleep(sleep_s)
                tries += 1
                delay *= 2
                continue
            raise

if __name__ == "__main__":
    sys.exit(main())
