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

def main():
    src = pathlib.Path(SRC)
    if not src.exists():
        print(f"[push_hf] Source not found: {src}", file=sys.stderr)
        return 1

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
            print(f"[push_hf] Done: {getattr(info, 'commit_url', '(no url)')}")
            return 0
        except HfHubHTTPError as e:
            resp = getattr(e, "response", None)
            if resp is not None and getattr(resp, "status_code", None) == 429 and tries < 6:
                import time, random
                sleep_s = delay + random.uniform(0, 0.6)
                print(f"[push_hf] 429 rate-limited, retry in {sleep_s:.1f}s (try {tries+1}/6)...", flush=True)
                time.sleep(sleep_s)
                tries += 1
                delay *= 2
                continue
            if resp is not None and getattr(resp, "status_code", None) == 429:
                print("[push_hf] final 429: skip this run (append shard).")
                return 0
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
