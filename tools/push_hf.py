# tools/push_hf.py
from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

# Compat HfHubHTTPError (multi versions)
try:
    from huggingface_hub.errors import HfHubHTTPError
except Exception:
    try:
        from huggingface_hub.utils._errors import HfHubHTTPError
    except Exception:
        class HfHubHTTPError(Exception):  # type: ignore
            pass

def _create_commit_compat(api: HfApi, **kwargs):
    try:
        return api.create_commit(max_workers=1, **kwargs)
    except TypeError:
        kwargs.pop("max_workers", None)
        return api.create_commit(**kwargs)

def main() -> int:
    p = argparse.ArgumentParser(description="Upload a file to a HF dataset repo")
    p.add_argument("--file", "-f", help="Local file to upload (ex: exports/data_health.csv)")
    p.add_argument("--dest", "-d", help="Path in repo (ex: exports/data_health.csv)")
    p.add_argument("--repo", "-r", help="HF dataset id (ex: user/velib-monitoring)")
    p.add_argument("--token", "-t", help="HF token (or set HF_TOKEN env var)")
    args = p.parse_args()

    # ---- Resolve inputs (CLI > env > defaults)
    repo_id = args.repo or os.environ.get("HF_DATASET_ID") or "Adrien97/velib-monitoring-historical"
    token   = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    src_env = os.environ.get("PUSH_SRC")  # legacy env support
    dst_env = os.environ.get("PUSH_DEST")
    src     = Path(args.file or src_env or "exports/velib.parquet")
    dest    = args.dest or dst_env or f"exports/{src.name}"

    print(f"[push_hf] cwd={os.getcwd()}", flush=True)
    print(f"[push_hf] repo={repo_id} type=dataset", flush=True)
    print(f"[push_hf] src={src}  dest={dest}", flush=True)

    if not src.exists():
        # Fallback helper
        fallback = Path("exports") / "velib.parquet"
        if fallback.exists():
            print(f"[push_hf] WARNING: {src} missing; using fallback {fallback}", flush=True)
            src = fallback
        else:
            print(f"[push_hf] Source not found: {src}", file=sys.stderr, flush=True)
            try:
                if Path("exports").exists():
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

    api = HfApi(token=token)
    print(f"[push_hf] Upload {src} -> {repo_id}:{dest}", flush=True)

    ops = [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(src))]

    tries, delay = 0, 1.5
    while True:
        try:
            info = _create_commit_compat(
                api,
                repo_id=repo_id,
                repo_type="dataset",
                operations=ops,
                commit_message=f"upload {dest}",
                token=token,
            )
            print(f"[push_hf] Done: {getattr(info, 'commit_url', '(no url)')}", flush=True)
            return 0

        except HfHubHTTPError as e:
            resp = getattr(e, "response", None)
            code = getattr(resp, "status_code", None)
            if code == 401:
                print("[push_hf] Unauthorized: set HF_TOKEN or pass --token", flush=True)
                return 1
            if code == 429 and tries < 6:
                import time, random
                sleep_s = delay + random.uniform(0, 0.6)
                print(f"[push_hf] 429 rate-limited, retry in {sleep_s:.1f}s (try {tries+1}/6)...", flush=True)
                time.sleep(sleep_s); tries += 1; delay *= 2; continue
            if code == 429:
                print("[push_hf] final 429: skip this run.", flush=True)
                return 0
            if tries < 3:
                import time, random
                sleep_s = delay + random.uniform(0, 0.6)
                print(f"[push_hf] HfHubHTTPError ({code}). retry in {sleep_s:.1f}s...", flush=True)
                time.sleep(sleep_s); tries += 1; delay *= 2; continue
            raise

        except Exception as e:
            if tries < 3:
                import time, random
                sleep_s = delay + random.uniform(0, 0.6)
                print(f"[push_hf] transient error: {e}. retry in {sleep_s:.1f}s...", flush=True)
                time.sleep(sleep_s); tries += 1; delay *= 2; continue
            raise

if __name__ == "__main__":
    sys.exit(main())
