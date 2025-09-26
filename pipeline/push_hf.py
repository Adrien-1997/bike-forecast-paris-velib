# pipeline/push_hf.py
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from huggingface_hub import HfApi
try:
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:
    try:
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        class HfHubHTTPError(Exception):  # type: ignore
            def __init__(self, *args, **kwargs):
                self.response = getattr(self, "response", None)
                super().__init__(*args, **kwargs)

REPO_ID = os.environ.get("HF_REPO_ID", "Adrien97/velib-monitoring-historical")
REPO_TYPE = "dataset"
api = HfApi()

# ----------------------------- backoff helpers -----------------------------
def _retry_after_or(backoff: int, e: Exception) -> int:
    try:
        ra = getattr(getattr(e, "response", None), "headers", {}).get("Retry-After")
        if ra and str(ra).isdigit():
            return max(backoff, int(ra))
    except Exception:
        pass
    return backoff

def _with_backoff(fn, *, max_retries: int = 6, first_backoff: int = 5, desc: str = "op", on_429_only: bool = False):
    backoff = first_backoff
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except HfHubHTTPError as e:  # type: ignore
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code == 429 and attempt < max_retries:
                delay = _retry_after_or(backoff, e)
                print(f"[push_hf] 429 on {desc} → retry in {delay}s (attempt {attempt}/{max_retries})")
                time.sleep(delay)
                backoff *= 2
                continue
            if on_429_only:
                raise
            if attempt < max_retries:
                delay = _retry_after_or(backoff, e)
                print(f"[push_hf] http {code} on {desc} → retry in {delay}s (attempt {attempt}/{max_retries})")
                time.sleep(delay)
                backoff *= 2
                continue
            raise
        except Exception as e:
            if attempt < max_retries:
                print(f"[push_hf] transient on {desc}: {e} → retry in {backoff}s (attempt {attempt}/{max_retries})")
                time.sleep(backoff)
                backoff *= 2
                continue
            raise

# ----------------------------- core ops -----------------------------
def _upload(local_path: Path, dest_path: str, msg: str, max_retries: int = 6):
    if not local_path.exists():
        print(f"[push_hf] missing local file: {local_path}", file=sys.stderr)
        sys.exit(1)

    dest_path = dest_path.replace("\\", "/")
    print(f"[push_hf] upload {local_path} -> {REPO_ID}:{dest_path}")

    def _do():
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=dest_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=msg,
        )

    _with_backoff(_do, max_retries=max_retries, first_backoff=5, desc=f"upload {dest_path}")

def _delete_remote_prefix(prefix: str, commit_message: str):
    prefix = prefix.replace("\\", "/")

    def _list():
        return [e.path for e in api.list_repo_tree(
            repo_id=REPO_ID, repo_type=REPO_TYPE, path=prefix, recursive=True
        )]

    try:
        files_under = _with_backoff(_list, max_retries=6, first_backoff=5, desc=f"list {prefix}")
    except HfHubHTTPError as e:
        code = getattr(e.response, "status_code", None)
        if code == 429:
            print(f"[push_hf] WARN: 429 listing {prefix}, skip cleanup this run")
            return
        raise

    to_delete = [p for p in files_under if p.startswith(prefix)]
    if not to_delete:
        print(f"[push_hf] nothing to delete under {prefix}")
        return

    print(f"[push_hf] delete {len(to_delete)} files under {prefix}")
    for i, fp in enumerate(to_delete, 1):
        def _del():
            api.delete_file(
                path_in_repo=fp,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                commit_message=commit_message if i == len(to_delete) else None,
            )
        try:
            _with_backoff(_del, max_retries=6, first_backoff=5, on_429_only=True, desc=f"delete {fp}")
        except HfHubHTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code == 429:
                print(f"[push_hf] WARN: 429 deleting {fp}, will retry next run")
                continue
            raise

    print("[push_hf] delete done")

# ----------------------------- high-level commands -----------------------------
def push_ingest(file_path: str):
    p = Path(file_path)
    if not p.exists():
        print(f"[push_hf] ingest file not found: {p}", file=sys.stderr)
        sys.exit(1)
    dest = str(p).replace("\\", "/")
    _upload(p, dest, f"staging snapshot {p.name}")

def push_daily(day: str):
    local = Path(f"data/daily/{day}.parquet")
    dest = f"data/daily/{day}.parquet"
    _upload(local, dest, f"daily {day}")

def push_monthly(month: Optional[str]):
    if month is None:
        month = datetime.now(timezone.utc).strftime("%Y-%m")
    local = Path(f"data/monthly/{month}.parquet")
    dest = f"data/monthly/{month}.parquet"
    _upload(local, dest, f"monthly {month}")

def delete_remote_staging_day(day: str):
    prefix = f"data/staging/{day}/"
    _delete_remote_prefix(prefix, commit_message=f"cleanup staging {day}")

# ----------------------------- CLI -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Push artifacts to Hugging Face dataset repo")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s_ing = sub.add_parser("ingest", help="push one staging snapshot file")
    s_ing.add_argument("--file", required=True, help="local path to snapshot parquet under data/staging/...")

    s_daily = sub.add_parser("daily", help="push daily parquet for a given day")
    s_daily.add_argument("--day", required=True, help="YYYY-MM-DD")

    s_month = sub.add_parser("monthly", help="push monthly parquet")
    s_month.add_argument("--month", default=None, help="YYYY-MM (default: current UTC)")

    s_del = sub.add_parser("delete_staging_day", help="delete remote staging day folder")
    s_del.add_argument("--day", required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    if args.cmd == "ingest":
        push_ingest(args.file)
    elif args.cmd == "daily":
        push_daily(args.day)
    elif args.cmd == "monthly":
        push_monthly(args.month)
    elif args.cmd == "delete_staging_day":
        delete_remote_staging_day(args.day)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
