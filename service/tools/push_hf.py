# push_hf.py
from __future__ import annotations
import os
from pathlib import Path
from huggingface_hub import HfApi, upload_folder

def main():
    HF_TOKEN = os.environ.get("HF_TOKEN")
    HF_REPO  = os.environ.get("HF_REPO")  # ex: "user/velib-parquet"
    if not HF_TOKEN or not HF_REPO:
        raise RuntimeError("Set HF_TOKEN and HF_REPO")

    api = HfApi(token=HF_TOKEN)

    # Choisis ce que tu pushes (dossiers locaux)
    to_push = [
        os.environ.get("DAILY_DIR", "data_local/daily"),
        os.environ.get("LATEST_DIR","data_local/latest"),
        os.environ.get("MONTHLY_DIR","data_local/monthly"),
        os.environ.get("TRAIN_EXPORT_DIR","exports"),
    ]
    to_push = [p for p in to_push if p and Path(p).exists()]

    for folder in to_push:
        print(f"[push_hf] upload_folder {folder} → {HF_REPO}")
        upload_folder(
            repo_id=HF_REPO,
            repo_type="dataset",
            folder_path=folder,
            path_in_repo=Path(folder).name,
            token=HF_TOKEN,
            allow_patterns=["*.parquet","*.csv","*.json"]
        )

    print("[push_hf] DONE")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
