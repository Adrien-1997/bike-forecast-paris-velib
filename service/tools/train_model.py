# service/tools/train_model.py
from __future__ import annotations
import os, sys, shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Tuple
from google.cloud import storage

def _parse_gs(uri: str) -> Tuple[str, str]:
    assert uri.startswith("gs://"), f"Bad GCS uri: {uri}"
    bkt, key = uri[5:].split("/", 1)
    return bkt, key

def _download(cli: storage.Client, src_uri: str, dst_path: Path) -> None:
    bkt, key = _parse_gs(src_uri)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cli.bucket(bkt).blob(key).download_to_filename(str(dst_path))

def main() -> int:
    # ENV attendus
    TRAIN_EXPORT_GCS = os.environ["TRAIN_EXPORT_GCS"]  # ex: gs://.../velib/exports/velib.parquet
    HORIZON_MIN      = int(os.environ.get("HORIZON_MIN", "15"))
    LOOKBACK_DAYS    = int(os.environ.get("LOOKBACK_DAYS", "30"))

    # Télécharge la base vers /tmp et expose TRAIN_EXPORT (consommé par ton train_model.py)
    local_base = Path("/tmp/exports/velib.parquet")
    cli = storage.Client()
    print(f"[train_job] download {TRAIN_EXPORT_GCS} → {local_base}", flush=True)
    _download(cli, TRAIN_EXPORT_GCS, local_base)

    # Variables pour ton script
    os.environ["TRAIN_EXPORT"] = str(local_base)
    os.environ["HORIZON_MIN"] = str(HORIZON_MIN)
    os.environ["LOOKBACK_DAYS"] = str(LOOKBACK_DAYS)

    # Appelle ton implémentation existante
    from service.train.forecast import train
    print(f"[train_job] start train(horizon={HORIZON_MIN}min, lookback={LOOKBACK_DAYS}d)", flush=True)
    metrics = train(horizon_minutes=HORIZON_MIN, lookback_days=LOOKBACK_DAYS)
    print("[train_job] metrics:", metrics, flush=True)

    # (optionnel) si ton train() sauvegarde un modèle, tu peux aussi l’uploader ici
    # via google.cloud.storage, comme on l’a fait pour la base, si besoin.

    return 0

if __name__ == "__main__":
    sys.exit(main())
