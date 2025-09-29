# service/tools/train_model.py
from __future__ import annotations

import os
import sys
import glob
from datetime import datetime, timedelta
from pathlib import Path
from subprocess import run, CalledProcessError

import duckdb
from google.cloud import storage

EXPORT_PATH = Path("exports/velib.parquet")

def _iso_date(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")

def _run_module(mod: str, env: dict | None = None) -> None:
    try:
        run([sys.executable, "-m", mod], check=True, env=env)
    except CalledProcessError as e:
        raise RuntimeError(f"{mod} failed (code={e.returncode})") from e

def _export_training_base(train_days: int) -> None:
    end = datetime.utcnow().date()
    start = end - timedelta(days=train_days)
    env = os.environ.copy()
    env["START_DAY"] = _iso_date(datetime.combine(start, datetime.min.time()))
    env["END_DAY"]   = _iso_date(datetime.combine(end,   datetime.min.time()))
    if not env.get("GCS_DB_DAILY"):
        raise RuntimeError("GCS_DB_DAILY env var is required (gs://.../velib/db/daily)")
    print(f"[train_model] export base: {env['START_DAY']} → {env['END_DAY']} (days={train_days})")
    _run_module("tools.export_training_base", env=env)

def _count_export_rows(path: Path = EXPORT_PATH) -> int:
    if not path.exists():
        return 0
    try:
        n = duckdb.sql(f"SELECT COUNT(*) FROM '{path.as_posix()}'").fetchone()[0]
        return int(n or 0)
    except Exception as e:
        print(f"[train_model] count error on {path}: {e}")
        return 0

def _run_training() -> None:
    print("[train_model] start training (train.forecast)")
    _run_module("train.forecast")

def _collect_artifacts() -> list[Path]:
    candidates: list[str] = []
    for base in ("models", "service/models", "exports"):
        p = Path(base)
        if p.exists():
            candidates += glob.glob(f"{base}/**/*.joblib", recursive=True)
            candidates += glob.glob(f"{base}/**/*.pkl", recursive=True)
            candidates += glob.glob(f"{base}/**/*.bin", recursive=True)
    files = sorted({Path(x) for x in candidates if Path(x).is_file()})
    print(f"[train_model] artifacts found: {[p.as_posix() for p in files]}")
    return list(files)

def _upload_to_gcs(files: list[Path]) -> None:
    dst_prefix = os.environ.get("MODELS_PREFIX")  # e.g., gs://bucket/velib/models
    if not dst_prefix:
        print("[train_model] MODELS_PREFIX not set → skip GCS upload")
        return
    assert dst_prefix.startswith("gs://"), "MODELS_PREFIX must start with gs://"
    if "/" in dst_prefix[5:]:
        bkt, key_prefix = dst_prefix[5:].split("/", 1)
    else:
        bkt, key_prefix = dst_prefix[5:], ""
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    cli = storage.Client()
    for f in files:
        stamped_key = f"{key_prefix}/{ts}/{f.name}" if key_prefix else f"{ts}/{f.name}"
        print(f"[train_model] upload → gs://{bkt}/{stamped_key}")
        cli.bucket(bkt).blob(stamped_key).upload_from_filename(f.as_posix())
        latest_key = f"{key_prefix}/latest/{f.name}" if key_prefix else f"latest/{f.name}"
        cli.bucket(bkt).blob(latest_key).upload_from_filename(f.as_posix())
        print(f"[train_model] upload → gs://{bkt}/{latest_key}")
    print("[train_model] upload done")

def main() -> int:
    try:
        # 1) Export (fenêtre par défaut)
        train_days = int(os.environ.get("TRAIN_DAYS", "45"))
        _export_training_base(train_days)
        n = _count_export_rows()
        print(f"[train_model] export rows = {n}")

        # 2) Si 0 ligne, tenter un fallback plus large (90 jours)
        if n == 0 and train_days < 90:
            print("[train_model] empty export → retry with 90 days")
            _export_training_base(90)
            n = _count_export_rows()
            print(f"[train_model] export rows after retry = {n}")

        # 3) Si toujours 0 → stop proprement
        if n == 0:
            print("[train_model] no data to train on → skip training")
            return 0

        # 4) Entraînement
        _run_training()

        # 5) Collecte & upload
        artifacts = _collect_artifacts()
        if artifacts:
            _upload_to_gcs(artifacts)
        else:
            print("[train_model] WARNING: no artifacts found to upload")

        print("[train_model] OK")
        return 0
    except Exception as e:
        print(f"[train_model] ERROR: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
