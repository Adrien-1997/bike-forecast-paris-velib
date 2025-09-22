FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates tzdata && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ENV PYTHONPATH=/workspace \
    PUSH_SRC=exports/velib.parquet \
    PUSH_DEST=exports/velib.parquet

COPY requirements-pipeline.txt .
RUN pip install -U pip && pip install -r requirements-pipeline.txt && pip install huggingface_hub

COPY src/ src/
COPY tools/ tools/
COPY exports/ exports/

CMD python -m src.ingest && python -m src.aggregate --input exports/snapshot.parquet --output exports/velib.parquet && python tools/push_hf.py
