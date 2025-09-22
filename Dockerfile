FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements-pipeline.txt .
RUN python -m pip install -U pip \
 && pip install --no-cache-dir -r requirements-pipeline.txt \
 && pip install --no-cache-dir huggingface_hub

# Copy project files
COPY . .

# Sleep 60s → ingest → aggregate → push_hf
CMD bash -lc "sleep 60 && PYTHONPATH=/app python -m src.ingest && PYTHONPATH=/app python -m src.aggregate && python tools/push_hf.py"
