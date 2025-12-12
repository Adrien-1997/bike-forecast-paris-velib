#!/usr/bin/env bash
set -euo pipefail

# Go to repo root (parent of scripts/)
cd "$(dirname "$0")/.."

echo
echo "==============================="
echo " Velib Forecast — Local start "
echo "==============================="
echo

# --- 1) Start API ---
echo "[API] Starting FastAPI (local backend)..."

PY="./.venv/Scripts/python.exe"
if [ ! -f "$PY" ]; then
  echo "ERROR: Python venv not found at $PY"
  exit 1
fi

"$PY" -m uvicorn api.app:app \
  --reload \
  --reload-dir api \
  --port 8081 \
  --env-file api/.env &
API_PID=$!

sleep 2
echo "[API] http://localhost:8081"
echo

# --- 2) Build UI ---
echo "[UI] Building Next.js app..."
( cd ui && npm run build )
echo "[UI] Build OK"
echo

# --- 3) Start UI (prod mode) ---
echo "[UI] Starting Next.js (production mode)..."
( cd ui && npm run start ) &
UI_PID=$!

echo
echo "[DONE]"
echo " API : http://localhost:8081"
echo " UI  : http://localhost:3000"
echo
echo "API PID=$API_PID | UI PID=$UI_PID"
echo "Press Ctrl+C to stop."
echo

# Stop both on Ctrl+C / termination
cleanup() {
  echo
  echo "[stop] Stopping..."
  kill "$UI_PID" 2>/dev/null || true
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

wait "$API_PID" "$UI_PID"