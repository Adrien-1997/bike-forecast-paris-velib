# Vélib’ Forecast — Local development

This document explains how to run the **Vélib’ Forecast application locally**
(API + UI) without Docker, using local data and configuration.

---

## Prerequisites

- **Python 3.11+** (recommended: 3.12)
- **Node.js 20+**
- **npm**
- (Windows) PowerShell  
- (Optional) Git Bash for Bash scripts

---

## Project structure (relevant parts)

```text
repo/
├─ api/
│  ├─ .env                # API configuration (local / gcs, paths, models, etc.)
│  ├─ requirements.txt
│
├─ ui/
│  ├─ package.json
│  ├─ .env.local          # UI configuration (API base)
│
├─ scripts/
│  ├─ start-local.ps1     # Windows launcher
│  ├─ start-local.sh      # Bash / Git Bash launcher
│
├─ .venv/                 # Python virtual environment (local)
```

---

## One-time setup

### 1) Python environment (API)

From the **repository root**:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r api\requirements.txt
```

This installs:
- FastAPI / Uvicorn
- Pandas / PyArrow
- Google Cloud Storage client
- All API runtime dependencies

---

### 2) Node dependencies (UI)

```powershell
cd ui
npm install
cd ..
```

---

## Environment configuration

### API

- The API reads **`api/.env`**
- This file is the **single source of truth** for:
  - local vs GCS mode
  - data paths
  - models (H15 / H60)
  - monitoring & serving prefixes

The API is started with:

```bash
uvicorn ... --env-file api/.env
```

so no manual environment variables are required.

---

### UI

- The UI reads **`ui/.env.local`**
- For local development, it must point to the local API:

```dotenv
NEXT_PUBLIC_API_BASE=http://localhost:8081
NEXT_PUBLIC_HTTP_TIMEOUT_MS=30000
```

⚠️ Any variable prefixed with `NEXT_PUBLIC_` is exposed to the browser.
Do not put secrets there.

---

## Run locally

### Windows (PowerShell)

```powershell
.\scripts\start-local.ps1
```

### Bash / Git Bash

```bash
chmod +x scripts/start-local.sh
./scripts/start-local.sh
```

What the script does:
1. Starts the API (FastAPI) with `api/.env`
2. Builds the UI (`next build`)
3. Starts the UI in production mode (`next start`)

---

## Services

- **API**: http://localhost:8081  
  - Swagger: http://localhost:8081/docs
- **UI**: http://localhost:3000

---

## Notes

- To stop the application: `Ctrl+C`
- If needed, kill the processes using the printed PIDs.
- For faster frontend iteration, you can also run:
  ```powershell
  cd ui
  npm run dev
  ```
  while keeping the API running.

---

## Production reference

In production (Cloud Run / Netlify):
- the API runs containerized with injected environment variables
- the UI uses a Netlify proxy (`/.netlify/functions/api-proxy`)

This local setup intentionally bypasses the proxy to talk directly to the API.
