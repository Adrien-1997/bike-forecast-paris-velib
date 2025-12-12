# scripts/start-local.ps1
$ErrorActionPreference = 'Stop'

# Go to repo root (parent of scripts/)
Set-Location -Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

Write-Host ''
Write-Host '==============================='
Write-Host ' Velib Forecast — Local start '
Write-Host '==============================='
Write-Host ''

# 1) API
Write-Host '[API] Starting FastAPI (local backend)...'
$venvPy = (Resolve-Path '.\.venv\Scripts\python.exe').Path

$apiArgs = @(
  '-m','uvicorn',
  'api.app:app',
  '--reload',
  '--reload-dir','api',
  '--port','8081',
  '--env-file','api/.env'
)

$apiProc = Start-Process -PassThru -NoNewWindow -FilePath $venvPy -ArgumentList $apiArgs
Start-Sleep -Seconds 2
Write-Host '[API] http://localhost:8081'
Write-Host ''

# 2) UI build
Write-Host '[UI] Building Next.js app...'
Push-Location ui
npm run build
Pop-Location
Write-Host '[UI] Build OK'
Write-Host ''

# 3) UI start
Write-Host '[UI] Starting Next.js (production mode)...'
$uiArgs = @('run','start')
$uiProc = Start-Process -PassThru -NoNewWindow -WorkingDirectory (Resolve-Path 'ui').Path -FilePath 'npm' -ArgumentList $uiArgs

Write-Host ''
Write-Host '[DONE]'
Write-Host ' API : http://localhost:8081'
Write-Host ' UI  : http://localhost:3000'
Write-Host ''
Write-Host ('API PID={0} | UI PID={1}' -f $apiProc.Id, $uiProc.Id)
Write-Host 'Press Ctrl+C to stop. If needed, kill the two PIDs above.'
Write-Host ''

Wait-Process -Id $apiProc.Id, $uiProc.Id
