<#
.SYNOPSIS
    PipeAugment AI - One-Command Local Development Launcher
    Launches:
      1. Warm GPU AI Worker (127.0.0.1:8001) in .venv-ai
      2. FastAPI Backend (127.0.0.1:8000) in backend/.venv
      3. Next.js Frontend (localhost:3000)
#>

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "==========================================================" -ForegroundColor DarkYellow
Write-Host "         PIPEAUGMENT AI — LOCAL DEV LAUNCHER             " -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor DarkYellow
Write-Host ""

# 1. Check Backend Python virtual environment
$BackendPython = Join-Path $ScriptDir "backend\.venv\Scripts\python.exe"
if (-not (Test-Path $BackendPython)) {
    Write-Host "[ERROR] Backend Python virtual environment not found at:" -ForegroundColor Red
    Write-Host "        $BackendPython" -ForegroundColor Gray
    Write-Host "Please ensure backend/.venv is initialized." -ForegroundColor Yellow
    exit 1
}

# 2. Check AI Python virtual environment
$AiPython = Join-Path $ScriptDir ".venv-ai\Scripts\python.exe"
if (-not (Test-Path $AiPython)) {
    Write-Host "[WARNING] AI virtual environment (.venv-ai) not found at: $AiPython" -ForegroundColor Yellow
    Write-Host "          Warm worker will be skipped; backend will run in single-shot mode." -ForegroundColor Gray
}

# 3. Check Frontend node_modules
$FrontendModules = Join-Path $ScriptDir "frontend\node_modules"
if (-not (Test-Path $FrontendModules)) {
    Write-Host "[ERROR] Frontend node_modules not found at:" -ForegroundColor Red
    Write-Host "        $FrontendModules" -ForegroundColor Gray
    Write-Host "Run 'cd frontend; npm install' before starting." -ForegroundColor Yellow
    exit 1
}

Write-Host "[✓] Backend Python environment verified." -ForegroundColor Green
if (Test-Path $AiPython) {
    Write-Host "[✓] AI GPU Python environment verified." -ForegroundColor Green
}
Write-Host "[✓] Frontend Node.js dependencies verified." -ForegroundColor Green
Write-Host ""

# 4. Start Warm GPU AI Worker (Port 8001) if .venv-ai exists
if (Test-Path $AiPython) {
    Write-Host "Starting Warm GPU AI Worker on http://127.0.0.1:8001 ..." -ForegroundColor Magenta
    $WorkerScript = Join-Path $ScriptDir "ai\scripts\ai_worker_server.py"
    $WorkerCmd = "cd `"$ScriptDir`"; `$host.UI.RawUI.WindowTitle = 'PipeAugment AI - Warm GPU Worker (127.0.0.1:8001)'; Write-Host 'Starting PipeAugment AI Warm GPU Worker...' -ForegroundColor Magenta; & `"$AiPython`" `"$WorkerScript`" --port 8001"
    Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $WorkerCmd
    Start-Sleep -Seconds 2
}

# 5. Start FastAPI Backend in dedicated window (Port 8000)
Write-Host "Starting FastAPI Backend on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
$BackendCmd = "cd `"$ScriptDir`"; `$host.UI.RawUI.WindowTitle = 'PipeAugment AI - FastAPI Backend (127.0.0.1:8000)'; Write-Host 'Starting PipeAugment AI Backend...' -ForegroundColor Cyan; & `"$BackendPython`" -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload"
Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $BackendCmd

# 6. Start Next.js Frontend in dedicated window (Port 3000)
Write-Host "Starting Next.js Frontend on http://localhost:3000 ..." -ForegroundColor Yellow
$FrontendDir = Join-Path $ScriptDir "frontend"
$FrontendCmd = "cd `"$FrontendDir`"; `$host.UI.RawUI.WindowTitle = 'PipeAugment AI - Next.js (localhost:3000)'; Write-Host 'Starting PipeAugment AI Frontend...' -ForegroundColor Yellow; npm run dev"
Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $FrontendCmd

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  ALL 3 SERVICES LAUNCHED IN SEPARATE DEDICATED WINDOWS   " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  • Frontend Web UI:  http://localhost:3000" -ForegroundColor White
Write-Host "  • Backend API Docs: http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "  • Health Check:     http://127.0.0.1:8000/api/v1/health" -ForegroundColor White
Write-Host "  • Warm AI Worker:   http://127.0.0.1:8001/health" -ForegroundColor White
Write-Host ""
Write-Host "To stop services: Close the respective PowerShell windows or press Ctrl+C in each." -ForegroundColor Gray
Write-Host "==========================================================" -ForegroundColor DarkYellow
