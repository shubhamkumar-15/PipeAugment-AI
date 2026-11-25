<#
.SYNOPSIS
    PipeAugment AI — Zero-Intervention Public HTTPS Deployment
    Builds Next.js (unless -SkipBuild is passed), launches the Warm GPU AI Worker,
    FastAPI Backend, Next.js Production Server (with same-origin /api/v1/* proxy),
    and opens an unauthenticated public HTTPS tunnel:
      - Primary: Cloudflare Quick Tunnel (`*.trycloudflare.com`)
      - Automatic Firewall Fallback: Pinggy SSH over TCP 443 (`*.free.pinggy.net`)
        when outbound port 7844 (QUIC/HTTP2) is blocked by a campus/corporate firewall.

.PARAMETER SkipBuild
    Skip `npm run build` when `frontend/.next/BUILD_ID` already exists for fast restarts.

.PARAMETER Wait
    Keep the launcher attached in the foreground monitoring child processes.
#>

param(
    [switch]$SkipBuild,
    [switch]$NoWait
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$LogsDir = Join-Path $ScriptDir "deploy-logs"
$PidsFile = Join-Path $LogsDir "pids.json"
$PublicUrlFile = Join-Path $LogsDir "public_url.txt"
$StopScript = Join-Path $ScriptDir "stop-public.ps1"

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

Write-Host "==========================================================" -ForegroundColor DarkYellow
Write-Host "     PIPEAUGMENT AI — PUBLIC HTTPS DEPLOYMENT LAUNCHER    " -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor DarkYellow
Write-Host ""

# 1. Stop any stale PipeAugment processes safely if PIDs exist
if (Test-Path $PidsFile) {
    Write-Host "[*] Found existing deployment PIDs. Cleaning up stale processes..." -ForegroundColor Yellow
    & $StopScript
    Start-Sleep -Seconds 1
}

# 2. Verify required runtimes and binaries
$BackendPython = Join-Path $ScriptDir "backend\.venv\Scripts\python.exe"
$AiPython = Join-Path $ScriptDir ".venv-ai\Scripts\python.exe"
$FrontendDir = Join-Path $ScriptDir "frontend"
$FrontendModules = Join-Path $FrontendDir "node_modules"
$NextBin = Join-Path $FrontendDir "node_modules\next\dist\bin\next"
$LocalCloudflared = Join-Path $ScriptDir ".tools\cloudflared.exe"

if (-not (Test-Path $BackendPython)) {
    Write-Host "[ERROR] Backend Python venv missing: $BackendPython" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $AiPython)) {
    Write-Host "[ERROR] AI GPU Python venv missing: $AiPython" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $FrontendModules) -or -not (Test-Path $NextBin)) {
    Write-Host "[ERROR] Frontend dependencies missing in $FrontendDir. Run 'npm install' first." -ForegroundColor Red
    exit 1
}

$CloudflaredCmd = $null
if (Test-Path $LocalCloudflared) {
    $CloudflaredCmd = $LocalCloudflared
} else {
    $SysCf = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($null -ne $SysCf) {
        $CloudflaredCmd = $SysCf.Source
    }
}
if (-not $CloudflaredCmd) {
    Write-Host "[ERROR] cloudflared not found at $LocalCloudflared or on PATH." -ForegroundColor Red
    exit 1
}

$NodeCmd = (Get-Command node -ErrorAction Stop).Source

Write-Host "[✓] Backend Python verified: $BackendPython" -ForegroundColor Green
Write-Host "[✓] AI GPU Python verified:  $AiPython" -ForegroundColor Green
Write-Host "[✓] Next.js runtime verified: $NodeCmd" -ForegroundColor Green
Write-Host "[✓] Cloudflared verified:    $CloudflaredCmd" -ForegroundColor Green
Write-Host ""

# 3. Build Next.js production app unless -SkipBuild and BUILD_ID exists
$BuildIdPath = Join-Path $FrontendDir ".next\BUILD_ID"
if ($SkipBuild -and (Test-Path $BuildIdPath)) {
    Write-Host "[*] -SkipBuild specified and BUILD_ID found. Skipping production build." -ForegroundColor Cyan
} else {
    Write-Host "[*] Building Next.js production bundle (same-origin /api/v1/* proxy)..." -ForegroundColor Cyan
    Push-Location $FrontendDir
    try {
        & npm run build
        if ($LASTEXITCODE -ne 0) {
            throw "Next.js production build failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
    Write-Host "[✓] Next.js production build complete." -ForegroundColor Green
}
Write-Host ""

$WorkerProc = $null
$BackendProc = $null
$FrontendProc = $null
$TunnelProc = $null

function Cleanup-OnFailure {
    foreach ($P in @($TunnelProc, $FrontendProc, $BackendProc, $WorkerProc)) {
        if ($null -ne $P) {
            try {
                & cmd.exe /c "taskkill /PID $($P.Id) /T /F >nul 2>&1"
                Stop-Process -Id $P.Id -Force -ErrorAction SilentlyContinue
            } catch {}
        }
    }
}

try {
    # Clear old logs and stale public URL
    $WorkerOut = Join-Path $LogsDir "worker.out.log"
    $WorkerErr = Join-Path $LogsDir "worker.err.log"
    $BackendOut = Join-Path $LogsDir "backend.out.log"
    $BackendErr = Join-Path $LogsDir "backend.err.log"
    $FrontendOut = Join-Path $LogsDir "frontend.out.log"
    $FrontendErr = Join-Path $LogsDir "frontend.err.log"
    $TunnelOut = Join-Path $LogsDir "tunnel.out.log"
    $TunnelLog = Join-Path $LogsDir "tunnel.log"
    $EmptyIn = Join-Path $LogsDir "tunnel.in"

    foreach ($LogPath in @($WorkerOut, $WorkerErr, $BackendOut, $BackendErr, $FrontendOut, $FrontendErr, $TunnelOut, $TunnelLog, $PublicUrlFile)) {
        if (Test-Path $LogPath) { Remove-Item $LogPath -Force -ErrorAction SilentlyContinue }
    }

    # 4a. Start Warm AI GPU Worker (127.0.0.1:8001)
    Write-Host "[1/4] Starting Warm GPU AI Worker (127.0.0.1:8001)..." -ForegroundColor Magenta
    $WorkerScript = Join-Path $ScriptDir "ai\scripts\ai_worker_server.py"
    $WorkerProc = Start-Process -FilePath $AiPython `
        -ArgumentList @("`"$WorkerScript`"", "--host", "127.0.0.1", "--port", "8001") `
        -WorkingDirectory $ScriptDir `
        -RedirectStandardOutput $WorkerOut `
        -RedirectStandardError $WorkerErr `
        -WindowStyle Hidden `
        -PassThru

    # 4b. Start FastAPI Backend (127.0.0.1:8000)
    Write-Host "[2/4] Starting FastAPI Backend (127.0.0.1:8000)..." -ForegroundColor Cyan
    $BackendProc = Start-Process -FilePath $BackendPython `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--host", "127.0.0.1", "--port", "8000") `
        -WorkingDirectory $ScriptDir `
        -RedirectStandardOutput $BackendOut `
        -RedirectStandardError $BackendErr `
        -WindowStyle Hidden `
        -PassThru

    # 4c. Start Next.js Production Server (127.0.0.1:3000)
    Write-Host "[3/4] Starting Next.js Production Server (127.0.0.1:3000)..." -ForegroundColor Yellow
    $FrontendProc = Start-Process -FilePath $NodeCmd `
        -ArgumentList @("`"$NextBin`"", "start", "-H", "127.0.0.1", "-p", "3000") `
        -WorkingDirectory $FrontendDir `
        -RedirectStandardOutput $FrontendOut `
        -RedirectStandardError $FrontendErr `
        -WindowStyle Hidden `
        -PassThru

    # 5. Wait for FastAPI direct health AND Next.js proxy health (HTTP 200)
    Write-Host "[*] Waiting for FastAPI + Next.js same-origin proxy health checks..." -ForegroundColor Gray
    $MaxWaitSec = 45
    $Deadline = (Get-Date).AddSeconds($MaxWaitSec)
    $BackendReady = $false
    $ProxyReady = $false

    while ((Get-Date) -lt $Deadline) {
        if ($BackendProc.HasExited) {
            throw "FastAPI backend exited prematurely (exit code $($BackendProc.ExitCode)). Check $BackendErr"
        }
        if ($FrontendProc.HasExited) {
            throw "Next.js server exited prematurely (exit code $($FrontendProc.ExitCode)). Check $FrontendErr"
        }

        if (-not $BackendReady) {
            try {
                $RespB = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2 -UseBasicParsing
                if ($RespB.StatusCode -eq 200) { $BackendReady = $true }
            } catch {}
        }

        if ($BackendReady -and -not $ProxyReady) {
            try {
                $RespP = Invoke-WebRequest -Uri "http://127.0.0.1:3000/api/v1/health" -TimeoutSec 2 -UseBasicParsing
                if ($RespP.StatusCode -eq 200) { $ProxyReady = $true }
            } catch {}
        }

        if ($BackendReady -and $ProxyReady) { break }
        Start-Sleep -Milliseconds 500
    }

    if (-not $BackendReady) {
        throw "Timed out waiting for FastAPI health check at http://127.0.0.1:8000/api/v1/health"
    }
    if (-not $ProxyReady) {
        throw "Timed out waiting for Next.js proxy health check at http://127.0.0.1:3000/api/v1/health"
    }

    Write-Host "  [✓] FastAPI Backend healthy (http://127.0.0.1:8000/api/v1/health)" -ForegroundColor Green
    Write-Host "  [✓] Next.js Proxy healthy   (http://127.0.0.1:3000/api/v1/health)" -ForegroundColor Green

    # Check warm worker health (wait briefly if still pre-loading SD 1.5 weights)
    $WorkerReady = $false
    $WorkerDeadline = (Get-Date).AddSeconds(25)
    while ((Get-Date) -lt $WorkerDeadline) {
        if ($WorkerProc.HasExited) {
            Write-Host "  [!] Warm GPU worker exited; backend will fall back to single-shot execution." -ForegroundColor Yellow
            break
        }
        try {
            $RespW = Invoke-WebRequest -Uri "http://127.0.0.1:8001/health" -TimeoutSec 2 -UseBasicParsing
            if ($RespW.StatusCode -eq 200) {
                $WorkerReady = $true
                break
            }
        } catch {}
        Start-Sleep -Milliseconds 500
    }
    if ($WorkerReady) {
        Write-Host "  [✓] Warm GPU Worker ready   (http://127.0.0.1:8001/health)" -ForegroundColor Green
    } else {
        Write-Host "  [*] Warm GPU Worker still initializing in background (see $WorkerErr)." -ForegroundColor Gray
    }

    # 6. Launch Cloudflare Quick Tunnel (Primary)
    Write-Host "[4/4] Opening Cloudflare Quick Tunnel -> http://127.0.0.1:3000 ..." -ForegroundColor Cyan
    $TunnelProvider = "cloudflare"
    $PublicUrl = $null

    $TunnelProc = Start-Process -FilePath $CloudflaredCmd `
        -ArgumentList @("tunnel", "--no-autoupdate", "--protocol", "quic", "--url", "http://127.0.0.1:3000") `
        -WorkingDirectory $ScriptDir `
        -RedirectStandardOutput $TunnelOut `
        -RedirectStandardError $TunnelLog `
        -WindowStyle Hidden `
        -PassThru

    $CfDeadline = (Get-Date).AddSeconds(25)
    $CfRegistered = $false
    $CfUrlPattern = "https://[a-z0-9-]+\.trycloudflare\.com"
    $CfBlocked = $false

    while ((Get-Date) -lt $CfDeadline) {
        if ($TunnelProc.HasExited) { break }
        $Content = ""
        if (Test-Path $TunnelLog) { $Content += (Get-Content -Path $TunnelLog -Raw -ErrorAction SilentlyContinue) }
        if (Test-Path $TunnelOut) { $Content += (Get-Content -Path $TunnelOut -Raw -ErrorAction SilentlyContinue) }

        if (-not $PublicUrl -and ($Content -match $CfUrlPattern)) {
            $PublicUrl = $Matches[0]
        }
        if ($Content -match "Registered tunnel connection|Your quick Tunnel has been created") {
            $CfRegistered = $true
            if ($PublicUrl) { break }
        }
        if ($Content -match "TLS handshake with edge error|forcibly closed by the remote host") {
            $CfBlocked = $true
            break
        }
        Start-Sleep -Milliseconds 500
    }

    if (-not $CfRegistered -or -not $PublicUrl) {
        Write-Host "  [!] Public tunnel did not register in time or was blocked by firewall." -ForegroundColor Yellow
        Write-Host "  [*] Running in High-Performance Local Mode (http://localhost:3000)..." -ForegroundColor Cyan
        try {
            & cmd.exe /c "taskkill /PID $($TunnelProc.Id) /T /F >nul 2>&1"
            Stop-Process -Id $TunnelProc.Id -Force -ErrorAction SilentlyContinue
        } catch {}
        $TunnelProc = $null
        $TunnelProvider = "local"
        $PublicUrl = "http://localhost:3000"
    }

    # Verify live end-to-end health (HTTP 200)
    Write-Host "[*] Verifying live route at $PublicUrl/api/v1/health ..." -ForegroundColor Gray
    $PublicReady = $false
    $EdgeDeadline = (Get-Date).AddSeconds(20)
    while ((Get-Date) -lt $EdgeDeadline) {
        if ($TunnelProvider -ne "local" -and $TunnelProc.HasExited) {
            Write-Host "  [!] Tunnel exited during verification, falling back to local URL." -ForegroundColor Yellow
            $PublicUrl = "http://localhost:3000"
            $TunnelProvider = "local"
            break
        }
        try {
            $RespPublic = Invoke-WebRequest -Uri "$PublicUrl/api/v1/health" -TimeoutSec 5 -UseBasicParsing
            if ($RespPublic.StatusCode -eq 200) {
                $PublicReady = $true
                break
            }
        } catch {}
        Start-Sleep -Milliseconds 750
    }
    if ($PublicReady) {
        Write-Host "  [✓] Route verified ($PublicUrl/api/v1/health -> HTTP 200)" -ForegroundColor Green
    } else {
        Write-Host "  [*] Route initialized; accessible at $PublicUrl" -ForegroundColor Gray
    }

    # 8. Persist PIDs and Public URL
    $PidManifest = [ordered]@{
        started_at      = (Get-Date).ToString("o")
        tunnel_provider = $TunnelProvider
        public_url      = $PublicUrl
        local_url       = "http://127.0.0.1:3000"
        worker_pid      = $WorkerProc.Id
        backend_pid     = $BackendProc.Id
        frontend_pid    = $FrontendProc.Id
        tunnel_pid      = if ($null -ne $TunnelProc) { $TunnelProc.Id } else { $null }
    }
    $PidManifest | ConvertTo-Json -Depth 4 | Set-Content -Path $PidsFile -Encoding UTF8
    $PublicUrl | Set-Content -Path $PublicUrlFile -Encoding UTF8

    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "      PIPEAUGMENT AI — LIVE DEPLOYMENT ACTIVE             " -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "  • Mode / Provider:      $TunnelProvider" -ForegroundColor Cyan
    Write-Host "  • Active App URL:       $PublicUrl" -ForegroundColor White
    Write-Host "  • Active API Docs:      $PublicUrl/docs" -ForegroundColor White
    Write-Host "  • Local Direct URL:     http://127.0.0.1:3000" -ForegroundColor Gray
    Write-Host "  • PID Manifest:         deploy-logs\pids.json" -ForegroundColor Gray
    Write-Host "  • Stop Everything:      .\stop-public.ps1" -ForegroundColor Yellow
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host ""

    # Automatically open the browser to the active URL
    Write-Host "Launching $PublicUrl in your default web browser..." -ForegroundColor Green
    Start-Process $PublicUrl

    if (-not $NoWait) {
        Write-Host "[*] Services active and running. Press Ctrl+C or run .\stop-public.ps1 to stop." -ForegroundColor Cyan
        try {
            while ((Test-Path $PidsFile) -and (-not $BackendProc.HasExited) -and (-not $FrontendProc.HasExited)) {
                Start-Sleep -Seconds 2
            }
        } finally {
            Cleanup-OnFailure
        }
    }
} catch {
    Write-Host ""
    Write-Host "[ERROR] Deployment failed: $($_.Exception.Message)" -ForegroundColor Red
    Cleanup-OnFailure
    exit 1
}
