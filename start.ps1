<#
.SYNOPSIS
    FPL Oracle 2026/27 - One-command Windows PowerShell Launcher.
    Starts the local decision support dashboard at http://127.0.0.1:8000.

.DESCRIPTION
    Everyday execution:
        .\start.ps1

    If PowerShell script execution is restricted on your machine, run with process-scoped policy:
        powershell -ExecutionPolicy Bypass -File .\start.ps1

.PARAMETER NoBrowser
    Do not automatically launch the default web browser.

.PARAMETER Port
    HTTP loopback port to bind (default: 8000).

.PARAMETER SetupOnly
    Perform environment verification and setup then exit without launching the server.
#>

[CmdletBinding()]
param (
    [switch]$NoBrowser,
    [int]$Port = 8000,
    [switch]$SetupOnly
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "==========================================================" -ForegroundColor Green
Write-Host " FPL Oracle 2026/27 - Decision Support and Intelligence   " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

# 1. Virtual Environment and Python Detection
$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "[!] Virtual environment not detected at .venv" -ForegroundColor Yellow
    Write-Host "[*] Searching for system Python 3.11+..." -ForegroundColor Cyan
    
    $SysPython = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $SysPython) {
        Write-Error "Python 3.11+ is required but was not found in PATH. Please install Python and try again."
        exit 1
    }
    
    Write-Host "[*] Creating virtual environment at .venv..." -ForegroundColor Cyan
    & $SysPython -m venv (Join-Path $ScriptDir ".venv")
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit 1
    }
    
    Write-Host "[*] Installing dependencies in editable mode..." -ForegroundColor Cyan
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
    & $VenvPython -m pip install -e $ScriptDir
    if ($LASTEXITCODE -ne 0) { throw "Dependency install failed" }
}

# Keep existing environments on the tested solver API, not only fresh installs.
& $VenvPython -c "import pulp, sys; sys.exit(0 if pulp.__version__ == '3.3.2' else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[*] Installing tested PuLP 3.3.2 solver runtime..." -ForegroundColor Cyan
    & $VenvPython -m pip install "pulp==3.3.2"
    if ($LASTEXITCODE -ne 0) { throw "PuLP runtime correction failed" }
}
& $VenvPython (Join-Path $ScriptDir "scripts/check_solver_runtime.py")
if ($LASTEXITCODE -ne 0) { throw "CBC solve verification failed; server not started" }

# 2. Setup-only check
& $VenvPython -c "import sys; assert sys.version_info >= (3,11); import fpl_oracle.server.main"
if ($LASTEXITCODE -ne 0) { throw "Environment verification failed" }

if ($SetupOnly) {
    Write-Host "[+] Environment setup verified successfully." -ForegroundColor Green
    exit 0
}

# 3. Check for Existing Running Instance or Port Conflict
$HealthUrl = "http://127.0.0.1:$Port/api/health"
$InstanceAlreadyRunning = $false

try {
    $HealthResponse = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($HealthResponse -and ($HealthResponse.status -eq "healthy" -or $HealthResponse.app -eq "fpl-oracle")) {
        $InstanceAlreadyRunning = $true
    }
} catch {
    # Port is not serving health, might be free or occupied by other process
}

if ($InstanceAlreadyRunning) {
    Write-Host "[+] FPL Oracle is already running at http://127.0.0.1:$Port" -ForegroundColor Green
    if (-not $NoBrowser) {
        Write-Host "[*] Opening browser..." -ForegroundColor Cyan
        Start-Process "http://127.0.0.1:$Port"
    }
    exit 0
}

# Check if port is held by an unrelated conflicting process
$PortConflict = $false
try {
    $TcpListener = [System.Net.Sockets.TcpClient]::new()
    $TcpListener.Connect("127.0.0.1", $Port)
    $TcpListener.Close()
    $PortConflict = $true
} catch {
    $PortConflict = $false
}

if ($PortConflict) {
    Write-Host "[!] Port $Port is currently occupied by another application." -ForegroundColor Red
    Write-Host "    Please specify an alternate port: .\start.ps1 -Port 8001" -ForegroundColor Yellow
    Write-Host "    Or terminate the conflicting process." -ForegroundColor Yellow
    exit 1
}

# 4. Launch Application Server
Write-Host "[*] Starting FPL Oracle server on http://127.0.0.1:$Port ..." -ForegroundColor Cyan
Write-Host "    Local data and rules loaded. Press Ctrl+C to stop." -ForegroundColor Gray

# If browser launch requested, poll in background job or wait loop
if (-not $NoBrowser) {
    $JobScript = {
        param($TargetUrl, $BrowserUrl)
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Seconds 1
            try {
                $r = Invoke-RestMethod -Uri $TargetUrl -TimeoutSec 1 -ErrorAction SilentlyContinue
                if ($r) {
                    Start-Process $BrowserUrl
                    break
                }
            } catch {}
        }
    }
    $BrowserJob = Start-Job -ScriptBlock $JobScript -ArgumentList $HealthUrl, "http://127.0.0.1:$Port" | Out-Null
}

# Execute server in foreground
try {
    & $VenvPython -m uvicorn fpl_oracle.server.main:app --host 127.0.0.1 --port $Port
} finally {
    if ($BrowserJob) {
        $BrowserJob | Stop-Job -ErrorAction SilentlyContinue
        $BrowserJob | Remove-Job -ErrorAction SilentlyContinue
    }
    Write-Host "`n[+] FPL Oracle server stopped cleanly." -ForegroundColor Green
}
