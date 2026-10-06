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
    & $VenvPython -m pip install -e $ScriptDir
}

# 2. Setup-only check
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
        param($TargetUrl)
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Seconds 1
            try {
                $r = Invoke-RestMethod -Uri $TargetUrl -TimeoutSec 1 -ErrorAction SilentlyContinue
                if ($r) {
                    Start-Process "http://127.0.0.1:8000"
                    break
                }
            } catch {}
        }
    }
    Start-Job -ScriptBlock $JobScript -ArgumentList $HealthUrl | Out-Null
}

# Execute server in foreground
try {
    & $VenvPython -m uvicorn fpl_oracle.server.main:app --host 127.0.0.1 --port $Port
} finally {
    Get-Job | Stop-Job -ErrorAction SilentlyContinue
    Get-Job | Remove-Job -ErrorAction SilentlyContinue
    Write-Host "`n[+] FPL Oracle server stopped cleanly." -ForegroundColor Green
}
