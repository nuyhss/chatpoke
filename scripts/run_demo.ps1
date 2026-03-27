param(
    [ValidateSet("start", "stop", "restart", "status", "logs")]
    [string]$Command = "start",
    [int]$Port = 8001,
    [string]$Host = "0.0.0.0",
    [switch]$NoReload
)

$ErrorActionPreference = "Stop"

$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunDir = Join-Path $ProjectDir ".run"
$PidFile = Join-Path $RunDir "demo_server_win.pid"
$LogFile = Join-Path $RunDir "demo_server_win.log"

$UvicornExe = Join-Path $ProjectDir ".venv\Scripts\uvicorn.exe"
$PythonExe = Join-Path $ProjectDir ".venv\Scripts\python.exe"

function Write-Info([string]$Message) {
    Write-Host "[INFO] $Message"
}

function Write-WarnMsg([string]$Message) {
    Write-Host "[WARN] $Message"
}

function Write-ErrMsg([string]$Message) {
    Write-Host "[ERROR] $Message"
}

function Ensure-Ready {
    if (-not (Test-Path $RunDir)) {
        New-Item -ItemType Directory -Path $RunDir | Out-Null
    }

    if (-not (Test-Path $UvicornExe) -or -not (Test-Path $PythonExe)) {
        Write-ErrMsg ".venv not found."
        Write-Host "Run:" 
        Write-Host "  cd $ProjectDir"
        Write-Host "  python -m venv .venv"
        Write-Host "  .\\.venv\\Scripts\\Activate.ps1"
        Write-Host "  pip install -r requirements.txt"
        exit 1
    }

    $envFile = Join-Path $ProjectDir ".env"
    $envExample = Join-Path $ProjectDir ".env.example"
    if (-not (Test-Path $envFile) -and (Test-Path $envExample)) {
        Copy-Item $envExample $envFile
        Write-Info ".env created from .env.example"
    }
}

function Get-ManagedPid {
    if (-not (Test-Path $PidFile)) { return 0 }
    try {
        $raw = (Get-Content $PidFile -ErrorAction Stop | Select-Object -First 1).Trim()
        if ([string]::IsNullOrWhiteSpace($raw)) { return 0 }
        return [int]$raw
    } catch {
        return 0
    }
}

function Test-ProcessAlive([int]$Pid) {
    if ($Pid -le 0) { return $false }
    try {
        Get-Process -Id $Pid -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Test-PortInUse([int]$TargetPort) {
    $netTcp = Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue
    if ($null -ne $netTcp) {
        $conn = Get-NetTCPConnection -State Listen -LocalPort $TargetPort -ErrorAction SilentlyContinue
        return ($null -ne $conn)
    }

    $line = netstat -ano | Select-String -Pattern ":$TargetPort\s"
    return ($null -ne $line)
}

function Test-Health {
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 5 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Start-Server {
    Ensure-Ready

    $pid = Get-ManagedPid
    if (Test-ProcessAlive $pid) {
        Write-Info "Server already running (PID: $pid)"
        Write-Host "      UI:   http://127.0.0.1:$Port/ui"
        Write-Host "      Docs: http://127.0.0.1:$Port/docs"
        return
    }

    if ($pid -gt 0 -and -not (Test-ProcessAlive $pid)) {
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }

    if (Test-PortInUse $Port) {
        Write-ErrMsg "Port $Port is already in use."
        Write-Host "Try another port:"
        Write-Host "  .\\scripts\\run_demo.ps1 -Command start -Port 8002"
        exit 1
    }

    $args = @(
        "main:app",
        "--app-dir", $ProjectDir,
        "--host", $Host,
        "--port", "$Port"
    )

    if (-not $NoReload) {
        $args += "--reload"
    }

    Write-Info "Starting demo server..."
    $proc = Start-Process -FilePath $UvicornExe -ArgumentList $args -WorkingDirectory $ProjectDir -RedirectStandardOutput $LogFile -RedirectStandardError $LogFile -PassThru

    Set-Content -Path $PidFile -Value $proc.Id
    Start-Sleep -Seconds 2

    if (-not (Test-ProcessAlive $proc.Id)) {
        Write-ErrMsg "Failed to start server."
        if (Test-Path $LogFile) {
            Write-Host "Last logs:"
            Get-Content $LogFile -Tail 80
        }
        exit 1
    }

    if (Test-Health) {
        Write-Host "[OK] Demo server started (PID: $($proc.Id))"
        Write-Host "     UI:   http://127.0.0.1:$Port/ui"
        Write-Host "     Docs: http://127.0.0.1:$Port/docs"
    } else {
        Write-WarnMsg "Server is running (PID: $($proc.Id)) but /health is not ready yet."
        Write-Host "         Check logs: .\\scripts\\run_demo.ps1 -Command logs"
    }
}

function Stop-Server {
    $pid = Get-ManagedPid

    if ($pid -le 0) {
        Write-Info "No PID file. Server may already be stopped."
        return
    }

    if (-not (Test-ProcessAlive $pid)) {
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        Write-Info "Stale PID file removed."
        return
    }

    Write-Info "Stopping server (PID: $pid)..."
    try {
        Stop-Process -Id $pid -ErrorAction Stop
    } catch {
        Write-WarnMsg "Graceful stop failed; forcing..."
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    }

    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    Write-Host "[OK] Server stopped."
}

function Show-Status {
    $pid = Get-ManagedPid

    if ($pid -gt 0 -and (Test-ProcessAlive $pid)) {
        Write-Info "Server running (PID: $pid)"
        if (Test-Health) {
            Write-Info "Health: OK"
        } else {
            Write-WarnMsg "Health: NOT READY"
        }
        Write-Host "      UI:   http://127.0.0.1:$Port/ui"
        Write-Host "      Docs: http://127.0.0.1:$Port/docs"
        return
    }

    if (Test-PortInUse $Port) {
        Write-WarnMsg "Port $Port is in use, but not by this script-managed PID."
    } else {
        Write-Info "Server is not running."
    }
}

function Show-Logs {
    if (-not (Test-Path $LogFile)) {
        Write-Info "No log file yet: $LogFile"
        return
    }
    Get-Content $LogFile -Tail 120
}

switch ($Command) {
    "start"   { Start-Server }
    "stop"    { Stop-Server }
    "restart" { Stop-Server; Start-Server }
    "status"  { Show-Status }
    "logs"    { Show-Logs }
    default {
        Write-ErrMsg "Unknown command: $Command"
        exit 1
    }
}
