param(
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $RootDir "backend"
$FrontendPath = Join-Path $RootDir "frontend\streamlit_app.py"
$ApiUrl = "http://localhost:8000/analyze"
$VenvPython = Join-Path $RootDir ".venv\Scripts\python.exe"
$PythonExe = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { "python" }
$BackendLog = Join-Path $RootDir "backend_startup.log"
$BackendErrorLog = Join-Path $RootDir "backend_startup_error.log"
$FrontendLog = Join-Path $RootDir "frontend_startup.log"
$FrontendErrorLog = Join-Path $RootDir "frontend_startup_error.log"

function Stop-AppProcesses {
    param (
        [System.Diagnostics.Process[]]$Processes
    )

    foreach ($Process in $Processes) {
        if ($null -ne $Process -and -not $Process.HasExited) {
            Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        }
    }
}

if (-not (Test-Path -LiteralPath $BackendDir)) {
    throw "Backend directory not found: $BackendDir"
}

if (-not (Test-Path -LiteralPath $FrontendPath)) {
    throw "Frontend file not found: $FrontendPath"
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    "Local virtual environment was not found. Creating it now..."
    python -m venv (Join-Path $RootDir ".venv")
    $PythonExe = $VenvPython
}

try {
    & $PythonExe --version | Out-Null
}
catch {
    throw "Python was not found. Install Python 3.11+ and make sure 'python' is available in PATH."
}

$RequiredModules = @("fastapi", "uvicorn", "streamlit", "sklearn", "pandas", "joblib")
$MissingModules = @()
foreach ($Module in $RequiredModules) {
    & $PythonExe -c "import $Module" 2>$null
    if ($LASTEXITCODE -ne 0) {
        $MissingModules += $Module
    }
}

if ($MissingModules.Count -gt 0) {
    "Missing Python dependencies: $($MissingModules -join ', ')"
    "Install them with:"
    "  install_dependencies.bat"
    ""
    throw "Cannot start app until dependencies are installed."
}

$env:MEDICAL_NLP_API_URL = $ApiUrl

$BackendProcess = $null
$FrontendProcess = $null

try {
    "Starting FastAPI backend..."
    $BackendProcess = Start-Process -FilePath $PythonExe -ArgumentList @("-m", "uvicorn", "app.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000") -WorkingDirectory $BackendDir -RedirectStandardOutput $BackendLog -RedirectStandardError $BackendErrorLog -PassThru

    "Waiting for backend health check..."
    $BackendReady = $false
    for ($Attempt = 1; $Attempt -le 30; $Attempt++) {
        try {
            $Response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get -TimeoutSec 2
            if ($Response.status -eq "ok") {
                $BackendReady = $true
                break
            }
        }
        catch {
            if ($BackendProcess.HasExited) {
                "Backend stopped during startup. Error log:"
                if (Test-Path -LiteralPath $BackendErrorLog) {
                    Get-Content -LiteralPath $BackendErrorLog -Tail 40
                }
                throw "Backend process stopped before becoming ready."
            }
            Start-Sleep -Seconds 1
        }
    }

    if (-not $BackendReady) {
        throw "Backend did not become ready on http://127.0.0.1:8000/health"
    }

    "Starting Streamlit frontend..."
    $FrontendArguments = "-m streamlit run `"$FrontendPath`" --server.port 8501"
    $FrontendProcess = Start-Process -FilePath $PythonExe -ArgumentList $FrontendArguments -WorkingDirectory $RootDir -RedirectStandardOutput $FrontendLog -RedirectStandardError $FrontendErrorLog -PassThru

    Start-Sleep -Seconds 3
    if ($FrontendProcess.HasExited) {
        "Frontend stopped during startup. Error log:"
        if (Test-Path -LiteralPath $FrontendErrorLog) {
            Get-Content -LiteralPath $FrontendErrorLog -Tail 60
        }
        throw "Frontend process stopped before becoming ready."
    }

    ""
    "Application is running."
    "Frontend: http://localhost:8501"
    "Backend API docs: http://localhost:8000/docs"
    "Backend analyze endpoint: $ApiUrl"
    ""
    "Press Ctrl+C in this terminal to stop both processes."

    while ($true) {
        Start-Sleep -Seconds 2
        if ($BackendProcess.HasExited) {
            throw "Backend process stopped unexpectedly."
        }
        if ($FrontendProcess.HasExited) {
            throw "Frontend process stopped unexpectedly."
        }
    }
}
catch {
    ""
    "Application failed to start:"
    $_.Exception.Message
    ""
    "Backend error log: $BackendErrorLog"
    if (Test-Path -LiteralPath $BackendErrorLog) {
        Get-Content -LiteralPath $BackendErrorLog -Tail 60
    }
    ""
    "Frontend error log: $FrontendErrorLog"
    if (Test-Path -LiteralPath $FrontendErrorLog) {
        Get-Content -LiteralPath $FrontendErrorLog -Tail 60
    }
    throw
}
finally {
    "Stopping application processes..."
    Stop-AppProcesses -Processes @($BackendProcess, $FrontendProcess)

    if (-not $NoPause) {
        ""
        "Press Enter to close this window."
        Read-Host | Out-Null
    }
}
