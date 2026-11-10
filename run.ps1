# JACCS Credit Scoring System — PowerShell Convenience Runner
# Usage: .\run.ps1 [command]
# Commands: setup, run, test, build, clean, health, help

param(
    [string]$Command = "help"
)

$ProjectRoot = $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
$PipExe = Join-Path $VenvPath "Scripts\pip.exe"
$MainEntry = Join-Path $ProjectRoot "src\frontend\main.py"
$RequirementsFile = Join-Path $ProjectRoot "requirements.txt"
$DistDir = Join-Path $ProjectRoot "dist"

# Colors
function Write-Success { param([string]$Msg) Write-Host "✓ $Msg" -ForegroundColor Green }
function Write-Warn    { param([string]$Msg) Write-Host "⚠ $Msg" -ForegroundColor Yellow }
function Write-Fail    { param([string]$Msg) Write-Host "✗ $Msg" -ForegroundColor Red }
function Write-Info    { param([string]$Msg) Write-Host "  $Msg" -ForegroundColor Cyan }

function Show-Header {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════╗" -ForegroundColor DarkGreen
    Write-Host "║  JACCS Credit Scoring System  v2.0.0         ║" -ForegroundColor DarkGreen
    Write-Host "╚══════════════════════════════════════════════╝" -ForegroundColor DarkGreen
    Write-Host ""
}

function Assert-Venv {
    if (-not (Test-Path $PythonExe)) {
        Write-Fail "Virtual environment not found. Run: .\run.ps1 setup"
        exit 1
    }
}

# ─── Commands ──────────────────────────────────────────────────────────────

function Invoke-Setup {
    Show-Header
    Write-Host "Setting up project environment..." -ForegroundColor White

    # Check Python
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Python not found. Please install Python 3.12+ and add it to PATH."
        exit 1
    }
    Write-Success "Found $pythonVersion"

    # Create venv
    if (-not (Test-Path $VenvPath)) {
        Write-Info "Creating virtual environment..."
        python -m venv $VenvPath
        if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to create venv"; exit 1 }
        Write-Success "Created .venv"
    } else {
        Write-Success "Virtual environment already exists"
    }

    # Install dependencies
    Write-Info "Installing dependencies (this may take a few minutes)..."
    & $PipExe install --upgrade pip --quiet
    & $PipExe install -r $RequirementsFile
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Dependency installation failed. Check requirements.txt."
        exit 1
    }
    Write-Success "All dependencies installed"

    # Create log directory
    $LogDir = Join-Path $env:USERPROFILE ".jaccs_credit_app\logs"
    if (-not (Test-Path $LogDir)) {
        New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
        Write-Success "Created log directory: $LogDir"
    }

    Write-Host ""
    Write-Success "Setup complete! Run the app with: .\run.ps1 run"
}

function Invoke-Run {
    Assert-Venv
    Show-Header
    Write-Info "Starting JACCS Credit Scoring System..."
    Set-Location $ProjectRoot
    & $PythonExe $MainEntry
}

function Invoke-Test {
    Assert-Venv
    Show-Header
    Write-Host "Running test suite..." -ForegroundColor White

    $TestsDir = Join-Path $ProjectRoot "tests"
    $CoverageArgs = @(
        "-m", "pytest",
        $TestsDir,
        "-v",
        "--tb=short",
        "--no-header",
        "-q"
    )

    # Try with coverage if available
    $CoverageInstalled = & $PythonExe -c "import pytest_cov" 2>&1
    if ($LASTEXITCODE -eq 0) {
        $CoverageArgs += @("--cov=src", "--cov-report=term-missing", "--cov-report=html:htmlcov")
        Write-Info "Running with coverage report..."
    } else {
        Write-Warn "pytest-cov not installed. Run: pip install pytest-cov for coverage reports."
    }

    Set-Location $ProjectRoot
    & $PythonExe @CoverageArgs

    if ($LASTEXITCODE -eq 0) {
        Write-Success "All tests passed!"
        if ($LASTEXITCODE -eq 0 -and (Test-Path "htmlcov\index.html")) {
            Write-Info "Coverage report: $ProjectRoot\htmlcov\index.html"
        }
    } else {
        Write-Fail "Some tests failed."
        exit 1
    }
}

function Invoke-Build {
    Assert-Venv
    Show-Header
    Write-Host "Building executable with PyInstaller..." -ForegroundColor White

    # Find model joblib file
    $ModelsDir = Join-Path $ProjectRoot "src\backend\models"
    $ModelFile = Get-ChildItem -Path $ModelsDir -Filter "*.joblib" -ErrorAction SilentlyContinue | Select-Object -First 1

    if (-not $ModelFile) {
        Write-Warn "No cached model found in src\backend\models\"
        Write-Warn "The built executable will train the model on first launch (may take 5-10 minutes)"
    }

    $IconFile = Join-Path $ProjectRoot "ltid_logo(1).ico"
    $BuildArgs = @(
        "--noconfirm",
        "--onefile",
        "--windowed",
        "--name", "JACCSScore",
        "--add-data", "$ProjectRoot\src;src/",
        "--add-data", "$ProjectRoot\src\data\img;img"
    )

    if (Test-Path $IconFile) {
        $BuildArgs += @("--icon", $IconFile)
    }

    if ($ModelFile) {
        $BuildArgs += @("--add-data", "$($ModelFile.FullName);src/backend/models/")
    }

    $BuildArgs += $MainEntry

    Write-Info "Running: pyinstaller $($BuildArgs -join ' ')"
    & $PipExe show pyinstaller | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Info "Installing PyInstaller..."
        & $PipExe install pyinstaller
    }

    Set-Location $ProjectRoot
    & $PythonExe -m PyInstaller @BuildArgs

    if ($LASTEXITCODE -eq 0) {
        $ExePath = Join-Path $DistDir "JACCSScore.exe"
        Write-Success "Build successful!"
        Write-Info "Executable: $ExePath"
    } else {
        Write-Fail "Build failed. Check output above for errors."
        exit 1
    }
}

function Invoke-Clean {
    Show-Header
    Write-Host "Cleaning temporary files..." -ForegroundColor White

    # Remove __pycache__ directories
    Get-ChildItem -Recurse -Path $ProjectRoot -Directory -Filter "__pycache__" |
        Where-Object { $_.FullName -notlike "*\.venv*" } |
        ForEach-Object {
            Remove-Item $_.FullName -Recurse -Force
            Write-Info "Removed: $($_.FullName)"
        }

    # Remove .pyc files
    Get-ChildItem -Recurse -Path $ProjectRoot -Filter "*.pyc" |
        Where-Object { $_.FullName -notlike "*\.venv*" } |
        ForEach-Object {
            Remove-Item $_.FullName -Force
        }

    # Remove build artifacts
    @("build", "dist", "*.spec.bak", ".pytest_cache", "htmlcov", ".coverage") |
        ForEach-Object {
            $item = Join-Path $ProjectRoot $_
            if (Test-Path $item) {
                Remove-Item $item -Recurse -Force
                Write-Info "Removed: $item"
            }
        }

    Write-Success "Clean complete!"
}

function Invoke-Health {
    Assert-Venv
    Show-Header
    Write-Host "Running system health check..." -ForegroundColor White
    Set-Location $ProjectRoot
    & $PythonExe -c "
import sys
sys.path.insert(0, '.')
from src.backend.health import health_check
status = health_check(check_db=False)
print(status.summary())
sys.exit(0 if status.is_healthy else 1)
"
    if ($LASTEXITCODE -eq 0) {
        Write-Success "System is healthy!"
    } else {
        Write-Warn "System is degraded. See details above."
    }
}

function Show-Help {
    Show-Header
    Write-Host "Available commands:" -ForegroundColor White
    Write-Host ""
    Write-Host "  .\run.ps1 setup     " -NoNewline; Write-Host "Create venv and install dependencies" -ForegroundColor Gray
    Write-Host "  .\run.ps1 run       " -NoNewline; Write-Host "Launch the Credit Scoring application" -ForegroundColor Gray
    Write-Host "  .\run.ps1 test      " -NoNewline; Write-Host "Run the pytest test suite" -ForegroundColor Gray
    Write-Host "  .\run.ps1 build     " -NoNewline; Write-Host "Build a standalone .exe with PyInstaller" -ForegroundColor Gray
    Write-Host "  .\run.ps1 clean     " -NoNewline; Write-Host "Remove __pycache__, .pyc files, and build artifacts" -ForegroundColor Gray
    Write-Host "  .\run.ps1 health    " -NoNewline; Write-Host "Run a system health check" -ForegroundColor Gray
    Write-Host "  .\run.ps1 help      " -NoNewline; Write-Host "Show this help message" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Environment variables:" -ForegroundColor White
    Write-Host "  JACCS_DEFAULT_ADMIN_PASSWORD  First-run admin password (default: auto-generated)"
    Write-Host "  SQL_SERVER                    Database server override"
    Write-Host "  JACCS_LOG_LEVEL               Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)"
    Write-Host ""
}

# ─── Dispatch ──────────────────────────────────────────────────────────────

switch ($Command.ToLower()) {
    "setup"   { Invoke-Setup }
    "run"     { Invoke-Run }
    "test"    { Invoke-Test }
    "build"   { Invoke-Build }
    "clean"   { Invoke-Clean }
    "health"  { Invoke-Health }
    "help"    { Show-Help }
    default   {
        Write-Fail "Unknown command: '$Command'"
        Show-Help
        exit 1
    }
}

