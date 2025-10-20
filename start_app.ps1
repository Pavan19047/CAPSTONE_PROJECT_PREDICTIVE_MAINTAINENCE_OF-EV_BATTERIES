#!/usr/bin/env pwsh
# EV Battery Digital Twin - Quick Launcher

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  ⚡ EV BATTERY DIGITAL TWIN LAUNCHER ⚡  " -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
Write-Host "🔍 Checking Python installation..." -ForegroundColor White
$pythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonCmd = "python3"
} else {
    Write-Host "❌ Python not found! Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Found: $pythonCmd" -ForegroundColor Green

# Check Python version
$pythonVersion = & $pythonCmd --version 2>&1
Write-Host "   Version: $pythonVersion" -ForegroundColor Gray
Write-Host ""

# Check if models directory exists
Write-Host "🔍 Checking for trained models..." -ForegroundColor White
if (Test-Path "models") {
    $modelCount = (Get-ChildItem -Path "models" -Filter "*.joblib").Count
    if ($modelCount -gt 0) {
        Write-Host "✅ Found $modelCount trained models" -ForegroundColor Green
    } else {
        Write-Host "⚠️  No model files found in models/ directory" -ForegroundColor Yellow
        Write-Host "   The app will run but predictions may not work" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  Models directory not found" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "models" -Force | Out-Null
    Write-Host "   Created models/ directory" -ForegroundColor Gray
}
Write-Host ""

# Check if app.py exists
if (-not (Test-Path "app.py")) {
    Write-Host "❌ app.py not found in current directory!" -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host "📦 Checking dependencies..." -ForegroundColor White
Write-Host "   Installing required packages (if needed)..." -ForegroundColor Gray

$packages = @("flask", "flask-cors", "joblib", "numpy", "pandas", "scikit-learn", "xgboost", "python-dotenv")
$packagesStr = $packages -join " "

& $pythonCmd -m pip install --quiet $packagesStr 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dependencies ready" -ForegroundColor Green
} else {
    Write-Host "⚠️  Some packages may not have installed correctly" -ForegroundColor Yellow
}
Write-Host ""

# Start the application
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "🚀 STARTING APPLICATION..." -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Dashboard will be available at:" -ForegroundColor White
Write-Host "   👉 http://localhost:5001" -ForegroundColor Green
Write-Host ""
Write-Host "📡 API Endpoints:" -ForegroundColor White
Write-Host "   • GET /api/battery/status" -ForegroundColor Gray
Write-Host "   • GET /api/battery/detailed" -ForegroundColor Gray
Write-Host "   • GET /api/battery/predictions" -ForegroundColor Gray
Write-Host "   • GET /api/health" -ForegroundColor Gray
Write-Host ""
Write-Host "💡 Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Start-Sleep -Seconds 2

# Launch the app
& $pythonCmd app.py

# If app exits
Write-Host ""
Write-Host "👋 Application stopped" -ForegroundColor Yellow
