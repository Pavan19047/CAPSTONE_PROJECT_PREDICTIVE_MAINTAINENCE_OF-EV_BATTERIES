# EV Battery Digital Twin - Windows Startup Script
# Run this script to start all services

Write-Host "🚀 Starting EV Battery Digital Twin System..." -ForegroundColor Green
Write-Host ""

# Check if Docker is running
Write-Host "📦 Checking Docker..." -ForegroundColor Cyan
$dockerRunning = docker info 2>$null
if (-not $dockerRunning) {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}
Write-Host "✅ Docker is running" -ForegroundColor Green
Write-Host ""

# Start Docker services
Write-Host "🐳 Starting Docker services..." -ForegroundColor Cyan
docker-compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to start Docker services" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Docker services started" -ForegroundColor Green
Write-Host ""

# Wait for services to be ready
Write-Host "⏳ Waiting for services to be ready (30 seconds)..." -ForegroundColor Cyan
Start-Sleep -Seconds 30
Write-Host "✅ Services should be ready" -ForegroundColor Green
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "⚠️  Virtual environment not found. Creating..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment and check dependencies
Write-Host "📦 Checking Python dependencies..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt --quiet

Write-Host ""
Write-Host "=" -repeat 70 -ForegroundColor Green
Write-Host "✅ SYSTEM READY!" -ForegroundColor Green
Write-Host "=" -repeat 70 -ForegroundColor Green
Write-Host ""

Write-Host "🌐 Access URLs:" -ForegroundColor Cyan
Write-Host "  📊 Grafana:     http://localhost:3000 (admin/admin)" -ForegroundColor White
Write-Host "  🧪 MLflow:      http://localhost:5000" -ForegroundColor White
Write-Host "  📈 Prometheus:  http://localhost:9090" -ForegroundColor White
Write-Host "  💾 MinIO:       http://localhost:9001 (minioadmin/minioadmin)" -ForegroundColor White
Write-Host "  🔌 API:         http://localhost:5001 (when started)" -ForegroundColor White
Write-Host ""

Write-Host "📋 Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Train models:      python src/models/train.py" -ForegroundColor White
Write-Host "  2. Start simulator:   python src/simulator/publisher.py" -ForegroundColor White
Write-Host "  3. Start predictor:   python src/inference/live_predictor.py" -ForegroundColor White
Write-Host "  4. Start API:         python src/api/app.py (optional)" -ForegroundColor White
Write-Host ""

Write-Host "💡 Tip: Open 3 separate terminals and run steps 2-4 in each" -ForegroundColor Cyan
Write-Host ""

# Ask if user wants to train models now
$train = Read-Host "Would you like to train ML models now? (y/n)"
if ($train -eq 'y' -or $train -eq 'Y') {
    Write-Host ""
    Write-Host "🎯 Training ML models..." -ForegroundColor Cyan
    python src/models/train.py
}

Write-Host ""
Write-Host "✨ Setup complete! Follow the next steps above." -ForegroundColor Green
