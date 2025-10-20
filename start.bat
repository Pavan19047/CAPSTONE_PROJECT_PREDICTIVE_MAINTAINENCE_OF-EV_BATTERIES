@echo off
echo ========================================
echo EV Battery Digital Twin - Quick Start
echo ========================================
echo.

REM Check if Docker is running
echo [1/4] Checking Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)
echo Docker is running!
echo.

REM Start Docker services
echo [2/4] Starting Docker services...
docker-compose up -d
if errorlevel 1 (
    echo ERROR: Failed to start Docker services
    pause
    exit /b 1
)
echo Docker services started!
echo.

REM Wait for services
echo [3/4] Waiting for services to be ready...
timeout /t 30 /nobreak >nul
echo Services ready!
echo.

REM Setup Python environment
echo [4/4] Setting up Python environment...
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
pip install -r requirements.txt --quiet
echo Python environment ready!
echo.

echo ========================================
echo SETUP COMPLETE!
echo ========================================
echo.
echo Access URLs:
echo   Grafana:     http://localhost:3000 (admin/admin)
echo   MLflow:      http://localhost:5000
echo   Prometheus:  http://localhost:9090
echo   MinIO:       http://localhost:9001
echo.
echo Next Steps:
echo   1. Train models:    python src/models/train.py
echo   2. Start simulator: python src/simulator/publisher.py
echo   3. Start predictor: python src/inference/live_predictor.py
echo.
echo Tip: Open 3 separate terminals for steps 2-3
echo.

set /p train="Train ML models now? (y/n): "
if /i "%train%"=="y" (
    echo.
    echo Training models...
    python src/models/train.py
)

echo.
echo Done! Follow the next steps above.
pause
