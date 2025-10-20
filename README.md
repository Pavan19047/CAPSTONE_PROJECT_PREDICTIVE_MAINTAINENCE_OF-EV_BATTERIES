# ⚡ EV Battery Digital Twin System

A production-grade Electric Vehicle Battery Digital Twin with real-time telemetry simulation, AI-powered predictive maintenance, time-series data storage, and comprehensive monitoring.

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 🎯 Features

- **Real-time Telemetry Simulation**: Realistic EV battery behavior with charging/discharging cycles
- **Advanced ML Monitoring**: Actual vs Predicted visualizations with 4 trained models (SoH, Temperature, RUL, Failure Probability)
- **AI/ML Predictions**: Scikit-learn RandomForest models for real-time predictions
- **Time-Series Database**: TimescaleDB (PostgreSQL) for efficient telemetry storage
- **Message Streaming**: Kafka (Redpanda) and MQTT support for IoT integration
- **Live Monitoring**: Grafana dashboards with 11 comprehensive panels
- **Metrics Collection**: Prometheus for system and business metrics with 15+ custom metrics
- **Continuous Learning**: Automatic model retraining with historical data
- **ML Ops**: MLflow for experiment tracking and model registry
- **REST API**: Flask-based API with health checks and comparison endpoints
- **3D Visualization**: Interactive battery visualization with Three.js
- **Docker Integration**: Fully containerized Grafana and Prometheus services

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EV Battery Digital Twin                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐      ┌──────────────┐      ┌──────────────┐   │
│  │  Simulator  │─────▶│ TimescaleDB  │◀─────│  Predictor   │   │
│  │  (Python)   │      │  (Postgres)  │      │   (XGBoost)  │   │
│  └──────┬──────┘      └──────────────┘      └──────┬───────┘   │
│         │                                            │           │
│         ├──────────┐                    ┌────────────┤           │
│         ▼          ▼                    ▼            ▼           │
│  ┌──────────┐  ┌──────────┐    ┌──────────────┐  ┌──────────┐ │
│  │  Kafka   │  │   MQTT   │    │  Prometheus  │  │  MLflow  │ │
│  │(Redpanda)│  │(Mosquitto)│    │   (Metrics)  │  │ (Models) │ │
│  └──────────┘  └──────────┘    └──────┬───────┘  └──────────┘ │
│                                        │                         │
│                                        ▼                         │
│                                 ┌──────────────┐                │
│                                 │   Grafana    │                │
│                                 │ (Dashboards) │                │
│                                 └──────────────┘                │
│                                                                  │
│  ┌──────────────┐           ┌─────────────────┐                │
│  │  REST API    │           │  3D Viz (Web)   │                │
│  │   (Flask)    │           │   (Three.js)    │                │
│  └──────────────┘           └─────────────────┘                │
│                                                                  │
└──────────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

- **Docker Desktop** (with Docker Compose)
- **Python 3.12+**
- **Git**
- **8GB RAM minimum** (16GB recommended)
- **10GB free disk space**

## 🚀 Quick Start Guide

### Step 1: Clone and Setup

```bash
# Clone the repository (or navigate to your project folder)
cd NEW_EV_BATTERY

# Create Python virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
# source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### Step 1.5: Train ML Models (New!)

```bash
# Train CPU-based RandomForest models for predictions
python train_simple_models.py
```

**Expected Output:**
- 4 models trained: SoH, Battery_Temperature, RUL, Failure_Probability
- Models saved to `models/` directory
- High accuracy (R² > 0.99) for SoH and Temperature predictions

### Step 2: Prepare Dataset

The dataset `EV_Predictive_Maintenance_Dataset_15min.csv` is already in your workspace. Ensure it's in the `datasets` folder:

```bash
# Move dataset to datasets folder if needed
# (It's likely already there)
```

### Step 3: Start Infrastructure Services

```bash
# Start all Docker services
docker-compose up -d

# Wait for services to be ready (30-60 seconds)
# Check service status
docker-compose ps
```

**Expected Services:**
- **Grafana** (port 3000) - Dashboards and visualization
- **Prometheus** (port 9090) - Metrics collection and storage
- TimescaleDB (port 5432)
- Redpanda/Kafka (port 9092)
- MinIO (ports 9000, 9001)
- MLflow (port 5000)
- Mosquitto/MQTT (port 1883)

**Important:** Ensure Docker containers `ev_grafana` and `ev_prometheus` are running:
```bash
docker ps --filter "name=ev_grafana" --filter "name=ev_prometheus"
```

### Step 4: Start Advanced Battery Digital Twin

Open a terminal window (keep this running):

```bash
# Activate virtual environment
.venv\Scripts\activate

# Start the advanced application with ML predictions
python app_advanced.py
```

**What it does:**
- Simulates realistic battery behavior (500ms updates)
- Loads 4 trained ML models for real-time predictions
- Exposes Prometheus metrics on port 9091
- Provides REST API on port 5002
- Tracks prediction accuracy in real-time
- Supports continuous learning with historical data

**Exposed Services:**
- **Web UI**: http://localhost:5002
- **Prometheus Metrics**: http://localhost:9091/metrics
- **REST API Endpoints**:
  - GET /api/battery/status
  - GET /api/battery/detailed
  - GET /api/battery/comparison (actual vs predicted)
  - GET /api/battery/history
  - GET /api/health

### Step 5: Configure Grafana Datasource

1. Open Grafana: http://localhost:3000 (login: admin/admin)
2. Go to **Connections** → **Data sources** → **Add new data source**
3. Select **Prometheus**
4. Configure:
   - **Name**: `prometheus` (lowercase, important!)
   - **URL**: `http://ev_prometheus:9090`
   - **Access**: Server (default)
5. Click **Save & Test** - should show "Successfully queried the Prometheus API"

### Step 6: Import Grafana Dashboard

1. In Grafana, go to **Dashboards** → **Import**
2. Click **Upload JSON file**
3. Select `grafana_ml_dashboard.json` from the project root
4. **Important**: Choose **"prometheus"** (lowercase) as the datasource
5. Click **Import**

**Dashboard Features:**
- 11 comprehensive panels
- Actual vs Predicted comparisons for SoH, Temperature, RUL
- Real-time gauges for SoC and SoH
- Electrical parameters (Voltage, Current, Power)
- Failure probability gauge
- ML prediction accuracy tracking
- Auto-refresh every 5 seconds

## 🎛️ Access Dashboards

Once everything is running, access the web interfaces:

| Service | URL | Credentials |
|---------|-----|-------------|
| **Grafana Dashboards** | http://localhost:3000 | admin / admin |
| **Battery Web UI** | http://localhost:5002 | (no auth) |
| **Prometheus Metrics** | http://localhost:9090 | (no auth) |
| **Prometheus Targets** | http://localhost:9090/targets | (no auth) |
| **Raw Metrics Endpoint** | http://localhost:9091/metrics | (no auth) |
| **MLflow Tracking** | http://localhost:5000 | (no auth) |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin |
| **3D Visualization** | Open `battery_3d_viz.html` in browser | (no auth) |

## 📊 Grafana ML Dashboard

### Dashboard Panels

The `grafana_ml_dashboard.json` includes 11 comprehensive panels:

1. **State of Health - Actual vs Predicted**
   - Time series comparing actual SoH with ML predictions
   - Dashed line for predicted values
   - Shows model accuracy visually

2. **Current SoH Gauge**
   - Real-time SoH percentage
   - Color-coded: Green (>85%), Orange (70-85%), Red (<70%)

3. **Current SoC Gauge**
   - Real-time State of Charge
   - Color-coded: Green (>50%), Orange (20-50%), Red (<20%)

4. **Battery Temperature - Actual vs Predicted**
   - Temperature trends with predictions
   - Threshold markers at 40°C and 60°C

5. **Remaining Useful Life - Actual vs Predicted**
   - RUL in cycles
   - Prediction accuracy visualization

6. **Electrical Parameters**
   - Voltage, Current, and Power on single chart
   - Multi-series time series

7. **Failure Probability Gauge**
   - Risk assessment (0-100%)
   - Color-coded: Green (<30%), Yellow (30-60%), Red (>60%)

8. **ML Model Prediction Accuracy**
   - Accuracy percentage for each model
   - RUL, SoH, Temperature, Failure Probability
   - Historical accuracy trends

9. **Total Predictions Made**
   - Counter of all predictions
   - Stat panel with trend indicator

10. **Model Retraining Events**
    - Continuous learning tracking
    - Shows when models are updated

11. **Additional Metrics**
    - System health indicators
    - Performance metrics

### Prometheus Metrics Exposed

The `app_advanced.py` exposes 15+ custom metrics:

**Battery Metrics:**
- `battery_soc_actual` - State of Charge (%)
- `battery_soh_actual` - State of Health (%)
- `battery_soh_predicted` - ML predicted SoH
- `battery_temp_actual` - Battery temperature (°C)
- `battery_temp_predicted` - ML predicted temperature
- `battery_rul_actual` - Remaining cycles
- `battery_rul_predicted` - ML predicted RUL
- `battery_failure_actual` - Failure probability
- `battery_failure_predicted` - ML predicted failure probability
- `battery_voltage` - Voltage (V)
- `battery_current` - Current (A)
- `battery_power` - Power consumption (kW)

**ML Performance Metrics:**
- `prediction_accuracy` - Model accuracy by metric type
- `predictions_total` - Total predictions counter
- `model_training_total` - Retraining events counter

## 🔌 REST API Endpoints

### Advanced API (app_advanced.py - Port 5002)

#### Battery Data
- `GET /api/battery/status` - Current battery status with actual values
- `GET /api/battery/detailed` - Detailed battery information including all parameters
- `GET /api/battery/comparison` - **Actual vs Predicted comparison** (key endpoint!)
- `GET /api/battery/history` - Historical data with predictions

#### System
- `GET /api/health` - Health check with model status
- `GET /metrics` - Prometheus metrics endpoint (port 9091)

**Example Requests:**
```bash
# Get actual vs predicted comparison
curl http://localhost:5002/api/battery/comparison

# Get detailed battery status
curl http://localhost:5002/api/battery/detailed

# Check health
curl http://localhost:5002/api/health

# View Prometheus metrics
curl http://localhost:9091/metrics
```

**Example Response (Comparison):**
```json
{
  "battery_id": "battery_001",
  "timestamp": "2025-10-20T19:00:00",
  "comparisons": {
    "soh": {
      "actual": 98.5,
      "predicted": 0.985,
      "accuracy": 99.2,
      "difference": 0.3
    },
    "temperature": {
      "actual": 35.2,
      "predicted": 35.1,
      "accuracy": 99.7,
      "difference": 0.1
    }
  }
}
```

## 🧪 Testing

Run unit tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_simulator.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🔧 Configuration

### Application Configuration (app_advanced.py)

Edit constants at the top of `app_advanced.py`:

```python
# Simulation speed
SIMULATION_INTERVAL = 0.5  # seconds (500ms updates)

# Model paths
MODELS_DIR = Path('models')

# Prometheus metrics
METRICS_PORT = 9091  # Exposed on all interfaces (0.0.0.0)

# Flask API
API_PORT = 5002
```

### Prometheus Configuration (prometheus.yml)

```yaml
scrape_configs:
  - job_name: 'battery_digital_twin'
    scrape_interval: 5s
    static_configs:
      - targets: ['192.168.31.96:9091']  # Your host IP
        labels:
          instance: 'battery_001'
          environment: 'development'
```

### Docker Network Configuration

Ensure containers are on the same network:
```bash
# Check network
docker network ls

# Connect Prometheus to Grafana network
docker network connect new_ev_battery_ev_network ev_prometheus
```

## 📈 Model Performance

The trained RandomForest models (CPU-based) achieve:

- **SoH Prediction:**
  - R² Score: 1.0000 (train & test)
  - Perfect predictions on dataset
  - Real-time inference: < 5ms

- **Battery Temperature Prediction:**
  - R² Score: 1.0000 (train & test)
  - Accurate thermal modeling
  - Real-time inference: < 5ms

- **RUL Prediction:**
  - R² Score: 0.0128 (train), -0.0008 (test)
  - Challenging due to high variance
  - Continuous learning improves over time

- **Failure Probability:**
  - R² Score: 0.0115 (train), -0.0007 (test)
  - Binary classification recommended for production
  - Threshold-based alerts work well

**Note:** Models are trained on normalized data (0-1 scale) from the dataset. Actual values in simulation use percentage scale (0-100) for better visualization.

## 🐛 Troubleshooting

### Grafana shows "No data"

**Problem:** Dashboard panels are empty

**Solutions:**
1. Check datasource name is lowercase `"prometheus"`:
   ```bash
   # Check via API
   curl http://admin:admin@localhost:3000/api/datasources
   ```
2. Verify Prometheus is scraping:
   ```bash
   # Check targets
   curl http://localhost:9090/api/v1/targets
   ```
3. Test metrics endpoint:
   ```bash
   curl http://localhost:9091/metrics | findstr battery_soh
   ```
4. Restart Prometheus container:
   ```bash
   docker restart ev_prometheus
   ```

### Docker Network Issues

**Problem:** Grafana can't reach Prometheus (`no such host`)

**Solution:**
```bash
# Check networks
docker inspect ev_grafana --format='{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}'
docker inspect ev_prometheus --format='{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}'

# Connect to same network
docker network connect new_ev_battery_ev_network ev_prometheus
```

### Models Not Loading

**Problem:** `No module named 'cuml'` warnings

**Solution:**
```bash
# Train CPU-based models
python train_simple_models.py

# Restart app
python app_advanced.py
```

### Port Conflicts

**Problem:** `Address already in use`

**Solution:**
```bash
# Find process using port
netstat -ano | findstr :9091

# Kill process (replace PID)
taskkill /PID <pid> /F

# Or change port in app_advanced.py
```

### Prometheus Not Scraping

**Problem:** Empty query results in Prometheus

**Solutions:**
1. Check Flask app is exposing metrics:
   ```bash
   curl http://localhost:9091/metrics
   ```
2. Verify Prometheus config:
   ```bash
   docker exec ev_prometheus cat /etc/prometheus/prometheus.yml
   ```
3. Check if target is reachable from container:
   ```bash
   docker exec ev_prometheus wget -q -O - http://192.168.31.96:9091/metrics
   ```

### Services not starting
```bash
# Check Docker logs
docker-compose logs -f

# Restart specific service
docker-compose restart timescaledb
```

## 📁 Project Structure

```
NEW_EV_BATTERY/
├── app_advanced.py                    # ⭐ Main application with ML predictions
├── train_simple_models.py             # ⭐ Train CPU-based ML models
├── grafana_ml_dashboard.json          # ⭐ Grafana dashboard (11 panels)
├── prometheus.yml                     # ⭐ Prometheus scrape configuration
├── battery_digital_twin_advanced.html # Web UI with Chart.js
├── battery_3d_viz.html                # 3D visualization
├── src/
│   ├── models/
│   │   └── train.py                   # Original ML training
│   ├── simulator/
│   │   └── publisher.py               # Telemetry simulator
│   ├── inference/
│   │   └── live_predictor.py          # Real-time predictions
│   ├── api/
│   │   └── app.py                     # REST API
│   └── utils.py                       # Utility functions
├── config/
│   ├── prometheus.yml                 # Prometheus config
│   ├── mosquitto.conf                 # MQTT config
│   ├── grafana_datasources.yml        # Grafana datasources
│   └── grafana_dashboards.yml         # Grafana dashboard config
├── docker/
│   └── init_db.sql                    # Database schema
├── monitoring/
│   └── grafana_dashboard.json         # Original dashboard
├── models/                            # ⭐ Trained ML models (.joblib)
│   ├── .gitkeep                       # Preserve directory in git
│   ├── SoH.joblib                     # SoH prediction model
│   ├── Battery_Temperature.joblib     # Temperature model
│   ├── RUL.joblib                     # RUL prediction model
│   └── Failure_Probability.joblib     # Failure prediction
├── datasets/                          # ⭐ CSV datasets
│   ├── .gitkeep                       # Preserve directory in git
│   └── EV_Predictive_Maintenance_Dataset_15min.csv
├── tests/                             # Unit tests
├── docker-compose.yml                 # Docker services definition
├── requirements.txt                   # Python dependencies
├── .env                               # Environment variables
├── .gitignore                         # ⭐ Git ignore rules
├── ADVANCED_SETUP.md                  # Detailed setup guide
├── GRAFANA_QUICK_START.md             # Quick Grafana setup
└── README.md                          # This file

⭐ = Recently added or updated files
```

## 🔄 Complete Application Workflow

### Day-to-Day Usage

**1. Start the System (Daily)**
```bash
# Start infrastructure
docker-compose up -d

# Wait 30 seconds for services to be ready
Start-Sleep -Seconds 30

# Start advanced battery digital twin (single application!)
.venv\Scripts\activate
python app_advanced.py
```

**What Runs:**
- ✅ Battery simulation (500ms updates)
- ✅ ML predictions (4 models)
- ✅ Prometheus metrics (port 9091)
- ✅ REST API (port 5002)
- ✅ Web UI with Chart.js
- ✅ Continuous learning

**2. Monitor System**
- **Grafana ML Dashboard**: http://localhost:3000/d/ev_battery_ml
- **Web UI**: http://localhost:5002
- **Prometheus**: http://localhost:9090
- **Prometheus Targets**: http://localhost:9090/targets
- **3D Visualization**: Open `battery_3d_viz.html`

**3. Query Data**
```bash
# Get actual vs predicted comparison
curl http://localhost:5002/api/battery/comparison

# Get detailed status
curl http://localhost:5002/api/battery/detailed

# Check Prometheus metrics
curl http://localhost:9091/metrics | findstr battery_soh

# Query Prometheus API
curl "http://localhost:9090/api/v1/query?query=battery_soh_actual"
```

**4. Stop the System**
```bash
# Stop Python app (Ctrl+C in terminal)

# Stop Docker services (optional - keeps data)
docker-compose stop

# Stop and remove containers (full cleanup)
docker-compose down
```

### Weekly Maintenance

```bash
# Re-train models with new data
python train_simple_models.py

# Restart app to load new models
python app_advanced.py

# Check Grafana for prediction accuracy trends
# Monitor the "ML Model Prediction Accuracy" panel

# Backup database
docker exec ev_timescaledb pg_dump -U twin twin_data > backup_$(Get-Date -Format "yyyyMMdd").sql

# Clean up old Prometheus data (optional)
docker exec ev_prometheus rm -rf /prometheus/data/*
docker restart ev_prometheus
```

## 🎨 3D Visualization

Open `battery_3d_viz.html` in any modern web browser to see:
- Real-time 3D battery pack visualization
- Dynamic cell coloring based on charge level
- Live metrics display (SoC, SoH, Temperature, RUL)
- Charging/discharging animations
- Energy particle effects

## 📊 Key Metrics

### Battery Health Indicators
- **SoC (State of Charge)**: 0-100%
  - 🟢 Good: > 50%
  - 🟡 Warning: 20-50%
  - 🔴 Critical: < 20%

- **SoH (State of Health)**: 70-100%
  - 🟢 Good: > 85%
  - 🟡 Warning: 75-85%
  - 🔴 Critical: < 75%

- **Temperature**: 20-80°C
  - 🟢 Optimal: 20-40°C
  - 🟡 Warning: 40-60°C
  - 🔴 Critical: > 60°C

- **RUL (Remaining Useful Life)**: Predicted cycles remaining
  - 🟢 Good: > 500 cycles
  - 🟡 Warning: 100-500 cycles
  - 🔴 Critical: < 100 cycles

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Created for EV Battery Digital Twin demonstration and research purposes.

## 🙏 Acknowledgments

- XGBoost for ML framework
- TimescaleDB for time-series database
- Redpanda for Kafka-compatible streaming
- Grafana for visualization
- MLflow for ML Ops

## 🆕 Recent Updates (October 20, 2025)

### Major Features Added

1. **Advanced ML Monitoring System** (`app_advanced.py`)
   - Integrated 4 ML models for real-time predictions
   - Actual vs Predicted visualizations
   - Continuous learning capabilities
   - 15+ Prometheus metrics exposed

2. **Comprehensive Grafana Dashboard** (`grafana_ml_dashboard.json`)
   - 11 professional panels
   - Actual vs Predicted comparisons
   - Real-time gauges and time series
   - ML accuracy tracking

3. **Docker Integration**
   - Containerized Grafana and Prometheus
   - Proper network configuration
   - Persistent storage volumes

4. **CPU-Based Models** (`train_simple_models.py`)
   - RandomForest models (no GPU required)
   - Perfect accuracy for SoH and Temperature
   - Fast training and inference

5. **Enhanced API**
   - `/api/battery/comparison` endpoint
   - Health checks with model status
   - Historical data with predictions

### Breaking Changes

- Port changed from 5001 to 5002 for advanced app
- Prometheus metrics now on port 9091 (was 9100)
- Datasource name must be lowercase `"prometheus"`
- Models trained on 0-1 scale (dataset format)

### Bug Fixes

- Fixed Grafana-Prometheus DNS resolution
- Resolved Docker network connectivity
- Fixed datasource name mismatch
- Corrected Prometheus scraping configuration

---

**Last Updated:** October 20, 2025

For questions or issues, please open an issue on GitHub.

## 📚 Additional Resources

- **ADVANCED_SETUP.md** - Detailed setup instructions with troubleshooting
- **GRAFANA_QUICK_START.md** - Quick reference for Grafana configuration
- **grafana_ml_dashboard.json** - Dashboard configuration with 11 panels
- **prometheus.yml** - Prometheus scrape configuration
