# ⚡ EV Battery Digital Twin System

A production-grade Electric Vehicle Battery Digital Twin with real-time telemetry simulation, AI-powered predictive maintenance, time-series data storage, and comprehensive monitoring.

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 🎯 Features

- **Real-time Telemetry Simulation**: Realistic EV battery behavior with charging/discharging cycles
- **AI/ML Predictions**: XGBoost models for RUL (Remaining Useful Life) and failure probability
- **Time-Series Database**: TimescaleDB (PostgreSQL) for efficient telemetry storage
- **Message Streaming**: Kafka (Redpanda) and MQTT support for IoT integration
- **Live Monitoring**: Grafana dashboards with real-time visualizations
- **Metrics Collection**: Prometheus for system and business metrics
- **ML Ops**: MLflow for experiment tracking and model registry
- **REST API**: Flask-based API for programmatic access
- **3D Visualization**: Interactive battery visualization with Three.js

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
- TimescaleDB (port 5432)
- Redpanda/Kafka (port 9092)
- MinIO (ports 9000, 9001)
- MLflow (port 5000)
- Mosquitto/MQTT (port 1883)
- Prometheus (port 9090)
- Grafana (port 3000)

### Step 4: Train ML Models

```bash
# Train XGBoost models for RUL prediction and failure classification
python src/models/train.py
```

**Expected Output:**
- Models saved to `models/` directory
- Model performance: R² > 0.90 for RUL prediction
- MLflow experiments logged

### Step 5: Start Telemetry Simulator

Open a **new terminal window** (keep this running):

```bash
# Activate virtual environment
.venv\Scripts\activate

# Start simulator
python src/simulator/publisher.py
```

**What it does:**
- Simulates realistic battery charging/discharging cycles
- Publishes telemetry every 2 seconds to:
  - TimescaleDB (storage)
  - Kafka (streaming)
  - MQTT (IoT devices)

### Step 6: Start Live Predictor

Open **another new terminal window** (keep this running):

```bash
# Activate virtual environment
.venv\Scripts\activate

# Start predictor service
python src/inference/live_predictor.py
```

**What it does:**
- Makes ML predictions every 5 seconds
- Updates database with RUL and failure probability
- Exposes Prometheus metrics on port 9100

### Step 7: Start REST API (Optional)

Open **another new terminal window**:

```bash
# Activate virtual environment
.venv\Scripts\activate

# Start API server
python src/api/app.py
```

**API available at:** http://localhost:5001

## 🎛️ Access Dashboards

Once everything is running, access the web interfaces:

| Service | URL | Credentials |
|---------|-----|-------------|
| **Grafana Dashboards** | http://localhost:3000 | admin / admin |
| **MLflow Tracking** | http://localhost:5000 | (no auth) |
| **Prometheus Metrics** | http://localhost:9090 | (no auth) |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin |
| **3D Visualization** | Open `battery_3d_viz.html` in browser | (no auth) |
| **REST API** | http://localhost:5001 | (no auth) |

## 📊 Grafana Dashboard Setup

1. **Access Grafana:** http://localhost:3000 (admin/admin)
2. **Import Dashboard:**
   - Click "+" → "Import"
   - Upload `monitoring/grafana_dashboard.json`
   - Select "TimescaleDB" as datasource
3. **View Real-time Data:**
   - SoC (State of Charge)
   - SoH (State of Health)
   - Temperature trends
   - RUL predictions
   - Failure probability

## 🔌 REST API Endpoints

### Battery Data
- `GET /api/battery/latest?battery_id=BATTERY_001` - Latest telemetry
- `GET /api/battery/history?battery_id=BATTERY_001&hours=1` - Historical data
- `GET /api/battery/predictions?battery_id=BATTERY_001` - ML predictions
- `GET /api/battery/stats?battery_id=BATTERY_001&hours=24` - Statistics

### System
- `GET /health` - Health check
- `GET /api/batteries/list` - List all batteries
- `GET /api/alerts` - Current alerts

**Example Request:**
```bash
curl http://localhost:5001/api/battery/latest
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

Edit `.env` file to customize:

```env
# Database
DB_HOST=localhost
DB_PORT=5432

# Simulation intervals
TELEMETRY_INTERVAL=2.0  # seconds
PREDICTION_INTERVAL=5.0  # seconds

# Ports
METRICS_PORT=9100
API_PORT=5001
```

## 📈 Model Performance

The trained XGBoost models achieve:

- **RUL Prediction:**
  - R² Score: > 0.90
  - RMSE: < 50 cycles
  - MAE: < 30 cycles

- **Failure Classification:**
  - Accuracy: > 0.92
  - AUC-ROC: > 0.95
  - F1 Score: > 0.90

## 🐛 Troubleshooting

### Services not starting
```bash
# Check Docker logs
docker-compose logs -f

# Restart specific service
docker-compose restart timescaledb
```

### Database connection issues
```bash
# Verify TimescaleDB is running
docker exec -it ev_timescaledb psql -U twin -d twin_data -c "SELECT version();"
```

### Models not found
```bash
# Re-train models
python src/models/train.py
```

### Ports already in use
```bash
# Stop conflicting services or change ports in docker-compose.yml
netstat -ano | findstr :5432
```

## 📁 Project Structure

```
NEW_EV_BATTERY/
├── src/
│   ├── models/
│   │   └── train.py              # ML training pipeline
│   ├── simulator/
│   │   └── publisher.py          # Telemetry simulator
│   ├── inference/
│   │   └── live_predictor.py     # Real-time predictions
│   ├── api/
│   │   └── app.py                # REST API
│   └── utils.py                  # Utility functions
├── config/
│   ├── prometheus.yml            # Prometheus config
│   ├── mosquitto.conf            # MQTT config
│   ├── grafana_datasources.yml   # Grafana datasources
│   └── grafana_dashboards.yml    # Grafana dashboard config
├── docker/
│   └── init_db.sql               # Database schema
├── monitoring/
│   └── grafana_dashboard.json    # Grafana dashboard
├── models/                       # Trained ML models (generated)
├── datasets/                     # Dataset files
├── tests/                        # Unit tests
├── docker-compose.yml            # Docker services
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables
├── battery_3d_viz.html           # 3D visualization
└── README.md                     # This file
```

## 🔄 Complete Application Workflow

### Day-to-Day Usage

**1. Start the System (Daily)**
```bash
# Start infrastructure
docker-compose up -d

# Wait 30 seconds for services to be ready
timeout 30

# Start simulator (Terminal 1)
.venv\Scripts\activate
python src/simulator/publisher.py

# Start predictor (Terminal 2)
.venv\Scripts\activate
python src/inference/live_predictor.py

# Start API (Terminal 3 - Optional)
.venv\Scripts\activate
python src/api/app.py
```

**2. Monitor System**
- Open Grafana: http://localhost:3000
- Check Prometheus: http://localhost:9090
- View 3D visualization: Open `battery_3d_viz.html`

**3. Query Data**
```bash
# Using API
curl http://localhost:5001/api/battery/latest

# Using database
docker exec -it ev_timescaledb psql -U twin -d twin_data -c "SELECT * FROM latest_battery_status;"
```

**4. Stop the System**
```bash
# Stop Python services (Ctrl+C in each terminal)

# Stop Docker services
docker-compose down
```

### Weekly Maintenance

```bash
# Re-train models with new data
python src/models/train.py

# Check MLflow for model performance
# Open http://localhost:5000

# Backup database
docker exec ev_timescaledb pg_dump -U twin twin_data > backup_$(date +%Y%m%d).sql
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

---

**Last Updated:** October 20, 2025

For questions or issues, please open an issue on GitHub.
