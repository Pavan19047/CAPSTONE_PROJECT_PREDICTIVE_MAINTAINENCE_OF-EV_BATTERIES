# 📖 Complete Step-by-Step Application Flow Guide

## 🎯 Overview

This guide walks you through the **complete workflow** of using the EV Battery Digital Twin system from start to finish, with detailed explanations of what happens at each step.

---

## 🚀 Phase 1: Initial Setup (One-Time)

### Step 1: Verify Prerequisites

**What to check:**
```powershell
# Check Python version (need 3.12+)
python --version

# Check Docker is installed
docker --version

# Check Docker Compose
docker-compose --version

# Check available memory (need 8GB+)
systeminfo | findstr /C:"Total Physical Memory"
```

**✅ Expected Output:**
- Python 3.12.x or higher
- Docker version 20.x or higher
- Docker Compose version 2.x or higher

---

### Step 2: Setup Python Environment

```powershell
# Navigate to project directory
cd c:\Users\pavan\OneDrive\Desktop\NEW_EV_BATTERY

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# OR for Command Prompt:
# .venv\Scripts\activate.bat

# Verify activation (should show (.venv) in prompt)
# Install all dependencies
pip install -r requirements.txt

# Verify installation
pip list
```

**What this does:**
- Creates isolated Python environment
- Installs 20+ packages (XGBoost, Flask, Kafka, MQTT, etc.)
- Prevents conflicts with system Python

**⏱️ Time:** ~2-3 minutes

---

### Step 3: Start Infrastructure Services

```powershell
# Start all Docker containers
docker-compose up -d

# Verify all services are running
docker-compose ps

# Check logs (optional)
docker-compose logs -f
```

**What starts:**
1. **TimescaleDB** (port 5432) - Database ready with schema
2. **Redpanda** (port 9092) - Kafka message broker
3. **MinIO** (ports 9000, 9001) - Object storage
4. **MLflow** (port 5000) - ML tracking server
5. **Mosquitto** (port 1883) - MQTT broker
6. **Prometheus** (port 9090) - Metrics collector
7. **Grafana** (port 3000) - Dashboard server

**✅ All should show "Up" status**

**⏱️ Time:** ~30-60 seconds for all services to be healthy

---

### Step 4: Train Machine Learning Models

```powershell
# Make sure virtual environment is activated
python src/models/train.py
```

**What happens internally:**

1. **Load Dataset** (`EV_Predictive_Maintenance_Dataset_15min.csv`)
   - Reads CSV with battery telemetry data
   - Checks for required columns

2. **Data Preprocessing**
   - Handles missing values (forward fill)
   - Removes outliers (IQR method)
   - Validates data integrity

3. **Feature Engineering**
   - Extracts 7 features: SoC, SoH, Voltage, Current, Temperature, Cycles, Power
   - Creates target variables: RUL, Failure Probability

4. **Train-Test Split** (80/20)
   - Splits data into training and testing sets
   - Ensures no data leakage

5. **Feature Scaling**
   - Applies StandardScaler
   - Normalizes features to mean=0, std=1
   - Saves scaler for later use

6. **Model Training**
   - **RUL Model**: XGBoost Regressor
     - 200 trees, max depth 8
     - Optimizes for RMSE
     - Target: R² > 0.90
   
   - **Failure Model**: XGBoost Classifier
     - 200 trees, max depth 6
     - Optimizes for AUC
     - Binary classification (fail/no fail)

7. **MLflow Logging**
   - Logs all parameters
   - Logs all metrics
   - Logs feature importance
   - Saves model artifacts to MinIO

8. **Model Saving**
   - Saves `rul_model.pkl`
   - Saves `failure_model.pkl`
   - Saves `scaler.pkl`
   - All saved to `models/` directory

**✅ Expected Output:**
```
✅ RUL Model - R² Score: 0.9234, RMSE: 42.15 cycles
✅ Failure Model - Accuracy: 0.9456, AUC: 0.9678
✅ Models saved to: models/
```

**⏱️ Time:** ~1-2 minutes

**🔍 View Results:**
- Open http://localhost:5000 (MLflow)
- See experiment tracking and model comparison

---

## 🎮 Phase 2: Running the System

### Step 5: Start Telemetry Simulator

**Open Terminal 1:**
```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Start simulator
python src/simulator/publisher.py
```

**What it does:**

**Initialization:**
1. Connects to TimescaleDB (port 5432)
2. Connects to Kafka/Redpanda (port 9092)
3. Connects to MQTT/Mosquitto (port 1883)
4. Creates BatterySimulator with initial state:
   - SoC: 85%
   - SoH: 95%
   - Temperature: 32°C
   - Cycles: 150

**Every 2 seconds:**
1. **Simulate Battery Physics**
   - If discharging: SoC decreases 0.2-0.8%
   - If charging: SoC increases 0.5-1.5%
   - Temperature = base + (current * 0.08) + noise
   - Voltage = 350 + (SoC/100 * 50) + noise
   - SoH degrades 0.01% per cycle

2. **Generate Telemetry JSON**
   ```json
   {
     "timestamp": "2025-10-20T10:30:45",
     "battery_id": "BATTERY_001",
     "soc": 84.3,
     "soh": 94.8,
     "voltage": 378.5,
     "current": 75.3,
     "temperature": 34.2,
     "charge_cycles": 152,
     "power_consumption": 38542.0,
     "is_charging": false
   }
   ```

3. **Publish to 3 Channels**
   - **Database**: INSERT into `ev_telemetry` table
   - **Kafka**: Send to `ev_battery_telemetry` topic
   - **MQTT**: Publish to `ev/battery/telemetry` topic

4. **Log to Console**
   ```
   📊 SoC: 84.3% | SoH: 94.8% | Temp: 34.2°C | Current: 75.3A | Cycles: 152 | 🔋 Discharging
   ```

**Behavioral Patterns:**
- **Discharging Phase** (SoC: 100% → 20%)
  - Current: +50 to +120A (positive = discharge)
  - Temperature rises with load
  - Duration: ~80 cycles (160 seconds)

- **Charging Phase** (SoC: 20% → 99%)
  - Current: -100 to -150A (negative = charge)
  - Temperature higher during fast charge
  - Duration: ~80 cycles (160 seconds)
  - Triggers when SoC < 20%
  - Completes when SoC > 99%
  - Increments cycle counter

**⚠️ Keep this terminal running!**

---

### Step 6: Start Live Predictor

**Open Terminal 2:**
```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Start predictor
python src/inference/live_predictor.py
```

**What it does:**

**Initialization:**
1. Load trained models from `models/`:
   - `rul_model.pkl`
   - `failure_model.pkl`
   - `scaler.pkl`
2. Connect to TimescaleDB
3. Start Prometheus metrics server (port 9100)

**Every 5 seconds:**

1. **Fetch Latest Telemetry**
   ```sql
   SELECT soc, soh, voltage, current, temperature, 
          charge_cycles, power_consumption
   FROM ev_telemetry
   WHERE battery_id = 'BATTERY_001'
   ORDER BY time DESC
   LIMIT 1
   ```

2. **Prepare Features**
   - Convert to DataFrame with 7 columns
   - Ensure correct column order
   - Scale using StandardScaler

3. **Make Predictions**
   - **RUL Prediction**:
     ```python
     rul = rul_model.predict(features_scaled)[0]
     # Returns: 850.5 (remaining cycles)
     ```
   
   - **Failure Probability**:
     ```python
     failure_prob = failure_model.predict_proba(features_scaled)[0][1]
     # Returns: 0.083 (8.3% risk)
     ```

4. **Update Database**
   ```sql
   UPDATE ev_telemetry
   SET rul_prediction = 850.5,
       failure_probability = 0.083,
       prediction_timestamp = NOW()
   WHERE battery_id = 'BATTERY_001'
   AND time = (SELECT MAX(time) FROM ev_telemetry)
   ```

5. **Update Prometheus Metrics**
   ```
   battery_soc_percent{battery_id="BATTERY_001"} 84.3
   battery_soh_percent{battery_id="BATTERY_001"} 94.8
   battery_temperature_celsius{battery_id="BATTERY_001"} 34.2
   battery_rul_cycles{battery_id="BATTERY_001"} 850.5
   battery_failure_probability{battery_id="BATTERY_001"} 0.083
   battery_predictions_total 1234
   ```

6. **Log to Console**
   ```
   🔮 Predictions - SoC: 84.3% | SoH: 94.8% | Temp: 34.2°C | RUL: 851 cycles | Failure Risk: 8.3%
   ```

**Metrics Endpoint:**
- Available at: http://localhost:9100/metrics
- Format: Prometheus exposition format
- Scraped by Prometheus every 15 seconds

**⚠️ Keep this terminal running!**

---

### Step 7: Start REST API (Optional)

**Open Terminal 3:**
```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Start API server
python src/api/app.py
```

**What it provides:**

**API Server** (Flask on port 5001)

**Available Endpoints:**

1. **GET** `/health`
   - Health check
   - Returns service status

2. **GET** `/api/battery/latest?battery_id=BATTERY_001`
   - Latest telemetry + predictions
   - Returns JSON with all fields

3. **GET** `/api/battery/history?battery_id=BATTERY_001&hours=1`
   - Historical data for past N hours
   - Up to 1000 records

4. **GET** `/api/battery/predictions?battery_id=BATTERY_001`
   - Current RUL and failure predictions
   - Returns JSON with prediction details

5. **GET** `/api/battery/stats?battery_id=BATTERY_001&hours=24`
   - Aggregated statistics
   - Averages, min, max over time period

6. **GET** `/api/batteries/list`
   - List all battery IDs in system
   - Returns array of strings

7. **GET** `/api/alerts`
   - Current alerts/warnings
   - Batteries in WARNING or CRITICAL state

**Example Usage:**
```powershell
# Get latest data
curl http://localhost:5001/api/battery/latest

# Get predictions
curl http://localhost:5001/api/battery/predictions

# Get 24-hour stats
curl http://localhost:5001/api/battery/stats?hours=24
```

**⚠️ Optional but recommended for API access**

---

## 📊 Phase 3: Monitoring & Visualization

### Step 8: Access Grafana Dashboard

**Browser:** http://localhost:3000

**Login:**
- Username: `admin`
- Password: `admin`
- (Change password when prompted)

**Import Dashboard:**
1. Click **+** (Create) → **Import**
2. Click **Upload JSON file**
3. Select `monitoring/grafana_dashboard.json`
4. Choose **TimescaleDB** as datasource
5. Click **Import**

**What you'll see:**

**Panel 1: State of Charge (SoC)**
- Line chart showing charge level over time
- Oscillates between 20% and 100%
- Green when > 50%, yellow 20-50%, red < 20%

**Panel 2: State of Health (SoH)**
- Line chart showing battery health degradation
- Slowly decreases from 95% towards 70%
- Shows long-term battery aging

**Panel 3: Battery Temperature**
- Line chart with color thresholds
- Blue (< 20°C), Green (20-40°C), Yellow (40-60°C), Red (> 60°C)
- Fluctuates based on current draw

**Panel 4: Remaining Useful Life (RUL)**
- Gauge showing predicted cycles remaining
- Updates every 5 seconds
- Green (> 500), Yellow (100-500), Red (< 100)

**Panel 5: Failure Probability**
- Percentage gauge (0-100%)
- Green (< 30%), Yellow (30-60%), Red (> 60%)
- Real-time risk assessment

**Panel 6: Voltage & Current**
- Dual-axis time series
- Voltage (V) on left axis - blue line
- Current (A) on right axis - orange line
- Current negative when charging

**Panel 7: Power Consumption**
- Area chart showing power draw
- Calculated as Voltage × Current
- Higher during acceleration/charging

**Panel 8: Current Statistics**
- Stat panel showing latest values
- All key metrics in one view
- Updates every 5 seconds

**Auto-Refresh:** Set to 5 seconds in top-right dropdown

---

### Step 9: View MLflow Experiments

**Browser:** http://localhost:5000

**What to explore:**

1. **Experiments Page**
   - Shows "EV_Battery_Digital_Twin" experiment
   - Lists all training runs

2. **Run Details**
   - Click any run to see:
     - **Parameters**: n_estimators, max_depth, learning_rate
     - **Metrics**: R², RMSE, MAE, accuracy, AUC
     - **Tags**: Model type, date, user
     - **Artifacts**: Model files in MinIO

3. **Compare Runs**
   - Select multiple runs
   - Click "Compare"
   - See metric comparisons and charts

4. **Model Registry** (optional)
   - Register best models
   - Version tracking
   - Stage transitions (Staging → Production)

---

### Step 10: Check Prometheus Metrics

**Browser:** http://localhost:9090

**Query Examples:**

```promql
# Current State of Charge
battery_soc_percent

# RUL over time
battery_rul_cycles

# Temperature trend
battery_temperature_celsius

# Prediction rate
rate(battery_predictions_total[5m])

# Average SoC last hour
avg_over_time(battery_soc_percent[1h])
```

**What to verify:**
- ✅ All metrics show recent data
- ✅ Targets shows `battery_predictor` as UP
- ✅ Graphs display correctly

---

### Step 11: View 3D Visualization

**Open File:** `battery_3d_viz.html` in Chrome/Edge/Firefox

**What you'll see:**

**3D Scene:**
- Rotating battery pack with 18 cells
- Cells glow based on charge level (green = full, red = empty)
- Energy particles floating around
- Smooth animations

**HUD Display:**
- Real-time metrics (simulated in browser):
  - SoC with progress bar
  - SoH with progress bar
  - Temperature with progress bar
  - RUL in cycles
  - Failure risk percentage

**Status Indicators:**
- Pulsing "Live Simulation Active" at bottom
- Color-coded health borders:
  - Green: Normal
  - Yellow: Warning
  - Red: Critical

**Interaction:**
- Automatically rotates
- Shows charging/discharging status

**Note:** This is a standalone simulation for demo purposes. To connect it to real data, you'd need to add WebSocket connection to the API.

---

## 🔄 Phase 4: Understanding the Data Flow

### Complete Cycle (Every 2 Seconds)

```
1. Simulator Generates Telemetry
   └─→ Calculates new battery state
   └─→ Creates JSON payload

2. Publish to 3 Channels
   ├─→ TimescaleDB
   │   └─→ INSERT with auto-timestamp
   │   └─→ Trigger updates status field
   │
   ├─→ Kafka
   │   └─→ Send to ev_battery_telemetry topic
   │   └─→ Available for stream processing
   │
   └─→ MQTT
       └─→ Publish to ev/battery/telemetry
       └─→ IoT devices can subscribe

3. Every 5 Seconds - Predictor Runs
   ├─→ SELECT latest FROM TimescaleDB
   ├─→ Scale features
   ├─→ Predict RUL & Failure
   ├─→ UPDATE database with predictions
   └─→ Update Prometheus metrics

4. Every 15 Seconds - Prometheus Scrapes
   └─→ GET http://localhost:9100/metrics
   └─→ Store time-series data

5. Grafana Queries
   ├─→ Every 5s: Query TimescaleDB
   │   └─→ Get latest telemetry + predictions
   │
   └─→ Display on dashboard panels
```

---

## 🎬 Typical Session Timeline

**00:00** - Start Docker services
**00:30** - Services healthy, train models
**02:30** - Models trained, start simulator
**02:31** - Data flowing to database/Kafka/MQTT
**02:35** - Start predictor
**02:36** - Predictions begin, metrics exposed
**02:40** - Open Grafana, see live data
**02:45** - Monitor for 5-10 minutes

**Patterns you'll see:**

- **First 160 seconds**: Battery discharges 100% → 20%
- **160-320 seconds**: Battery charges 20% → 100%
- **Every cycle**: Charge counter increments, SoH decreases slightly
- **RUL**: Slowly decreases as cycles accumulate
- **Temperature**: Fluctuates 25-45°C based on load

---

## 🛑 Phase 5: Stopping the System

### Graceful Shutdown

**Terminal 1 (Simulator):**
```
Press Ctrl+C
```
- Closes database connection
- Disconnects from Kafka
- Disconnects from MQTT

**Terminal 2 (Predictor):**
```
Press Ctrl+C
```
- Stops metrics server
- Closes database connection

**Terminal 3 (API):**
```
Press Ctrl+C
```
- Stops Flask server

**Stop Docker Services:**
```powershell
# Stop but keep data
docker-compose stop

# OR stop and remove containers (keep volumes)
docker-compose down

# OR completely remove everything including data
docker-compose down -v
```

---

## 📊 What to Monitor

### Key Metrics

| Metric | Normal Range | Action Needed |
|--------|-------------|---------------|
| SoC | 20-100% | Alert if < 15% |
| SoH | 85-100% | Monitor if < 85% |
| Temperature | 25-40°C | Alert if > 60°C |
| RUL | 500-1000 | Maintenance if < 100 |
| Failure Risk | 0-15% | Alert if > 70% |

### System Health

```powershell
# Check all services
docker-compose ps

# View logs
docker-compose logs -f [service_name]

# Check database
docker exec -it ev_timescaledb psql -U twin -d twin_data -c "SELECT COUNT(*) FROM ev_telemetry;"

# Check Prometheus targets
# Open http://localhost:9090/targets
```

---

## 💡 Pro Tips

1. **Multiple Terminals**: Use tmux or Windows Terminal with split panes
2. **Log Monitoring**: Keep logs visible to catch issues early
3. **Auto-Refresh**: Set Grafana to auto-refresh every 5s
4. **Data Export**: Use API to export data for analysis
5. **Experiments**: Try different battery scenarios in simulator
6. **Alerts**: Set up Grafana alerts for critical conditions

---

## 🎓 Next Steps

- **Customize Simulation**: Edit simulator parameters
- **Add Batteries**: Extend to multi-battery fleet
- **Create Alerts**: Set up email/Slack notifications
- **Deploy**: Move to production Kubernetes cluster
- **Integrate**: Connect to real IoT devices
- **Analyze**: Export data for deeper analysis

---

**🎉 You're now ready to fully operate the EV Battery Digital Twin system!**
