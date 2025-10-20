# 🚀 Quick Start Guide - EV Battery Digital Twin

## ⚡ Fast Setup (5 Minutes)

### Prerequisites Check
- ✅ Docker Desktop installed and running
- ✅ Python 3.12+ installed
- ✅ 8GB RAM available

### Step-by-Step Instructions

#### 1️⃣ Install Dependencies (2 minutes)

```powershell
# Navigate to project folder
cd NEW_EV_BATTERY

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install Python packages
pip install -r requirements.txt
```

#### 2️⃣ Start Infrastructure (1 minute)

```powershell
# Start all Docker services
docker-compose up -d

# Wait 30 seconds for services to initialize
timeout 30
```

#### 3️⃣ Train ML Models (2 minutes)

```powershell
# Train XGBoost models
python src/models/train.py
```

**✅ Expected:** Models saved to `models/` folder with R² > 0.90

---

## 🎮 Running the Application

### Option A: Automated Startup (Recommended)

```powershell
# Run the startup script
.\start.ps1
```

This will:
- ✅ Check Docker is running
- ✅ Start all services
- ✅ Set up Python environment
- ✅ Optionally train models

### Option B: Manual Startup

**Terminal 1 - Simulator:**
```powershell
.\.venv\Scripts\Activate.ps1
python src/simulator/publisher.py
```

**Terminal 2 - Predictor:**
```powershell
.\.venv\Scripts\Activate.ps1
python src/inference/live_predictor.py
```

**Terminal 3 - API (Optional):**
```powershell
.\.venv\Scripts\Activate.ps1
python src/api/app.py
```

---

## 🌐 Access the Dashboards

Open these URLs in your browser:

| Service | URL | Login |
|---------|-----|-------|
| 📊 **Grafana** | http://localhost:3000 | admin / admin |
| 🧪 **MLflow** | http://localhost:5000 | - |
| 📈 **Prometheus** | http://localhost:9090 | - |
| 💾 **MinIO** | http://localhost:9001 | minioadmin / minioadmin |
| 🔌 **API** | http://localhost:5001 | - |
| 🎨 **3D Viz** | `battery_3d_viz.html` | - |

---

## 📊 Grafana Dashboard Setup

1. Go to http://localhost:3000
2. Login: `admin` / `admin` (change password if prompted)
3. Click **+** → **Import Dashboard**
4. Click **Upload JSON file**
5. Select `monitoring/grafana_dashboard.json`
6. Click **Import**

**✅ You should see 8 panels with live battery data!**

---

## 🧪 Test the System

### Check Live Data
```powershell
# Get latest battery status
curl http://localhost:5001/api/battery/latest

# View in database
docker exec -it ev_timescaledb psql -U twin -d twin_data -c "SELECT * FROM latest_battery_status;"
```

### Verify Predictions
```powershell
# Check predictor metrics
curl http://localhost:9100/metrics

# View in Prometheus
# Go to http://localhost:9090
# Query: battery_rul_cycles
```

---

## 📱 What You Should See

### In Grafana:
- 📈 SoC (State of Charge) oscillating between 20-100%
- 📉 SoH (State of Health) slowly degrading
- 🌡️ Temperature fluctuating 25-45°C
- 🔮 RUL predictions updating every 5 seconds
- ⚠️ Failure probability showing risk level

### In Simulator Terminal:
```
📊 SoC: 85.3% | SoH: 94.8% | Temp: 34.2°C | Current: 75.3A | Cycles: 152 | 🔋 Discharging
📊 SoC: 84.8% | SoH: 94.8% | Temp: 35.1°C | Current: 82.1A | Cycles: 152 | 🔋 Discharging
```

### In Predictor Terminal:
```
🔮 Predictions - SoC: 84.8% | SoH: 94.8% | Temp: 35.1°C | RUL: 848 cycles | Failure Risk: 8.3%
```

---

## 🛑 Stop the System

```powershell
# Stop Python services
# Press Ctrl+C in each terminal

# Stop Docker services
docker-compose down
```

**Keep data:**
```powershell
docker-compose down
```

**Clean everything (including data):**
```powershell
docker-compose down -v
```

---

## ⚠️ Troubleshooting

### "Port already in use"
```powershell
# Check what's using the port
netstat -ano | findstr :5432

# Change port in docker-compose.yml or stop conflicting service
```

### "Docker not running"
```powershell
# Start Docker Desktop
# Wait for it to fully start (whale icon in taskbar)
```

### "Models not found"
```powershell
# Re-train models
python src/models/train.py
```

### "No data in Grafana"
```powershell
# Ensure simulator is running
# Check database connection in Grafana settings
# Verify TimescaleDB is running: docker ps
```

---

## 💡 Usage Tips

1. **Keep 3 terminals open**: Simulator, Predictor, and one for commands
2. **Monitor Grafana**: Set to auto-refresh every 5 seconds
3. **Watch the logs**: Helpful for debugging
4. **Use the API**: Access data programmatically
5. **3D Visualization**: Open in browser for cool visuals!

---

## 🎯 What to Monitor

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| SoC | > 50% | 20-50% | < 20% |
| SoH | > 85% | 75-85% | < 75% |
| Temp | 20-40°C | 40-60°C | > 60°C |
| RUL | > 500 | 100-500 | < 100 |
| Failure Risk | < 30% | 30-60% | > 60% |

---

## 🎓 Next Steps

- 📚 Read full `README.md` for detailed documentation
- 🧪 Run tests: `pytest tests/ -v`
- 🔧 Customize `.env` for your settings
- 📊 Create custom Grafana dashboards
- 🚀 Deploy to production (see README)

---

## 🆘 Need Help?

- Check `README.md` for comprehensive guide
- Review logs: `docker-compose logs -f`
- Test connectivity: `docker ps` and `curl` endpoints

---

**⚡ Enjoy your EV Battery Digital Twin! ⚡**
