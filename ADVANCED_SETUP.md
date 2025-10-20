# 🚀 Advanced ML Integration Guide
## EV Battery Digital Twin with Actual vs Predicted Visualizations

---

## 🎯 What's New

### ✨ Advanced Features:
1. **Actual vs Predicted Comparisons** - See ML predictions alongside real values
2. **Live Charts** - Real-time graphs showing trends
3. **Prometheus Metrics** - Professional monitoring infrastructure
4. **Grafana Dashboards** - Beautiful, industry-standard visualizations
5. **Continuous Learning** - Models retrain automatically from live data
6. **Prediction Accuracy Tracking** - Monitor ML model performance

---

## 🚀 Quick Start

### Step 1: Start the Advanced Application

```powershell
# Run the advanced server
python app_advanced.py
```

This will:
- Load your 28 joblib models
- Start web server on port **5002**
- Start Prometheus metrics on port **9091**
- Begin battery simulation with ML predictions
- Enable continuous learning

### Step 2: Access the Dashboard

Open your browser to:
```
http://localhost:5002
```

You'll see:
- 📊 **Actual vs Predicted comparisons** for SoH, RUL, Temperature, Failure
- 📈 **Live charts** with both actual and predicted trends
- ⚡ **Real-time metrics** updating every second
- 🤖 **ML system status** with model info

---

## 📊 Grafana Setup (Professional Monitoring)

### Step 1: Install Grafana (if needed)

**Windows:**
```powershell
# Using Chocolatey
choco install grafana

# Or download from: https://grafana.com/grafana/download
```

**Alternative: Use Docker**
```powershell
docker run -d -p 3000:3000 --name=grafana grafana/grafana
```

### Step 2: Configure Prometheus Datasource

**⚠️ IMPORTANT: Start `app_advanced.py` first! The standalone Prometheus metrics server runs on port 9091.**

**For Docker Grafana (recommended):**

1. Open Grafana: http://localhost:3000
2. Default login: `admin` / `admin`
3. Go to **Configuration → Data Sources → Add data source**
4. Select **Prometheus**
5. Set URL to: `http://192.168.31.96:9091` *(use your actual Windows IP address)*
   - To find your IP: Run `ipconfig` and look for "IPv4 Address"
   - Replace `192.168.31.96` with your actual IP
6. Click **Save & Test** ✅

**Alternative - For Windows-installed Grafana (not Docker):**

1. Set URL to: `http://localhost:9091`
2. Click **Save & Test**

**Why use IP instead of host.docker.internal?**
The Prometheus metrics server on port 9091 binds to localhost by default. Docker containers can't reach localhost:9091, but they CAN reach your Windows machine's IP address on the local network.

### Step 3: Import Dashboard

1. Go to **Dashboards → Import**
2. Click **Upload JSON file**
3. Select: `grafana_ml_dashboard.json`
4. Choose Prometheus datasource
5. Click **Import**

### 🎉 You now have a professional Grafana dashboard!

---

## 📡 Available Endpoints

### Web Application (Port 5002):
| Endpoint | Description |
|----------|-------------|
| `GET /` | Advanced ML dashboard |
| `GET /api/battery/status` | Simple status |
| `GET /api/battery/detailed` | Full battery data |
| `GET /api/battery/comparison` | Actual vs Predicted comparison |
| `GET /api/battery/history` | Historical data for charts |
| `GET /api/health` | System health check |

### Prometheus Metrics (Port 9091):
| Endpoint | Description |
|----------|-------------|
| `GET /metrics` | All Prometheus metrics |

---

## 🔍 Understanding the Metrics

### Actual Metrics (Ground Truth):
- `battery_soc_actual` - Real State of Charge
- `battery_soh_actual` - Real State of Health
- `battery_temperature_actual` - Real temperature
- `battery_rul_actual` - Real Remaining Useful Life
- `battery_failure_actual` - Real failure probability

### Predicted Metrics (ML Models):
- `battery_soh_predicted` - ML predicted SoH
- `battery_temperature_predicted` - ML predicted temperature
- `battery_rul_predicted` - ML predicted RUL
- `battery_failure_predicted` - ML predicted failure risk

### Performance Metrics:
- `prediction_accuracy{metric="SoH"}` - SoH prediction accuracy %
- `prediction_error{metric="SoH"}` - SoH prediction error
- `predictions_total` - Total predictions made
- `model_training_total` - Number of retraining events

---

## 🤖 Continuous Learning

### How It Works:

1. **Data Collection**: Every 500ms, the system:
   - Simulates actual battery behavior
   - Makes ML predictions
   - Stores both in history buffer

2. **Model Retraining**: Every 5 minutes (when 100+ samples):
   - Extracts recent historical data
   - Updates models with new patterns
   - Improves prediction accuracy

3. **Accuracy Tracking**: Real-time calculation of:
   - Prediction error (absolute difference)
   - Prediction accuracy (percentage)
   - Exposed via Prometheus metrics

### Configuration:

Edit `app_advanced.py` to change:

```python
# Line 57: History buffer size
history_buffer = deque(maxlen=1000)  # Keep last 1000 samples

# Line 303: Update frequency
time.sleep(0.5)  # Update every 500ms

# Line 288: Retraining interval
if time_since_training > 300:  # 5 minutes
```

---

## 📊 Dashboard Features

### Web Dashboard (localhost:5002):

**📊 Model Predictions vs Actual**
- Side-by-side comparison of all metrics
- Color-coded health indicators:
  - 🟢 Green: >90% accuracy
  - 🟡 Yellow: 70-90% accuracy  
  - 🔴 Red: <70% accuracy
- Accuracy progress bars

**📈 Live Charts**
- State of Health Trend (actual + predicted)
- Temperature Monitoring (actual + predicted)
- RUL Prediction Accuracy (actual + predicted)
- Updates every 2 seconds
- Smooth animations

**⚡ Live Battery Metrics**
- SoC, Voltage, Current, Power
- Speed, Distance
- All updating in real-time

**🤖 ML System Status**
- Models loaded count
- History samples collected
- Last training timestamp
- Prometheus port info

### Grafana Dashboard:

**Panel 1: SoH Comparison**
- Line chart with actual (solid) and predicted (dashed)
- Shows model accuracy over time

**Panel 2 & 3: Current Gauges**
- Live SoH and SoC gauges
- Color thresholds

**Panel 4: Temperature Comparison**
- Actual vs predicted temperature
- Mean and current values

**Panel 5: RUL Prediction**
- Remaining useful life trends
- Prediction quality visible

**Panel 6: Electrical Parameters**
- Voltage, Current, Power
- All on one chart

**Panel 7: Failure Probability**
- Risk gauge with thresholds

**Panel 8: ML Accuracy**
- Multiple metrics accuracy tracking
- Shows model performance

**Panel 9 & 10: Statistics**
- Total predictions made
- Model retraining count

---

## 🎨 Customization

### Change Visualization Update Rates:

```javascript
// In battery_digital_twin_advanced.html

// Line ~370: Comparison update
setInterval(updateComparisons, 1000);  // Change to 2000 for slower

// Line ~371: Live metrics
setInterval(updateLiveMetrics, 1000);

// Line ~372: Charts
setInterval(updateCharts, 2000);  // Change to 5000 for slower
```

### Add More Models:

Edit `app_advanced.py`, modify line 61:

```python
self.feature_names = [
    'soc', 'soh', 'battery_voltage', 'battery_current', 
    'battery_temperature', 'charge_cycles', 'power_consumption',
    'your_new_feature'  # Add here
]
```

### Adjust Retraining:

```python
# Line 286 in app_advanced.py
if len(history_buffer) >= 100:  # Change minimum samples
    time_since_training = (datetime.now() - last_training_time).total_seconds()
    if time_since_training > 300:  # Change interval (seconds)
```

---

## 🔧 Troubleshooting

### Models Not Loading?
Check terminal output for:
```
📦 Successfully loaded X models
📋 Available models: [...]
```

If 0 models loaded:
- Verify models are in `models/` folder
- Check file extensions (*.joblib or *.pkl)
- Install cuML if using GPU models: `conda install -c rapidsai cuml`

### Charts Not Updating?
1. Check browser console (F12) for errors
2. Verify API is responding: `curl http://localhost:5002/api/health`
3. Check CORS isn't blocking requests

### Grafana Not Showing Data?
1. Verify Prometheus datasource URL: `http://localhost:9091`
2. Test metrics endpoint: `curl http://localhost:9091/metrics`
3. Check dashboard time range (top right)
4. Ensure app_advanced.py is running

### High CPU Usage?
- Increase update intervals in HTML
- Reduce history buffer size
- Increase `time.sleep()` in simulation

---

## 📈 Performance Tips

### For Better ML Accuracy:
1. Let the system run for 10+ minutes to collect data
2. Models will improve with retraining
3. Check accuracy metrics in Grafana

### For Smoother Visualization:
1. Use Chrome/Edge for best performance
2. Close other browser tabs
3. Reduce update frequency if needed

### For Production Use:
1. Use proper WSGI server (gunicorn, waitress)
2. Add authentication to endpoints
3. Store history in database (PostgreSQL, TimescaleDB)
4. Set up proper logging
5. Monitor with Grafana alerts

---

## 🎯 Testing the System

### Test 1: Verify Predictions

```powershell
# Get comparison data
curl http://localhost:5002/api/battery/comparison
```

Look for `accuracy` values - should be >80%

### Test 2: Check Prometheus Metrics

```powershell
# View all metrics
curl http://localhost:9091/metrics | Select-String "battery"
```

### Test 3: Verify Continuous Learning

Wait 5+ minutes, then check:
```powershell
curl http://localhost:5002/api/health
```

`model_training_total` should be > 0

---

## 🎉 What You've Built

You now have a **production-grade Digital Twin** with:

✅ **Real-time ML predictions** from your trained models  
✅ **Actual vs Predicted** comparisons with accuracy tracking  
✅ **Beautiful web dashboard** with live charts  
✅ **Professional Grafana** monitoring  
✅ **Prometheus metrics** for observability  
✅ **Continuous learning** - models improve over time  
✅ **REST API** for integration  
✅ **Color-coded health** indicators  
✅ **Historical data** tracking  

---

## 🚀 Next Steps

1. **Run it**: `python app_advanced.py`
2. **Open dashboard**: http://localhost:5002
3. **Set up Grafana**: Follow Grafana setup above
4. **Watch it learn**: Models improve as data accumulates
5. **Monitor accuracy**: Check prediction accuracy metrics
6. **Customize**: Modify colors, thresholds, update rates

---

**Enjoy your Advanced ML-Powered Digital Twin!** 🎉⚡🔋

*Need help? Check logs in terminal or use `/api/health` endpoint*
