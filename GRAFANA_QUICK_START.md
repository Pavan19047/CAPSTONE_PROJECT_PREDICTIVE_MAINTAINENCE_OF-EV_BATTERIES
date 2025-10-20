# 🎯 Quick Grafana Setup Guide

## Current Status:
✅ **App Running**: Port 5002 (Web UI), Port 9091 (Metrics)  
✅ **Prometheus Running**: Port 9090  
✅ **Grafana Running**: Port 3000  

---

## 🚀 Configure Grafana (3 Easy Steps)

### Step 1: Add Prometheus Datasource

1. Open: **http://localhost:3000**
2. Login: `admin` / `admin`
3. Go to: **⚙️ Configuration → Data Sources → Add data source**
4. Select: **Prometheus**
5. Set URL to: **`http://ev_prometheus:9090`** (Docker network)
   - OR if that doesn't work: **`http://localhost:9090`**
6. Click: **Save & Test** ✅

### Step 2: Verify Metrics in Prometheus

1. Open: **http://localhost:9090**
2. Check: **Status → Targets**
   - Should see: `battery_digital_twin` target
   - State should be: **UP**
3. Go to: **Graph**
4. Enter query: `battery_soh_actual`
5. Click: **Execute**
   - Should see battery health data!

### Step 3: Import Grafana Dashboard

1. In Grafana, go to: **➕ Dashboards → Import**
2. Click: **Upload JSON file**
3. Select: `grafana_ml_dashboard.json`
4. Choose datasource: **Prometheus**
5. Click: **Import** 🎉

---

## 🔍 Troubleshooting

### If Prometheus target is DOWN:

**Check app is running:**
```powershell
curl http://192.168.31.96:9091/metrics | Select-String "battery"
```

**Restart Prometheus:**
```powershell
docker restart ev_prometheus
```

### If Grafana can't connect:

**Option 1: Use Docker network name**
```
http://ev_prometheus:9090
```

**Option 2: Use host network**
```
http://host.docker.internal:9090
```

**Option 3: Use localhost (if Grafana is not in Docker)**
```
http://localhost:9090
```

---

## 📊 Available Metrics:

- `battery_soc_actual` - State of Charge
- `battery_soh_actual` - State of Health  
- `battery_temperature_actual` - Temperature
- `battery_rul_actual` - Remaining Useful Life
- `battery_failure_actual` - Failure Probability
- `battery_voltage` - Voltage
- `battery_current` - Current
- `battery_power` - Power
- `predictions_total` - Total predictions
- `model_training_total` - Retraining events

---

## 🎉 Once Connected:

Your Grafana dashboard will show:
- 📈 Live battery metrics
- 📊 Real-time gauges for SoH, SoC
- ⚡ Electrical parameters
- 🔢 Prediction statistics

**Enjoy your monitoring dashboard!** 🚀
