# Predictive Maintenance of EV Batteries using a Digital Twin and Machine Learning

Author: [Your Name]
Affiliation: [Your Department, Your University]
Date: October 20, 2025
Repository: https://github.com/Pavan19047/CAPSTONE_PROJECT_PREDICTIVE_MAINTAINENCE_OF-EV_BATTERIES

## Abstract
Electric vehicle (EV) battery packs are safety‑critical and cost‑dominant subsystems whose degradation dynamics are nonlinear and usage‑dependent. Traditional maintenance strategies (corrective or scheduled) either react too late or over‑service components. This work presents an open, reproducible digital‑twin system that integrates real‑time telemetry, machine learning (ML) prediction, and operational monitoring for EV battery predictive maintenance. A physics‑informed simulator generates high‑frequency telemetry (500 ms) that is ingested into a time‑series database (TimescaleDB) and streamed via Kafka/Redpanda and MQTT for IoT interoperability. A set of scikit‑learn Random Forest models provide live predictions for State of Health (SoH), Battery Temperature, Remaining Useful Life (RUL), and Failure Probability, exposed through a Flask REST API and instrumented with Prometheus metrics for Grafana dashboards. On the provided dataset, the SoH and temperature models achieve near‑perfect fit (R² ≈ 1.0), while RUL and failure risk are challenging with low R², highlighting opportunities for temporal and survival modeling. End‑to‑end inference latency remains below 5 ms per prediction on CPU, and the monitoring stack provides an 11‑panel dashboard summarizing live and predicted parameters. The contribution is a unified, end‑to‑end blueprint that combines a digital twin, ML inference, and cloud‑native observability to enable proactive, scalable EV battery maintenance.

## 1. Introduction
Predictive maintenance (PdM) seeks to anticipate failures by analyzing operational data to estimate component health and remaining life. In EVs, battery packs dictate safety, range, and lifecycle cost, making accurate health monitoring essential. Batteries exhibit complex electro‑thermal dynamics and age via cycling, calendar effects, and thermal stress. Traditional threshold‑based heuristics cannot reliably capture these coupled effects across diverse duty cycles.

Digital twins—virtual replicas synchronized with their physical counterparts—offer a principled approach to PdM by combining physics, data, and ML. A battery digital twin can continuously assimilate telemetry, infer latent health variables, forecast RUL, and surface actionable alerts to operators.

This work addresses three gaps: (i) integrated, open implementations that connect high‑frequency telemetry to prediction and monitoring; (ii) practical pipelines for near‑real‑time inference with production‑grade observability; and (iii) reproducible artifacts (code, configs, models, dashboards) that students and researchers can adapt. We propose and implement an end‑to‑end system that couples a battery telemetry simulator, a time‑series storage layer, ML inference services, a REST API, and a metrics‑first monitoring stack. The significance is twofold: an easily deployable reference architecture for EV battery PdM and an experimental platform to evaluate learning algorithms and operational trade‑offs.

## 2. System Overview
Figure 1 conceptually depicts the architecture: high‑frequency telemetry is simulated and published to multiple sinks, persisted in TimescaleDB, scored by ML models, exposed via REST endpoints, and visualized in Grafana and a web UI.

Conceptual flow (Figure 1):
1) Telemetry generator (500 ms) → 2) Message brokers (Kafka/Redpanda, MQTT) and TimescaleDB → 3) ML predictor → 4) REST API → 5) Prometheus metrics → 6) Grafana dashboards and web visualization.

Data path in the provided implementation:
- Simulation: `src/simulator/publisher.py` generates realistic SoC/SoH/temperature/voltage/current dynamics, publishes to TimescaleDB, Kafka, and MQTT.
- Storage: TimescaleDB schema (`docker/init_db.sql`) creates a hypertable `ev_telemetry`, a `latest_battery_status` view, and an hourly continuous aggregate.
- Inference: `app_advanced.py` loads CPU models (`models/*.joblib`), computes live predictions (SoH, Temperature, RUL, Failure Probability), and exposes Prometheus metrics; `src/inference/live_predictor.py` demonstrates a DB‑backed inference service that reads the latest row, predicts, and writes back predictions.
- API and UI: Flask endpoints (`app_advanced.py` and `src/api/app.py`) surface status, comparisons, and history; Grafana dashboards (`grafana_ml_dashboard.json`) present 11 panels; `battery_3d_viz.html` provides an interactive 3D visualization.

## 3. Technical Architecture (A–Z Guide)

### 3.1 Tech Stack Overview
- Programming language: Python 3.12 (ecosystem maturity and rich scientific libraries).
- Web framework: Flask (`app_advanced.py`, `src/api/app.py`) for lightweight REST services and CORS.
- ML toolkit: scikit‑learn (RandomForestRegressor) for CPU‑efficient, low‑latency inference; `joblib` for model serialization.
- Data science: NumPy, pandas for feature handling and preprocessing.
- Time‑series storage: TimescaleDB (PostgreSQL 15) for hypertables, continuous aggregates, retention, and compression (`docker/init_db.sql`).
- Messaging: Redpanda/Kafka (`kafka-python`) for streaming; Eclipse Mosquitto (MQTT, `paho‑mqtt`) for IoT compatibility.
- Monitoring: Prometheus (`prometheus_client`) with custom business KPIs and Grafana dashboards for operational visibility.
- MLOps: MLflow (optional in `docker-compose.yml`) with MinIO S3 for artifact storage.
- Containerization and orchestration: Docker Compose to provision TimescaleDB, Prometheus, Grafana, Mosquitto, Redpanda, MLflow, and MinIO.
- Visualization: Grafana dashboards (11 panels) and browser‑based 3D visualization (`battery_3d_viz.html`, Three.js) plus a charting UI in `battery_digital_twin_advanced.html`.

Rationale: All components are open‑source, widely adopted, and container‑ready. The combination supports real‑time ingest, millisecond‑level predictions, and strong observability.

### 3.2 Frontend Layer
- 3D Visualization: `battery_3d_viz.html` uses WebGL/Three.js to render a battery pack with cell coloring mapped to charge/temperature, supporting intuitive spatial inspection (e.g., thermal hotspots).
- Operational UI: `battery_digital_twin_advanced.html` presents live charts for actual vs predicted metrics and health status.
- Grafana Dashboards: `grafana_ml_dashboard.json` defines 11 panels, including: SoH actual vs predicted, temperature actual vs predicted, RUL comparison, SoC/SoH gauges, electrical parameters (voltage/current/power), failure probability gauge, prediction accuracy trends, and counters for predictions and retraining events.
- Interaction: Users observe deviations between actual and ML‑predicted series, drill into history (`/api/battery/history`), and assess risk via gauges and alert panels.

### 3.3 Backend Layer
Two complementary backends are provided:
1) Advanced battery twin service (`app_advanced.py`):
   - Endpoints: `GET /api/battery/status`, `GET /api/battery/detailed`, `GET /api/battery/comparison`, `GET /api/battery/history`, `GET /api/health`, and `GET /metrics`.
   - Simulation: physics‑aware charging/discharging, thermal dynamics, and health status classification (Good/Warning/Critical) at 500 ms cadence.
   - Inference: `AdvancedModelPredictor` loads models from `models/` and scores SoH, temperature, RUL, and failure probability on each tick.
   - Metrics: exposes >15 Prometheus metrics on port 9091 (e.g., `battery_soh_actual`, `battery_temp_predicted`, `prediction_accuracy`, `predictions_total`).

2) Database‑centric REST API (`src/api/app.py`):
   - Endpoints: `GET /health`, `GET /api/battery/latest`, `GET /api/battery/history?hours=`, `GET /api/battery/predictions`, `GET /api/battery/stats?hours=`, `GET /api/batteries/list`, and `GET /api/alerts`.
   - Use case: For deployments where telemetry is primarily consumed from TimescaleDB and the API provides aggregated views and alerts.

### 3.4 Machine Learning Models
- Training: `train_simple_models.py` trains four Random Forest regressors (SoH, Battery_Temperature, RUL, Failure_Probability) on `datasets/EV_Predictive_Maintenance_Dataset_15min.csv` using 80/20 splits. Models are persisted as `models/*.joblib`.
- Features: `['SoC','SoH','Battery_Voltage','Battery_Current','Battery_Temperature','Charge_Cycles','Power_Consumption']`.
- Metrics: Reported R² on the provided dataset—SoH and temperature ≈ 1.0000 (train/test), RUL ≈ 0.01 (train) and < 0 (test), failure probability ≈ 0.01 (train) and < 0 (test). The latter two indicate a need for temporal features and/or different objectives (classification/survival analysis).
- Inference: In `app_advanced.py`, features are assembled from live state and models are scored on each simulation step; `src/inference/live_predictor.py` illustrates DB‑driven inference, reading the latest row, predicting, and writing back `rul_prediction` and `failure_probability` with Prometheus counters.

### 3.5 Data Management
- Schema: `docker/init_db.sql` provisions `ev_telemetry` as a hypertable with indexes on `(battery_id,time)` and key columns (SoC, SoH, temperature, predictions). A `latest_battery_status` view simplifies API queries. Continuous aggregates (`ev_telemetry_hourly`) and policies provide roll‑ups, retention (30 days), and compression (>7 days).
- Ingestion: `src/simulator/publisher.py` writes telemetry to TimescaleDB and publishes to Kafka (`ev_battery_telemetry`) and MQTT (`ev/battery/telemetry`).
- Flow: simulation → storage/brokers → (optional) predictor updates predictions → API/Grafana read paths. The design supports both pull‑based (DB/API) and push‑based (brokers) consumers.

### 3.6 Monitoring & Visualization
- Prometheus: Scrapes the metrics endpoint every 5–15 s (`prometheus.yml`, `config/prometheus.yml`). Exposed metrics include actual/predicted SoH/temperature/RUL/failure risk, electrical measurements, prediction accuracy histograms, and event counters.
- Grafana: The 11‑panel dashboard renders live gauges, time series comparisons, and model KPIs. Alert panels highlight thermal excess, low SoC, elevated failure risk, and low RUL.

### 3.7 Deployment & Infrastructure
- Docker Compose (`docker-compose.yml`) provisions: TimescaleDB (5432), Redpanda/Kafka (9092/29092), Mosquitto (1883), Prometheus (9090), Grafana (3000), MinIO (9000/9001), MLflow (5000). Services share the `ev_network` bridge with persistent volumes.
- Application runtime: The advanced app runs locally (`python app_advanced.py`) exposing the web UI at 5002 and metrics at 9091. Ensure Prometheus scrapes the host metrics (e.g., `host.docker.internal:9091` from Docker, or your host IP).
- Minimal workflow:
  - `pip install -r requirements.txt`; `python train_simple_models.py`;
  - `docker-compose up -d` (wait 30–60 s);
  - `python app_advanced.py` (simulator + inference + /metrics);
  - Import `grafana_ml_dashboard.json` into Grafana and select the Prometheus datasource.
- See `ADVANCED_SETUP.md` for troubleshooting and networking guidance.

### 3.8 Testing & Quality Assurance
- Unit tests: `tests/test_simulator.py` validates simulator invariants (charge/discharge behavior, temperature and voltage ranges, telemetry formatting). Run with `pytest -v`.
- Data checks: `verify_dataset.py` (included) supports dataset sanity checks before training.
- Performance validation: empirical inference latency on CPU is < 5 ms/model; simulator generates updates every 500 ms without backlog; Prometheus scrape intervals of 5–15 s balance timeliness and overhead.

### 3.9 Performance & Scalability
- Throughput and latency: Simulation at 2 Hz (500 ms); per‑sample multi‑model inference at < 20 ms total on commodity CPU; metrics export is constant‑time.
- Horizontal scale: Partition by `battery_id` across predictor instances; use Kafka partitions for telemetry fan‑out; TimescaleDB scales reads via continuous aggregates and indexing; Grafana/Prometheus scale with sharding or remote‑write for fleets.
- Storage management: Retention and compression policies bound storage growth while preserving recent high‑resolution data.

### 3.10 Security & Reliability
- Health and resilience: `/api/health` and `/metrics` facilitate liveness/readiness; Compose services restart policies and volumes preserve state.
- Data integrity: TimescaleDB constraints and triggers compute status fields consistently.
- Hardening recommendations: enable TLS and auth for MQTT and API, restrict Prometheus exposure to internal networks, store secrets in `.env`/vault, and apply role‑based DB access beyond the demo defaults.

### 3.11 Future Scope & Research Directions
- Learning: sequence models (LSTM/Transformer), physics‑informed neural networks, and multi‑task learning for SoH/RUL co‑estimation; survival analysis for failure risk.
- Features: online feature store; anomaly detection; uncertainty quantification and conformal prediction for risk‑aware alerts.
- Systems: edge deployment on vehicle ECUs, federated digital twins across fleets, and CI/CD for models via MLflow registries.
- Data: integration with real CAN/BMS streams and calibration to cell chemistry‑specific parameters.

## 4. Results and Discussion
On the supplied dataset, the Random Forest regressors achieve R² ≈ 1.0 on SoH and temperature (train/test), demonstrating that static features capture most of the variance in these targets under the dataset’s sampling assumptions. In contrast, RUL and failure probability exhibit near‑zero or negative R² on held‑out data, indicating that non‑temporal regressors lack the temporal/degradation context necessary for generalization. Operationally, the end‑to‑end system demonstrates low inference latency (< 5 ms/model) and responsive dashboards; the simulator’s physics‑inspired rules create realistic charge/discharge and thermal trajectories that are sufficient to evaluate monitoring logic and alerting thresholds.

Limitations: (i) RUL modeling requires temporal histories and survival objectives; (ii) simulated data lacks real sensor noise, drift, and calibration errors; (iii) the demo security posture (no TLS/ACLs) is suitable for local evaluation only. Nevertheless, the architecture is extensible and the codebase modular, supporting rapid experimentation with model families and deployment patterns.

## 5. Conclusion
We presented an end‑to‑end digital‑twin system for EV battery predictive maintenance that unifies real‑time telemetry, ML inference, REST APIs, and monitoring. The open implementation, grounded in TimescaleDB, Flask, scikit‑learn, Prometheus, and Grafana, offers a reproducible baseline for research and teaching. While static regressors suffice for instantaneous SoH/temperature estimation on the reference dataset, RUL and failure risk motivate temporal and survival modeling. By combining a digital twin with metrics‑first observability, the system supports proactive maintenance strategies that can enhance EV safety and lifecycle economics.

## 6. References
[1] Project repository: P. Pavan, “Predictive Maintenance of EV Batteries using Digital Twin and Machine Learning,” GitHub, 2025. Available: https://github.com/Pavan19047/CAPSTONE_PROJECT_PREDICTIVE_MAINTAINENCE_OF-EV_BATTERIES

[2] F. Pedregosa et al., “Scikit‑learn: Machine Learning in Python,” Journal of Machine Learning Research, 12, pp. 2825–2830, 2011. Available: https://scikit-learn.org/

[3] TimescaleDB. “TimescaleDB: an open‑source time‑series database powered by PostgreSQL.” Available: https://www.timescale.com/

[4] The Prometheus Authors. “Prometheus: Monitoring system & time series database.” Available: https://prometheus.io/

[5] Grafana Labs. “Grafana: The open observability platform.” Available: https://grafana.com/

[6] Redpanda Data. “Redpanda: Kafka API compatible streaming platform.” Available: https://redpanda.com/

[7] Eclipse Foundation. “Eclipse Mosquitto—An open source MQTT broker.” Available: https://mosquitto.org/

[8] MLflow Authors. “MLflow: A platform for the ML lifecycle.” Available: https://mlflow.org/

[9] MinIO, Inc. “MinIO High Performance Object Storage.” Available: https://min.io/

[10] Three.js. “JavaScript 3D library.” Available: https://threejs.org/

Note: Cite additional datasets and domain references specific to your work as appropriate.

---
Appendix A (Reproducibility): Code references
- Simulation and multi‑channel publishing: `src/simulator/publisher.py`
- Advanced twin + inference + metrics + REST: `app_advanced.py`
- DB‑centric REST API: `src/api/app.py`
- DB schema and policies: `docker/init_db.sql`
- Live predictor (DB‑driven): `src/inference/live_predictor.py`
- Training script: `train_simple_models.py`
- Monitoring assets: `grafana_ml_dashboard.json`, `config/prometheus.yml`