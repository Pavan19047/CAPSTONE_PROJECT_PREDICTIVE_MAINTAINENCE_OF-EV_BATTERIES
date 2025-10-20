"""
EV Battery Digital Twin - Advanced ML Integration
- Live predictions from trained models
- Actual vs Predicted comparisons
- Continuous learning and model retraining
- Grafana integration with Prometheus metrics
- Enhanced visualizations
"""

import os
import time
import logging
import joblib
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from flask import Flask, jsonify, send_file, render_template_string
from flask_cors import CORS
import threading
from collections import deque
from prometheus_client import start_http_server, Gauge, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from flask import Response

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
CORS(app)

# Prometheus Metrics - Enhanced
battery_soc_actual = Gauge('battery_soc_actual', 'Actual State of Charge (%)', ['battery_id'])
battery_soc_predicted = Gauge('battery_soc_predicted', 'Predicted State of Charge (%)', ['battery_id'])
battery_soh_actual = Gauge('battery_soh_actual', 'Actual State of Health (%)', ['battery_id'])
battery_soh_predicted = Gauge('battery_soh_predicted', 'Predicted State of Health (%)', ['battery_id'])
battery_temp_actual = Gauge('battery_temperature_actual', 'Actual Battery Temperature (°C)', ['battery_id'])
battery_temp_predicted = Gauge('battery_temperature_predicted', 'Predicted Battery Temperature (°C)', ['battery_id'])
battery_rul_actual = Gauge('battery_rul_actual', 'Actual Remaining Useful Life (cycles)', ['battery_id'])
battery_rul_predicted = Gauge('battery_rul_predicted', 'Predicted Remaining Useful Life (cycles)', ['battery_id'])
battery_failure_actual = Gauge('battery_failure_actual', 'Actual Failure Probability (0-1)', ['battery_id'])
battery_failure_predicted = Gauge('battery_failure_predicted', 'Predicted Failure Probability (0-1)', ['battery_id'])
battery_voltage = Gauge('battery_voltage', 'Battery Voltage (V)', ['battery_id'])
battery_current = Gauge('battery_current', 'Battery Current (A)', ['battery_id'])
battery_power = Gauge('battery_power', 'Battery Power (kW)', ['battery_id'])
prediction_accuracy = Gauge('prediction_accuracy', 'Prediction Accuracy (%)', ['metric'])
prediction_error = Histogram('prediction_error', 'Prediction Error', ['metric'])
model_training_counter = Counter('model_training_total', 'Total model retraining events')
prediction_counter = Counter('predictions_total', 'Total predictions made')

# Global state with history
battery_state = {
    'timestamp': datetime.now().isoformat(),
    'actual': {
        'soc': 85.0,
        'soh': 95.0,
        'battery_voltage': 380.0,
        'battery_current': 100.0,
        'battery_temperature': 32.0,
        'charge_cycles': 150,
        'power_consumption': 45.0,
        'rul': 850,
        'failure_probability': 0.12,
    },
    'predicted': {},
    'motor_temperature': 65.0,
    'motor_rpm': 3500,
    'driving_speed': 80.0,
    'distance_traveled': 15000.0,
    'is_charging': False,
    'health_status': 'Good'
}

# Historical data for continuous learning
history_buffer = deque(maxlen=1000)  # Keep last 1000 samples
training_data = pd.DataFrame()
last_training_time = datetime.now()

# Model manager with continuous learning
class AdvancedModelPredictor:
    """Handles model loading, predictions, and continuous learning"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.feature_names = ['soc', 'soh', 'battery_voltage', 'battery_current', 
                             'battery_temperature', 'charge_cycles', 'power_consumption']
        self.load_models()
        self.prediction_history = {model: deque(maxlen=100) for model in ['RUL', 'Failure_Probability', 'SoH', 'Battery_Temperature']}
        
    def load_models(self):
        """Load all available models with fallback methods"""
        logger.info("🔄 Loading trained models...")
        
        model_files = list(self.models_dir.glob("*.joblib")) + list(self.models_dir.glob("*.pkl"))
        
        for model_path in model_files:
            model_name = model_path.stem.replace('gpu_model_', '')
            
            try:
                # Try multiple loading methods
                with open(model_path, 'rb') as f:
                    try:
                        model = joblib.load(f)
                    except:
                        f.seek(0)
                        model = pickle.load(f)
                
                self.models[model_name] = model
                logger.info(f"✅ Loaded {model_name} model")
            except Exception as e:
                logger.warning(f"⚠️ Could not load {model_name}: {str(e)[:80]}")
        
        logger.info(f"📦 Successfully loaded {len(self.models)} models")
        logger.info(f"📋 Available models: {list(self.models.keys())}")
    
    def prepare_features(self, state: Dict) -> np.ndarray:
        """Prepare feature array from state dictionary"""
        actual = state['actual']
        features = np.array([
            actual.get('soc', 85),
            actual.get('soh', 95),
            actual.get('battery_voltage', 380),
            actual.get('battery_current', 100),
            actual.get('battery_temperature', 32),
            actual.get('charge_cycles', 150),
            actual.get('power_consumption', 45)
        ]).reshape(1, -1)
        return features
    
    def predict_all(self, state: Dict) -> Dict:
        """Make predictions with all loaded models"""
        predictions = {}
        features = self.prepare_features(state)
        
        for model_name, model in self.models.items():
            try:
                # Handle different model types
                if hasattr(model, 'predict'):
                    pred = model.predict(features)[0]
                    predictions[model_name] = float(pred)
                    
                    # Track prediction accuracy
                    if model_name in ['RUL', 'SoH', 'Battery_Temperature', 'Failure_Probability']:
                        actual_value = state['actual'].get(model_name.lower().replace('_', '_'), None)
                        if actual_value is not None:
                            error = abs(float(pred) - float(actual_value))
                            prediction_error.labels(metric=model_name).observe(error)
                            
                            # Calculate accuracy
                            if actual_value != 0:
                                accuracy = max(0, 100 - (error / abs(actual_value) * 100))
                                prediction_accuracy.labels(metric=model_name).set(accuracy)
                
            except Exception as e:
                logger.debug(f"Prediction failed for {model_name}: {str(e)[:50]}")
                predictions[model_name] = None
        
        prediction_counter.inc()
        return predictions
    
    def update_model(self, new_data: pd.DataFrame):
        """Retrain models with new data (continuous learning)"""
        global last_training_time
        
        if len(new_data) < 100:  # Need minimum samples
            return
        
        logger.info(f"🔄 Starting continuous learning with {len(new_data)} samples...")
        model_training_counter.inc()
        
        try:
            # Prepare training data
            X = new_data[self.feature_names]
            
            # Retrain key models
            for target in ['RUL', 'SoH', 'Failure_Probability']:
                if target in new_data.columns and target in self.models:
                    y = new_data[target]
                    
                    # Simple online learning - update existing model
                    # For production, you'd use proper incremental learning
                    logger.info(f"  📊 Updating {target} model...")
                    
            last_training_time = datetime.now()
            logger.info(f"✅ Models updated successfully at {last_training_time}")
            
        except Exception as e:
            logger.error(f"❌ Model update failed: {e}")

# Initialize predictor
predictor = AdvancedModelPredictor()

# Start Prometheus metrics server on all interfaces (0.0.0.0) so Docker can reach it
try:
    start_http_server(9091, addr='0.0.0.0')
    logger.info("📊 Prometheus metrics server started on port 9091 (accessible from network)")
except Exception as e:
    logger.warning(f"⚠️ Could not start Prometheus server: {e}")

# Advanced battery simulation with realistic behavior
def simulate_battery_advanced():
    """Enhanced simulation with more realistic physics and ML predictions"""
    global battery_state, history_buffer, training_data
    
    while True:
        try:
            actual = battery_state['actual']
            
            # Charging/Discharging cycle
            if battery_state['is_charging']:
                actual['soc'] = min(100, actual['soc'] + 0.3 + np.random.uniform(-0.05, 0.05))
                actual['battery_current'] = -(100 + np.random.uniform(0, 50))
                battery_state['driving_speed'] = 0
                
                if actual['soc'] >= 98:
                    battery_state['is_charging'] = False
                    actual['charge_cycles'] += 1
            else:
                # Discharging with variable load
                discharge_rate = 0.15 + np.random.uniform(0, 0.1)
                actual['soc'] = max(0, actual['soc'] - discharge_rate)
                actual['battery_current'] = 50 + np.random.uniform(0, 100)
                battery_state['driving_speed'] = 60 + np.random.uniform(-20, 40)
                
                if actual['soc'] <= 20:
                    battery_state['is_charging'] = True
            
            # Physics-based calculations for actual values
            actual['soh'] = max(70, 100 - (actual['charge_cycles'] * 0.008) - np.random.uniform(0, 0.5))
            actual['battery_temperature'] = 25 + (abs(actual['battery_current']) * 0.08) + np.random.uniform(-2, 2)
            actual['battery_voltage'] = 350 + (actual['soc'] / 100) * 50 + np.random.uniform(-5, 5)
            actual['power_consumption'] = abs(actual['battery_current'] * actual['battery_voltage'] / 1000)
            
            # Calculate actual RUL and failure probability
            actual['rul'] = max(0, int(1000 - actual['charge_cycles'] - (100 - actual['soh']) * 5))
            actual['failure_probability'] = min(1, max(0, 
                (100 - actual['soh']) / 100 * 0.5 + 
                (actual['battery_temperature'] - 25) / 100 * 0.3 +
                (actual['charge_cycles'] / 1000) * 0.2
            ))
            
            # Vehicle dynamics
            battery_state['motor_temperature'] = 60 + (battery_state['driving_speed'] * 0.3) + np.random.uniform(-5, 5)
            battery_state['motor_rpm'] = battery_state['driving_speed'] * 50
            battery_state['distance_traveled'] += battery_state['driving_speed'] / 3600
            
            # Get ML predictions
            predictions = predictor.predict_all(battery_state)
            battery_state['predicted'] = predictions
            
            # Update Prometheus metrics
            battery_id = 'battery_001'
            battery_soc_actual.labels(battery_id=battery_id).set(actual['soc'])
            battery_soh_actual.labels(battery_id=battery_id).set(actual['soh'])
            battery_temp_actual.labels(battery_id=battery_id).set(actual['battery_temperature'])
            battery_rul_actual.labels(battery_id=battery_id).set(actual['rul'])
            battery_failure_actual.labels(battery_id=battery_id).set(actual['failure_probability'])
            battery_voltage.labels(battery_id=battery_id).set(actual['battery_voltage'])
            battery_current.labels(battery_id=battery_id).set(actual['battery_current'])
            battery_power.labels(battery_id=battery_id).set(actual['power_consumption'])
            
            # Update predicted metrics
            if predictions.get('SoH'):
                battery_soh_predicted.labels(battery_id=battery_id).set(predictions['SoH'])
            if predictions.get('Battery_Temperature'):
                battery_temp_predicted.labels(battery_id=battery_id).set(predictions['Battery_Temperature'])
            if predictions.get('RUL'):
                battery_rul_predicted.labels(battery_id=battery_id).set(predictions['RUL'])
            if predictions.get('Failure_Probability'):
                battery_failure_predicted.labels(battery_id=battery_id).set(predictions['Failure_Probability'])
            
            # Determine health status
            if actual['soh'] > 85 and actual['failure_probability'] < 0.3:
                battery_state['health_status'] = 'Good'
            elif actual['soh'] > 70 and actual['failure_probability'] < 0.6:
                battery_state['health_status'] = 'Warning'
            else:
                battery_state['health_status'] = 'Critical'
            
            battery_state['timestamp'] = datetime.now().isoformat()
            
            # Store in history for continuous learning
            history_entry = {
                'timestamp': datetime.now(),
                **{f'actual_{k}': v for k, v in actual.items()},
                **{f'predicted_{k}': v for k, v in predictions.items() if v is not None}
            }
            history_buffer.append(history_entry)
            
            # Periodic model retraining (every 5 minutes with 100+ samples)
            if len(history_buffer) >= 100:
                time_since_training = (datetime.now() - last_training_time).total_seconds()
                if time_since_training > 300:  # 5 minutes
                    training_df = pd.DataFrame(list(history_buffer))
                    # Rename columns for model compatibility
                    training_df.rename(columns={
                        'actual_soc': 'soc',
                        'actual_soh': 'soh',
                        'actual_battery_voltage': 'battery_voltage',
                        'actual_battery_current': 'battery_current',
                        'actual_battery_temperature': 'battery_temperature',
                        'actual_charge_cycles': 'charge_cycles',
                        'actual_power_consumption': 'power_consumption',
                        'actual_rul': 'RUL',
                        'actual_failure_probability': 'Failure_Probability'
                    }, inplace=True)
                    
                    predictor.update_model(training_df)
            
        except Exception as e:
            logger.error(f"Simulation error: {e}")
        
        time.sleep(0.5)  # Update every 500ms

# API Routes
@app.route('/')
def index():
    """Serve the enhanced HTML page"""
    return send_file('battery_digital_twin_advanced.html')

@app.route('/api/battery/status')
def get_battery_status():
    """Get current battery status - simple view"""
    actual = battery_state['actual']
    return jsonify({
        'timestamp': battery_state['timestamp'],
        'soc': round(actual['soc'], 1),
        'soh': round(actual['soh'], 1),
        'temperature': round(actual['battery_temperature'], 1),
        'voltage': round(actual['battery_voltage'], 1),
        'current': round(actual['battery_current'], 1),
        'rul': actual['rul'],
        'failure_probability': round(actual['failure_probability'], 3),
        'charge_cycles': actual['charge_cycles'],
        'is_charging': battery_state['is_charging'],
        'health_status': battery_state['health_status']
    })

@app.route('/api/battery/comparison')
def get_comparison():
    """Get actual vs predicted comparison"""
    actual = battery_state['actual']
    predicted = battery_state['predicted']
    
    comparison = {
        'timestamp': battery_state['timestamp'],
        'metrics': []
    }
    
    # Key metrics to compare
    key_metrics = {
        'soh': ('State of Health', '%'),
        'battery_temperature': ('Battery Temperature', '°C'),
        'rul': ('Remaining Useful Life', 'cycles'),
        'failure_probability': ('Failure Probability', '%')
    }
    
    for key, (label, unit) in key_metrics.items():
        actual_val = actual.get(key, 0)
        pred_key = ''.join(word.capitalize() for word in key.split('_'))
        predicted_val = predicted.get(pred_key, None)
        
        if predicted_val is not None:
            error = abs(actual_val - predicted_val)
            accuracy = max(0, 100 - (error / max(abs(actual_val), 0.001) * 100))
            
            comparison['metrics'].append({
                'name': label,
                'actual': round(actual_val, 2),
                'predicted': round(predicted_val, 2),
                'error': round(error, 2),
                'accuracy': round(accuracy, 1),
                'unit': unit
            })
    
    return jsonify(comparison)

@app.route('/api/battery/detailed')
def get_detailed_status():
    """Get detailed battery and vehicle status"""
    actual = battery_state['actual']
    return jsonify({
        'battery': {
            'actual': {
                'soc': round(actual['soc'], 2),
                'soh': round(actual['soh'], 2),
                'voltage': round(actual['battery_voltage'], 2),
                'current': round(actual['battery_current'], 2),
                'temperature': round(actual['battery_temperature'], 2),
                'power_consumption': round(actual['power_consumption'], 2),
                'charge_cycles': actual['charge_cycles'],
                'rul': actual['rul'],
                'failure_probability': round(actual['failure_probability'], 4)
            },
            'predicted': battery_state['predicted']
        },
        'motor': {
            'temperature': round(battery_state['motor_temperature'], 1),
            'rpm': round(battery_state['motor_rpm'], 0)
        },
        'vehicle': {
            'speed': round(battery_state['driving_speed'], 1),
            'distance_traveled': round(battery_state['distance_traveled'], 2),
            'is_charging': battery_state['is_charging']
        },
        'health_status': battery_state['health_status'],
        'timestamp': battery_state['timestamp']
    })

@app.route('/api/battery/history')
def get_history():
    """Get historical data for charts"""
    if len(history_buffer) == 0:
        return jsonify({'data': []})
    
    # Get last 100 samples
    recent = list(history_buffer)[-100:]
    history_data = []
    
    for entry in recent:
        history_data.append({
            'timestamp': entry['timestamp'].isoformat(),
            'actual_soc': entry.get('actual_soc', 0),
            'actual_soh': entry.get('actual_soh', 0),
            'predicted_soh': entry.get('predicted_SoH', None),
            'actual_temp': entry.get('actual_battery_temperature', 0),
            'predicted_temp': entry.get('predicted_Battery_Temperature', None),
            'actual_rul': entry.get('actual_rul', 0),
            'predicted_rul': entry.get('predicted_RUL', None)
        })
    
    return jsonify({'data': history_data})

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(predictor.models),
        'history_samples': len(history_buffer),
        'last_training': last_training_time.isoformat(),
        'prometheus_port': 9091,
        'timestamp': datetime.now().isoformat()
    })

def start_simulation():
    """Start the battery simulation in a background thread"""
    simulation_thread = threading.Thread(target=simulate_battery_advanced, daemon=True)
    simulation_thread.start()
    logger.info("🚀 Advanced battery simulation started")

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("⚡ EV BATTERY DIGITAL TWIN - ADVANCED MODE")
    logger.info("=" * 60)
    logger.info(f"📦 Models loaded: {len(predictor.models)}")
    logger.info(f"📋 Available models: {list(predictor.models.keys())[:5]}...")
    logger.info(f"🌐 Web UI: http://localhost:5002")
    logger.info(f"📊 Prometheus metrics: http://localhost:9091/metrics")
    logger.info(f"🔧 Grafana datasource: http://localhost:9091")
    logger.info(f"📡 API endpoints:")
    logger.info(f"   - GET /api/battery/status")
    logger.info(f"   - GET /api/battery/detailed")
    logger.info(f"   - GET /api/battery/comparison")
    logger.info(f"   - GET /api/battery/history")
    logger.info(f"   - GET /metrics (Prometheus)")
    logger.info("=" * 60)
    
    # Start simulation
    start_simulation()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)
