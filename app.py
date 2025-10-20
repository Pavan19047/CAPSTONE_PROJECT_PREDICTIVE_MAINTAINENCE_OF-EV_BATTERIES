"""
EV Battery Digital Twin - Complete Web Application
Integrates ML models, real-time data, and 3D visualization
"""

import os
import time
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from flask import Flask, jsonify, send_file, render_template_string
from flask_cors import CORS
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
CORS(app)

# Global state
battery_state = {
    'timestamp': datetime.now().isoformat(),
    'soc': 85.0,
    'soh': 95.0,
    'battery_voltage': 380.0,
    'battery_current': 100.0,
    'battery_temperature': 32.0,
    'charge_cycles': 150,
    'power_consumption': 45.0,
    'rul': 850,
    'failure_probability': 0.12,
    'motor_temperature': 65.0,
    'motor_rpm': 3500,
    'driving_speed': 80.0,
    'distance_traveled': 15000.0,
    'is_charging': False,
    'predictions': {},
    'health_status': 'Good'
}

# Model loader
class ModelPredictor:
    """Handles loading and predictions from trained models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all available models"""
        logger.info("🔄 Loading trained models...")
        
        # Try to import cuML if available
        try:
            import cuml
            logger.info("✅ cuML GPU library available")
        except ImportError:
            logger.warning("⚠️ cuML not available - will use CPU fallback for predictions")
        
        # Key models to load
        key_models = [
            'RUL', 'Failure_Probability', 'SoH', 'Battery_Temperature',
            'Battery_Voltage', 'Power_Consumption', 'Component_Health_Score'
        ]
        
        for model_name in key_models:
            model_path = self.models_dir / f"gpu_model_{model_name}.joblib"
            if model_path.exists():
                try:
                    # Try loading with joblib (works with most models)
                    import pickle
                    with open(model_path, 'rb') as f:
                        # Try multiple loading methods
                        try:
                            self.models[model_name] = joblib.load(f)
                        except:
                            f.seek(0)
                            self.models[model_name] = pickle.load(f)
                    logger.info(f"✅ Loaded {model_name} model")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load {model_name}: {str(e)[:50]}")
            else:
                logger.warning(f"⚠️ Model not found: {model_path}")
        
        # If no models loaded, use simulation-based predictions
        if len(self.models) == 0:
            logger.warning("⚠️ No models loaded - using physics-based simulation")
        else:
            logger.info(f"📦 Loaded {len(self.models)} models")
    
    def predict(self, features: Dict) -> Dict:
        """Make predictions using loaded models"""
        predictions = {}
        
        # Create feature array from current state
        feature_values = [
            features.get('soc', 85),
            features.get('soh', 95),
            features.get('battery_voltage', 380),
            features.get('battery_current', 100),
            features.get('battery_temperature', 32),
            features.get('charge_cycles', 150),
            features.get('power_consumption', 45)
        ]
        
        # Make predictions with each model
        for model_name, model in self.models.items():
            try:
                # Most models expect 2D input
                X = np.array(feature_values).reshape(1, -1)
                prediction = model.predict(X)[0]
                predictions[model_name] = float(prediction)
            except Exception as e:
                logger.debug(f"Prediction failed for {model_name}: {e}")
                predictions[model_name] = None
        
        return predictions

# Initialize predictor
predictor = ModelPredictor()

# Simulation logic
def simulate_battery():
    """Simulate realistic battery behavior"""
    global battery_state
    
    while True:
        try:
            # Charging/Discharging cycle
            if battery_state['is_charging']:
                battery_state['soc'] = min(100, battery_state['soc'] + 0.3)
                battery_state['battery_current'] = -(100 + np.random.uniform(0, 50))
                battery_state['driving_speed'] = 0
                
                if battery_state['soc'] >= 98:
                    battery_state['is_charging'] = False
                    battery_state['charge_cycles'] += 1
            else:
                # Discharging
                discharge_rate = 0.15 + np.random.uniform(0, 0.1)
                battery_state['soc'] = max(0, battery_state['soc'] - discharge_rate)
                battery_state['battery_current'] = 50 + np.random.uniform(0, 100)
                battery_state['driving_speed'] = 60 + np.random.uniform(-20, 40)
                
                if battery_state['soc'] <= 20:
                    battery_state['is_charging'] = True
            
            # Update dependent variables
            battery_state['soh'] = max(70, 100 - (battery_state['charge_cycles'] * 0.008))
            battery_state['battery_temperature'] = 25 + (abs(battery_state['battery_current']) * 0.08) + np.random.uniform(-2, 2)
            battery_state['battery_voltage'] = 350 + (battery_state['soc'] / 100) * 50 + np.random.uniform(-5, 5)
            battery_state['power_consumption'] = abs(battery_state['battery_current'] * battery_state['battery_voltage'] / 1000)
            battery_state['motor_temperature'] = 60 + (battery_state['driving_speed'] * 0.3) + np.random.uniform(-5, 5)
            battery_state['motor_rpm'] = battery_state['driving_speed'] * 50
            battery_state['distance_traveled'] += battery_state['driving_speed'] / 3600  # km per second
            
            # Get ML predictions
            predictions = predictor.predict(battery_state)
            battery_state['predictions'] = predictions
            
            # Update key metrics from predictions or calculations
            if predictions.get('RUL') is not None:
                battery_state['rul'] = max(0, int(predictions['RUL']))
            else:
                battery_state['rul'] = max(0, int(1000 - battery_state['charge_cycles']))
            
            if predictions.get('Failure_Probability') is not None:
                battery_state['failure_probability'] = min(1, max(0, float(predictions['Failure_Probability'])))
            else:
                battery_state['failure_probability'] = min(1, (100 - battery_state['soh']) / 100 + (battery_state['battery_temperature'] - 25) / 100)
            
            # Update SoH from prediction if available
            if predictions.get('SoH') is not None:
                battery_state['soh'] = min(100, max(0, float(predictions['SoH'])))
            
            # Determine health status
            if battery_state['soh'] > 85 and battery_state['failure_probability'] < 0.3:
                battery_state['health_status'] = 'Good'
            elif battery_state['soh'] > 70 and battery_state['failure_probability'] < 0.6:
                battery_state['health_status'] = 'Warning'
            else:
                battery_state['health_status'] = 'Critical'
            
            battery_state['timestamp'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Simulation error: {e}")
        
        time.sleep(0.5)  # Update every 500ms

# API Routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_file('battery_3d_viz.html')

@app.route('/api/battery/status')
def get_battery_status():
    """Get current battery status"""
    return jsonify({
        'timestamp': battery_state['timestamp'],
        'soc': round(battery_state['soc'], 1),
        'soh': round(battery_state['soh'], 1),
        'temperature': round(battery_state['battery_temperature'], 1),
        'voltage': round(battery_state['battery_voltage'], 1),
        'current': round(battery_state['battery_current'], 1),
        'rul': battery_state['rul'],
        'failure_probability': round(battery_state['failure_probability'], 3),
        'charge_cycles': battery_state['charge_cycles'],
        'is_charging': battery_state['is_charging'],
        'health_status': battery_state['health_status']
    })

@app.route('/api/battery/detailed')
def get_detailed_status():
    """Get detailed battery and vehicle status"""
    return jsonify({
        'battery': {
            'soc': round(battery_state['soc'], 2),
            'soh': round(battery_state['soh'], 2),
            'voltage': round(battery_state['battery_voltage'], 2),
            'current': round(battery_state['battery_current'], 2),
            'temperature': round(battery_state['battery_temperature'], 2),
            'power_consumption': round(battery_state['power_consumption'], 2),
            'charge_cycles': battery_state['charge_cycles'],
            'rul': battery_state['rul'],
            'failure_probability': round(battery_state['failure_probability'], 4)
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
        'predictions': battery_state['predictions'],
        'health_status': battery_state['health_status'],
        'timestamp': battery_state['timestamp']
    })

@app.route('/api/battery/predictions')
def get_predictions():
    """Get ML model predictions"""
    return jsonify({
        'predictions': battery_state['predictions'],
        'timestamp': battery_state['timestamp']
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(predictor.models),
        'timestamp': datetime.now().isoformat()
    })

def start_simulation():
    """Start the battery simulation in a background thread"""
    simulation_thread = threading.Thread(target=simulate_battery, daemon=True)
    simulation_thread.start()
    logger.info("🚀 Battery simulation started")

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("⚡ EV BATTERY DIGITAL TWIN - STARTING")
    logger.info("=" * 60)
    logger.info(f"📦 Models loaded: {len(predictor.models)}")
    logger.info(f"🌐 Server will start on http://localhost:5001")
    logger.info(f"📊 API endpoints available:")
    logger.info(f"   - GET /api/battery/status")
    logger.info(f"   - GET /api/battery/detailed")
    logger.info(f"   - GET /api/battery/predictions")
    logger.info(f"   - GET /api/health")
    logger.info("=" * 60)
    
    # Start simulation
    start_simulation()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
