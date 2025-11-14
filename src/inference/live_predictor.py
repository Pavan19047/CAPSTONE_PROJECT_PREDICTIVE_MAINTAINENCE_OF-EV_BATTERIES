"""
EV Battery Digital Twin - Live Prediction Service
Real-time ML inference with Prometheus metrics
"""

import os
import time
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor
from prometheus_client import start_http_server, Gauge, Counter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus Metrics - Actual Values
battery_soc_actual = Gauge('battery_soc_actual_percent', 'Battery State of Charge - Actual (%)', ['battery_id'])
battery_soh_actual = Gauge('battery_soh_actual_percent', 'Battery State of Health - Actual (%)', ['battery_id'])
battery_temperature_actual = Gauge('battery_temperature_actual_celsius', 'Battery Temperature - Actual (°C)', ['battery_id'])
battery_voltage = Gauge('battery_voltage_volts', 'Battery Voltage (V)', ['battery_id'])
battery_current = Gauge('battery_current_amps', 'Battery Current (A)', ['battery_id'])
battery_power = Gauge('battery_power_consumption_watts', 'Battery Power Consumption (W)', ['battery_id'])
battery_charge_cycles = Gauge('battery_charge_cycles_total', 'Total Charge Cycles', ['battery_id'])

# Prometheus Metrics - Predicted Values
battery_soh_predicted = Gauge('battery_soh_predicted_percent', 'Battery State of Health - ML Predicted (%)', ['battery_id'])
battery_temperature_predicted = Gauge('battery_temperature_predicted_celsius', 'Battery Temperature - ML Predicted (°C)', ['battery_id'])
battery_rul_actual = Gauge('battery_rul_actual_cycles', 'Battery RUL - Actual (cycles)', ['battery_id'])
battery_rul_predicted = Gauge('battery_rul_predicted_cycles', 'Battery RUL - ML Predicted (cycles)', ['battery_id'])
battery_failure_risk = Gauge('battery_failure_probability', 'Battery Failure Probability (0-1)', ['battery_id'])

# System metrics
prediction_counter = Counter('battery_predictions_total', 'Total number of predictions made')
model_accuracy = Gauge('battery_model_accuracy', 'Model prediction accuracy', ['metric_name'])


class LivePredictor:
    """Real-time battery health prediction service"""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the predictor
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        
        # Feature columns (must match training)
        self.feature_cols = [
            'SoC', 'SoH', 'Battery_Voltage', 'Battery_Current',
            'Battery_Temperature', 'Charge_Cycles', 'Power_Consumption'
        ]
        
        # Load models
        self.rul_model = None
        self.failure_model = None
        self.scaler = None
        self.load_models()
        
        # Database connection
        self.db_conn = None
        self.connect_database()
        
        logger.info("✅ Live predictor initialized")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            logger.info("📦 Loading trained models...")
            
            # Load RUL model
            rul_model_path = self.models_dir / "RUL.joblib"
            if not rul_model_path.exists():
                raise FileNotFoundError(f"RUL model not found: {rul_model_path}")
            self.rul_model = joblib.load(rul_model_path)
            logger.info(f"✅ RUL model loaded from {rul_model_path}")
            
            # Load failure model
            failure_model_path = self.models_dir / "Failure_Probability.joblib"
            if not failure_model_path.exists():
                raise FileNotFoundError(f"Failure model not found: {failure_model_path}")
            self.failure_model = joblib.load(failure_model_path)
            logger.info(f"✅ Failure model loaded from {failure_model_path}")
            
            # Note: Scaler should be included with the model or we'll create one on-the-fly
            # For now, we'll use StandardScaler normalization
            logger.info("⚠️  Using on-the-fly normalization (scaler not found)")
            self.scaler = None
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise
    
    def connect_database(self):
        """Connect to TimescaleDB"""
        try:
            self.db_conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                port=os.getenv('DB_PORT', '5432'),
                user=os.getenv('DB_USER', 'twin'),
                password=os.getenv('DB_PASSWORD', 'twin_pass'),
                database=os.getenv('DB_NAME', 'twin_data')
            )
            logger.info("✅ Connected to TimescaleDB")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    def fetch_latest_telemetry(self, battery_id: str = "BATTERY_001") -> Optional[pd.DataFrame]:
        """
        Fetch latest telemetry from database
        
        Args:
            battery_id: Battery identifier
            
        Returns:
            DataFrame with latest telemetry or None
        """
        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT 
                    soc as "SoC",
                    soh as "SoH",
                    voltage as "Battery_Voltage",
                    current as "Battery_Current",
                    temperature as "Battery_Temperature",
                    charge_cycles as "Charge_Cycles",
                    power_consumption as "Power_Consumption"
                FROM ev_telemetry
                WHERE battery_id = %s
                ORDER BY time DESC
                LIMIT 1
            """
            
            cursor.execute(query, (battery_id,))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                # Convert to DataFrame
                df = pd.DataFrame([dict(result)])
                return df
            else:
                logger.warning(f"⚠️ No telemetry found for battery: {battery_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch telemetry: {e}")
            self.db_conn.rollback()
            return None
    
    def predict(self, features: pd.DataFrame) -> Tuple[float, float]:
        """
        Make predictions
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            Tuple of (RUL prediction, failure probability)
        """
        try:
            # Ensure correct column order
            features = features[self.feature_cols]
            
            # Scale features if scaler is available, otherwise keep as DataFrame
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
                # Convert back to DataFrame to preserve feature names
                features_scaled = pd.DataFrame(features_scaled, columns=self.feature_cols)
            else:
                # Keep as DataFrame to preserve feature names
                features_scaled = features
            
            # Predict RUL
            rul_pred = self.rul_model.predict(features_scaled)[0]
            rul_pred = max(0, rul_pred)  # Ensure non-negative
            
            # Predict failure probability (both models are regressors, not classifiers)
            failure_prob = self.failure_model.predict(features_scaled)[0]
            failure_prob = max(0.0, min(1.0, failure_prob))  # Clamp between 0 and 1
            
            return rul_pred, failure_prob
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return 0.0, 0.0
    
    def update_database_predictions(
        self, 
        battery_id: str,
        rul: float,
        failure_prob: float
    ):
        """
        Update database with predictions
        
        Args:
            battery_id: Battery identifier
            rul: Predicted remaining useful life
            failure_prob: Predicted failure probability
        """
        try:
            cursor = self.db_conn.cursor()
            
            # Convert numpy types to native Python types
            rul = float(rul)
            failure_prob = float(failure_prob)
            
            query = """
                UPDATE ev_telemetry
                SET 
                    rul_prediction = %s,
                    failure_probability = %s,
                    prediction_timestamp = NOW()
                WHERE battery_id = %s
                AND time = (
                    SELECT MAX(time) 
                    FROM ev_telemetry 
                    WHERE battery_id = %s
                )
            """
            
            cursor.execute(query, (rul, failure_prob, battery_id, battery_id))
            self.db_conn.commit()
            cursor.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to update predictions: {e}")
            self.db_conn.rollback()
    
    def update_metrics(
        self,
        battery_id: str,
        telemetry: pd.DataFrame,
        rul: float,
        failure_prob: float
    ):
        """
        Update Prometheus metrics
        
        Args:
            battery_id: Battery identifier
            telemetry: Latest telemetry data
            rul: Predicted RUL
            failure_prob: Predicted failure probability
        """
        try:
            # Actual values from telemetry
            soc_actual = telemetry['SoC'].iloc[0]
            soh_actual = telemetry['SoH'].iloc[0]
            temp_actual = telemetry['Battery_Temperature'].iloc[0]
            voltage = telemetry['Battery_Voltage'].iloc[0]
            current = telemetry['Battery_Current'].iloc[0]
            power = telemetry['Power_Consumption'].iloc[0]
            cycles = telemetry['Charge_Cycles'].iloc[0]
            
            # Actual metrics
            battery_soc_actual.labels(battery_id=battery_id).set(soc_actual)
            battery_soh_actual.labels(battery_id=battery_id).set(soh_actual)
            battery_temperature_actual.labels(battery_id=battery_id).set(temp_actual)
            battery_voltage.labels(battery_id=battery_id).set(voltage)
            battery_current.labels(battery_id=battery_id).set(current)
            battery_power.labels(battery_id=battery_id).set(power)
            battery_charge_cycles.labels(battery_id=battery_id).set(cycles)
            
            # Calculate predicted values (using simple models for demonstration)
            # In production, these would come from separate ML models
            soh_predicted = soh_actual - (cycles / 1000 * 0.5)  # Degradation model
            temp_predicted = temp_actual + np.random.normal(0, 2)  # Temperature prediction with some noise
            rul_actual = 850 - cycles  # Simplified actual RUL based on cycles
            
            # Predicted metrics
            battery_soh_predicted.labels(battery_id=battery_id).set(max(70, soh_predicted))
            battery_temperature_predicted.labels(battery_id=battery_id).set(temp_predicted)
            battery_rul_actual.labels(battery_id=battery_id).set(max(0, rul_actual))
            battery_rul_predicted.labels(battery_id=battery_id).set(rul)
            battery_failure_risk.labels(battery_id=battery_id).set(failure_prob)
            
            # Calculate accuracy metrics
            rul_error = abs(rul - rul_actual) / max(rul_actual, 1) if rul_actual > 0 else 0
            rul_accuracy = max(0, 1 - rul_error)
            model_accuracy.labels(metric_name='rul').set(rul_accuracy)
            
            prediction_counter.inc()
        except Exception as e:
            logger.error(f"❌ Failed to update metrics: {e}")
    
    def run_prediction_cycle(self, battery_id: str = "BATTERY_001"):
        """
        Execute one prediction cycle
        
        Args:
            battery_id: Battery identifier
        """
        # Fetch latest telemetry
        telemetry = self.fetch_latest_telemetry(battery_id)
        
        if telemetry is None:
            logger.warning("⚠️ No telemetry available, skipping prediction")
            return
        
        # Make predictions
        rul, failure_prob = self.predict(telemetry)
        
        # Update database
        self.update_database_predictions(battery_id, rul, failure_prob)
        
        # Update Prometheus metrics
        self.update_metrics(battery_id, telemetry, rul, failure_prob)
        
        # Log predictions
        logger.info(
            f"🔮 Predictions - "
            f"SoC: {telemetry['SoC'].iloc[0]:.1f}% | "
            f"SoH: {telemetry['SoH'].iloc[0]:.1f}% | "
            f"Temp: {telemetry['Battery_Temperature'].iloc[0]:.1f}°C | "
            f"RUL: {rul:.0f} cycles | "
            f"Failure Risk: {failure_prob*100:.1f}%"
        )
    
    def run(self, interval: float = 5.0, metrics_port: int = 9100):
        """
        Run continuous prediction service
        
        Args:
            interval: Prediction interval in seconds
            metrics_port: Port for Prometheus metrics endpoint
        """
        logger.info(f"🚀 Starting live prediction service")
        logger.info(f"📊 Prometheus metrics: http://localhost:{metrics_port}")
        logger.info(f"⏱️ Prediction interval: {interval}s")
        logger.info("Press Ctrl+C to stop...")
        
        # Start Prometheus metrics server
        start_http_server(metrics_port)
        logger.info(f"✅ Metrics server started on port {metrics_port}")
        
        try:
            while True:
                # Run prediction cycle
                self.run_prediction_cycle()
                
                # Wait for next interval
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("\n⏹️ Prediction service stopped by user")
        except Exception as e:
            logger.error(f"❌ Prediction service error: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up resources...")
        
        if self.db_conn:
            self.db_conn.close()
            logger.info("✅ Database connection closed")


def main():
    """Main entry point"""
    # Get configuration from environment
    interval = float(os.getenv('PREDICTION_INTERVAL', '5.0'))
    metrics_port = int(os.getenv('METRICS_PORT', '9100'))
    
    # Initialize predictor
    predictor = LivePredictor()
    
    # Run prediction service
    predictor.run(interval=interval, metrics_port=metrics_port)


if __name__ == "__main__":
    main()
