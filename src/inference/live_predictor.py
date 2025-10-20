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

# Prometheus Metrics
battery_soc = Gauge('battery_soc_percent', 'Battery State of Charge (%)', ['battery_id'])
battery_soh = Gauge('battery_soh_percent', 'Battery State of Health (%)', ['battery_id'])
battery_temperature = Gauge('battery_temperature_celsius', 'Battery Temperature (°C)', ['battery_id'])
battery_rul = Gauge('battery_rul_cycles', 'Battery Remaining Useful Life (cycles)', ['battery_id'])
battery_failure_risk = Gauge('battery_failure_probability', 'Battery Failure Probability (0-1)', ['battery_id'])
prediction_counter = Counter('battery_predictions_total', 'Total number of predictions made')


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
            rul_model_path = self.models_dir / "rul_model.pkl"
            if not rul_model_path.exists():
                raise FileNotFoundError(f"RUL model not found: {rul_model_path}")
            self.rul_model = joblib.load(rul_model_path)
            logger.info(f"✅ RUL model loaded from {rul_model_path}")
            
            # Load failure model
            failure_model_path = self.models_dir / "failure_model.pkl"
            if not failure_model_path.exists():
                raise FileNotFoundError(f"Failure model not found: {failure_model_path}")
            self.failure_model = joblib.load(failure_model_path)
            logger.info(f"✅ Failure model loaded from {failure_model_path}")
            
            # Load scaler
            scaler_path = self.models_dir / "scaler.pkl"
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found: {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            logger.info(f"✅ Scaler loaded from {scaler_path}")
            
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
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict RUL
            rul_pred = self.rul_model.predict(features_scaled)[0]
            rul_pred = max(0, rul_pred)  # Ensure non-negative
            
            # Predict failure probability
            failure_proba = self.failure_model.predict_proba(features_scaled)[0][1]
            
            return rul_pred, failure_proba
            
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
            battery_soc.labels(battery_id=battery_id).set(telemetry['SoC'].iloc[0])
            battery_soh.labels(battery_id=battery_id).set(telemetry['SoH'].iloc[0])
            battery_temperature.labels(battery_id=battery_id).set(telemetry['Battery_Temperature'].iloc[0])
            battery_rul.labels(battery_id=battery_id).set(rul)
            battery_failure_risk.labels(battery_id=battery_id).set(failure_prob)
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
