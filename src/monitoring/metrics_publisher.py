"""
Metrics publisher for EV Battery ML model predictions
Publishes predictions and model metrics to Prometheus
"""

import time
from prometheus_client import start_http_server, Gauge, Counter
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MetricsPublisher:
    def __init__(self, port=9091):
        """Initialize the metrics publisher"""
        # Actual vs Predicted metrics
        self.battery_soh_actual = Gauge('battery_soh_actual', 'Actual State of Health', ['battery_id'])
        self.battery_soh_predicted = Gauge('battery_soh_predicted', 'Predicted State of Health', ['battery_id'])
        
        self.battery_rul_actual = Gauge('battery_rul_actual', 'Actual Remaining Useful Life', ['battery_id'])
        self.battery_rul_predicted = Gauge('battery_rul_predicted', 'Predicted Remaining Useful Life', ['battery_id'])
        
        self.battery_temperature_actual = Gauge('battery_temperature_actual', 'Actual Battery Temperature', ['battery_id'])
        self.battery_temperature_predicted = Gauge('battery_temperature_predicted', 'Predicted Battery Temperature', ['battery_id'])
        
        self.battery_failure_actual = Gauge('battery_failure_actual', 'Current Failure Probability', ['battery_id'])
        self.battery_failure_predicted = Gauge('battery_failure_predicted', 'Predicted Failure Probability', ['battery_id'])
        
        # Model performance metrics
        self.prediction_accuracy = Gauge('prediction_accuracy', 'ML Model Prediction Accuracy', ['metric'])
        self.prediction_rmse = Gauge('prediction_rmse', 'Root Mean Square Error', ['metric'])
        self.prediction_mae = Gauge('prediction_mae', 'Mean Absolute Error', ['metric'])
        
        # Other battery metrics
        self.battery_voltage = Gauge('battery_voltage', 'Battery Voltage', ['battery_id'])
        self.battery_current = Gauge('battery_current', 'Battery Current', ['battery_id'])
        self.battery_power = Gauge('battery_power', 'Battery Power', ['battery_id'])
        self.battery_soc = Gauge('battery_soc', 'State of Charge', ['battery_id'])
        
        # Counters
        self.predictions_total = Counter('predictions_total', 'Total number of predictions made')
        self.model_training_total = Counter('model_training_total', 'Total number of model retraining events')
        
        # Start Prometheus HTTP server
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")
    
    def publish_predictions(self, battery_id, predictions, actuals=None):
        """
        Publish model predictions and actuals
        
        Args:
            battery_id: ID of the battery
            predictions: Dictionary of predictions
            actuals: Dictionary of actual values (optional)
        """
        # Update predictions
        if 'soh' in predictions:
            self.battery_soh_predicted.labels(battery_id=battery_id).set(predictions['soh'])
        
        if 'rul' in predictions:
            self.battery_rul_predicted.labels(battery_id=battery_id).set(predictions['rul'])
        
        if 'temperature' in predictions:
            self.battery_temperature_predicted.labels(battery_id=battery_id).set(predictions['temperature'])
        
        if 'failure_prob' in predictions:
            self.battery_failure_predicted.labels(battery_id=battery_id).set(predictions['failure_prob'])
        
        # Update actuals if provided
        if actuals:
            if 'soh' in actuals:
                self.battery_soh_actual.labels(battery_id=battery_id).set(actuals['soh'])
            
            if 'rul' in actuals:
                self.battery_rul_actual.labels(battery_id=battery_id).set(actuals['rul'])
            
            if 'temperature' in actuals:
                self.battery_temperature_actual.labels(battery_id=battery_id).set(actuals['temperature'])
            
            if 'failure_prob' in actuals:
                self.battery_failure_actual.labels(battery_id=battery_id).set(actuals['failure_prob'])
            
            # Update battery state metrics
            if 'voltage' in actuals:
                self.battery_voltage.labels(battery_id=battery_id).set(actuals['voltage'])
            
            if 'current' in actuals:
                self.battery_current.labels(battery_id=battery_id).set(actuals['current'])
            
            if 'power' in actuals:
                self.battery_power.labels(battery_id=battery_id).set(actuals['power'])
            
            if 'soc' in actuals:
                self.battery_soc.labels(battery_id=battery_id).set(actuals['soc'])
        
        # Increment predictions counter
        self.predictions_total.inc()
    
    def update_model_metrics(self, metric_name, accuracy, rmse=None, mae=None):
        """
        Update model performance metrics
        
        Args:
            metric_name: Name of the metric (RUL, SoH, etc.)
            accuracy: Prediction accuracy (0-100)
            rmse: Root Mean Square Error (optional)
            mae: Mean Absolute Error (optional)
        """
        self.prediction_accuracy.labels(metric=metric_name).set(accuracy)
        
        if rmse is not None:
            self.prediction_rmse.labels(metric=metric_name).set(rmse)
        
        if mae is not None:
            self.prediction_mae.labels(metric=metric_name).set(mae)
    
    def record_training_event(self):
        """Record a model retraining event"""
        self.model_training_total.inc()