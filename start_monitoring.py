#!/usr/bin/env python3
"""
Start the EV Battery monitoring service
"""

import logging
from src.monitoring.prediction_service import BatteryPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point"""
    try:
        logger.info("Starting EV Battery Monitoring Service")
        predictor = BatteryPredictor()
        predictor.run_prediction_loop()
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()