"""
EV Battery Digital Twin - Utility Functions
Helper functions for database, logging, and data processing
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import psycopg2


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Setup a logger with standard formatting
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def get_db_connection(
    host: Optional[str] = None,
    port: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None
):
    """
    Create database connection with environment variable fallbacks
    
    Returns:
        psycopg2 connection object
    """
    return psycopg2.connect(
        host=host or os.getenv('DB_HOST', 'localhost'),
        port=port or os.getenv('DB_PORT', '5432'),
        user=user or os.getenv('DB_USER', 'twin'),
        password=password or os.getenv('DB_PASSWORD', 'twin_pass'),
        database=database or os.getenv('DB_NAME', 'twin_data')
    )


def validate_telemetry(data: Dict[str, Any]) -> bool:
    """
    Validate telemetry data structure
    
    Args:
        data: Telemetry dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        'battery_id', 'soc', 'soh', 'voltage', 'current',
        'temperature', 'charge_cycles', 'power_consumption'
    ]
    
    for field in required_fields:
        if field not in data:
            return False
    
    # Validate ranges
    if not (0 <= data['soc'] <= 100):
        return False
    if not (0 <= data['soh'] <= 100):
        return False
    if data['temperature'] < -40 or data['temperature'] > 100:
        return False
    
    return True


def format_telemetry_for_display(data: Dict[str, Any]) -> str:
    """
    Format telemetry data for console display
    
    Args:
        data: Telemetry dictionary
        
    Returns:
        Formatted string
    """
    return (
        f"Battery: {data.get('battery_id', 'N/A')} | "
        f"SoC: {data.get('soc', 0):.1f}% | "
        f"SoH: {data.get('soh', 0):.1f}% | "
        f"Temp: {data.get('temperature', 0):.1f}°C | "
        f"Voltage: {data.get('voltage', 0):.1f}V | "
        f"Current: {data.get('current', 0):.1f}A | "
        f"Cycles: {data.get('charge_cycles', 0)}"
    )


def calculate_battery_health_score(
    soc: float,
    soh: float,
    temperature: float,
    failure_prob: float
) -> float:
    """
    Calculate overall battery health score (0-100)
    
    Args:
        soc: State of Charge
        soh: State of Health
        temperature: Battery temperature
        failure_prob: Failure probability
        
    Returns:
        Health score (0-100)
    """
    # Weight factors
    soh_weight = 0.4
    soc_weight = 0.2
    temp_weight = 0.2
    failure_weight = 0.2
    
    # Normalize temperature (optimal: 20-40°C)
    temp_score = 100
    if temperature < 20:
        temp_score = 50 + (temperature / 20) * 50
    elif temperature > 40:
        temp_score = max(0, 100 - (temperature - 40) * 2)
    
    # Calculate weighted score
    health_score = (
        soh * soh_weight +
        soc * soc_weight +
        temp_score * temp_weight +
        (1 - failure_prob) * 100 * failure_weight
    )
    
    return max(0, min(100, health_score))


def save_json(data: Dict[str, Any], filepath: str):
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file
    
    Args:
        filepath: Input file path
        
    Returns:
        Dictionary from JSON
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def retry_on_failure(func, max_retries: int = 3, delay: float = 1.0):
    """
    Retry function on failure
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
    """
    import time
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
