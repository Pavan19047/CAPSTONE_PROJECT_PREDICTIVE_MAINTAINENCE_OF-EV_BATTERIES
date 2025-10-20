"""
EV Battery Digital Twin - REST API
Flask application with endpoints for battery data and predictions
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from flask import Flask, jsonify, request
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Database connection pool
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'user': os.getenv('DB_USER', 'twin'),
    'password': os.getenv('DB_PASSWORD', 'twin_pass'),
    'database': os.getenv('DB_NAME', 'twin_data')
}


def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(**db_config)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        conn.close()
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'service': 'EV Battery Digital Twin API'
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503


@app.route('/api/battery/latest', methods=['GET'])
def get_latest_battery_data():
    """Get latest battery telemetry"""
    battery_id = request.args.get('battery_id', 'BATTERY_001')
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT *
            FROM latest_battery_status
            WHERE battery_id = %s
        """
        
        cursor.execute(query, (battery_id,))
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if result:
            return jsonify({
                'status': 'success',
                'data': dict(result),
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': f'No data found for battery: {battery_id}'
            }), 404
            
    except Exception as e:
        logger.error(f"Failed to fetch latest data: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/battery/history', methods=['GET'])
def get_battery_history():
    """Get historical battery telemetry"""
    battery_id = request.args.get('battery_id', 'BATTERY_001')
    hours = int(request.args.get('hours', 1))
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT *
            FROM ev_telemetry
            WHERE battery_id = %s
            AND time > NOW() - INTERVAL '%s hours'
            ORDER BY time DESC
            LIMIT 1000
        """
        
        cursor.execute(query, (battery_id, hours))
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'status': 'success',
            'data': [dict(row) for row in results],
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/battery/predictions', methods=['GET'])
def get_predictions():
    """Get current RUL and failure risk predictions"""
    battery_id = request.args.get('battery_id', 'BATTERY_001')
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT 
                battery_id,
                rul_prediction,
                failure_probability,
                prediction_timestamp,
                soc,
                soh,
                temperature,
                charge_cycles
            FROM latest_battery_status
            WHERE battery_id = %s
        """
        
        cursor.execute(query, (battery_id,))
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if result:
            return jsonify({
                'status': 'success',
                'predictions': dict(result),
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': f'No predictions found for battery: {battery_id}'
            }), 404
            
    except Exception as e:
        logger.error(f"Failed to fetch predictions: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/battery/stats', methods=['GET'])
def get_battery_stats():
    """Get aggregated battery statistics"""
    battery_id = request.args.get('battery_id', 'BATTERY_001')
    hours = int(request.args.get('hours', 24))
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT 
                COUNT(*) as total_records,
                AVG(soc) as avg_soc,
                MIN(soc) as min_soc,
                MAX(soc) as max_soc,
                AVG(soh) as avg_soh,
                AVG(temperature) as avg_temperature,
                MAX(temperature) as max_temperature,
                AVG(voltage) as avg_voltage,
                AVG(current) as avg_current,
                MAX(charge_cycles) as total_cycles,
                AVG(rul_prediction) as avg_rul,
                AVG(failure_probability) as avg_failure_risk
            FROM ev_telemetry
            WHERE battery_id = %s
            AND time > NOW() - INTERVAL '%s hours'
        """
        
        cursor.execute(query, (battery_id, hours))
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'status': 'success',
            'stats': dict(result),
            'period_hours': hours,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to fetch stats: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/batteries/list', methods=['GET'])
def list_batteries():
    """List all batteries in the system"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT DISTINCT battery_id
            FROM ev_telemetry
            ORDER BY battery_id
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        battery_ids = [row['battery_id'] for row in results]
        
        return jsonify({
            'status': 'success',
            'batteries': battery_ids,
            'count': len(battery_ids),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to list batteries: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get current alerts for batteries"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT 
                battery_id,
                soc,
                soh,
                temperature,
                failure_probability,
                rul_prediction,
                status,
                time
            FROM latest_battery_status
            WHERE status IN ('WARNING', 'CRITICAL')
            OR soc < 20
            OR temperature > 60
            OR failure_probability > 0.7
            OR rul_prediction < 100
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'status': 'success',
            'alerts': [dict(row) for row in results],
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to fetch alerts: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'timestamp': datetime.now().isoformat()
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500


if __name__ == '__main__':
    port = int(os.getenv('API_PORT', 5001))
    logger.info(f"🚀 Starting EV Battery API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
