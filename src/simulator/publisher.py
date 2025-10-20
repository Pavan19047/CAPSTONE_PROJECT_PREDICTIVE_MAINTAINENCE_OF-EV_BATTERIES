"""
EV Battery Digital Twin - Telemetry Simulator
Simulates realistic battery behavior and publishes to DB, Kafka, and MQTT
"""

import os
import json
import time
import logging
import random
from datetime import datetime
from typing import Dict, Any
from dataclasses import dataclass, asdict

import psycopg2
from psycopg2.extras import RealDictCursor
from kafka import KafkaProducer
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BatteryState:
    """Battery state representation"""
    battery_id: str = "BATTERY_001"
    soc: float = 85.0  # State of Charge (%)
    soh: float = 95.0  # State of Health (%)
    voltage: float = 380.0  # Voltage (V)
    current: float = 100.0  # Current (A)
    temperature: float = 32.0  # Temperature (°C)
    charge_cycles: int = 150  # Total cycles
    power_consumption: float = 38000.0  # Power (W)
    is_charging: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class BatterySimulator:
    """Simulates realistic EV battery behavior"""
    
    def __init__(self):
        """Initialize the battery simulator"""
        self.state = BatteryState()
        
        # Simulation parameters
        self.discharge_rate = (0.2, 0.8)  # % per cycle
        self.charge_rate = (0.5, 1.5)  # % per cycle
        self.soh_degradation = 0.01  # % per cycle
        self.charge_threshold_low = 20.0  # Start charging
        self.charge_threshold_high = 99.0  # Stop charging
        
        logger.info("✅ Battery simulator initialized")
    
    def simulate_step(self):
        """Execute one simulation step"""
        if self.state.is_charging:
            # Charging behavior
            charge_increment = random.uniform(*self.charge_rate)
            self.state.soc = min(100.0, self.state.soc + charge_increment)
            self.state.current = -(100 + random.uniform(0, 50))  # Negative = charging
            
            # Check if fully charged
            if self.state.soc >= self.charge_threshold_high:
                self.state.is_charging = False
                self.state.charge_cycles += 1
                logger.info(f"⚡ Charging complete! Cycle: {self.state.charge_cycles}, SoC: {self.state.soc:.1f}%")
        else:
            # Discharging behavior
            discharge_decrement = random.uniform(*self.discharge_rate)
            self.state.soc = max(0.0, self.state.soc - discharge_decrement)
            self.state.current = 50 + random.uniform(0, 70)  # Positive = discharging
            
            # Check if needs charging
            if self.state.soc <= self.charge_threshold_low:
                self.state.is_charging = True
                logger.info(f"🔋 Low battery! Starting charge at SoC: {self.state.soc:.1f}%")
        
        # Update State of Health (degrades over time)
        self.state.soh = max(70.0, 100.0 - (self.state.charge_cycles * self.soh_degradation))
        
        # Temperature calculation (depends on current draw)
        base_temp = 25.0
        current_factor = abs(self.state.current) * 0.08
        noise = random.uniform(-2, 2)
        self.state.temperature = base_temp + current_factor + noise
        self.state.temperature = max(20.0, min(80.0, self.state.temperature))  # Clamp
        
        # Voltage calculation (depends on SoC)
        self.state.voltage = 350.0 + (self.state.soc / 100.0) * 50.0 + random.uniform(-5, 5)
        
        # Power consumption
        self.state.power_consumption = abs(self.state.voltage * self.state.current)
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Get current telemetry data"""
        return {
            'timestamp': datetime.now().isoformat(),
            'battery_id': self.state.battery_id,
            'soc': round(self.state.soc, 2),
            'soh': round(self.state.soh, 2),
            'voltage': round(self.state.voltage, 2),
            'current': round(self.state.current, 2),
            'temperature': round(self.state.temperature, 2),
            'charge_cycles': self.state.charge_cycles,
            'power_consumption': round(self.state.power_consumption, 2),
            'is_charging': self.state.is_charging
        }


class TelemetryPublisher:
    """Publishes telemetry to PostgreSQL, Kafka, and MQTT"""
    
    def __init__(self):
        """Initialize all publishers"""
        self.simulator = BatterySimulator()
        
        # Database connection
        self.db_conn = None
        self.connect_database()
        
        # Kafka producer
        self.kafka_producer = None
        self.connect_kafka()
        
        # MQTT client
        self.mqtt_client = None
        self.connect_mqtt()
        
        logger.info("✅ All publishers initialized")
    
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
            self.db_conn = None
    
    def connect_kafka(self):
        """Connect to Kafka (Redpanda)"""
        try:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                retries=3
            )
            logger.info("✅ Connected to Kafka")
        except Exception as e:
            logger.error(f"❌ Kafka connection failed: {e}")
            self.kafka_producer = None
    
    def connect_mqtt(self):
        """Connect to MQTT broker"""
        try:
            self.mqtt_client = mqtt.Client(client_id="battery_simulator")
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            
            mqtt_host = os.getenv('MQTT_HOST', 'localhost')
            mqtt_port = int(os.getenv('MQTT_PORT', '1883'))
            
            self.mqtt_client.connect(mqtt_host, mqtt_port, 60)
            self.mqtt_client.loop_start()
            logger.info("✅ Connected to MQTT broker")
        except Exception as e:
            logger.error(f"❌ MQTT connection failed: {e}")
            self.mqtt_client = None
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            logger.info("✅ MQTT connected successfully")
        else:
            logger.error(f"❌ MQTT connection failed with code: {rc}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        logger.warning(f"⚠️ MQTT disconnected with code: {rc}")
    
    def publish_to_database(self, telemetry: Dict[str, Any]):
        """Publish telemetry to TimescaleDB"""
        if not self.db_conn:
            return
        
        try:
            cursor = self.db_conn.cursor()
            query = """
                INSERT INTO ev_telemetry (
                    battery_id, soc, soh, voltage, current, temperature,
                    charge_cycles, power_consumption, is_charging
                ) VALUES (
                    %(battery_id)s, %(soc)s, %(soh)s, %(voltage)s, %(current)s,
                    %(temperature)s, %(charge_cycles)s, %(power_consumption)s, %(is_charging)s
                )
            """
            cursor.execute(query, telemetry)
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            logger.error(f"❌ Database publish failed: {e}")
            self.db_conn.rollback()
            # Try to reconnect
            self.connect_database()
    
    def publish_to_kafka(self, telemetry: Dict[str, Any]):
        """Publish telemetry to Kafka"""
        if not self.kafka_producer:
            return
        
        try:
            topic = os.getenv('KAFKA_TOPIC', 'ev_battery_telemetry')
            self.kafka_producer.send(topic, value=telemetry)
        except Exception as e:
            logger.error(f"❌ Kafka publish failed: {e}")
    
    def publish_to_mqtt(self, telemetry: Dict[str, Any]):
        """Publish telemetry to MQTT"""
        if not self.mqtt_client:
            return
        
        try:
            topic = os.getenv('MQTT_TOPIC', 'ev/battery/telemetry')
            payload = json.dumps(telemetry)
            self.mqtt_client.publish(topic, payload, qos=1)
        except Exception as e:
            logger.error(f"❌ MQTT publish failed: {e}")
    
    def publish_telemetry(self):
        """Get telemetry and publish to all channels"""
        telemetry = self.simulator.get_telemetry()
        
        # Publish to all channels
        self.publish_to_database(telemetry)
        self.publish_to_kafka(telemetry)
        self.publish_to_mqtt(telemetry)
        
        # Log telemetry
        logger.info(
            f"📊 SoC: {telemetry['soc']:.1f}% | "
            f"SoH: {telemetry['soh']:.1f}% | "
            f"Temp: {telemetry['temperature']:.1f}°C | "
            f"Current: {telemetry['current']:.1f}A | "
            f"Cycles: {telemetry['charge_cycles']} | "
            f"{'⚡ Charging' if telemetry['is_charging'] else '🔋 Discharging'}"
        )
    
    def run(self, interval: float = 2.0):
        """Run continuous simulation"""
        logger.info(f"🚀 Starting battery simulation (interval: {interval}s)")
        logger.info("Press Ctrl+C to stop...")
        
        try:
            while True:
                # Simulate battery step
                self.simulator.simulate_step()
                
                # Publish telemetry
                self.publish_telemetry()
                
                # Wait for next interval
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("\n⏹️ Simulation stopped by user")
        except Exception as e:
            logger.error(f"❌ Simulation error: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up connections"""
        logger.info("🧹 Cleaning up connections...")
        
        if self.db_conn:
            self.db_conn.close()
            logger.info("✅ Database connection closed")
        
        if self.kafka_producer:
            self.kafka_producer.close()
            logger.info("✅ Kafka producer closed")
        
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            logger.info("✅ MQTT client disconnected")


def main():
    """Main entry point"""
    publisher = TelemetryPublisher()
    
    # Get interval from environment or use default
    interval = float(os.getenv('TELEMETRY_INTERVAL', '2.0'))
    
    # Run simulation
    publisher.run(interval=interval)


if __name__ == "__main__":
    main()
