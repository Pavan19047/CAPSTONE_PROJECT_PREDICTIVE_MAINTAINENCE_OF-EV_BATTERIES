"""
EV Battery Data Simulator
Continuously generates and publishes realistic battery telemetry data to TimescaleDB
"""
import psycopg2
import time
import random
import math
from datetime import datetime

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'twin_data',
    'user': 'twin',
    'password': 'twin_pass'
}

class BatterySimulator:
    def __init__(self, battery_id="BATTERY_001"):
        self.battery_id = battery_id
        # Initial battery state
        self.soc = 85.0  # State of Charge
        self.soh = 95.0  # State of Health
        self.voltage = 380.0
        self.current = 100.0
        self.temperature = 32.0
        self.charge_cycles = 150
        self.is_charging = False
        self.time_step = 0
        
    def simulate_step(self):
        """Simulate one time step of battery operation"""
        self.time_step += 1
        
        # Simulate charging/discharging cycle
        if self.soc < 20:
            self.is_charging = True
        elif self.soc > 95:
            self.is_charging = False
            
        # Update SoC based on charging state
        if self.is_charging:
            self.soc = min(100, self.soc + random.uniform(0.3, 0.8))
            self.current = -random.uniform(80, 120)  # Negative = charging
        else:
            self.soc = max(0, self.soc - random.uniform(0.1, 0.4))
            self.current = random.uniform(80, 150)  # Positive = discharging
            
        # Update voltage based on SoC
        self.voltage = 350 + (self.soc / 100) * 50 + random.uniform(-5, 5)
        
        # Update temperature with some thermal dynamics
        base_temp = 30 + (abs(self.current) / 10)
        self.temperature = self.temperature * 0.8 + base_temp * 0.2 + random.uniform(-2, 2)
        self.temperature = max(20, min(80, self.temperature))
        
        # Slowly degrade SoH
        if random.random() < 0.01:  # 1% chance per step
            self.soh = max(70, self.soh - random.uniform(0.01, 0.05))
            
        # Update charge cycles
        if self.is_charging and self.soc > 90 and random.random() < 0.1:
            self.charge_cycles += 1
            
        # Calculate power consumption
        power_consumption = abs(self.voltage * self.current)
        
        # Calculate RUL (Remaining Useful Life) - simplified model
        rul_prediction = max(0, (self.soh - 70) * 40 + random.uniform(-20, 20))
        
        # Calculate failure probability based on multiple factors
        failure_prob = 0.0
        if self.soh < 80:
            failure_prob += (80 - self.soh) / 100
        if self.temperature > 50:
            failure_prob += (self.temperature - 50) / 100
        if self.soc < 15:
            failure_prob += (15 - self.soc) / 100
        failure_prob = min(1.0, max(0.0, failure_prob))
        
        return {
            'battery_id': self.battery_id,
            'soc': round(self.soc, 2),
            'soh': round(self.soh, 2),
            'voltage': round(self.voltage, 2),
            'current': round(self.current, 2),
            'temperature': round(self.temperature, 2),
            'charge_cycles': self.charge_cycles,
            'power_consumption': round(power_consumption, 2),
            'is_charging': self.is_charging,
            'rul_prediction': round(rul_prediction, 2),
            'failure_probability': round(failure_prob, 4)
        }

def insert_telemetry(conn, data):
    """Insert telemetry data into TimescaleDB"""
    try:
        cursor = conn.cursor()
        query = """
        INSERT INTO ev_telemetry (
            battery_id, soc, soh, voltage, current, temperature,
            charge_cycles, power_consumption, is_charging,
            rul_prediction, failure_probability, prediction_timestamp
        ) VALUES (
            %(battery_id)s, %(soc)s, %(soh)s, %(voltage)s, %(current)s, %(temperature)s,
            %(charge_cycles)s, %(power_consumption)s, %(is_charging)s,
            %(rul_prediction)s, %(failure_probability)s, NOW()
        )
        """
        cursor.execute(query, data)
        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        print(f"Error inserting data: {e}")
        conn.rollback()
        return False

def main():
    print("🔋 EV Battery Digital Twin Simulator")
    print("=" * 50)
    print(f"Starting simulation at {datetime.now()}")
    print("Press Ctrl+C to stop\n")
    
    # Connect to database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Connected to TimescaleDB")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return
    
    # Initialize simulator
    simulator = BatterySimulator()
    
    iteration = 0
    try:
        while True:
            iteration += 1
            
            # Generate telemetry data
            data = simulator.simulate_step()
            
            # Insert into database
            if insert_telemetry(conn, data):
                status_icon = "🔌" if data['is_charging'] else "⚡"
                print(f"{status_icon} [{datetime.now().strftime('%H:%M:%S')}] "
                      f"SoC: {data['soc']:>5.1f}% | "
                      f"SoH: {data['soh']:>5.1f}% | "
                      f"Temp: {data['temperature']:>4.1f}°C | "
                      f"Voltage: {data['voltage']:>5.1f}V | "
                      f"Current: {data['current']:>6.1f}A | "
                      f"RUL: {data['rul_prediction']:>6.1f} | "
                      f"Failure: {data['failure_probability']*100:>4.1f}%")
            
            # Wait 5 seconds between readings
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Simulation stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        conn.close()
        print("✅ Database connection closed")
        print(f"Total iterations: {iteration}")

if __name__ == "__main__":
    main()
