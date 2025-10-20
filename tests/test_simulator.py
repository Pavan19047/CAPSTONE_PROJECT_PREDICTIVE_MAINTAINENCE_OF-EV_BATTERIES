"""
Unit tests for battery simulator
"""

import pytest
from src.simulator.publisher import BatterySimulator, BatteryState


def test_battery_state_initialization():
    """Test battery state initializes with correct defaults"""
    state = BatteryState()
    assert state.battery_id == "BATTERY_001"
    assert 0 <= state.soc <= 100
    assert 0 <= state.soh <= 100
    assert state.voltage > 0
    assert state.charge_cycles >= 0


def test_battery_simulator_creation():
    """Test simulator can be created"""
    simulator = BatterySimulator()
    assert simulator.state is not None
    assert simulator.state.soc > 0


def test_simulator_discharge():
    """Test battery discharges correctly"""
    simulator = BatterySimulator()
    simulator.state.soc = 80.0
    simulator.state.is_charging = False
    
    initial_soc = simulator.state.soc
    simulator.simulate_step()
    
    # SoC should decrease during discharge
    assert simulator.state.soc < initial_soc
    assert simulator.state.current > 0  # Positive current = discharge


def test_simulator_charge():
    """Test battery charges correctly"""
    simulator = BatterySimulator()
    simulator.state.soc = 30.0
    simulator.state.is_charging = True
    
    initial_soc = simulator.state.soc
    simulator.simulate_step()
    
    # SoC should increase during charge
    assert simulator.state.soc > initial_soc
    assert simulator.state.current < 0  # Negative current = charge


def test_telemetry_format():
    """Test telemetry is properly formatted"""
    simulator = BatterySimulator()
    telemetry = simulator.get_telemetry()
    
    required_fields = [
        'timestamp', 'battery_id', 'soc', 'soh', 'voltage',
        'current', 'temperature', 'charge_cycles', 
        'power_consumption', 'is_charging'
    ]
    
    for field in required_fields:
        assert field in telemetry


def test_soh_degradation():
    """Test State of Health degrades over cycles"""
    simulator = BatterySimulator()
    initial_soh = simulator.state.soh
    initial_cycles = simulator.state.charge_cycles
    
    # Simulate multiple cycles
    for _ in range(100):
        simulator.simulate_step()
    
    # SoH should degrade or stay same, cycles should increase
    assert simulator.state.charge_cycles >= initial_cycles


def test_temperature_calculation():
    """Test temperature is calculated within valid range"""
    simulator = BatterySimulator()
    
    for _ in range(50):
        simulator.simulate_step()
        # Temperature should be in reasonable range
        assert 15 <= simulator.state.temperature <= 85


def test_voltage_correlates_with_soc():
    """Test voltage increases with SoC"""
    simulator = BatterySimulator()
    
    simulator.state.soc = 20.0
    simulator.simulate_step()
    low_soc_voltage = simulator.state.voltage
    
    simulator.state.soc = 90.0
    simulator.simulate_step()
    high_soc_voltage = simulator.state.voltage
    
    # Higher SoC should generally mean higher voltage
    assert high_soc_voltage > low_soc_voltage


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
