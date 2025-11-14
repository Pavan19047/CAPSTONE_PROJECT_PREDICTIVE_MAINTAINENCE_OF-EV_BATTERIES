-- EV Battery Digital Twin - TimescaleDB Initialization Script
-- Create hypertable for time-series telemetry data

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Drop table if exists (for clean reinstall)
DROP TABLE IF EXISTS ev_telemetry CASCADE;

-- Create main telemetry table
CREATE TABLE ev_telemetry (
    time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    battery_id VARCHAR(50) NOT NULL DEFAULT 'BATTERY_001',
    
    -- Core Battery Metrics
    soc DOUBLE PRECISION NOT NULL,              -- State of Charge (%)
    soh DOUBLE PRECISION NOT NULL,              -- State of Health (%)
    voltage DOUBLE PRECISION NOT NULL,          -- Battery Voltage (V)
    current DOUBLE PRECISION NOT NULL,          -- Battery Current (A)
    temperature DOUBLE PRECISION NOT NULL,      -- Battery Temperature (°C)
    charge_cycles INTEGER NOT NULL,             -- Total Charge Cycles
    power_consumption DOUBLE PRECISION NOT NULL, -- Power Consumption (W)
    
    -- Operational Status
    is_charging BOOLEAN NOT NULL DEFAULT FALSE,
    status VARCHAR(20) DEFAULT 'NORMAL',        -- NORMAL, WARNING, CRITICAL
    
    -- ML Predictions (populated by predictor service)
    rul_prediction DOUBLE PRECISION,            -- Remaining Useful Life (cycles)
    failure_probability DOUBLE PRECISION,       -- Failure Risk (0-1)
    prediction_timestamp TIMESTAMPTZ,           -- When prediction was made
    
    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Convert to hypertable (partition by time)
SELECT create_hypertable('ev_telemetry', 'time');

-- Create indexes for better query performance
CREATE INDEX idx_battery_id ON ev_telemetry(battery_id, time DESC);
CREATE INDEX idx_soc ON ev_telemetry(soc);
CREATE INDEX idx_soh ON ev_telemetry(soh);
CREATE INDEX idx_temperature ON ev_telemetry(temperature);
CREATE INDEX idx_rul_prediction ON ev_telemetry(rul_prediction);
CREATE INDEX idx_failure_probability ON ev_telemetry(failure_probability);

-- Create continuous aggregate for hourly statistics
CREATE MATERIALIZED VIEW ev_telemetry_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    battery_id,
    AVG(soc) AS avg_soc,
    AVG(soh) AS avg_soh,
    AVG(temperature) AS avg_temperature,
    AVG(voltage) AS avg_voltage,
    AVG(current) AS avg_current,
    MAX(charge_cycles) AS max_cycles,
    AVG(rul_prediction) AS avg_rul,
    AVG(failure_probability) AS avg_failure_prob
FROM ev_telemetry
GROUP BY bucket, battery_id;

-- Add refresh policy (refresh every 30 minutes)
SELECT add_continuous_aggregate_policy('ev_telemetry_hourly',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '30 minutes');

-- Create retention policy (keep data for 30 days)
SELECT add_retention_policy('ev_telemetry', INTERVAL '30 days');

-- Create compression policy (compress data older than 7 days)
ALTER TABLE ev_telemetry SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'battery_id'
);

SELECT add_compression_policy('ev_telemetry', INTERVAL '7 days');

-- Insert sample data for testing
INSERT INTO ev_telemetry (
    battery_id, soc, soh, voltage, current, temperature, 
    charge_cycles, power_consumption, is_charging
) VALUES 
    ('BATTERY_001', 85.0, 95.0, 380.0, 100.0, 32.0, 150, 38000.0, FALSE),
    ('BATTERY_001', 84.5, 95.0, 378.5, 105.0, 33.0, 150, 39500.0, FALSE),
    ('BATTERY_001', 84.0, 95.0, 377.0, 110.0, 34.0, 150, 41000.0, FALSE);

-- Create view for latest battery status
CREATE OR REPLACE VIEW latest_battery_status AS
SELECT DISTINCT ON (battery_id)
    battery_id,
    time,
    soc,
    soh,
    voltage,
    current,
    temperature,
    charge_cycles,
    power_consumption,
    is_charging,
    status,
    rul_prediction,
    failure_probability,
    prediction_timestamp
FROM ev_telemetry
ORDER BY battery_id, time DESC;

-- Create function to get battery health status
CREATE OR REPLACE FUNCTION get_battery_health(
    p_soc DOUBLE PRECISION,
    p_soh DOUBLE PRECISION,
    p_temperature DOUBLE PRECISION,
    p_failure_prob DOUBLE PRECISION
) RETURNS VARCHAR(20) AS $$
BEGIN
    IF p_soc < 15 OR p_soh < 75 OR p_temperature > 60 OR p_failure_prob > 0.7 THEN
        RETURN 'CRITICAL';
    ELSIF p_soc < 30 OR p_soh < 85 OR p_temperature > 45 OR p_failure_prob > 0.3 THEN
        RETURN 'WARNING';
    ELSE
        RETURN 'NORMAL';
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to auto-update status based on health
CREATE OR REPLACE FUNCTION update_battery_status()
RETURNS TRIGGER AS $$
BEGIN
    NEW.status := get_battery_health(
        NEW.soc,
        NEW.soh,
        NEW.temperature,
        COALESCE(NEW.failure_probability, 0)
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_battery_status
BEFORE INSERT OR UPDATE ON ev_telemetry
FOR EACH ROW
EXECUTE FUNCTION update_battery_status();

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO twin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO twin;

-- Display setup confirmation
DO $$
BEGIN
    RAISE NOTICE '✅ TimescaleDB initialization complete!';
    RAISE NOTICE '📊 Hypertable created: ev_telemetry';
    RAISE NOTICE '📈 Continuous aggregate: ev_telemetry_hourly';
    RAISE NOTICE '🔍 View created: latest_battery_status';
    RAISE NOTICE '⚡ Database ready for EV Battery Digital Twin';
END $$;
