CREATE TABLE IF NOT EXISTS vehicles (
    id SERIAL PRIMARY KEY,
    plate_number VARCHAR(20) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS logs (
    id SERIAL PRIMARY KEY,
    vehicle_id INT REFERENCES vehicles(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    action VARCHAR(10) CHECK (action IN ('IN', 'OUT'))
);

CREATE TABLE IF NOT EXISTS parking_status (
    vehicle_id INT REFERENCES vehicles(id),
    is_parked BOOLEAN NOT NULL DEFAULT TRUE,
    PRIMARY KEY (vehicle_id)
);
