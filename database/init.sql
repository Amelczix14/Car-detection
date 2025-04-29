CREATE TABLE plates (
    id SERIAL PRIMARY KEY,
    plate_number TEXT UNIQUE NOT NULL
);

CREATE TABLE logs (
    id SERIAL PRIMARY KEY,
    plate_number TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT CHECK (status IN ('GRANTED', 'DENIED'))
);
