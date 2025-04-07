import React, { useState, useEffect } from 'react';

const App = () => {
    const [vehicles, setVehicles] = useState([]);
    const [history, setHistory] = useState([]);

    const fetchParkingState = async () => {
        const response = await fetch('http://localhost:8000/parking-state');
        const data = await response.json();
        setVehicles(data);
    };

    const fetchHistory = async () => {
        const response = await fetch('http://localhost:8000/history');
        const data = await response.json();
        setHistory(data);
    };

    useEffect(() => {
        fetchParkingState();
        fetchHistory();
    }, []);

    return (
        <div>
            <h1>Aktualny stan parkingu</h1>
            <ul>
                {vehicles.map(vehicle => (
                    <li key={vehicle.id}>{vehicle.plate_number}</li>
                ))}
            </ul>
            
            <h1>Historia wjazdów</h1>
            <ul>
                {history.map(entry => (
                    <li key={entry.id}>{entry.plate_number} - {entry.timestamp} ({entry.status})</li>
                ))}
            </ul>
        </div>
    );
};

export default App;
