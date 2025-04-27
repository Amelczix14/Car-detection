import React, { useEffect, useState } from 'react';
import axios from 'axios';

const App = () => {
  const [parkingStatus, setParkingStatus] = useState([]);
  const [logs, setLogs] = useState([]);
  const [search, setSearch] = useState('');

  const fetchData = () => {
    axios.get('http://localhost:8000/status/').then(res => setParkingStatus(res.data));
    axios.get('http://localhost:8000/logs/').then(res => setLogs(res.data));
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const filteredLogs = logs.filter(log => log.plate_number.toLowerCase().includes(search.toLowerCase()));

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-4xl font-bold mb-6 text-center">🚗 Monitoring Parkingu</h1>
      <input type="text" placeholder="Wyszukaj tablicę..." value={search} onChange={e => setSearch(e.target.value)} className="p-2 border rounded w-full mb-4" />
      <h2 className="text-2xl mb-2">🔓 Aktualnie na parkingu:</h2>
      <ul>{parkingStatus.length === 0 ? <p>Brak pojazdów.</p> : parkingStatus.map((s, i) => (<li key={i}><strong>{s.plate_number}</strong> — {new Date(s.timestamp).toLocaleString()}</li>))}</ul>
      <h2 className="text-2xl mt-6 mb-2">📄 Historia:</h2>
      <table className="min-w-full bg-white border">
        <thead><tr><th className="py-2 px-4 border">Numer</th><th className="py-2 px-4 border">Akcja</th><th className="py-2 px-4 border">Data</th></tr></thead>
        <tbody>{filteredLogs.map(log => (<tr key={log.id}><td className="py-2 px-4 border">{log.plate_number}</td><td className={`py-2 px-4 border ${log.action === 'IN' ? 'text-green-600' : 'text-red-600'}`}>{log.action}</td><td className="py-2 px-4 border">{new Date(log.timestamp).toLocaleString()}</td></tr>))}</tbody>
      </table>
    </div>
  );
};

export default App;
