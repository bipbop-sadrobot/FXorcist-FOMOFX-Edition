import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

function App() {
  const [equityData, setEquityData] = useState([]);

  // Simulated WebSocket data for now
  useEffect(() => {
    const interval = setInterval(() => {
      setEquityData(prev => [
        ...prev, 
        { 
          time: new Date().toLocaleTimeString(), 
          value: Math.random() * 100 
        }
      ].slice(-20)); // Keep last 20 points
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">FXorcist Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Live Equity Curve</CardTitle>
          </CardHeader>
          <CardContent>
            <LineChart width={500} height={300} data={equityData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="value" stroke="#8884d8" />
            </LineChart>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default App;