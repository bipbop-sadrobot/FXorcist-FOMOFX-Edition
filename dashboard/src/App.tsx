import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';

// Simulated data generation function
const generateMarketData = () => {
  const timestamps = [];
  const prices = [];
  const returns = [];
  
  let currentPrice = 1.0;
  let cumulativeReturn = 0;
  
  for (let i = 0; i < 100; i++) {
    const timestamp = new Date(Date.now() - (100 - i) * 24 * 60 * 60 * 1000);
    const dailyReturn = (Math.random() - 0.5) * 0.02; // Random return between -1% and 1%
    currentPrice *= (1 + dailyReturn);
    cumulativeReturn += dailyReturn;
    
    timestamps.push(timestamp);
    prices.push(currentPrice);
    returns.push(cumulativeReturn);
  }
  
  return { timestamps, prices, returns };
};

const FXorcistDashboard: React.FC = () => {
  const [marketData, setMarketData] = useState(() => generateMarketData());

  return (
    <div className="container mx-auto p-6 bg-gray-100 dark:bg-gray-900 min-h-screen">
      <h1 className="text-3xl font-bold mb-6 text-center text-gray-800 dark:text-gray-200">
        FXorcist Trading Dashboard
      </h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Price Chart */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
          <h2 className="text-xl font-semibold mb-4 text-gray-700 dark:text-gray-300">
            Price Movement
          </h2>
          <Plot
            data={[{
              x: marketData.timestamps,
              y: marketData.prices,
              type: 'scatter',
              mode: 'lines',
              name: 'Price',
              line: { color: 'blue' }
            }]}
            layout={{
              autosize: true,
              height: 300,
              margin: { l: 50, r: 50, b: 50, t: 50 },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent'
            }}
            config={{ responsive: true }}
          />
        </div>

        {/* Cumulative Returns Chart */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
          <h2 className="text-xl font-semibold mb-4 text-gray-700 dark:text-gray-300">
            Cumulative Returns
          </h2>
          <Plot
            data={[{
              x: marketData.timestamps,
              y: marketData.returns,
              type: 'scatter',
              mode: 'lines',
              name: 'Cumulative Return',
              line: { color: 'green' }
            }]}
            layout={{
              autosize: true,
              height: 300,
              margin: { l: 50, r: 50, b: 50, t: 50 },
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent'
            }}
            config={{ responsive: true }}
          />
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="mt-6 bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-700 dark:text-gray-300">
          Performance Metrics
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg">
            <h3 className="text-sm text-gray-600 dark:text-gray-400">Total Return</h3>
            <p className="text-lg font-bold text-green-600">
              {(marketData.returns[marketData.returns.length - 1] * 100).toFixed(2)}%
            </p>
          </div>
          <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg">
            <h3 className="text-sm text-gray-600 dark:text-gray-400">Volatility</h3>
            <p className="text-lg font-bold text-blue-600">
              {(Math.std(marketData.returns) * 100).toFixed(2)}%
            </p>
          </div>
          <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg">
            <h3 className="text-sm text-gray-600 dark:text-gray-400">Sharpe Ratio</h3>
            <p className="text-lg font-bold text-purple-600">1.2</p>
          </div>
          <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg">
            <h3 className="text-sm text-gray-600 dark:text-gray-400">Max Drawdown</h3>
            <p className="text-lg font-bold text-red-600">-5.6%</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FXorcistDashboard;