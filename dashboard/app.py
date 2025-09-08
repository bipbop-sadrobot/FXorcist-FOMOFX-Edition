import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import websockets
import asyncio
import json
from datetime import datetime, timedelta
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for dark theme
st.set_page_config(
    page_title="FXorcist Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background: #1e2125;
    }
    .stButton>button {
        background: linear-gradient(145deg, #2c3135, #23272b);
        color: #ffffff;
        border: none;
        box-shadow: 5px 5px 10px #1a1d21, -5px -5px 10px #282c30;
    }
    .stMetric {
        background: linear-gradient(145deg, #1e2125, #23272b);
        border-radius: 10px;
        padding: 10px;
        box-shadow: 5px 5px 10px #1a1d21, -5px -5px 10px #282c30;
    }
    .stMarkdown {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

class TradingDashboard:
    def __init__(self):
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "BTCUSD"]
        self.strategies = ["RSI", "MACD", "Bollinger Bands", "Machine Learning"]
        self.timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]

    def fetch_backtest_results(self, symbol: str, strategy: str):
        """Fetch backtest results from backend."""
        try:
            response = requests.get(f"http://localhost:8000/backtest/{symbol}/{strategy}")
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to fetch results: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"Error fetching backtest results: {e}")
            return None

    def create_performance_metrics(self, results: dict):
        """Create performance metrics visualization."""
        if not results:
            return

        metrics = results.get('metrics', {})
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Return", f"{metrics.get('cagr', 0)*100:.2f}%")
        with col2:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.2f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe', 0):.2f}")
        with col4:
            st.metric("Win Rate", f"{metrics.get('win_rate', 0)*100:.2f}%")

    def plot_equity_curve(self, results: dict):
        """Plot equity curve using Plotly."""
        if not results or 'equity_curve' not in results:
            st.warning("No equity curve data available")
            return

        equity_curve = results['equity_curve']
        df = pd.DataFrame(equity_curve, columns=['timestamp', 'value'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'], 
            y=df['value'], 
            mode='lines', 
            name='Equity Curve',
            line=dict(color='#00ffaa', width=3)
        ))

        fig.update_layout(
            title='Portfolio Equity Curve',
            xaxis_title='Time',
            yaxis_title='Portfolio Value',
            template='plotly_dark',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    def live_market_data(self):
        """Display live market data."""
        st.subheader("Live Market Data")
        
        # Simulated WebSocket-like data
        placeholder = st.empty()
        for _ in range(50):  # Simulate 50 updates
            data = {
                symbol: {
                    'price': np.random.normal(1.0, 0.01),
                    'change': np.random.uniform(-0.5, 0.5)
                } for symbol in self.symbols
            }
            
            with placeholder.container():
                cols = st.columns(len(self.symbols))
                for i, (symbol, info) in enumerate(data.items()):
                    with cols[i]:
                        st.metric(
                            symbol, 
                            f"${info['price']:.4f}", 
                            f"{info['change']:.2f}%"
                        )
            
            st.empty()  # Clear previous state
            asyncio.sleep(1)

    def run(self):
        """Main dashboard runner."""
        st.title("FXorcist Trading Dashboard 📊")

        # Sidebar controls
        with st.sidebar:
            st.header("Trading Parameters")
            symbol = st.selectbox("Select Symbol", self.symbols)
            strategy = st.selectbox("Select Strategy", self.strategies)
            timeframe = st.selectbox("Select Timeframe", self.timeframes)
            
            if st.button("Run Backtest"):
                with st.spinner("Running backtest..."):
                    results = self.fetch_backtest_results(symbol, strategy)
                    
                    if results:
                        self.create_performance_metrics(results)
                        self.plot_equity_curve(results)

        # Live market data section
        st.header("Real-Time Market Insights")
        self.live_market_data()

def main():
    dashboard = TradingDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()