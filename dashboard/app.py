import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import quantstats as qs
import numpy as np
from datetime import datetime
import os

# Set page config
st.set_page_config(
    page_title="FXorcist Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme + neumorphism
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .stButton>button {
        background: linear-gradient(145deg, #1e2125, #23272b);
        border: none; border-radius: 10px; padding: 10px 20px;
        box-shadow: 5px 5px 10px #1a1d21, -5px -5px 10px #282c30;
        color: white; font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(145deg, #1e2125, #23272b);
        border-radius: 15px; padding: 20px; margin: 10px 0;
        box-shadow: 5px 5px 15px #1a1d21, -5px -5px 15px #282c30;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("FXorcist Controls")
symbol = st.sidebar.selectbox("Symbol", ["EURUSD", "GBPUSD", "USDJPY"], index=0)
start_date = st.sidebar.date_input("Start Date", datetime(2024, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2024, 12, 31))
strategy = st.sidebar.selectbox("Strategy", ["RSI", "MACD", "Bollinger"], index=0)

# Mock data (replace with real backtest results)
dates = pd.date_range(start_date, end_date, freq='D')
np.random.seed(42)
returns = np.random.randn(len(dates)) * 0.01
prices = 100 * np.cumprod(1 + returns)
drawdown = prices / np.maximum.accumulate(prices) - 1

# Main content
st.title("FXorcist Trading Dashboard")

# Key Metrics (Rich-styled)
col1, col2, col3, col4 = st.columns(4)
metrics = [
    ("Total Return", f"{(prices[-1]/prices[0]-1)*100:.1f}%"),
    ("Max Drawdown", f"{drawdown.min()*100:.1f}%"),
    ("Sharpe Ratio", "1.8"),
    ("Win Rate", "58%")
]
for col, (label, value) in zip([col1, col2, col3, col4], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{label}</h3>
            <h2>{value}</h2>
        </div>
        """, unsafe_allow_html=True)

# Interactive Equity Curve + Drawdown
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    subplot_titles=("Equity Curve", "Drawdown"),
                    vertical_spacing=0.1, row_heights=[0.7, 0.3])

fig.add_trace(
    go.Scatter(x=dates, y=prices, name="Equity", line=dict(color='#00ff00', width=2)),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=dates, y=drawdown*100, name="Drawdown", fill='tozeroy', 
               line=dict(color='#ff0000', width=1)),
    row=2, col=1
)

fig.update_layout(
    height=600, 
    template="plotly_dark",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# QuantStats Tearsheet (embedded HTML)
st.subheader("Performance Analytics")
returns_series = pd.Series(returns, index=dates)
report_path = 'tearsheet.html'
qs.reports.html(returns_series, output=report_path, title='Strategy Report')
with open(report_path, 'r') as f:
    html_content = f.read()
st.components.v1.html(html_content, height=800, scrolling=True)

# Footer
st.markdown("---")
st.caption("FXorcist — Event-Driven Forex Backtesting & Optimization")