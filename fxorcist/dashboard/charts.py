import plotly.graph_objects as go
import pandas as pd

def candlestick_fig(df: pd.DataFrame, volume: bool = True) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty:
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    if volume and 'Volume' in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', yaxis='y2', marker={'opacity':0.3}))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False))
    fig.update_layout(xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=25,b=10))
    return fig

def equity_curve_fig(returns: pd.Series) -> go.Figure:
    if returns is None or len(returns)==0:
        fig = go.Figure(); fig.add_annotation(text='No returns', x=0.5, y=0.5, showarrow=False); return fig
    cumulative = (1+returns).cumprod()
    fig = go.Figure(go.Scatter(x=cumulative.index, y=cumulative.values, mode='lines', name='Equity'))
    fig.update_layout(margin=dict(l=10,r=10,t=25,b=10))
    return fig

def drawdown_fig(returns: pd.Series) -> go.Figure:
    cumulative = (1+returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative-peak)/peak
    fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown.values, mode='lines', name='Drawdown'))
    return fig
