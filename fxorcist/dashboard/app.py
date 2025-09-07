import streamlit as st
from fxorcist.data.loader import list_available_symbols, load_symbol
from fxorcist.dashboard.charts import candlestick_fig, equity_curve_fig, drawdown_fig
from fxorcist.pipeline.vectorized_backtest import sma_strategy_returns, simple_metrics
import pandas as pd
import io

st.set_page_config(page_title="FXorcist", layout="wide")

@st.cache_data(ttl=600)
def _get_symbols(base_dir: str | None = None):
    return list_available_symbols(base_dir)

@st.cache_data(ttl=600)
def _load(symbol: str, base_dir: str | None = None):
    return load_symbol(symbol, base_dir=base_dir, allow_synthetic_fallback=True)

def _df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    import pandas as pd
    with io.BytesIO() as b:
        with pd.ExcelWriter(b, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='data', index=True)
        return b.getvalue()

def streamlit_app():
    st.title("FXorcist — Strategy Explorer")
    left, right = st.columns([3,1])
    symbols = _get_symbols() or ['EURUSD','GBPUSD','AUDUSD']
    with right:
        st.header('Controls')
        symbol = st.selectbox('Symbol', symbols)
        fast = st.slider('Fast MA', 5, 40, 10)
        slow = st.slider('Slow MA', 21, 200, 50)
        txn_cost = st.number_input('Transaction cost', min_value=0.0, max_value=0.01, value=0.0001, format="%.6f")
        show_rsi = st.checkbox('Show RSI')
        run_backtest = st.button('Run Backtest')
        st.markdown('---')
        if st.button('Download data (Excel)'):
            df = _load(symbol)
            st.download_button('Download Excel', _df_to_excel_bytes(df), file_name=f"{symbol}.xlsx")
    with left:
        df = _load(symbol)
        st.subheader('Price Chart')
        st.plotly_chart(candlestick_fig(df), use_container_width=True)
        if run_backtest:
            with st.spinner('Running backtest...'):
                rets = sma_strategy_returns(df, fast=fast, slow=slow, transaction_cost=txn_cost)
                metrics = simple_metrics(rets)
                st.metric('Sharpe', f"{metrics['sharpe']:.3f}")
                st.metric('Total Return', f"{metrics['total_return']:.3%}")
                st.metric('Max Drawdown', f"{metrics['max_drawdown']:.3%}")
                st.plotly_chart(equity_curve_fig(rets), use_container_width=True)
                st.plotly_chart(drawdown_fig(rets), use_container_width=True)
        st.sidebar.header('Advanced')
        if st.sidebar.checkbox('Show Raw Data'):
            st.write(df.head(50))

if __name__ == '__main__':
    streamlit_app()
