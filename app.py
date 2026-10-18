"""Local-only Streamlit visualization of the research backtest."""
import streamlit as st
from bot import load_prices, backtest

st.set_page_config(page_title="Crypto AI Paper Lab", layout="wide")
st.title("Crypto AI Paper Lab")
st.caption("Research simulation only. No exchange orders, accounts, or real money.")
symbol = st.selectbox("Asset", ["BTC-USD", "ETH-USD"])
capital = st.number_input("Virtual starting balance ($)", min_value=100, value=10000, step=100)
fee = st.slider("Fee per position change (basis points)", 0, 50, 10)
spread = st.slider("Spread/slippage per position change (basis points)", 0, 50, 5)
threshold = st.slider("ML probability threshold", .50, .80, .55, .01)
if st.button("Run historical simulation", type="primary"):
    try:
        with st.spinner("Downloading data and running chronological backtest..."):
            result, metrics = backtest(load_prices(symbol=symbol), capital, fee, spread, threshold)
        a,b,c = st.columns(3)
        a.metric("Virtual ending balance", f"${metrics['ending_equity']:,.2f}")
        b.metric("Simulated return", f"{metrics['total_return_pct']:.2f}%")
        c.metric("Max drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
        st.line_chart(result[["equity", "buy_hold_equity"]])
        st.dataframe(result.tail(30), use_container_width=True)
        st.download_button("Download results CSV", result.to_csv(index_label="Date"),
                           "backtest_results.csv", "text/csv")
        st.warning("Historical performance does not predict future results. This is NOT live paper trading.")
    except Exception as exc:
        st.error(f"Simulation failed: {exc}")
