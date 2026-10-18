"""Local research dashboard; all orders are virtual."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from bot import load_prices, backtest
from paper_trade import step

st.set_page_config(page_title="Crypto AI Paper Lab", layout="wide")
st.title("Crypto AI Paper Lab")
st.caption("SIMULATION ONLY • No real-money execution • Local virtual ledger")

historical, paper = st.tabs(["Historical backtest", "Forward paper portfolio"])

with historical:
    symbol = st.selectbox("Asset", ["BTC-USD", "ETH-USD"])
    capital = st.number_input("Virtual starting balance ($)", min_value=100, value=10000, step=100)
    fee = st.slider("Fee (basis points)", 0, 50, 10)
    spread = st.slider("Spread/slippage (basis points)", 0, 50, 5)
    threshold = st.slider("Model probability threshold", .50, .80, .55, .01)
    if st.button("Run historical simulation", type="primary"):
        try:
            result, metrics = backtest(load_prices(symbol=symbol), capital, fee, spread, threshold)
            a,b,c = st.columns(3)
            a.metric("Virtual ending balance", f"${metrics['ending_equity']:,.2f}")
            b.metric("Simulated return", f"{metrics['total_return_pct']:.2f}%")
            c.metric("Max drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
            st.line_chart(result[["equity", "buy_hold_equity"]])
            st.dataframe(result.tail(30), use_container_width=True)
            st.download_button("Download historical CSV", result.to_csv(index_label="Date"),
                               "backtest_results.csv", "text/csv")
        except Exception as exc:
            st.error(str(exc))
    st.warning("Backtest results are historical and do not predict future returns.")

with paper:
    pair = st.selectbox("Binance market", ["BTCUSDT", "ETHUSDT"])
    state_path = Path(f"paper_state_{pair}.json")
    st.info("Each click checks the latest fully closed hourly candle. Signals execute no earlier than the next processed candle. No exchange orders are sent.")
    if st.button("Process latest closed candle"):
        try:
            step(pair, str(state_path))
            st.success("Paper simulation updated (or latest candle already processed).")
        except Exception as exc:
            st.error(f"Paper simulation failed: {exc}")
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            history = pd.DataFrame(state.get("history", []))
            cash, units = float(state["cash"]), float(state["units"])
            latest_close = float(history["close"].iloc[-1]) if not history.empty else 0
            a,b,c = st.columns(3)
            a.metric("Virtual cash (USDT)", f"{cash:,.2f}")
            b.metric(f"Virtual {pair[:-4]} units", f"{units:.6f}")
            c.metric("Marked equity (USDT)", f"{cash + units*latest_close:,.2f}")
            st.write("Pending model decision:", "BUY / HOLD" if state.get("pending_signal") == 1 else "CASH / EXIT" if state.get("pending_signal") == 0 else "None")
            if not history.empty:
                st.line_chart(history.set_index("candle")["equity"])
                st.dataframe(history.iloc[::-1], use_container_width=True)
                st.download_button("Download paper history", history.to_csv(index=False),
                                   f"paper_history_{pair}.csv", "text/csv")
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            st.error(f"Invalid local ledger: {exc}")
    else:
        st.caption("No paper ledger yet. Process a closed candle to create one.")
    st.warning("Virtual fills use candle-close approximations. Run locally after each hourly candle; this is not a 24/7 hosted bot.")
