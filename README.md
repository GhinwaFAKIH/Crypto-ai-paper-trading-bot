# Crypto AI Paper Trading Bot

Research-only BTC/ETH crypto backtesting with Python, Random Forest and Streamlit.

**Simulation only. No exchange orders, API keys, leverage, or real money.**

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Windows activation: `.venv\Scripts\activate`.

CLI: `python bot.py --symbol BTC-USD`

For local data: `python bot.py --csv prices.csv`. CSV needs `Date` and `Close` columns, and at least 350 daily positive prices.

## Model

- Features: historical returns, moving averages, volatility.
- Random Forest predicts next-day direction.
- Chronological 60/20/20 training/validation/test split. Validation reserved, not yet used for tuning.
- Signal delayed one bar, avoiding same-close fills.
- Simulated fees and spread/slippage applied when position changes.
- Compare against buy-and-hold, report return and maximum drawdown.

## Important limitations

This is a **historical backtester**, not live paper trading yet. Close-based fills, spreads, and fees are approximations; no order book, liquidity, taxes, or market impact is modeled. Historical results are not forecasts. Do not tune thresholds against the held-out test set.

Next steps: unit tests, walk-forward validation, a persistent virtual portfolio ledger, and forward paper trading.