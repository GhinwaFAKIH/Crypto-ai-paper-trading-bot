"""One-shot virtual paper decision. Run once after each closed hourly candle.

No brokerage access, no real orders. Persistent JSON ledger is local only.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from bot import features
from market_data import hourly_candles

def step(symbol="BTCUSDT", path="paper_state.json", fee_bps=10, spread_bps=5):
    if fee_bps < 0 or spread_bps < 0:
        raise ValueError("Costs must be nonnegative")
    df = hourly_candles(symbol)
    close = df["close"]
    X = features(close).dropna()
    # Train on observations whose next-close label is already known.
    train_x = X.iloc[:-1]
    future = close.shift(-1)
    train_y = (future.loc[train_x.index] > close.loc[train_x.index]).astype(int)
    if len(train_x) < 100 or train_y.nunique() < 2:
        raise ValueError("Insufficient historical training data")
    model = RandomForestClassifier(n_estimators=100, max_depth=4,
                                   min_samples_leaf=15, random_state=42)
    model.fit(train_x, train_y)
    p_up = float(model.predict_proba(X.iloc[[-1]])[0, list(model.classes_).index(1)])
    signal = int(p_up >= .55)
    candle = X.index[-1].isoformat()
    price = float(close.loc[X.index[-1]])
    target = Path(path)
    state = json.loads(target.read_text()) if target.exists() else {
        "symbol": symbol, "cash": 10000.0, "units": 0.0,
        "last_candle": None, "history": [], "pending_signal": None
    }
    if state["symbol"] != symbol:
        raise ValueError("State file belongs to another symbol")
    if state["last_candle"] == candle:
        print("Already processed this candle; no duplicate decision")
        return state
    # Previous signal is filled at this candle's CLOSE (one full bar delay).
    previous = state.get("pending_signal")
    action = "HOLD"
    if previous == 1 and state["units"] == 0:
        execution_price = price * (1 + spread_bps/10000)
        units = state["cash"] / (execution_price * (1 + fee_bps/10000))
        state["cash"] -= units * execution_price * (1 + fee_bps/10000)
        state["units"] = units
        action = "VIRTUAL_BUY"
    elif previous == 0 and state["units"] > 0:
        execution_price = price * (1 - spread_bps/10000)
        state["cash"] += state["units"] * execution_price * (1 - fee_bps/10000)
        state["units"] = 0.0
        action = "VIRTUAL_SELL"
    state["pending_signal"] = signal
    state["last_candle"] = candle
    equity = state["cash"] + state["units"] * price
    state["history"].append({"candle": candle, "close": price, "prob_up": p_up,
                             "next_signal": signal, "action": action, "equity": equity})
    target.write_text(json.dumps(state, indent=2))
    print(f"{candle} {symbol}: {action}; p_up={p_up:.3f}; virtual equity={equity:.2f}")
    return state

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", choices=["BTCUSDT", "ETHUSDT"], default="BTCUSDT")
    parser.add_argument("--state", default="paper_state.json")
    args = parser.parse_args()
    step(args.symbol, args.state)
