"""Crypto research backtester: NO exchange connections or real orders."""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def load_prices(csv=None, symbol="BTC-USD"):
    if csv:
        raw = pd.read_csv(csv)
        if not {"Date", "Close"}.issubset(raw.columns):
            raise ValueError("CSV requires Date and Close columns")
        raw["Date"] = pd.to_datetime(raw["Date"], utc=True)
        close = raw.set_index("Date")["Close"].astype(float)
    else:
        import yfinance as yf
        raw = yf.download(symbol, period="5y", interval="1d", auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError("No data downloaded; try --csv")
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index, utc=True)
    close = close.sort_index()
    close = close[~close.index.duplicated(keep="last")].dropna()
    if len(close) < 350 or (close <= 0).any():
        raise ValueError("Need at least 350 positive daily closing prices")
    return close

def features(close):
    ret = close.pct_change()
    return pd.DataFrame({
        "return_1": ret,
        "return_3": close.pct_change(3),
        "return_7": close.pct_change(7),
        "ma_ratio_7": close / close.rolling(7).mean() - 1,
        "ma_ratio_30": close / close.rolling(30).mean() - 1,
        "volatility_14": ret.rolling(14).std(),
    })

def backtest(close, capital=10000, fee_bps=10, spread_bps=5, threshold=.55):
    if capital <= 0 or min(fee_bps, spread_bps) < 0 or not .5 <= threshold <= 1:
        raise ValueError("Invalid capital, costs, or threshold")
    X = features(close)
    next_return = close.pct_change().shift(-1)
    valid = X.notna().all(axis=1) & next_return.notna()
    X = X.loc[valid]
    y = (next_return.loc[valid] > 0).astype(int)
    n = len(X)
    train_end, test_start = int(n*.6), int(n*.8)
    if train_end < 100 or n-test_start < 30:
        raise ValueError("Insufficient training/test observations")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=20,
        random_state=42, n_jobs=-1
    )
    model.fit(X.iloc[:train_end], y.iloc[:train_end])
    probabilities = pd.Series(
        model.predict_proba(X.iloc[test_start:])[:, list(model.classes_).index(1)],
        index=X.index[test_start:], name="prob_up"
    )
    # Decision after close t; first possible execution at close t+1.
    # Returns realized from close t+1 to close t+2; avoids same-close fills.
    decisions = (probabilities > threshold).astype(float)
    position = decisions.shift(1).fillna(0.0)
    daily_returns = close.pct_change().reindex(position.index).fillna(0)
    changes = position.diff().abs().fillna(position.iloc[0])
    cost = changes * (fee_bps + spread_bps) / 10000
    strategy_returns = position * daily_returns - cost
    equity = capital * (1 + strategy_returns).cumprod()
    benchmark = capital * (1 + daily_returns).cumprod()
    result = pd.DataFrame({
        "close": close.reindex(position.index), "prob_up": probabilities,
        "position": position, "market_return": daily_returns,
        "cost": cost, "strategy_return": strategy_returns,
        "equity": equity, "buy_hold_equity": benchmark
    })
    drawdown = equity / equity.cummax() - 1
    metrics = {
        "ending_equity": float(equity.iloc[-1]),
        "buy_hold_ending_equity": float(benchmark.iloc[-1]),
        "total_return_pct": float((equity.iloc[-1]/capital-1)*100),
        "max_drawdown_pct": float(drawdown.min()*100),
        "position_changes": int(changes.sum()),
        "test_days": len(result),
    }
    return result, metrics

def main():
    p = argparse.ArgumentParser(description="Research-only crypto backtest (no real trades)")
    p.add_argument("--csv", help="CSV with Date,Close columns")
    p.add_argument("--symbol", default="BTC-USD")
    p.add_argument("--capital", type=float, default=10000)
    p.add_argument("--fee-bps", type=float, default=10)
    p.add_argument("--spread-bps", type=float, default=5)
    p.add_argument("--threshold", type=float, default=.55)
    p.add_argument("--output", default="backtest_results.csv")
    args = p.parse_args()
    results, metrics = backtest(load_prices(args.csv, args.symbol), args.capital,
                                args.fee_bps, args.spread_bps, args.threshold)
    results.to_csv(args.output, index_label="Date")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"Saved {args.output} — SIMULATED ONLY")

if __name__ == "__main__":
    main()
