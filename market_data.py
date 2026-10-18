"""Public Binance hourly OHLCV. Read-only: never authenticates or submits orders."""
import json
from urllib.request import Request, urlopen
import pandas as pd

ENDPOINT = "https://data-api.binance.vision/api/v3/klines"
ALLOWED = {"BTCUSDT", "ETHUSDT"}

def hourly_candles(symbol="BTCUSDT", limit=1000):
    if symbol not in ALLOWED or not 100 <= limit <= 1000:
        raise ValueError("Only BTCUSDT/ETHUSDT and 100-1000 candles supported")
    url = f"{ENDPOINT}?symbol={symbol}&interval=1h&limit={limit}"
    req = Request(url, headers={"User-Agent": "CryptoPaperResearch/1.0"})
    with urlopen(req, timeout=20) as response:
        rows = json.load(response)
    if not isinstance(rows, list) or not rows:
        raise ValueError("No market data returned")
    frame = pd.DataFrame(rows)
    frame = frame.iloc[:, :7]
    frame.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    # Binance may include the currently forming candle. Exclude it.
    now = pd.Timestamp.now(tz="UTC")
    frame = frame[frame["close_time"] < now].copy()
    if frame.empty:
        raise ValueError("No fully closed candles")
    return frame.set_index("open_time").sort_index()
