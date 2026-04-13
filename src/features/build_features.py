import pandas as pd
import ta

def build_features(df):

    df = df.sort_values(["ticker", "Date"])

    # Returns
    df["return_1d"] = df.groupby("ticker")["Adj Close"].pct_change()
    df["return_5d"] = df.groupby("ticker")["Adj Close"].pct_change(5)

    # Moving averages
    df["ma20"] = df.groupby("ticker")["Adj Close"].transform(lambda x: x.rolling(20).mean())
    df["ma50"] = df.groupby("ticker")["Adj Close"].transform(lambda x: x.rolling(50).mean())

    # RSI
    df["rsi"] = df.groupby("ticker")["Adj Close"].transform(
        lambda x: ta.momentum.RSIIndicator(x).rsi()
    )

    # Volatility
    df["volatility_20"] = df.groupby("ticker")["return_1d"].transform(
        lambda x: x.rolling(20).std()
    )

    return df