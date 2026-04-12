"""
build_features.py  —  Fase 1 + 2
Calcola tutti gli indicatori tecnici usati dall'entry score avanzato.

Indicatori aggiunti rispetto alla versione precedente:
  - ma100, ma200          medie mobili più lente
  - adx                   forza del trend (ADX 14)
  - volume_ratio          volume odierno / media 20 giorni
  - momentum_3m           rendimento 63 giorni (≈ 3 mesi)
  - momentum_1m           rendimento 21 giorni
  - distance_from_ma200   distanza % dal prezzo alla MA200
  - ma50_slope            pendenza MA50 su 5 giorni (già presente)
  - atr_pct               volatilità normalizzata (ATR / prezzo)
  - above_ma200           flag 0/1
  - sp500_above_ma50      flag 0/1 (richiede colonna SP500 nel dataset)
  - vix_below_ma10        flag 0/1 (richiede colonna VIX nel dataset)
"""

import pandas as pd
import numpy as np
import ta


# ---------------------------------------------------------------------------
# Helper: calcola ADX su una singola serie (High, Low, Close)
# ---------------------------------------------------------------------------
def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Ritorna la serie ADX usando la libreria ta."""
    indicator = ta.trend.ADXIndicator(high=high, low=low, close=close, window=window)
    return indicator.adx()


# ---------------------------------------------------------------------------
# Helper: ATR percentuale
# ---------------------------------------------------------------------------
def _atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=window).average_true_range()
    return atr / close


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature tecniche al dataframe prezzi.

    Colonne attese in input:
        Date, ticker, Adj Close, High, Low, Volume
        (opzionali: VIX, SP500 — vengono usate se presenti)

    Ritorna il dataframe arricchito con tutte le feature.
    """
    df = df.copy()
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 1. Returns
    # ------------------------------------------------------------------
    df["return_1d"] = df.groupby("ticker")["Adj Close"].pct_change()
    df["return_5d"]  = df.groupby("ticker")["Adj Close"].pct_change(5)
    df["momentum_1m"] = df.groupby("ticker")["Adj Close"].pct_change(21)   # ~1 mese
    df["momentum_3m"] = df.groupby("ticker")["Adj Close"].pct_change(63)   # ~3 mesi

    # ------------------------------------------------------------------
    # 2. Moving averages
    # ------------------------------------------------------------------
    for w in [20, 50, 100, 200]:
        df[f"ma{w}"] = df.groupby("ticker")["Adj Close"].transform(
            lambda x: x.rolling(w).mean()
        )

    # Flag: prezzo sopra ogni MA
    df["above_ma50"]  = (df["Adj Close"] > df["ma50"]).astype(int)
    df["above_ma200"] = (df["Adj Close"] > df["ma200"]).astype(int)

    # Distanza % dal prezzo alla MA50 e MA200
    df["distance_from_ma50"]  = (df["Adj Close"] / df["ma50"])  - 1
    df["distance_from_ma200"] = (df["Adj Close"] / df["ma200"]) - 1

    # Pendenza MA50 (differenza a 5 giorni → trend positivo?)
    df["ma50_slope"] = df.groupby("ticker")["ma50"].transform(lambda x: x.diff(5))

    # Golden cross: MA20 sopra MA50
    df["ma20_above_ma50"] = (df["ma20"] > df["ma50"]).astype(int)

    # ------------------------------------------------------------------
    # 3. RSI
    # ------------------------------------------------------------------
    df["rsi"] = df.groupby("ticker")["Adj Close"].transform(
        lambda x: ta.momentum.RSIIndicator(close=x, window=14).rsi()
    )

    # ------------------------------------------------------------------
    # 4. ADX  (richiede High e Low)
    # ------------------------------------------------------------------
    if "High" in df.columns and "Low" in df.columns:
        adx_list = []
        for ticker, grp in df.groupby("ticker"):
            adx_series = _adx(grp["High"], grp["Low"], grp["Adj Close"])
            adx_series.index = grp.index
            adx_list.append(adx_series)
        df["adx"] = pd.concat(adx_list).sort_index()
    else:
        df["adx"] = np.nan

    # Flag: trend confermato
    df["adx_strong"] = (df["adx"] > 25).astype(int)

    # ------------------------------------------------------------------
    # 5. Volatilità
    # ------------------------------------------------------------------
    df["volatility_20"] = df.groupby("ticker")["return_1d"].transform(
        lambda x: x.rolling(20).std()
    )

    if "High" in df.columns and "Low" in df.columns:
        atr_list = []
        for ticker, grp in df.groupby("ticker"):
            atr_series = _atr_pct(grp["High"], grp["Low"], grp["Adj Close"])
            atr_series.index = grp.index
            atr_list.append(atr_series)
        df["atr_pct"] = pd.concat(atr_list).sort_index()
    else:
        df["atr_pct"] = df["volatility_20"]   # fallback

    # ------------------------------------------------------------------
    # 6. Volume ratio  (Volume odierno / media 20gg)
    # ------------------------------------------------------------------
    if "Volume" in df.columns:
        df["volume_ma20"] = df.groupby("ticker")["Volume"].transform(
            lambda x: x.rolling(20).mean()
        )
        df["volume_ratio"] = df["Volume"] / df["volume_ma20"]
        # Flag: volume sopra media (conferma istituzionale)
        df["volume_above_avg"] = (df["volume_ratio"] > 1.0).astype(int)
    else:
        df["volume_ratio"] = np.nan
        df["volume_above_avg"] = 0

    # ------------------------------------------------------------------
    # 7. Indicatori macro (VIX, SP500) — se presenti nel dataset
    # ------------------------------------------------------------------
    if "SP500" in df.columns:
        df["sp500_ma50"] = df["SP500"].rolling(50).mean()
        df["sp500_above_ma50"] = (df["SP500"] > df["sp500_ma50"]).astype(int)
    else:
        df["sp500_above_ma50"] = 0

    if "VIX" in df.columns:
        df["vix_ma10"] = df["VIX"].rolling(10).mean()
        df["vix_below_ma10"] = (df["VIX"] < df["vix_ma10"]).astype(int)
    else:
        df["vix_below_ma10"] = 0

    return df