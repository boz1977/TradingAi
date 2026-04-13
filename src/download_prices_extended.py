"""
download_prices_extended.py  —  Fase 4
Scarica prezzi storici estesi dal 2000 per tutti i ticker FTSE MIB.

Rispetto al download_prices.py originale:
  - Parte dal 2000 invece del 2015 (~2.5x più storia)
  - Scarica anche High, Low, Volume (servono per ADX e volume_ratio)
  - Gestisce i ticker che esistono solo da una certa data (IPO più recenti)
  - Salva un file di metadati con la data di primo dato disponibile

Utilizzo:
    python src/download_prices_extended.py
"""

import os
import sys
import pandas as pd
import yfinance as yf

# Compatibile sia con "python src/download_prices_extended.py"
# che con "from src.download_prices_extended import ..."
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tickers import FTSE_MIB_TICKERS


START_DATE    = "2000-01-01"
OUTPUT_DIR    = "data/raw"
PRICES_FILE   = os.path.join(OUTPUT_DIR, "prices_extended.csv")
FAILED_FILE   = os.path.join(OUTPUT_DIR, "failed_tickers_extended.csv")
METADATA_FILE = os.path.join(OUTPUT_DIR, "ticker_metadata.csv")

# Ticker con IPO recente — partiamo da una data realistica
TICKER_START_OVERRIDE = {
    "NEXI.MI":  "2019-04-01",
    "BMED.MI":  "2017-10-01",
    "INW.MI":   "2015-02-01",
}


def download_prices(tickers: list, default_start: str = START_DATE) -> tuple:
    all_data = []
    failed   = []
    metadata = []

    for ticker in tickers:
        start = TICKER_START_OVERRIDE.get(ticker, default_start)
        try:
            print(f"Downloading {ticker} (da {start})...")
            df = yf.download(ticker, start=start, auto_adjust=False, progress=False)

            if df.empty:
                print(f"  WARNING: nessun dato per {ticker}")
                failed.append({"ticker": ticker, "reason": "empty dataframe"})
                continue

            # Normalizza colonne (yfinance a volte ritorna MultiIndex)
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df.reset_index(inplace=True)
            df["ticker"] = ticker

            # Assicura che ci siano le colonne necessarie
            required = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
            missing  = [c for c in required if c not in df.columns]
            if missing:
                print(f"  WARNING: colonne mancanti {missing} per {ticker}")
                failed.append({"ticker": ticker, "reason": f"missing columns: {missing}"})
                continue

            rows_before = len(df)
            df = df.dropna(subset=["Adj Close", "High", "Low", "Volume"])
            rows_after  = len(df)

            print(f"  OK — {rows_after} righe "
                  f"({df['Date'].min().date()} → {df['Date'].max().date()})")

            all_data.append(df)
            metadata.append({
                "ticker":       ticker,
                "first_date":   df["Date"].min().date(),
                "last_date":    df["Date"].max().date(),
                "total_rows":   rows_after,
                "rows_dropped": rows_before - rows_after,
            })

        except Exception as e:
            print(f"  ERRORE su {ticker}: {e}")
            failed.append({"ticker": ticker, "reason": str(e)})

    prices_df  = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    failed_df  = pd.DataFrame(failed)
    metadata_df= pd.DataFrame(metadata)

    return prices_df, failed_df, metadata_df


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    prices_df, failed_df, metadata_df = download_prices(FTSE_MIB_TICKERS)

    if not prices_df.empty:
        prices_df.to_csv(PRICES_FILE, index=False)
        print(f"\n✅ Prezzi salvati:  {PRICES_FILE}")
        print(f"   Ticker scaricati: {prices_df['ticker'].nunique()}")
        print(f"   Righe totali:     {len(prices_df)}")
        print(f"   Periodo:          {prices_df['Date'].min()} → {prices_df['Date'].max()}")
    else:
        print("ERRORE: nessun dato scaricato.")

    if not failed_df.empty:
        failed_df.to_csv(FAILED_FILE, index=False)
        print(f"\n⚠️  Ticker falliti: {FAILED_FILE}")
        print(failed_df.to_string(index=False))

    metadata_df.to_csv(METADATA_FILE, index=False)
    print(f"\n📋 Metadata: {METADATA_FILE}")
    print(metadata_df.to_string(index=False))