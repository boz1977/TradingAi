import os
import pandas as pd
import yfinance as yf
from src.tickers import FTSE_MIB_TICKERS


def download_prices(tickers, start="2015-01-01"):
    all_data = []
    failed = []

    for ticker in tickers:
        try:
            print(f"Downloading {ticker}...")
            df = yf.download(ticker, start=start, auto_adjust=False, progress=False)

            if df.empty:
                print(f"WARNING: nessun dato per {ticker}")
                failed.append({"ticker": ticker, "reason": "empty dataframe"})
                continue

            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df.reset_index(inplace=True)
            df["ticker"] = ticker

            all_data.append(df)

        except Exception as e:
            print(f"ERRORE su {ticker}: {e}")
            failed.append({"ticker": ticker, "reason": str(e)})

    if not all_data:
        raise Exception("Nessun dato scaricato")

    prices_df = pd.concat(all_data, ignore_index=True)
    failed_df = pd.DataFrame(failed)

    return prices_df, failed_df


if __name__ == "__main__":
    prices_df, failed_df = download_prices(FTSE_MIB_TICKERS)

    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    prices_file = os.path.join(output_dir, "prices.csv")
    failed_file = os.path.join(output_dir, "failed_tickers.csv")

    prices_df.to_csv(prices_file, index=False)
    failed_df.to_csv(failed_file, index=False)

    print(f"\n✅ Prezzi salvati in: {prices_file}")
    print(f"✅ Ticker falliti salvati in: {failed_file}")
    print(f"✅ Ticker scaricati: {prices_df['ticker'].nunique()}")