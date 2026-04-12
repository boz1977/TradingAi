"""
download_macro_extended.py  —  Fase 4
Scarica serie macro estese dal 2000 da FRED e yfinance.

Serie scaricate:
  Da FRED:
    VIX        — volatilità implicita SP500
    SP500      — indice S&P 500
    BRENT      — petrolio Brent
    IT10Y      — rendimento BTP 10 anni (proxy rischio Italia)
    DE10Y      — rendimento Bund 10 anni (benchmark)

  Da yfinance (non disponibili su FRED):
    FTSEMIB    — indice FTSE MIB (.FTMIB o ^FTSEMIB)
    EURUSD     — cambio Euro/Dollaro

  Derivate (calcolate):
    BTP_BUND_SPREAD — IT10Y - DE10Y (rischio paese Italia)

Il dataset macro viene unito ai prezzi per ticker tramite la colonna Date.

Utilizzo:
    python src/download_macro_extended.py
"""

import os
import requests
import pandas as pd
import yfinance as yf


FRED_API_KEY = "1677d4a6bba57761a61c7caa5fca1b12"
FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES = {
    "VIX":   "VIXCLS",
    "SP500": "SP500",
    "BRENT": "DCOILBRENTEU",
    "IT10Y": "IRLTLT01ITM156N",   # rendimento BTP 10Y mensile Italia
    "DE10Y": "IRLTLT01DEM156N",   # rendimento Bund 10Y mensile Germania
}

YF_SERIES = {
    "FTSEMIB": "FTSEMIB.MI",
    "EURUSD":  "EURUSD=X",
}

OUTPUT_DIR   = "data/raw"
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "macro_extended.csv")
FAILED_FILE  = os.path.join(OUTPUT_DIR, "macro_failed_extended.csv")


def _download_fred(series_name: str, series_id: str) -> pd.DataFrame | None:
    params = {
        "series_id": series_id,
        "api_key":   FRED_API_KEY,
        "file_type": "json",
    }
    try:
        r = requests.get(FRED_BASE, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if "observations" not in payload:
            print(f"  Risposta inattesa FRED per {series_id}")
            return None
        df = pd.DataFrame(payload["observations"])[["date", "value"]].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"]  = pd.to_datetime(df["date"])
        df = df.dropna(subset=["value"])
        df.columns = ["Date", series_name]
        print(f"  FRED {series_name} ({series_id}): {len(df)} righe "
              f"({df['Date'].min().date()} → {df['Date'].max().date()})")
        return df
    except Exception as e:
        print(f"  ERRORE FRED {series_name}: {e}")
        return None


def _download_yf(series_name: str, ticker_symbol: str, start: str = "2000-01-01") -> pd.DataFrame | None:
    try:
        df = yf.download(ticker_symbol, start=start, auto_adjust=True, progress=False)
        if df.empty:
            print(f"  WARNING yfinance {series_name}: nessun dato")
            return None
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df[["Close"]].copy()
        df.index.name = "Date"
        df = df.reset_index()
        df.columns = ["Date", series_name]
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df.dropna()
        print(f"  yfinance {series_name} ({ticker_symbol}): {len(df)} righe "
              f"({df['Date'].min().date()} → {df['Date'].max().date()})")
        return df
    except Exception as e:
        print(f"  ERRORE yfinance {series_name}: {e}")
        return None


def download_all_macro() -> tuple:
    dfs    = []
    failed = []

    print("\n--- FRED ---")
    for name, series_id in FRED_SERIES.items():
        df = _download_fred(name, series_id)
        if df is not None:
            dfs.append(df)
        else:
            failed.append({"series": name, "source": "FRED", "id": series_id})

    print("\n--- yfinance ---")
    for name, symbol in YF_SERIES.items():
        df = _download_yf(name, symbol)
        if df is not None:
            dfs.append(df)
        else:
            failed.append({"series": name, "source": "yfinance", "id": symbol})

    if not dfs:
        raise RuntimeError("Nessuna serie macro scaricata.")

    # Unisci tutto su Date con outer join
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="Date", how="outer")

    merged = merged.sort_values("Date").reset_index(drop=True)

    # Calcola BTP/Bund spread
    if "IT10Y" in merged.columns and "DE10Y" in merged.columns:
        merged["BTP_BUND_SPREAD"] = merged["IT10Y"] - merged["DE10Y"]
        print(f"\n  Calcolato BTP_BUND_SPREAD "
              f"(media: {merged['BTP_BUND_SPREAD'].mean():.2f}%)")

    return merged, pd.DataFrame(failed)


def build_daily_macro(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Le serie FRED sono spesso mensili o hanno buchi nei weekend.
    Espande a frequenza giornaliera con forward-fill.
    """
    # Range date completo dal minimo al massimo
    date_range = pd.date_range(
        start=macro_df["Date"].min(),
        end=macro_df["Date"].max(),
        freq="D"
    )
    daily = pd.DataFrame({"Date": date_range})
    daily = daily.merge(macro_df, on="Date", how="left")

    # Forward fill per i weekend e i giorni mancanti
    numeric_cols = daily.select_dtypes(include="number").columns
    daily[numeric_cols] = daily[numeric_cols].ffill()

    return daily


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 55)
    print("DOWNLOAD MACRO ESTESO — Fase 4")
    print("=" * 55)

    macro_raw, failed_df = download_all_macro()

    # Versione daily con ffill
    macro_daily = build_daily_macro(macro_raw)

    macro_daily.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Macro salvate:  {OUTPUT_FILE}")
    print(f"   Righe totali:   {len(macro_daily)}")
    print(f"   Colonne:        {list(macro_daily.columns)}")
    print(f"   Periodo:        {macro_daily['Date'].min().date()} → {macro_daily['Date'].max().date()}")

    if not failed_df.empty:
        failed_df.to_csv(FAILED_FILE, index=False)
        print(f"\n⚠️  Serie non scaricate: {FAILED_FILE}")
        print(failed_df.to_string(index=False))

    # Anteprima
    print(f"\nAnteprima ultime 5 righe:")
    print(macro_daily.tail().to_string(index=False))