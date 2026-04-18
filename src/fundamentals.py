"""
fundamentals.py  —  Fase 2
Scarica e gestisce i dati fondamentali per i titoli italiani.

Indicatori scaricati da yfinance:
  - P/E ratio (trailing e forward)
  - P/B ratio (price to book)
  - Dividend yield
  - EPS (earnings per share)
  - Revenue growth YoY
  - Debt/Equity ratio
  - ROE (return on equity)
  - Free cash flow yield

Utilizzo nel modello:
  - Filtra titoli fondamentalmente "sani" prima dell'entry score tecnico
  - Aggiunge feature al modello AI (P/E relativo al settore, dividend yield)
  - Evita ingressi su titoli con debt/equity molto alto

Utilizzo:
    python src/fundamentals.py
    python src/fundamentals.py --ticker ENI.MI
"""

import os, sys, json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import DB
from tickers import ITALY_LIQUID_TICKERS, SECTOR_MAP

ROOT       = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FUND_FILE  = OUTPUT_DIR / "fundamentals.csv"

# Soglie fondamentali per il filtro (configurabili)
MAX_PE_RATIO      = 40.0   # P/E > 40 = molto caro
MIN_PE_RATIO      = 3.0    # P/E < 3 = sospetto o loss
MAX_DEBT_EQUITY   = 3.0    # D/E > 3 = troppo indebitato (escludi banche)
MIN_ROE           = -0.05  # ROE < -5% = azienda in perdita strutturale

# Settori esclusi dal filtro D/E (banche e assicurazioni usano leva per natura)
FINANCIAL_SECTORS = {"financials"}


# =============================================================================
# DOWNLOAD FONDAMENTALI
# =============================================================================
def fetch_fundamentals(ticker: str) -> dict:
    """
    Scarica i fondamentali per un singolo ticker da yfinance.
    Ritorna un dizionario con tutti i valori disponibili.
    """
    result = {
        "ticker":           ticker,
        "fetched_at":       datetime.now().strftime("%Y-%m-%d"),
        "pe_trailing":      None,
        "pe_forward":       None,
        "pb_ratio":         None,
        "dividend_yield":   None,
        "eps_ttm":          None,
        "eps_forward":      None,
        "revenue_growth":   None,
        "earnings_growth":  None,
        "debt_equity":      None,
        "roe":              None,
        "roa":              None,
        "current_ratio":    None,
        "free_cashflow":    None,
        "market_cap":       None,
        "beta":             None,
        "52w_high":         None,
        "52w_low":          None,
        "avg_volume":       None,
        "sector":           SECTOR_MAP.get(ticker, "other"),
    }

    try:
        t    = yf.Ticker(ticker)
        info = t.info

        if not info:
            return result

        # Valutazione
        result["pe_trailing"]    = info.get("trailingPE")
        result["pe_forward"]     = info.get("forwardPE")
        result["pb_ratio"]       = info.get("priceToBook")
        result["dividend_yield"] = info.get("dividendYield")
        result["eps_ttm"]        = info.get("trailingEps")
        result["eps_forward"]    = info.get("forwardEps")

        # Crescita
        result["revenue_growth"]  = info.get("revenueGrowth")
        result["earnings_growth"] = info.get("earningsGrowth")

        # Solidità finanziaria
        result["debt_equity"]   = info.get("debtToEquity")
        result["roe"]           = info.get("returnOnEquity")
        result["roa"]           = info.get("returnOnAssets")
        result["current_ratio"] = info.get("currentRatio")
        result["free_cashflow"] = info.get("freeCashflow")

        # Mercato
        result["market_cap"]    = info.get("marketCap")
        result["beta"]          = info.get("beta")
        result["52w_high"]      = info.get("fiftyTwoWeekHigh")
        result["52w_low"]       = info.get("fiftyTwoWeekLow")
        result["avg_volume"]    = info.get("averageVolume")

    except Exception as e:
        print(f"  [WARN] {ticker}: {e}")

    return result


def fetch_all_fundamentals(
    tickers: list = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Scarica fondamentali per tutti i ticker.
    """
    tickers = tickers or ITALY_LIQUID_TICKERS
    results = []

    for i, ticker in enumerate(tickers):
        if verbose:
            print(f"  [{i+1}/{len(tickers)}] {ticker}...", end=" ")
        data = fetch_fundamentals(ticker)
        results.append(data)
        if verbose:
            pe = data.get("pe_trailing")
            dy = data.get("dividend_yield")
            print(f"PE={pe:.1f if pe else 'N/A'}  DY={dy*100:.1f if dy else 'N/A'}%")

    df = pd.DataFrame(results)
    df.to_csv(FUND_FILE, index=False)
    if verbose:
        print(f"\n[OK] Fondamentali salvati: {FUND_FILE}")
    return df


# =============================================================================
# ARRICCHIMENTO CON METRICHE RELATIVE AL SETTORE
# =============================================================================
def add_sector_relative_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge metriche relative al settore:
    - pe_vs_sector:  P/E del titolo vs mediana del settore
    - dy_vs_sector:  dividend yield vs mediana del settore
    Un P/E sotto la mediana settoriale = titolo relativamente economico.
    """
    df = df.copy()

    for metric in ["pe_trailing", "dividend_yield", "pb_ratio"]:
        col_rel = f"{metric}_vs_sector"
        df[col_rel] = np.nan

        for sector in df["sector"].dropna().unique():
            mask   = df["sector"] == sector
            values = df.loc[mask, metric].dropna()
            if len(values) < 2:
                continue
            median = values.median()
            if median != 0 and not np.isnan(median):
                df.loc[mask, col_rel] = df.loc[mask, metric] / median - 1

    return df


# =============================================================================
# FILTRO FONDAMENTALE
# =============================================================================
def fundamental_filter(
    ticker: str,
    fund_df: pd.DataFrame = None,
    verbose: bool = False,
) -> dict:
    """
    Applica i filtri fondamentali a un singolo ticker.

    Ritorna:
        {
            "pass":   True/False
            "score":  0-5 (punteggio qualità fondamentale)
            "reason": stringa con motivazione se filtrato
            "flags":  lista di warning
        }
    """
    result = {"pass": True, "score": 0, "reason": None, "flags": []}

    if fund_df is None:
        if FUND_FILE.exists():
            fund_df = pd.read_csv(FUND_FILE)
        else:
            # Nessun dato: passa comunque (non bloccare per mancanza dati)
            result["flags"].append("fondamentali non disponibili")
            return result

    row = fund_df[fund_df["ticker"] == ticker]
    if row.empty:
        result["flags"].append("ticker non trovato nei fondamentali")
        return result

    row = row.iloc[0]
    sector = row.get("sector", "other")
    is_financial = sector in FINANCIAL_SECTORS

    # ── Controlli ──
    pe = row.get("pe_trailing")
    if pd.notna(pe):
        if pe > MAX_PE_RATIO:
            result["flags"].append(f"P/E alto ({pe:.1f} > {MAX_PE_RATIO})")
        elif pe < MIN_PE_RATIO and pe > 0:
            result["flags"].append(f"P/E anomalo ({pe:.1f})")
        elif pe > 0 and pe <= 20:
            result["score"] += 2   # P/E ragionevole
        elif pe > 0:
            result["score"] += 1

    # Debt/Equity — ignora per banche/assicurazioni
    de = row.get("debt_equity")
    if pd.notna(de) and not is_financial:
        if de > MAX_DEBT_EQUITY * 100:  # yfinance restituisce in percentuale
            result["pass"]   = False
            result["reason"] = f"Debt/Equity troppo alto ({de:.0f}%)"
            return result
        elif de <= 100:
            result["score"] += 1

    # ROE
    roe = row.get("roe")
    if pd.notna(roe):
        if roe < MIN_ROE:
            result["flags"].append(f"ROE negativo ({roe*100:.1f}%)")
        elif roe > 0.10:
            result["score"] += 1
        if roe > 0.20:
            result["score"] += 1   # ROE eccellente

    # Dividend yield (bonus)
    dy = row.get("dividend_yield")
    if pd.notna(dy) and dy > 0.02:
        result["score"] += 1   # dividendo significativo

    # Revenue growth
    rg = row.get("revenue_growth")
    if pd.notna(rg) and rg > 0:
        result["score"] += 1

    if verbose:
        print(f"  {ticker}: pass={result['pass']} score={result['score']} flags={result['flags']}")

    return result


def filter_universe_by_fundamentals(
    tickers: list,
    min_score: int = 2,
    fund_df: pd.DataFrame = None,
    verbose: bool = True,
) -> list:
    """
    Filtra una lista di ticker applicando i criteri fondamentali.
    Ritorna solo i ticker che passano il filtro con score >= min_score.
    """
    if fund_df is None and FUND_FILE.exists():
        fund_df = pd.read_csv(FUND_FILE)

    approved  = []
    rejected  = []

    for ticker in tickers:
        check = fundamental_filter(ticker, fund_df, verbose=False)
        if check["pass"] and check["score"] >= min_score:
            approved.append(ticker)
        else:
            reason = check["reason"] or f"score basso ({check['score']}/{min_score})"
            rejected.append(f"{ticker} ({reason})")

    if verbose:
        print(f"\nFiltro fondamentale:")
        print(f"  Approvati: {len(approved)}/{len(tickers)}")
        if rejected:
            print(f"  Esclusi:   {', '.join(rejected[:10])}")

    return approved


# =============================================================================
# FEATURE PER IL MODELLO AI
# =============================================================================
def get_fundamental_features(ticker: str, fund_df: pd.DataFrame = None) -> dict:
    """
    Estrae le feature fondamentali per il modello AI.
    Usato in build_ai_features() nel daily_screener.
    """
    defaults = {
        "pe_trailing":          15.0,
        "pe_vs_sector":         0.0,
        "dividend_yield":       0.03,
        "dy_vs_sector":         0.0,
        "pb_ratio":             1.5,
        "debt_equity_norm":     1.0,
        "roe":                  0.10,
        "revenue_growth":       0.05,
        "fundamental_score":    3,
        "has_dividend":         1,
    }

    if fund_df is None:
        if FUND_FILE.exists():
            fund_df = pd.read_csv(FUND_FILE)
        else:
            return defaults

    row = fund_df[fund_df["ticker"] == ticker]
    if row.empty:
        return defaults

    row   = row.iloc[0]
    check = fundamental_filter(ticker, fund_df)

    pe  = row.get("pe_trailing")
    dy  = row.get("dividend_yield")
    de  = row.get("debt_equity")
    roe = row.get("roe")
    rg  = row.get("revenue_growth")

    return {
        "pe_trailing":       float(pe)  if pd.notna(pe)  else 15.0,
        "pe_vs_sector":      float(row.get("pe_trailing_vs_sector", 0)) if pd.notna(row.get("pe_trailing_vs_sector")) else 0.0,
        "dividend_yield":    float(dy)  if pd.notna(dy)  else 0.0,
        "dy_vs_sector":      float(row.get("dividend_yield_vs_sector", 0)) if pd.notna(row.get("dividend_yield_vs_sector")) else 0.0,
        "pb_ratio":          float(row.get("pb_ratio", 1.5)) if pd.notna(row.get("pb_ratio")) else 1.5,
        "debt_equity_norm":  min(float(de) / 100, 5.0) if pd.notna(de) else 1.0,
        "roe":               float(roe) if pd.notna(roe) else 0.0,
        "revenue_growth":    float(rg)  if pd.notna(rg)  else 0.0,
        "fundamental_score": check["score"],
        "has_dividend":      int(pd.notna(dy) and dy > 0.01),
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--quick",  action="store_true",
                        help="Solo FTSE MIB (31 titoli)")
    args = parser.parse_args()

    print("=" * 55)
    print("DOWNLOAD FONDAMENTALI — Fase 2")
    print("=" * 55)

    from tickers import FTSE_MIB_TICKERS, ITALY_LIQUID_TICKERS

    if args.ticker:
        data = fetch_fundamentals(args.ticker)
        print(json.dumps({k: v for k, v in data.items() if v is not None}, indent=2))
    else:
        tickers = FTSE_MIB_TICKERS if args.quick else ITALY_LIQUID_TICKERS
        print(f"\nScaricamento fondamentali per {len(tickers)} titoli...")
        df = fetch_all_fundamentals(tickers)
        df = add_sector_relative_metrics(df)
        df.to_csv(FUND_FILE, index=False)

        print(f"\n=== TOP 10 PER QUALITA' FONDAMENTALE ===")
        approved = filter_universe_by_fundamentals(tickers, min_score=3, fund_df=df)
        print(f"Titoli con score >= 3: {len(approved)}")
        print(approved)
