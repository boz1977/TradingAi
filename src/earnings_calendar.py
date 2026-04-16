"""
earnings_calendar.py  —  Fase 1
Scarica e gestisce il calendario degli earnings trimestrali.

Integrato nel daily_screener: se un titolo ha earnings nei prossimi
N giorni, il segnale viene soppresso o marcato come "rischio earnings".

Fonti:
  - yfinance  (ticker.calendar — gratuito, affidabile)
  - Fallback manuale per titoli FTSE MIB non coperti da yfinance

Utilizzo standalone:
    python src/earnings_calendar.py             # aggiorna tutti i ticker
    python src/earnings_calendar.py --check     # mostra earnings prossimi 14 giorni
"""

import os, sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import DB
from tickers import FTSE_MIB_TICKERS

DAYS_AHEAD_BLOCK = 3   # blocca segnali se earnings entro N giorni
DAYS_AHEAD_WARN  = 7   # warning se earnings entro N giorni


# =============================================================================
# DOWNLOAD EARNINGS DA YFINANCE
# =============================================================================
def fetch_earnings_for_ticker(ticker: str) -> list:
    """
    Scarica le prossime date di earnings per un ticker via yfinance.
    Ritorna lista di dict con 'date' e 'period'.
    """
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar

        # yfinance ritorna un dict o DataFrame a seconda della versione
        if cal is None:
            return []

        dates = []

        # Formato dict (versione recente yfinance)
        if isinstance(cal, dict):
            earnings_date = cal.get("Earnings Date")
            if earnings_date:
                if isinstance(earnings_date, list):
                    for d in earnings_date:
                        dates.append({
                            "date":   pd.Timestamp(d).strftime("%Y-%m-%d"),
                            "period": "Q",
                        })
                else:
                    dates.append({
                        "date":   pd.Timestamp(earnings_date).strftime("%Y-%m-%d"),
                        "period": "Q",
                    })

        # Formato DataFrame (versione vecchia yfinance)
        elif isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.columns:
                for d in cal["Earnings Date"]:
                    if pd.notna(d):
                        dates.append({
                            "date":   pd.Timestamp(d).strftime("%Y-%m-%d"),
                            "period": "Q",
                        })

        return dates

    except Exception as e:
        print(f"  [WARN] {ticker}: {e}")
        return []


def update_earnings_calendar(
    tickers: list = None,
    db: DB = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Aggiorna il calendario earnings per tutti i ticker nel DB.
    Tipicamente da chiamare una volta a settimana.
    """
    tickers = tickers or FTSE_MIB_TICKERS
    db      = db or DB()

    if verbose:
        print(f"Aggiornamento earnings calendar per {len(tickers)} ticker...")

    all_dates = []
    for ticker in tickers:
        dates = fetch_earnings_for_ticker(ticker)
        if dates:
            db.save_earnings(ticker, dates)
            for d in dates:
                all_dates.append({"ticker": ticker, **d})
            if verbose:
                print(f"  [OK] {ticker}: {[d['date'] for d in dates]}")
        else:
            if verbose:
                print(f"  [ ] {ticker}: nessuna data trovata")

    return pd.DataFrame(all_dates) if all_dates else pd.DataFrame()


# =============================================================================
# CONTROLLO SEGNALI
# =============================================================================
def check_earnings_risk(
    ticker: str,
    signal_date: datetime = None,
    db: DB = None,
    days_block: int = DAYS_AHEAD_BLOCK,
    days_warn:  int = DAYS_AHEAD_WARN,
) -> dict:
    """
    Controlla se un ticker ha earnings nei prossimi giorni.

    Ritorna:
        {
            "block":  True/False — sopprimi il segnale
            "warn":   True/False — mostra warning ma lascia passare
            "date":   "2024-04-15" — data earnings più vicina
            "days":   5 — giorni al prossimo earnings
            "reason": "earnings in 3 giorni"
        }
    """
    db          = db or DB()
    signal_date = signal_date or datetime.now()

    result = {"block": False, "warn": False, "date": None, "days": None, "reason": None}

    earnings = db.has_earnings_soon(ticker, days_ahead=days_warn)
    if not earnings:
        return result

    report_date = datetime.strptime(earnings["report_date"], "%Y-%m-%d")
    days_to     = (report_date - signal_date).days

    result["date"] = earnings["report_date"]
    result["days"] = days_to

    if days_to <= days_block:
        result["block"]  = True
        result["reason"] = f"earnings tra {days_to} giorni ({earnings['report_date']})"
    elif days_to <= days_warn:
        result["warn"]   = True
        result["reason"] = f"earnings tra {days_to} giorni ({earnings['report_date']})"

    return result


def get_upcoming_earnings(
    tickers: list = None,
    days_ahead: int = 14,
    db: DB = None,
) -> pd.DataFrame:
    """
    Ritorna tutti gli earnings nei prossimi N giorni per i ticker specificati.
    Utile per la dashboard.
    """
    db      = db or DB()
    tickers = tickers or FTSE_MIB_TICKERS
    today   = datetime.now().strftime("%Y-%m-%d")
    end     = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    df = db.get_earnings_in_range(today, end, tickers)
    if df.empty:
        return df

    df["days_to_earnings"] = df["report_date"].apply(
        lambda d: (datetime.strptime(d, "%Y-%m-%d") - datetime.now()).days
    )
    return df.sort_values("days_to_earnings")


# =============================================================================
# INTEGRAZIONE CON DAILY SCREENER
# =============================================================================
def filter_signals_for_earnings(
    signals_df: pd.DataFrame,
    db: DB = None,
    days_block: int = DAYS_AHEAD_BLOCK,
    days_warn:  int = DAYS_AHEAD_WARN,
) -> pd.DataFrame:
    """
    Filtra/annota i segnali in base agli earnings imminenti.

    - Rimuove i segnali con earnings entro DAYS_AHEAD_BLOCK giorni
    - Aggiunge colonna 'earnings_warning' per quelli entro DAYS_AHEAD_WARN
    """
    if signals_df.empty:
        return signals_df

    db = db or DB()
    filtered = []
    blocked  = []

    for _, row in signals_df.iterrows():
        ticker = row["ticker"]
        risk   = check_earnings_risk(ticker, db=db,
                                     days_block=days_block, days_warn=days_warn)

        if risk["block"]:
            blocked.append(f"{ticker} ({risk['reason']})")
            continue

        row_copy = row.copy()
        row_copy["earnings_warning"] = risk["reason"] if risk["warn"] else None
        filtered.append(row_copy)

    if blocked:
        print(f"  [OK] Segnali bloccati per earnings: {', '.join(blocked)}")

    return pd.DataFrame(filtered) if filtered else pd.DataFrame()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check",  action="store_true",
                        help="Mostra earnings nei prossimi 14 giorni")
    parser.add_argument("--days",   type=int, default=14,
                        help="Giorni in avanti da controllare")
    args = parser.parse_args()

    db = DB()

    if args.check:
        print(f"\nEarnings nei prossimi {args.days} giorni:")
        df = get_upcoming_earnings(days_ahead=args.days, db=db)
        if df.empty:
            print("  Nessuno — aggiorna prima il calendario con python src/earnings_calendar.py")
        else:
            print(df[["ticker","report_date","days_to_earnings"]].to_string(index=False))
    else:
        print("=" * 50)
        print("AGGIORNAMENTO EARNINGS CALENDAR — Fase 1")
        print("=" * 50)
        df = update_earnings_calendar(db=db)
        print(f"\n[OK] {len(df)} date earnings salvate nel DB")

        upcoming = get_upcoming_earnings(days_ahead=14, db=db)
        if not upcoming.empty:
            print(f"\nProssimi earnings (14 giorni):")
            print(upcoming[["ticker","report_date","days_to_earnings"]].to_string(index=False))