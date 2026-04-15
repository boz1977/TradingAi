"""
signal_history.py  —  Storico segnali con risultato
Ogni volta che daily_screener genera un segnale, questo modulo
lo salva in uno storico persistente e, dopo la chiusura del trade,
ne registra il risultato finale (win/loss/aperto).

Utilizzo da daily_screener.py (aggiungere alla fine):
    from signal_history import record_signals
    record_signals(df_signals)

Utilizzo standalone per aggiornare i risultati:
    python src/signal_history.py --update
"""

import os, sys, json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR     = Path(__file__).parent.parent / "data" / "processed"
HISTORY_FILE = DATA_DIR / "signal_history.csv"

STOP_LOSS_PCT     = 0.04
TAKE_PROFIT_PCT   = 0.12
TRAILING_STOP_PCT = 0.06
VIX_EXIT          = 25.0
TRANSACTION_COST  = 0.0015
MAX_HOLDING_DAYS  = 120


# =============================================================================
# SALVA NUOVI SEGNALI
# =============================================================================
def record_signals(signals_df: pd.DataFrame, overwrite_today: bool = True):
    """
    Aggiunge i segnali odierni allo storico.
    Se il ticker è già presente oggi, non lo duplica.
    """
    if signals_df.empty:
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    new_rows = []
    for _, row in signals_df.iterrows():
        new_rows.append({
            "signal_date":   row.get("date", today),
            "ticker":        row["ticker"],
            "sector":        row.get("sector", ""),
            "entry_price":   row.get("close"),
            "entry_score":   row.get("entry_score"),
            "ai_prob":       row.get("ai_prob"),
            "signal_type":   row.get("signal", "OK"),
            "rsi":           row.get("rsi"),
            "momentum_3m":   row.get("momentum_3m"),
            "vix":           row.get("vix"),
            "status":        "open",        # open / win / loss / expired
            "exit_date":     None,
            "exit_price":    None,
            "exit_reason":   None,
            "net_return_pct":None,
            "holding_days":  None,
        })

    new_df = pd.DataFrame(new_rows)

    if HISTORY_FILE.exists():
        existing = pd.read_csv(HISTORY_FILE)
        if overwrite_today:
            # rimuovi righe di oggi per evitare duplicati
            existing = existing[existing["signal_date"] != today]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(HISTORY_FILE, index=False)
    print(f"[OK] Storico aggiornato: {len(new_df)} segnali aggiunti ({HISTORY_FILE})")


# =============================================================================
# AGGIORNA RISULTATI (scarica prezzi e calcola outcome)
# =============================================================================
def update_results(max_age_days: int = MAX_HOLDING_DAYS):
    """
    Per ogni segnale 'open' nello storico, scarica i prezzi successivi
    e calcola se è diventato win, loss o è ancora aperto.
    """
    if not HISTORY_FILE.exists():
        print("[WARN] Nessuno storico trovato.")
        return pd.DataFrame()

    import yfinance as yf

    history = pd.read_csv(HISTORY_FILE)
    history["signal_date"] = pd.to_datetime(history["signal_date"])

    open_signals = history[history["status"] == "open"].copy()
    if open_signals.empty:
        print("[OK] Nessun segnale aperto da aggiornare.")
        return history

    print(f"Aggiornamento {len(open_signals)} segnali aperti...")

    # Scarica prezzi per i ticker necessari
    tickers_needed = open_signals["ticker"].unique().tolist()
    today = datetime.now()

    for ticker in tickers_needed:
        rows = open_signals[open_signals["ticker"] == ticker]
        oldest_date = rows["signal_date"].min()
        start = (oldest_date - timedelta(days=2)).strftime("%Y-%m-%d")

        try:
            df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
            if df.empty:
                continue
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.index   = pd.to_datetime(df.index).tz_localize(None)
        except Exception as e:
            print(f"  [ERRORE] {ticker}: {e}")
            continue

        for idx, row in rows.iterrows():
            signal_date  = pd.Timestamp(row["signal_date"])
            entry_price  = row["entry_price"]

            # Trova prezzi dopo la data del segnale
            future = df[df.index > signal_date]["Adj Close"]
            if future.empty or pd.isna(entry_price):
                continue

            stop_price     = entry_price * (1 - STOP_LOSS_PCT)
            target_price   = entry_price * (1 + TAKE_PROFIT_PCT)
            trailing_active= False
            highest_price  = entry_price
            current_stop   = stop_price

            exit_date   = None
            exit_price  = None
            exit_reason = None

            for date, price in future.items():
                holding = (date - signal_date).days
                if holding > max_age_days:
                    exit_date, exit_price, exit_reason = date, price, "expired"
                    break

                # trailing activation
                if not trailing_active and price >= target_price:
                    trailing_active = True
                    highest_price   = price
                    current_stop    = highest_price * (1 - TRAILING_STOP_PCT)
                elif trailing_active and price > highest_price:
                    highest_price = price
                    current_stop  = highest_price * (1 - TRAILING_STOP_PCT)

                # stop hit
                if price <= current_stop:
                    exit_reason = "trailing_stop" if trailing_active else "stop_loss"
                    exit_date, exit_price = date, price
                    break

            # ancora aperto
            if exit_date is None:
                age = (today - signal_date).days
                if age <= max_age_days:
                    last_price = float(future.iloc[-1])
                    cur_ret = (last_price / entry_price - 1) * 100
                    history.at[idx, "exit_price"]     = last_price
                    history.at[idx, "net_return_pct"] = round(cur_ret - TRANSACTION_COST * 2 * 100, 2)
                    history.at[idx, "holding_days"]   = age
                    # status rimane "open"
                    continue

            if exit_date:
                gross_ret = (exit_price / entry_price - 1) * 100
                net_ret   = gross_ret - TRANSACTION_COST * 2 * 100
                status    = "win" if net_ret > 0 else "loss"

                history.at[idx, "status"]        = status
                history.at[idx, "exit_date"]     = exit_date.strftime("%Y-%m-%d")
                history.at[idx, "exit_price"]    = round(exit_price, 4)
                history.at[idx, "exit_reason"]   = exit_reason
                history.at[idx, "net_return_pct"]= round(net_ret, 2)
                history.at[idx, "holding_days"]  = (exit_date - signal_date).days

    history.to_csv(HISTORY_FILE, index=False)
    closed = history[history["status"] != "open"]
    print(f"[OK] Storico aggiornato: {len(closed)} trade chiusi su {len(history)} totali")
    return history


# =============================================================================
# STATISTICHE STORICO
# =============================================================================
def get_stats(history: pd.DataFrame) -> dict:
    if history.empty:
        return {}

    closed = history[history["status"].isin(["win","loss"])]
    if closed.empty:
        return {"total": len(history), "open": len(history), "closed": 0}

    wins   = closed[closed["status"] == "win"]
    losses = closed[closed["status"] == "loss"]
    pf     = wins["net_return_pct"].sum() / abs(losses["net_return_pct"].sum()) \
             if not losses.empty and losses["net_return_pct"].sum() != 0 else float("inf")

    return {
        "total":        len(history),
        "open":         len(history[history["status"] == "open"]),
        "closed":       len(closed),
        "wins":         len(wins),
        "losses":       len(losses),
        "win_rate":     len(wins) / len(closed) * 100,
        "profit_factor":pf,
        "avg_return":   closed["net_return_pct"].mean(),
        "best":         closed["net_return_pct"].max(),
        "worst":        closed["net_return_pct"].min(),
        "by_signal":    {
            "FORTE": closed[closed["signal_type"]=="FORTE"]["net_return_pct"].mean() if "FORTE" in closed["signal_type"].values else None,
            "OK":    closed[closed["signal_type"]=="OK"]["net_return_pct"].mean()    if "OK"    in closed["signal_type"].values else None,
        },
        "by_score": closed.groupby("entry_score")["net_return_pct"].mean().to_dict() if "entry_score" in closed.columns else {},
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="Aggiorna risultati segnali aperti")
    args = parser.parse_args()

    if args.update:
        history = update_results()
        stats   = get_stats(history)
        print("\n=== STATISTICHE STORICO ===")
        for k, v in stats.items():
            if not isinstance(v, dict):
                print(f"  {k}: {v}")
    else:
        if HISTORY_FILE.exists():
            df = pd.read_csv(HISTORY_FILE)
            print(f"Storico: {len(df)} segnali")
            print(df.tail(10).to_string(index=False))
        else:
            print("Nessuno storico trovato. Lancia daily_screener.py per iniziare.")
