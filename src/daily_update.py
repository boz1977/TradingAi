"""
daily_update.py  —  Aggiornamento serale completo
Script da lanciare ogni sera dopo la chiusura del mercato (18:30).

Fa tre cose in sequenza:
  1. Aggiorna le posizioni reali (prezzi, P&L, alert)
  2. Genera i nuovi segnali per domani
  3. Manda un'email con tutto: posizioni aperte + alert + nuovi segnali

Questo è lo script che va nel GitHub Actions / Task Scheduler.

Utilizzo:
    python src/daily_update.py
"""

import os, sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = Path(__file__).parent.parent


def run_daily_update():
    print("=" * 60)
    print(f"AGGIORNAMENTO SERALE — {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 60)

    # 1. Aggiorna posizioni reali
    print("\n[1/3] Aggiornamento posizioni reali...")
    try:
        from real_portfolio import RealPortfolio
        rp     = RealPortfolio()
        alerts = rp.update_all()
        summary= rp.get_summary()
        open_positions = rp.get_open_positions()
        print(f"  Posizioni aperte: {summary['n_open']}, Alert: {len(alerts)}")
    except Exception as e:
        print(f"  [WARN] Portafoglio reale non disponibile: {e}")
        alerts = []
        summary = {}
        open_positions = None
        rp = None

    # 2. Genera nuovi segnali
    print("\n[2/3] Generazione segnali per domani...")
    signals_df = None
    try:
        from daily_screener import run_screener
        signals_df = run_screener()
        n_segnali  = len(signals_df) if signals_df is not None and not signals_df.empty else 0
        print(f"  Segnali trovati: {n_segnali}")
    except Exception as e:
        print(f"  [WARN] Screener non disponibile: {e}")
        import pandas as pd
        signals_df = pd.DataFrame()

    # 3. Manda email
    print("\n[3/3] Invio email...")
    try:
        from notify_email import send_daily_email
        send_daily_email(
            signals_df     = signals_df,
            open_positions = open_positions,
            alerts         = alerts,
            portfolio_summary = summary,
        )
    except Exception as e:
        print(f"  [WARN] Email non inviata: {e}")

    print("\n[OK] Aggiornamento serale completato")


if __name__ == "__main__":
    run_daily_update()