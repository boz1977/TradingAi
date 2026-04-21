"""
real_portfolio.py  —  Portafoglio Reale
Gestisce le posizioni reali aperte dall'utente.

Flusso:
  1. Utente vede segnale nell'app → decide di entrare
  2. Registra l'operazione: ticker, prezzo ingresso, size, data
  3. Ogni sera: aggiorna prezzi, calcola P&L, verifica stop loss / trailing
  4. Se condizione di uscita → manda alert nell'email serale
  5. Utente decide se uscire → registra l'uscita nell'app

Utilizzo:
    from real_portfolio import RealPortfolio
    rp = RealPortfolio()
    rp.open_position("TRN.MI", entry_price=10.32, size_eur=3333, entry_date="2026-04-22")
    rp.update_all()   # aggiorna prezzi e calcola alert
    rp.close_position("TRN.MI", exit_price=10.88, exit_date="2026-05-10", reason="manuale")
"""

import os, sys, sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT    = Path(__file__).parent.parent
DB_PATH = ROOT / "trading.db"

# Parametri default (sovrascrivibili da config.json)
DEFAULT_STOP_LOSS     = 0.04
DEFAULT_TAKE_PROFIT   = 0.12
DEFAULT_TRAILING_STOP = 0.06
VIX_EXIT_THRESHOLD    = 25.0
TRANSACTION_COST      = 0.0015


# =============================================================================
# SCHEMA DB — tabella posizioni reali
# =============================================================================
SCHEMA_REAL = """
CREATE TABLE IF NOT EXISTS real_positions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    status          TEXT DEFAULT 'open',
    entry_date      TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    size_eur        REAL NOT NULL,
    shares          REAL NOT NULL,
    stop_loss_price REAL NOT NULL,
    take_profit_price REAL NOT NULL,
    trailing_active INTEGER DEFAULT 0,
    highest_price   REAL,
    current_price   REAL,
    current_pnl_eur REAL,
    current_pnl_pct REAL,
    days_open       INTEGER DEFAULT 0,
    exit_date       TEXT,
    exit_price      REAL,
    exit_reason     TEXT,
    net_pnl_eur     REAL,
    net_pnl_pct     REAL,
    notes           TEXT,
    ai_prob         REAL,
    entry_score     INTEGER,
    alert           TEXT,
    updated_at      TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS real_transactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER,
    type        TEXT,
    date        TEXT,
    ticker      TEXT,
    price       REAL,
    shares      REAL,
    amount_eur  REAL,
    notes       TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);
"""


# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================
class RealPortfolio:

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self._init_db()
        self._load_config()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._conn()
        conn.executescript(SCHEMA_REAL)
        conn.commit()
        conn.close()

    def _load_config(self):
        cfg_path = ROOT / "config.json"
        if cfg_path.exists():
            import json
            cfg = json.loads(cfg_path.read_text())
            self.stop_loss     = cfg.get("stop_loss_pct",     DEFAULT_STOP_LOSS)
            self.take_profit   = cfg.get("take_profit_pct",   DEFAULT_TAKE_PROFIT)
            self.trailing_stop = cfg.get("trailing_stop_pct", DEFAULT_TRAILING_STOP)
        else:
            self.stop_loss     = DEFAULT_STOP_LOSS
            self.take_profit   = DEFAULT_TAKE_PROFIT
            self.trailing_stop = DEFAULT_TRAILING_STOP

    # ─────────────────────────────────────────────────────────────────────────
    # OPERAZIONI
    # ─────────────────────────────────────────────────────────────────────────
    def open_position(
        self,
        ticker:      str,
        entry_price: float,
        size_eur:    float,
        entry_date:  str   = None,
        notes:       str   = "",
        ai_prob:     float = None,
        entry_score: int   = None,
        stop_loss_pct:    float = None,
        take_profit_pct:  float = None,
    ) -> int:
        """
        Registra una nuova posizione aperta.
        Ritorna l'ID della posizione.
        """
        sl  = stop_loss_pct   or self.stop_loss
        tp  = take_profit_pct or self.take_profit
        ed  = entry_date or datetime.now().strftime("%Y-%m-%d")

        cost    = size_eur * TRANSACTION_COST
        shares  = (size_eur - cost) / entry_price
        sl_price= entry_price * (1 - sl)
        tp_price= entry_price * (1 + tp)

        conn = self._conn()
        cur  = conn.execute("""
            INSERT INTO real_positions
            (ticker, status, entry_date, entry_price, size_eur, shares,
             stop_loss_price, take_profit_price, highest_price,
             current_price, current_pnl_eur, current_pnl_pct,
             notes, ai_prob, entry_score, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,0,0,?,?,?,?)
        """, (
            ticker, "open", ed, entry_price, size_eur, shares,
            sl_price, tp_price, entry_price,
            entry_price,
            notes, ai_prob, entry_score,
            datetime.now().strftime("%Y-%m-%d %H:%M"),
        ))
        pos_id = cur.lastrowid

        # Log transazione
        conn.execute("""
            INSERT INTO real_transactions (position_id, type, date, ticker, price, shares, amount_eur)
            VALUES (?,?,?,?,?,?,?)
        """, (pos_id, "buy", ed, ticker, entry_price, shares, size_eur))

        conn.commit()
        conn.close()
        print(f"[OK] Posizione aperta: {ticker} @ €{entry_price:.3f} | "
              f"Size: €{size_eur:.0f} | Stop: €{sl_price:.3f} | Target: €{tp_price:.3f}")
        return pos_id

    def close_position(
        self,
        ticker:     str,
        exit_price: float,
        exit_date:  str  = None,
        reason:     str  = "manuale",
        position_id: int = None,
    ) -> dict:
        """Chiude una posizione aperta."""
        xd   = exit_date or datetime.now().strftime("%Y-%m-%d")
        conn = self._conn()

        if position_id:
            row = conn.execute(
                "SELECT * FROM real_positions WHERE id=? AND status='open'", (position_id,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM real_positions WHERE ticker=? AND status='open' ORDER BY entry_date DESC LIMIT 1",
                (ticker,)
            ).fetchone()

        if not row:
            conn.close()
            print(f"[WARN] Nessuna posizione aperta trovata per {ticker}")
            return {}

        row       = dict(row)
        shares    = row["shares"]
        entry_p   = row["entry_price"]
        entry_d   = datetime.strptime(row["entry_date"], "%Y-%m-%d")
        exit_d    = datetime.strptime(xd, "%Y-%m-%d")
        holding   = (exit_d - entry_d).days

        proceeds  = shares * exit_price * (1 - TRANSACTION_COST)
        cost_base = row["size_eur"]
        net_pnl   = proceeds - cost_base
        net_pct   = (exit_price / entry_p - 1 - 2 * TRANSACTION_COST) * 100

        conn.execute("""
            UPDATE real_positions
            SET status='closed', exit_date=?, exit_price=?,
                exit_reason=?, net_pnl_eur=?, net_pnl_pct=?,
                days_open=?, updated_at=?
            WHERE id=?
        """, (xd, exit_price, reason, net_pnl, net_pct,
              holding, datetime.now().strftime("%Y-%m-%d %H:%M"),
              row["id"]))

        conn.execute("""
            INSERT INTO real_transactions (position_id, type, date, ticker, price, shares, amount_eur)
            VALUES (?,?,?,?,?,?,?)
        """, (row["id"], "sell", xd, ticker, exit_price, shares, proceeds))

        conn.commit()
        conn.close()

        print(f"[OK] Posizione chiusa: {ticker} @ €{exit_price:.3f} | "
              f"P&L: €{net_pnl:+.2f} ({net_pct:+.1f}%) | {holding} giorni")
        return {"ticker": ticker, "net_pnl_eur": net_pnl, "net_pnl_pct": net_pct}

    # ─────────────────────────────────────────────────────────────────────────
    # AGGIORNAMENTO PREZZI
    # ─────────────────────────────────────────────────────────────────────────
    def update_all(self) -> list:
        """
        Scarica i prezzi attuali per tutte le posizioni aperte,
        aggiorna P&L, trailing stop e genera alert.
        Ritorna lista di alert da includere nell'email.
        """
        conn  = self._conn()
        rows  = conn.execute("SELECT * FROM real_positions WHERE status='open'").fetchall()
        conn.close()

        if not rows:
            print("Nessuna posizione aperta.")
            return []

        tickers = list(set(r["ticker"] for r in rows))
        prices  = self._fetch_prices(tickers)

        # VIX attuale
        vix = self._fetch_vix()

        alerts = []
        conn   = self._conn()

        for row in rows:
            row   = dict(row)
            tid   = row["id"]
            tk    = row["ticker"]
            price = prices.get(tk)

            if price is None:
                continue

            entry_p  = row["entry_price"]
            sl_price = row["stop_loss_price"]
            tp_price = row["take_profit_price"]
            highest  = row["highest_price"] or entry_p
            trailing = bool(row["trailing_active"])
            ed       = datetime.strptime(row["entry_date"], "%Y-%m-%d")
            days     = (datetime.now() - ed).days

            # Aggiorna trailing
            if not trailing and price >= tp_price:
                trailing  = True
                highest   = price
                sl_price  = highest * (1 - self.trailing_stop)
                print(f"  [TRAILING] {tk}: attivato @ €{price:.3f}, nuovo stop €{sl_price:.3f}")

            elif trailing and price > highest:
                highest   = price
                sl_price  = highest * (1 - self.trailing_stop)

            # P&L attuale
            pnl_pct = (price / entry_p - 1) * 100
            pnl_eur = (price - entry_p) * row["shares"]

            # Alert conditions
            alert = None

            if vix and vix >= VIX_EXIT_THRESHOLD:
                alert = f"ESCI — VIX={vix:.1f} sopra soglia {VIX_EXIT_THRESHOLD}"
            elif price <= sl_price:
                kind  = "trailing stop" if trailing else "stop loss"
                alert = f"ESCI — {kind} colpito @ €{price:.3f} (stop €{sl_price:.3f})"
            elif pnl_pct >= self.take_profit * 100 and not trailing:
                alert = f"ATTENZIONE — vicino al target ({pnl_pct:+.1f}%), trailing in attivazione"
            elif pnl_pct <= -(self.stop_loss * 100 * 0.75):
                alert = f"ATTENZIONE — perdita del {pnl_pct:.1f}%, stop loss a €{sl_price:.3f}"

            if alert:
                alerts.append({
                    "ticker":   tk,
                    "alert":    alert,
                    "price":    price,
                    "pnl_pct":  pnl_pct,
                    "pnl_eur":  pnl_eur,
                    "days":     days,
                })

            conn.execute("""
                UPDATE real_positions
                SET current_price=?, current_pnl_eur=?, current_pnl_pct=?,
                    stop_loss_price=?, highest_price=?, trailing_active=?,
                    days_open=?, alert=?, updated_at=?
                WHERE id=?
            """, (
                price, pnl_eur, pnl_pct,
                sl_price, highest, int(trailing),
                days, alert, datetime.now().strftime("%Y-%m-%d %H:%M"),
                tid,
            ))

        conn.commit()
        conn.close()

        if alerts:
            print(f"\n[!] {len(alerts)} alert generati:")
            for a in alerts:
                print(f"  {a['ticker']}: {a['alert']}")
        else:
            print("[OK] Nessun alert — tutte le posizioni nella norma.")

        return alerts

    def _fetch_prices(self, tickers: list) -> dict:
        prices = {}
        for tk in tickers:
            try:
                df = yf.download(tk, period="2d", auto_adjust=False, progress=False)
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                if not df.empty:
                    prices[tk] = float(df["Adj Close"].iloc[-1])
            except Exception as e:
                print(f"  [WARN] Prezzo {tk}: {e}")
        return prices

    def _fetch_vix(self) -> float | None:
        try:
            df = yf.download("^VIX", period="2d", auto_adjust=True, progress=False)
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            return float(df["Close"].iloc[-1]) if not df.empty else None
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # LETTURA DATI
    # ─────────────────────────────────────────────────────────────────────────
    def get_open_positions(self) -> pd.DataFrame:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM real_positions WHERE status='open' ORDER BY entry_date"
        ).fetchall()
        conn.close()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    def get_closed_positions(self) -> pd.DataFrame:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM real_positions WHERE status='closed' ORDER BY exit_date DESC"
        ).fetchall()
        conn.close()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    def get_summary(self) -> dict:
        open_pos = self.get_open_positions()
        closed   = self.get_closed_positions()

        invested   = open_pos["size_eur"].sum()       if not open_pos.empty else 0
        open_pnl   = open_pos["current_pnl_eur"].sum() if not open_pos.empty else 0
        closed_pnl = closed["net_pnl_eur"].sum()      if not closed.empty  else 0
        total_pnl  = open_pnl + closed_pnl

        wins  = closed[closed["net_pnl_eur"] > 0] if not closed.empty else pd.DataFrame()
        loss  = closed[closed["net_pnl_eur"] < 0] if not closed.empty else pd.DataFrame()
        wr    = len(wins) / len(closed) * 100 if not closed.empty else 0

        alerts = open_pos[open_pos["alert"].notna()]["ticker"].tolist() if not open_pos.empty and "alert" in open_pos.columns else []

        return {
            "n_open":      len(open_pos),
            "n_closed":    len(closed),
            "invested_eur":invested,
            "open_pnl_eur":open_pnl,
            "closed_pnl_eur": closed_pnl,
            "total_pnl_eur":  total_pnl,
            "win_rate":    wr,
            "alerts":      alerts,
        }


# =============================================================================
# MAIN — aggiornamento serale
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="Aggiorna prezzi e alert")
    parser.add_argument("--summary", action="store_true", help="Mostra riepilogo")
    args = parser.parse_args()

    rp = RealPortfolio()

    if args.update or not (args.summary):
        print("Aggiornamento posizioni reali...")
        alerts = rp.update_all()

    summary = rp.get_summary()
    print(f"\n=== PORTAFOGLIO REALE ===")
    print(f"Posizioni aperte:  {summary['n_open']}")
    print(f"Capitale investito: €{summary['invested_eur']:,.0f}")
    print(f"P&L aperto:        €{summary['open_pnl_eur']:+,.2f}")
    print(f"P&L realizzato:    €{summary['closed_pnl_eur']:+,.2f}")
    print(f"P&L totale:        €{summary['total_pnl_eur']:+,.2f}")
    if summary['alerts']:
        print(f"\n[!] Alert: {', '.join(summary['alerts'])}")

    open_pos = rp.get_open_positions()
    if not open_pos.empty:
        print("\nPosizioni aperte:")
        cols = ["ticker","entry_date","entry_price","current_price",
                "current_pnl_pct","current_pnl_eur","stop_loss_price","days_open","alert"]
        cols_ok = [c for c in cols if c in open_pos.columns]
        print(open_pos[cols_ok].to_string(index=False))