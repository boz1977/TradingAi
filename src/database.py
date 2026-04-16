"""
database.py  —  Fase 1
Layer SQLite per la persistenza dei dati tra i deploy di Streamlit Cloud.

Sostituisce i CSV per:
  - signal_history  (segnali generati + risultati)
  - screener_runs   (ogni run dello screener con metadata)
  - trade_outcomes  (trade chiusi con P&L)
  - model_metrics   (metriche del modello nel tempo)

I CSV rimangono come export/import ma il DB è la fonte di verità.

Utilizzo:
    from database import DB
    db = DB()
    db.save_signals(df)
    history = db.get_signal_history()
"""

import os, sys, sqlite3, json
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# DB nella root del progetto — persiste tra i deploy se usi Streamlit Cloud
# con persistent storage, oppure in data/ per uso locale
ROOT    = Path(__file__).parent.parent
DB_PATH = ROOT / "trading.db"


# =============================================================================
# SCHEMA
# =============================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_date     TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    sector          TEXT,
    entry_price     REAL,
    entry_score     INTEGER,
    ai_prob         REAL,
    signal_type     TEXT,
    rsi             REAL,
    adx             REAL,
    volume_ratio    REAL,
    momentum_1m     REAL,
    momentum_3m     REAL,
    vix             REAL,
    status          TEXT DEFAULT 'open',
    exit_date       TEXT,
    exit_price      REAL,
    exit_reason     TEXT,
    net_return_pct  REAL,
    holding_days    INTEGER,
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(signal_date, ticker)
);

CREATE TABLE IF NOT EXISTS screener_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date        TEXT NOT NULL,
    run_time        TEXT,
    n_signals       INTEGER,
    n_forte         INTEGER,
    vix             REAL,
    tickers_signal  TEXT,
    notes           TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS earnings_calendar (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    report_date     TEXT NOT NULL,
    period          TEXT,
    eps_estimate    REAL,
    fetched_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(ticker, report_date)
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    version         TEXT,
    train_date      TEXT,
    n_trades        INTEGER,
    auc_cv          REAL,
    win_rate_filtered REAL,
    avg_return_filtered REAL,
    features_json   TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_signals_date   ON signals(signal_date);
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);
CREATE INDEX IF NOT EXISTS idx_earnings_date  ON earnings_calendar(report_date);
CREATE INDEX IF NOT EXISTS idx_earnings_ticker ON earnings_calendar(ticker);
"""


# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================
class DB:
    def __init__(self, path: Path = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNALS
    # ─────────────────────────────────────────────────────────────────────────
    def save_signals(self, df: pd.DataFrame, overwrite_today: bool = True) -> int:
        """
        Salva i segnali dello screener nel DB.
        Ritorna il numero di righe inserite.
        """
        if df.empty:
            return 0

        today = datetime.now().strftime("%Y-%m-%d")
        inserted = 0

        with self._conn() as conn:
            if overwrite_today:
                conn.execute(
                    "DELETE FROM signals WHERE signal_date = ?", (today,)
                )

            for _, row in df.iterrows():
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO signals
                        (signal_date, ticker, sector, entry_price, entry_score,
                         ai_prob, signal_type, rsi, adx, volume_ratio,
                         momentum_1m, momentum_3m, vix, status)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'open')
                    """, (
                        row.get("date", today),
                        row.get("ticker"),
                        row.get("sector"),
                        row.get("close"),
                        row.get("entry_score"),
                        row.get("ai_prob"),
                        row.get("signal", "OK"),
                        row.get("rsi"),
                        row.get("adx"),
                        row.get("volume_ratio"),
                        row.get("momentum_1m"),
                        row.get("momentum_3m"),
                        row.get("vix"),
                    ))
                    inserted += conn.execute("SELECT changes()").fetchone()[0]
                except Exception as e:
                    print(f"  [WARN] save_signals {row.get('ticker')}: {e}")

        return inserted

    def save_screener_run(self, df: pd.DataFrame, vix: float = None):
        """Logga ogni run dello screener."""
        tickers = ",".join(df["ticker"].tolist()) if not df.empty else ""
        n_forte = len(df[df.get("signal", "") == "FORTE"]) if not df.empty else 0
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO screener_runs
                (run_date, run_time, n_signals, n_forte, vix, tickers_signal)
                VALUES (?,?,?,?,?,?)
            """, (
                datetime.now().strftime("%Y-%m-%d"),
                datetime.now().strftime("%H:%M"),
                len(df),
                n_forte,
                vix,
                tickers,
            ))

    def get_signal_history(
        self,
        days_back: int = 365,
        status: list = None,
        ticker: str = None,
    ) -> pd.DataFrame:
        """Recupera lo storico segnali con filtri opzionali."""
        conditions = ["signal_date >= ?"]
        params     = [(datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")]

        if status:
            placeholders = ",".join("?" * len(status))
            conditions.append(f"status IN ({placeholders})")
            params.extend(status)

        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker)

        where = " AND ".join(conditions)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM signals WHERE {where} ORDER BY signal_date DESC",
                params
            ).fetchall()

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    def update_signal_outcome(
        self,
        signal_id: int,
        status: str,
        exit_date: str,
        exit_price: float,
        exit_reason: str,
        net_return_pct: float,
        holding_days: int,
    ):
        """Aggiorna il risultato di un singolo segnale."""
        with self._conn() as conn:
            conn.execute("""
                UPDATE signals
                SET status=?, exit_date=?, exit_price=?,
                    exit_reason=?, net_return_pct=?, holding_days=?
                WHERE id=?
            """, (status, exit_date, exit_price,
                  exit_reason, net_return_pct, holding_days, signal_id))

    def get_open_signals(self) -> pd.DataFrame:
        """Tutti i segnali ancora aperti."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM signals WHERE status='open' ORDER BY signal_date"
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    def get_stats(self) -> dict:
        """Statistiche aggregate dello storico."""
        with self._conn() as conn:
            total  = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            open_n = conn.execute("SELECT COUNT(*) FROM signals WHERE status='open'").fetchone()[0]
            wins   = conn.execute("SELECT COUNT(*) FROM signals WHERE status='win'").fetchone()[0]
            losses = conn.execute("SELECT COUNT(*) FROM signals WHERE status='loss'").fetchone()[0]

            avg_ret = conn.execute(
                "SELECT AVG(net_return_pct) FROM signals WHERE status IN ('win','loss')"
            ).fetchone()[0]

            best = conn.execute(
                "SELECT ticker, net_return_pct, signal_date FROM signals ORDER BY net_return_pct DESC LIMIT 1"
            ).fetchone()
            worst = conn.execute(
                "SELECT ticker, net_return_pct, signal_date FROM signals ORDER BY net_return_pct ASC LIMIT 1"
            ).fetchone()

            by_ticker = conn.execute("""
                SELECT ticker,
                       COUNT(*) as n,
                       SUM(CASE WHEN status='win' THEN 1 ELSE 0 END) as wins,
                       AVG(CASE WHEN status IN ('win','loss') THEN net_return_pct END) as avg_ret
                FROM signals
                WHERE status IN ('win','loss')
                GROUP BY ticker
                ORDER BY avg_ret DESC
            """).fetchall()

        closed = wins + losses
        return {
            "total":        total,
            "open":         open_n,
            "closed":       closed,
            "wins":         wins,
            "losses":       losses,
            "win_rate":     wins / closed * 100 if closed > 0 else 0,
            "avg_return":   round(avg_ret, 2) if avg_ret else 0,
            "best":         dict(best) if best else None,
            "worst":        dict(worst) if worst else None,
            "by_ticker":    [dict(r) for r in by_ticker],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # EARNINGS CALENDAR
    # ─────────────────────────────────────────────────────────────────────────
    def save_earnings(self, ticker: str, dates: list):
        """Salva le date degli earnings per un ticker."""
        with self._conn() as conn:
            for d in dates:
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO earnings_calendar (ticker, report_date, period)
                        VALUES (?, ?, ?)
                    """, (ticker, d.get("date"), d.get("period")))
                except Exception:
                    pass

    def get_earnings_in_range(
        self,
        start_date: str,
        end_date: str,
        tickers: list = None,
    ) -> pd.DataFrame:
        """Earnings nel range di date specificato."""
        conditions = ["report_date BETWEEN ? AND ?"]
        params     = [start_date, end_date]

        if tickers:
            placeholders = ",".join("?" * len(tickers))
            conditions.append(f"ticker IN ({placeholders})")
            params.extend(tickers)

        where = " AND ".join(conditions)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM earnings_calendar WHERE {where} ORDER BY report_date",
                params
            ).fetchall()

        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    def has_earnings_soon(self, ticker: str, days_ahead: int = 5) -> dict | None:
        """
        Controlla se un ticker ha earnings nei prossimi N giorni.
        Ritorna il record earnings se trovato, None altrimenti.
        """
        today    = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        with self._conn() as conn:
            row = conn.execute("""
                SELECT * FROM earnings_calendar
                WHERE ticker = ? AND report_date BETWEEN ? AND ?
                ORDER BY report_date
                LIMIT 1
            """, (ticker, today, end_date)).fetchone()
        return dict(row) if row else None

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL METRICS
    # ─────────────────────────────────────────────────────────────────────────
    def save_model_metrics(self, metrics: dict):
        """Salva le metriche di un training del modello."""
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO model_metrics
                (version, train_date, n_trades, auc_cv,
                 win_rate_filtered, avg_return_filtered, features_json)
                VALUES (?,?,?,?,?,?,?)
            """, (
                metrics.get("version", "v2"),
                datetime.now().strftime("%Y-%m-%d"),
                metrics.get("n_trades"),
                metrics.get("auc_cv"),
                metrics.get("win_rate_filtered"),
                metrics.get("avg_return_filtered"),
                json.dumps(metrics.get("features", [])),
            ))

    def get_model_history(self) -> pd.DataFrame:
        """Storico delle versioni del modello."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM model_metrics ORDER BY train_date DESC"
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    # ─────────────────────────────────────────────────────────────────────────
    # EXPORT / IMPORT CSV  (compatibilità con il vecchio sistema)
    # ─────────────────────────────────────────────────────────────────────────
    def export_to_csv(self, output_dir: Path = None):
        """Esporta tutte le tabelle come CSV (backup / compatibilità)."""
        output_dir = output_dir or ROOT / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        tables = ["signals", "screener_runs", "earnings_calendar", "model_metrics"]
        for table in tables:
            with self._conn() as conn:
                rows = conn.execute(f"SELECT * FROM {table}").fetchall()
            if rows:
                df = pd.DataFrame([dict(r) for r in rows])
                df.to_csv(output_dir / f"{table}.csv", index=False)
                print(f"  [OK] Exported {table}: {len(df)} rows")

    def import_from_csv(self, csv_path: Path):
        """
        Importa uno storico CSV esistente (signal_history.csv)
        nel nuovo DB SQLite.
        """
        if not csv_path.exists():
            print(f"  [WARN] File non trovato: {csv_path}")
            return 0

        df = pd.read_csv(csv_path)
        print(f"  Importazione {len(df)} righe da {csv_path.name}...")

        col_map = {
            "close": "entry_price",
            "signal": "signal_type",
        }
        df = df.rename(columns=col_map)

        with self._conn() as conn:
            inserted = 0
            for _, row in df.iterrows():
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO signals
                        (signal_date, ticker, sector, entry_price, entry_score,
                         ai_prob, signal_type, rsi, momentum_3m, vix,
                         status, exit_date, exit_price, exit_reason,
                         net_return_pct, holding_days)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        row.get("signal_date"),
                        row.get("ticker"),
                        row.get("sector"),
                        row.get("entry_price"),
                        row.get("entry_score"),
                        row.get("ai_prob"),
                        row.get("signal_type", "OK"),
                        row.get("rsi"),
                        row.get("momentum_3m"),
                        row.get("vix"),
                        row.get("status", "open"),
                        row.get("exit_date") if pd.notna(row.get("exit_date", None)) else None,
                        row.get("exit_price") if pd.notna(row.get("exit_price", None)) else None,
                        row.get("exit_reason") if pd.notna(row.get("exit_reason", None)) else None,
                        row.get("net_return_pct") if pd.notna(row.get("net_return_pct", None)) else None,
                        row.get("holding_days") if pd.notna(row.get("holding_days", None)) else None,
                    ))
                    inserted += conn.execute("SELECT changes()").fetchone()[0]
                except Exception as e:
                    print(f"    [WARN] {row.get('ticker')}: {e}")

        print(f"  [OK] Importate {inserted} righe nel DB")
        return inserted


# =============================================================================
# MAIN — inizializza DB e importa storico esistente
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("DATABASE SETUP — Fase 1")
    print("=" * 50)

    db = DB()
    print(f"[OK] Database creato: {DB_PATH}")

    # Importa storico CSV esistente se presente
    old_history = ROOT / "data" / "processed" / "signal_history.csv"
    if old_history.exists():
        print(f"\nImportazione storico esistente...")
        db.import_from_csv(old_history)
    else:
        print("\nNessuno storico CSV esistente — DB inizializzato vuoto.")

    # Mostra stato
    stats = db.get_stats()
    print(f"\nStato DB:")
    print(f"  Segnali totali: {stats['total']}")
    print(f"  Aperti:         {stats['open']}")
    print(f"  Chiusi:         {stats['closed']}")
    print(f"  Win rate:       {stats['win_rate']:.1f}%")
    print(f"\n[OK] {DB_PATH}")