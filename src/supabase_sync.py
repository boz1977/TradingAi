"""
supabase_sync.py  —  Persistenza dati su Supabase
Risolve il problema dei dati che si azzerano ad ogni restart di Streamlit Cloud.

Supabase è un database PostgreSQL hosted gratuito.
Piano free: 500MB storage, illimitati per il nostro uso.

Setup (una volta sola — 5 minuti):
  1. Vai su supabase.com → "New project" (gratuito)
  2. Aspetta 2 minuti che si avvii
  3. Vai su Settings → API → copia "Project URL" e "anon public key"
  4. Su Streamlit Cloud: Settings → Secrets → aggiungi:
       SUPABASE_URL = "https://xxxx.supabase.co"
       SUPABASE_KEY = "eyJhbGci..."
  5. Lancia una volta: python src/supabase_sync.py --setup

Utilizzo automatico:
  - All'avvio dell'app: scarica i dati da Supabase nel DB locale
  - Dopo ogni operazione: salva nel DB locale E su Supabase
"""

import os, sys, json, sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT    = Path(__file__).parent.parent
DB_PATH = ROOT / "trading.db"

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Tabelle da sincronizzare
SYNC_TABLES = [
    "signals",
    "real_positions",
    "real_transactions",
    "screener_runs",
    "earnings_calendar",
    "model_metrics",
]


# =============================================================================
# CONNESSIONE SUPABASE
# =============================================================================
def _get_client():
    """Ritorna il client Supabase."""
    try:
        from supabase import create_client
        if not SUPABASE_URL or not SUPABASE_KEY:
            return None
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except ImportError:
        return None
    except Exception as e:
        print(f"  [WARN] Supabase connection: {e}")
        return None


def is_configured() -> bool:
    """True se Supabase è configurato."""
    return bool(SUPABASE_URL and SUPABASE_KEY)


# =============================================================================
# SETUP — crea le tabelle su Supabase
# =============================================================================
SUPABASE_TABLES_SQL = {
    "signals": """
        CREATE TABLE IF NOT EXISTS signals (
            id              BIGSERIAL PRIMARY KEY,
            signal_date     TEXT,
            ticker          TEXT,
            sector          TEXT,
            entry_price     FLOAT,
            entry_score     INT,
            ai_prob         FLOAT,
            signal_type     TEXT,
            rsi             FLOAT,
            momentum_3m     FLOAT,
            vix             FLOAT,
            status          TEXT DEFAULT 'open',
            exit_date       TEXT,
            exit_price      FLOAT,
            exit_reason     TEXT,
            net_return_pct  FLOAT,
            holding_days    INT,
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(signal_date, ticker)
        );
    """,
    "real_positions": """
        CREATE TABLE IF NOT EXISTS real_positions (
            id                  BIGSERIAL PRIMARY KEY,
            ticker              TEXT,
            status              TEXT DEFAULT 'open',
            entry_date          TEXT,
            entry_price         FLOAT,
            size_eur            FLOAT,
            shares              FLOAT,
            stop_loss_price     FLOAT,
            take_profit_price   FLOAT,
            trailing_active     INT DEFAULT 0,
            highest_price       FLOAT,
            current_price       FLOAT,
            current_pnl_eur     FLOAT,
            current_pnl_pct     FLOAT,
            days_open           INT DEFAULT 0,
            exit_date           TEXT,
            exit_price          FLOAT,
            exit_reason         TEXT,
            net_pnl_eur         FLOAT,
            net_pnl_pct         FLOAT,
            notes               TEXT,
            ai_prob             FLOAT,
            entry_score         INT,
            alert               TEXT,
            updated_at          TEXT,
            created_at          TIMESTAMPTZ DEFAULT NOW()
        );
    """,
    "real_transactions": """
        CREATE TABLE IF NOT EXISTS real_transactions (
            id          BIGSERIAL PRIMARY KEY,
            position_id INT,
            type        TEXT,
            date        TEXT,
            ticker      TEXT,
            price       FLOAT,
            shares      FLOAT,
            amount_eur  FLOAT,
            notes       TEXT,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );
    """,
}


def setup_supabase():
    """Crea le tabelle su Supabase via SQL Editor."""
    if not is_configured():
        print("Supabase non configurato. Aggiungi SUPABASE_URL e SUPABASE_KEY.")
        return False

    print("Setup Supabase...")
    print("\nEsegui questi SQL nell'SQL Editor di Supabase (supabase.com → SQL Editor):")
    for table, sql in SUPABASE_TABLES_SQL.items():
        print(f"\n-- Tabella: {table}")
        print(sql)
    print("\nOppure usa l'interfaccia Table Editor per creare le tabelle manualmente.")
    return True


# =============================================================================
# SYNC: SQLite → Supabase (upload)
# =============================================================================
def push_to_supabase(tables: list = None) -> dict:
    """
    Carica i dati dal DB locale su Supabase.
    Usato dopo ogni operazione importante (nuovo segnale, nuova posizione...).
    """
    client = _get_client()
    if not client:
        return {}

    tables = tables or ["signals", "real_positions", "real_transactions"]
    results = {}

    if not DB_PATH.exists():
        print("  [WARN] DB locale non trovato")
        return {}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    for table in tables:
        try:
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
            if not rows:
                continue

            data = [dict(r) for r in rows]

            # Upsert su Supabase (insert or update)
            response = client.table(table).upsert(data).execute()
            results[table] = len(data)
            print(f"  [OK] {table}: {len(data)} righe sincronizzate")

        except Exception as e:
            print(f"  [WARN] {table}: {e}")
            results[table] = 0

    conn.close()
    return results


# =============================================================================
# SYNC: Supabase → SQLite (download)
# =============================================================================
def pull_from_supabase(tables: list = None) -> dict:
    """
    Scarica i dati da Supabase nel DB locale.
    Chiamato all'avvio dell'app per ripristinare i dati.
    """
    client = _get_client()
    if not client:
        return {}

    tables = tables or SYNC_TABLES
    results = {}

    # Inizializza DB locale se non esiste
    from database import DB
    db = DB()

    conn = sqlite3.connect(DB_PATH)

    for table in tables:
        try:
            response = client.table(table).select("*").execute()
            rows = response.data

            if not rows:
                results[table] = 0
                continue

            df = pd.DataFrame(rows)

            # Scrivi nel DB locale (replace)
            df.to_sql(table, conn, if_exists="replace", index=False)
            results[table] = len(df)
            print(f"  [OK] {table}: {len(df)} righe scaricate")

        except Exception as e:
            print(f"  [WARN] {table}: {e}")
            results[table] = 0

    conn.commit()
    conn.close()
    return results


# =============================================================================
# FUNZIONE DA CHIAMARE ALL'AVVIO DELL'APP
# =============================================================================
def ensure_data_loaded() -> bool:
    """
    All'avvio dell'app, verifica se il DB locale è vuoto/mancante
    e in quel caso scarica i dati da Supabase.

    Ritorna True se i dati sono disponibili, False se il DB è vuoto.
    """
    if not is_configured():
        return DB_PATH.exists()

    # Controlla se il DB è vuoto
    needs_restore = False
    if not DB_PATH.exists():
        needs_restore = True
    else:
        try:
            conn = sqlite3.connect(DB_PATH)
            n = conn.execute("SELECT COUNT(*) FROM real_positions").fetchone()[0]
            conn.close()
            needs_restore = (n == 0)
        except Exception:
            needs_restore = True

    if needs_restore:
        print("DB locale vuoto — scaricamento dati da Supabase...")
        results = pull_from_supabase()
        total = sum(results.values())
        print(f"  Ripristinati {total} record totali")
        return total > 0

    return True


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup",  action="store_true", help="Mostra SQL per setup Supabase")
    parser.add_argument("--push",   action="store_true", help="Carica dati locali su Supabase")
    parser.add_argument("--pull",   action="store_true", help="Scarica dati da Supabase in locale")
    parser.add_argument("--status", action="store_true", help="Mostra stato connessione")
    args = parser.parse_args()

    if args.status:
        print(f"Supabase configurato: {is_configured()}")
        print(f"URL: {SUPABASE_URL[:30]}..." if SUPABASE_URL else "URL: non impostato")
        client = _get_client()
        print(f"Connessione: {'OK' if client else 'FALLITA'}")

    elif args.setup:
        setup_supabase()

    elif args.push:
        print("Upload dati su Supabase...")
        results = push_to_supabase()
        print(f"Completato: {results}")

    elif args.pull:
        print("Download dati da Supabase...")
        results = pull_from_supabase()
        print(f"Completato: {results}")

    else:
        print("Uso: python src/supabase_sync.py --setup|--push|--pull|--status")
