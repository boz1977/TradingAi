"""
daily_screener.py  —  Fase 5
Scarica i prezzi aggiornati di oggi, calcola le feature e interroga
il modello AI per trovare i titoli candidati all'ingresso domani.

Da lanciare ogni sera dopo la chiusura del mercato (es. 18:00).

Output a video e su file:
    data/processed/screener_YYYYMMDD.csv  — segnali del giorno
    data/processed/screener_latest.csv    — ultimo run (sovrascitto ogni volta)

Utilizzo:
    python src/daily_screener.py
    python src/daily_screener.py --date 2024-03-15   # replay su data passata
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import joblib

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tickers import FTSE_MIB_TICKERS
from universe_selection import select_universe
from train_model import SECTOR_MAP, score_trade

# Fase 1: DB e earnings
try:
    from database import DB
    from earnings_calendar import filter_signals_for_earnings, update_earnings_calendar
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

# Fase 2: fondamentali e sentiment
try:
    from fundamentals import get_fundamental_features, filter_universe_by_fundamentals
    from sentiment import get_sentiment_feature, compute_sentiment
    _FASE2_AVAILABLE = True
except ImportError:
    _FASE2_AVAILABLE = False


# =============================================================================
# CONFIGURAZIONE
# =============================================================================
MODEL_PATH  = "models/trade_scorer_v2.joblib"
FEAT_PATH   = "models/feature_names_v2.json"

# Soglie per i segnali
AI_PROB_THRESHOLD     = 0.60   # soglia minima per segnale
AI_PROB_STRONG        = 0.70   # soglia per segnale forte
ENTRY_SCORE_THRESHOLD = 6
VIX_MAX               = 20
RSI_MAX               = 62
MOMENTUM_3M_MIN       = -0.05  # gate duro: non entrare se calo > 5% in 3 mesi

# Lookback per scaricare abbastanza storia per gli indicatori
HISTORY_DAYS = 300   # ~14 mesi, sufficiente per MA200 + ADX

OUTPUT_DIR = "data/processed"

SECTOR_ENC = {
    "consumer":0, "energy":1, "financials":2, "healthcare":3,
    "industrials":4, "technology":5, "telecom":6, "utilities":7, "other":8
}


# =============================================================================
# DOWNLOAD DATI FRESCHI
# =============================================================================
def download_fresh_prices(tickers: list, days: int = HISTORY_DAYS) -> pd.DataFrame:
    """Scarica i prezzi più recenti per tutti i ticker."""
    end   = datetime.today()
    start = end - timedelta(days=days)

    all_data = []
    failed   = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                             auto_adjust=False, progress=False)
            if df.empty:
                failed.append(ticker)
                continue
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df.reset_index(inplace=True)
            df["ticker"] = ticker
            all_data.append(df)
        except Exception as e:
            print(f"  Errore {ticker}: {e}")
            failed.append(ticker)

    if failed:
        print(f"  Ticker non scaricati: {failed}")

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


def download_fresh_macro() -> dict:
    """
    Scarica i valori macro più recenti.
    Ritorna un dizionario con i valori attuali.
    """
    macro = {}

    # VIX
    try:
        vix_df = yf.download("^VIX", period="5d", auto_adjust=True, progress=False)
        vix_df.columns = [col[0] if isinstance(col, tuple) else col for col in vix_df.columns]
        macro["VIX"] = float(vix_df["Close"].iloc[-1])
    except Exception as e:
        print(f"  Errore VIX: {e}")
        macro["VIX"] = None

    # SP500
    try:
        sp_df = yf.download("^GSPC", period="5d", auto_adjust=True, progress=False)
        sp_df.columns = [col[0] if isinstance(col, tuple) else col for col in sp_df.columns]
        macro["SP500"] = float(sp_df["Close"].iloc[-1])
    except Exception as e:
        macro["SP500"] = None

    # FTSE MIB
    try:
        mib_df = yf.download("FTSEMIB.MI", period="5d", auto_adjust=True, progress=False)
        mib_df.columns = [col[0] if isinstance(col, tuple) else col for col in mib_df.columns]
        macro["FTSEMIB"] = float(mib_df["Close"].iloc[-1])
    except Exception:
        macro["FTSEMIB"] = None

    # BTP 10Y (proxy: ETF iShares BTP)
    try:
        btp_df = yf.download("IBTS.MI", period="5d", auto_adjust=True, progress=False)
        btp_df.columns = [col[0] if isinstance(col, tuple) else col for col in btp_df.columns]
        macro["BTP_PROXY"] = float(btp_df["Close"].iloc[-1])
    except Exception:
        macro["BTP_PROXY"] = None

    return macro


# =============================================================================
# CALCOLO INDICATORI SU UN SINGOLO TICKER
# =============================================================================
def compute_indicators(df: pd.DataFrame) -> dict:
    """
    Dato il dataframe storico di un ticker, calcola tutti gli indicatori
    necessari per l'entry score e il modello AI.
    Ritorna un dizionario con l'ultima riga di indicatori.
    """
    import ta

    df = df.sort_values("Date").copy()

    if len(df) < 60:
        return {}

    close = df["Adj Close"]

    # Moving averages
    ma20  = close.rolling(20).mean()
    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    ma50_slope = ma50.diff(5)

    # RSI
    rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    # ADX
    adx = None
    if "High" in df.columns and "Low" in df.columns:
        try:
            adx = ta.trend.ADXIndicator(
                high=df["High"], low=df["Low"], close=close
            ).adx()
        except Exception:
            pass

    # Volume ratio
    vol_ratio = None
    if "Volume" in df.columns:
        vol_ma20  = df["Volume"].rolling(20).mean()
        vol_ratio = df["Volume"] / vol_ma20

    # Momentum
    mom_1m = close.pct_change(21)
    mom_3m = close.pct_change(63)

    # Prendi ultima riga
    idx = -1
    last_close = float(close.iloc[idx])
    last_ma20  = float(ma20.iloc[idx])  if pd.notna(ma20.iloc[idx])  else None
    last_ma50  = float(ma50.iloc[idx])  if pd.notna(ma50.iloc[idx])  else None
    last_ma200 = float(ma200.iloc[idx]) if pd.notna(ma200.iloc[idx]) else None
    last_slope = float(ma50_slope.iloc[idx]) if pd.notna(ma50_slope.iloc[idx]) else None
    last_rsi   = float(rsi.iloc[idx])   if pd.notna(rsi.iloc[idx])   else None
    last_adx   = float(adx.iloc[idx])   if (adx is not None and pd.notna(adx.iloc[idx])) else None
    last_vr    = float(vol_ratio.iloc[idx]) if (vol_ratio is not None and pd.notna(vol_ratio.iloc[idx])) else None
    last_mom1m = float(mom_1m.iloc[idx]) if pd.notna(mom_1m.iloc[idx]) else None
    last_mom3m = float(mom_3m.iloc[idx]) if pd.notna(mom_3m.iloc[idx]) else None

    return {
        "date":         df["Date"].iloc[idx],
        "close":        last_close,
        "ma20":         last_ma20,
        "ma50":         last_ma50,
        "ma200":        last_ma200,
        "ma50_slope":   last_slope,
        "rsi":          last_rsi,
        "adx":          last_adx,
        "volume_ratio": last_vr,
        "momentum_1m":  last_mom1m,
        "momentum_3m":  last_mom3m,
        "above_ma50":   int(last_close > last_ma50)  if last_ma50  else 0,
        "above_ma200":  int(last_close > last_ma200) if last_ma200 else 0,
        "distance_from_ma50":  (last_close / last_ma50  - 1) if last_ma50  else None,
        "distance_from_ma200": (last_close / last_ma200 - 1) if last_ma200 else None,
    }


# =============================================================================
# CALCOLO ENTRY SCORE
# =============================================================================
def calculate_entry_score(ind: dict, vix: float) -> int:
    score = 0
    if ind.get("ma50")  and ind["close"] > ind["ma50"]:                          score += 1
    if ind.get("ma200") and ind["close"] > ind["ma200"]:                         score += 1
    if ind.get("ma20")  and ind.get("ma50") and ind["ma20"] > ind["ma50"]:       score += 1
    if ind.get("ma50_slope") and ind["ma50_slope"] > 0:                          score += 1
    if ind.get("rsi")   and ind["rsi"] < RSI_MAX:                                score += 1
    if ind.get("adx")   and ind["adx"] >= 20:                                    score += 1
    if ind.get("volume_ratio") and ind["volume_ratio"] >= 1.0:                   score += 1
    if ind.get("momentum_3m") and ind["momentum_3m"] >= 0:                       score += 1
    if vix and vix < VIX_MAX:                                                    score += 1
    return score


# =============================================================================
# COSTRUZIONE FEATURE AI
# =============================================================================
def build_ai_features(
    ticker: str,
    ind: dict,
    vix: float,
    macro: dict,
    ticker_stats: dict,
    trade_count: int,
    streak: int,
    today: datetime,
) -> dict:
    sector     = SECTOR_MAP.get(ticker, "other")
    sector_enc = SECTOR_ENC.get(sector, 8)
    hist_wr    = ticker_stats.get(ticker, {}).get("win_rate", 0.43)
    hist_n     = ticker_stats.get(ticker, {}).get("count", 30)
    month      = today.month
    quarter    = (month - 1) // 3 + 1
    dow        = today.weekday()
    year       = today.year
    vix_val    = vix if vix else 18.0
    entry_score= calculate_entry_score(ind, vix)

    return {
        "entry_vix":            vix_val,
        "entry_vix_bucket":     min(4, max(0, int((vix_val - 10) // 3))),
        "vix_low":              int(vix_val < 15),
        "vix_medium":           int(15 <= vix_val < 20),
        "vix_high":             int(vix_val >= 20),
        "entry_month":          month,
        "entry_quarter":        quarter,
        "entry_dayofweek":      dow,
        "month_sin":            np.sin(2 * np.pi * month / 12),
        "month_cos":            np.cos(2 * np.pi * month / 12),
        "dow_sin":              np.sin(2 * np.pi * dow / 5),
        "dow_cos":              np.cos(2 * np.pi * dow / 5),
        "sector_enc":           sector_enc,
        "ticker_win_rate_hist": hist_wr,
        "ticker_trade_count":   hist_n,
        "ticker_prev_trades":   trade_count,
        "ticker_streak":        streak,
        "entry_score":          entry_score,
        "is_score_strategy":    0,
        "bull_period":          int((2019 <= year <= 2021) or year >= 2023),
        "covid_period":         0,
        "entry_year":           year,
        # Feature macro estese (v2)
        "post_qe_period":       int(2015 <= year <= 2018),
        "pre_2008_crisis":      int(year < 2008),
        "gfc_period":           0,
        "eurozone_crisis":      0,
        "spread_rising_at_entry":   0,  # aggiornare con dati BTP live
        "btp_bund_spread_at_entry": 1.5,  # valore medio storico come default
        "sp500_ret_1m_at_entry":    0,
        "sp500_ret_3m_at_entry":    0,
        "sp500_above_ma200_at_entry": 1,
        "ftsemib_ret_1m_at_entry":  0,
        "ftsemib_above_ma50_at_entry": 1,
        "eurusd_trend_at_entry":    1,
        "beta":                 0.9,
        "is_defensive":         int(sector in ["utilities", "telecom", "healthcare"]),
        "is_financial":         int(sector == "financials"),
        "is_cyclical":          int(sector in ["consumer", "industrials", "energy"]),
    }

    # Fase 2: aggiungi feature fondamentali e sentiment se disponibili
    if _FASE2_AVAILABLE:
        try:
            fund_feats = get_fundamental_features(ticker)
            features.update(fund_feats)
        except Exception:
            pass
        try:
            sent_feats = get_sentiment_feature(ticker)
            features.update(sent_feats)
        except Exception:
            pass

    return features


# =============================================================================
# MAIN SCREENER
# =============================================================================
def run_screener(
    reference_date: datetime | None = None,
    approved_tickers: list | None = None,
) -> pd.DataFrame:

    today = reference_date or datetime.today()
    date_str = today.strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"DAILY SCREENER  —  {date_str}")
    print(f"{'='*60}")

    # 1. Carica modello
    if not os.path.exists(MODEL_PATH):
        print(f"Modello non trovato: {MODEL_PATH}")
        print("Esegui prima: python src/retrain_model_extended.py")
        return pd.DataFrame()

    model       = joblib.load(MODEL_PATH)
    feature_cols= json.load(open(FEAT_PATH))
    print(f"Modello caricato: {MODEL_PATH}")

    # 2. Universe selection
    if approved_tickers is None:
        summary_file = "data/processed/target_stop_trailing_vix_regime_summary.csv"
        if os.path.exists(summary_file):
            summary_df       = pd.read_csv(summary_file)
            approved_tickers = select_universe(summary_df, verbose=False)
        else:
            approved_tickers = FTSE_MIB_TICKERS
    print(f"Ticker universe: {len(approved_tickers)}  {approved_tickers}")

    # 3. Dati macro live
    print("\nDownload macro live...")
    macro = download_fresh_macro()
    vix   = macro.get("VIX")
    print(f"  VIX attuale:    {vix:.1f}" if vix else "  VIX: N/D")
    print(f"  SP500:          {macro.get('SP500', 'N/D')}")
    print(f"  FTSE MIB:       {macro.get('FTSEMIB', 'N/D')}")

    if vix and vix >= VIX_MAX:
        print(f"\n[STOP] VIX = {vix:.1f} >= {VIX_MAX} — nessun ingresso oggi")
        return pd.DataFrame()

    # 4. Ticker stats storiche
    ticker_stats = {}
    tl_file = "data/processed/target_stop_trailing_vix_regime_trade_log.csv"
    if os.path.exists(tl_file):
        tl = pd.read_csv(tl_file)
        for t, g in tl.groupby("ticker"):
            ticker_stats[t] = {
                "win_rate": (g["net_return_pct"] > 0).mean(),
                "count":    len(g),
            }

    # 5. Download prezzi freschi
    print(f"\nDownload prezzi ({HISTORY_DAYS} giorni storia)...")
    prices_df = download_fresh_prices(approved_tickers)
    if prices_df.empty:
        print("Errore: nessun prezzo scaricato.")
        return pd.DataFrame()
    prices_df["Date"] = pd.to_datetime(prices_df["Date"]).dt.tz_localize(None)
    print(f"  Scaricati {prices_df['ticker'].nunique()} ticker")

    # 6. Calcola indicatori e score per ogni ticker
    print("\nAnalisi segnali...")
    signals = []

    for ticker in approved_tickers:
        df_t = prices_df[prices_df["ticker"] == ticker].copy()
        if df_t.empty or len(df_t) < 60:
            continue

        ind   = compute_indicators(df_t)
        if not ind:
            continue

        score = calculate_entry_score(ind, vix)

        # Gate rapidissimi prima di chiamare il modello AI
        if score < ENTRY_SCORE_THRESHOLD:
            continue
        if ind.get("momentum_3m") and ind["momentum_3m"] < MOMENTUM_3M_MIN:
            continue

        # AI scoring
        ai_feats = build_ai_features(
            ticker, ind, vix, macro, ticker_stats, 0, 0, today
        )
        ai_prob = score_trade(model, ai_feats, feature_cols)

        if ai_prob < AI_PROB_THRESHOLD:
            continue

        sector = SECTOR_MAP.get(ticker, "other")
        signals.append({
            "ticker":        ticker,
            "sector":        sector,
            "date":          date_str,
            "close":         round(ind["close"], 3),
            "entry_score":   score,
            "ai_prob":       round(ai_prob, 3),
            "signal":        "FORTE" if ai_prob >= AI_PROB_STRONG else "OK",
            "rsi":           round(ind["rsi"], 1)   if ind.get("rsi")   else None,
            "adx":           round(ind["adx"], 1)   if ind.get("adx")   else None,
            "volume_ratio":  round(ind["volume_ratio"], 2) if ind.get("volume_ratio") else None,
            "momentum_1m":   round(ind["momentum_1m"] * 100, 1) if ind.get("momentum_1m") else None,
            "momentum_3m":   round(ind["momentum_3m"] * 100, 1) if ind.get("momentum_3m") else None,
            "above_ma50":    ind.get("above_ma50", 0),
            "above_ma200":   ind.get("above_ma200", 0),
            "dist_ma50_pct": round(ind["distance_from_ma50"]  * 100, 1) if ind.get("distance_from_ma50")  else None,
            "dist_ma200_pct":round(ind["distance_from_ma200"] * 100, 1) if ind.get("distance_from_ma200") else None,
            "vix":           round(vix, 1) if vix else None,
        })

    # 7. Report
    if not signals:
        print(f"\n  Nessun segnale oggi (VIX={vix:.1f if vix else 'N/D'}, soglia score={ENTRY_SCORE_THRESHOLD}, soglia AI={AI_PROB_THRESHOLD})")
        return pd.DataFrame()

    df_signals = pd.DataFrame(signals).sort_values("ai_prob", ascending=False)

    print(f"\n{'='*60}")
    print(f"SEGNALI TROVATI: {len(df_signals)}")
    print(f"{'='*60}")

    for _, row in df_signals.iterrows():
        label = "[FORTE]" if row["signal"] == "FORTE" else "[OK]   "
        print(f"\n{label}  {row['ticker']:<10} {row['sector']:<12}")
        print(f"         Prezzo:    {row['close']}")
        print(f"         AI prob:   {row['ai_prob']:.1%}  |  Score: {row['entry_score']}/9")
        print(f"         RSI: {row['rsi']}  ADX: {row['adx']}  Vol ratio: {row['volume_ratio']}")
        print(f"         Mom 1M: {row['momentum_1m']}%  Mom 3M: {row['momentum_3m']}%")
        print(f"         Dist MA50: {row['dist_ma50_pct']}%  Dist MA200: {row['dist_ma200_pct']}%")
        if "sentiment_score" in row and pd.notna(row.get("sentiment_score")):
            sent = row["sentiment_score"]
            label = "positivo" if sent > 0.1 else "negativo" if sent < -0.1 else "neutro"
            print(f"         Sentiment: {sent:+.2f} ({label})")

    # 8. Filtro earnings — blocca segnali se earnings entro 3 giorni
    if _DB_AVAILABLE:
        try:
            db = DB()
            df_signals = filter_signals_for_earnings(df_signals, db=db)
            if df_signals.empty:
                print("\n[STOP] Tutti i segnali bloccati per earnings imminenti.")
                return pd.DataFrame()
        except Exception as e:
            print(f"  [WARN] Filtro earnings non applicato: {e}")

    # 9. Salva su CSV e DB
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_dated  = os.path.join(OUTPUT_DIR, f"screener_{today.strftime('%Y%m%d')}.csv")
    out_latest = os.path.join(OUTPUT_DIR, "screener_latest.csv")
    df_signals.to_csv(out_dated,  index=False)
    df_signals.to_csv(out_latest, index=False)
    print(f"\nSegnali salvati: {out_dated}")

    if _DB_AVAILABLE:
        try:
            db = DB()
            db.save_signals(df_signals)
            db.save_screener_run(df_signals, vix=vix)
            print(f"[OK] Salvati nel DB: {len(df_signals)} segnali")
        except Exception as e:
            print(f"  [WARN] DB non aggiornato: {e}")

    return df_signals


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily screener FTSE MIB")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Data di riferimento YYYY-MM-DD (default: oggi)"
    )
    args = parser.parse_args()

    ref_date   = datetime.strptime(args.date, "%Y-%m-%d") if args.date else None
    df_signals = run_screener(reference_date=ref_date)

    # Salva nello storico segnali
    try:
        from signal_history import record_signals
        record_signals(df_signals)
    except Exception as e:
        print(f"  [WARN] Storico non aggiornato: {e}")

    # Notifica email automatica (commenta se non vuoi l'email)
    try:
        from notify_email import send_email
        if df_signals is not None and not df_signals.empty:
            vix_val  = df_signals["vix"].iloc[0] if "vix" in df_signals.columns else None
            date_str = df_signals["date"].iloc[0] if "date" in df_signals.columns else datetime.now().strftime("%d/%m/%Y")
            latest   = os.path.join(OUTPUT_DIR, "screener_latest.csv")
            send_email(df_signals, date_str, vix_val, attach_csv=latest)
    except Exception as e:
        print(f"  Email non inviata: {e}")
