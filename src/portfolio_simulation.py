"""
portfolio_simulation.py  —  Simulazione portafoglio 365 giorni
Esegue il daily screener su ogni giorno di borsa degli ultimi 365 giorni,
simula gli ingressi/uscite e costruisce un portafoglio virtuale.

Regole portafoglio:
  - Capitale iniziale: 10.000€ (configurabile)
  - Max posizioni contemporanee: 3
  - Ogni posizione: capitale / max_posizioni (uguale peso)
  - Ingresso: prezzo di chiusura del giorno del segnale + 1
  - Uscita: stop loss -4%, trailing stop 6% dal massimo, VIX exit > 25
  - Un titolo già in portafoglio non viene riacquistato

Utilizzo:
    python src/portfolio_simulation.py
    python src/portfolio_simulation.py --capital 50000 --max-positions 5
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tickers import FTSE_MIB_TICKERS
from universe_selection import select_universe
from train_model import SECTOR_MAP, score_trade
from daily_screener import (
    compute_indicators, calculate_entry_score,
    build_ai_features, SECTOR_ENC,
    AI_PROB_THRESHOLD, AI_PROB_STRONG,
    ENTRY_SCORE_THRESHOLD, VIX_MAX, MOMENTUM_3M_MIN,
)


# =============================================================================
# PARAMETRI PORTAFOGLIO
# =============================================================================
INITIAL_CAPITAL   = 10_000.0
MAX_POSITIONS     = 3
TRANSACTION_COST  = 0.0015   # 0.15% per lato
STOP_LOSS_PCT     = 0.04
TAKE_PROFIT_PCT   = 0.12
TRAILING_STOP_PCT = 0.06
VIX_EXIT          = 25.0

MODEL_PATH = "models/trade_scorer_v2.joblib"
FEAT_PATH  = "models/feature_names_v2.json"
OUTPUT_DIR = "data/processed"
CHART_DIR  = "data/charts"


# =============================================================================
# DOWNLOAD DATI STORICI (un solo blocco per tutto l'anno)
# =============================================================================
def download_all_history(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Scarica tutti i prezzi in un solo blocco per efficienza."""
    print(f"Download prezzi {start} → {end} per {len(tickers)} ticker...")
    all_data = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=False, progress=False)
            if df.empty:
                continue
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df.reset_index(inplace=True)
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            df["ticker"] = ticker
            all_data.append(df)
        except Exception as e:
            print(f"  Errore {ticker}: {e}")
    combined = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    print(f"  Scaricati {combined['ticker'].nunique()} ticker, {len(combined)} righe")
    return combined


def download_vix_history(start: str, end: str) -> pd.Series:
    """Scarica la serie storica del VIX."""
    try:
        df = yf.download("^VIX", start=start, end=end,
                         auto_adjust=True, progress=False)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df["Close"]
    except Exception as e:
        print(f"  Errore VIX storico: {e}")
        return pd.Series(dtype=float)


def download_benchmark(start: str, end: str) -> pd.Series:
    """Scarica FTSE MIB come benchmark."""
    try:
        df = yf.download("FTSEMIB.MI", start=start, end=end,
                         auto_adjust=True, progress=False)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df["Close"]
    except Exception:
        try:
            df = yf.download("^FTMIB", start=start, end=end,
                             auto_adjust=True, progress=False)
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df["Close"]
        except Exception as e:
            print(f"  Errore benchmark: {e}")
            return pd.Series(dtype=float)


# =============================================================================
# SCREENER SU DATI STORICI (senza download live)
# =============================================================================
def screen_on_date(
    date: datetime,
    prices_df: pd.DataFrame,
    vix_series: pd.Series,
    model,
    feature_cols: list,
    approved_tickers: list,
    ticker_stats: dict,
    history_days: int = 252,
) -> list:
    """
    Esegue lo screener su una data storica specifica.
    Usa solo i dati disponibili fino a quella data (no look-ahead).
    """
    # VIX del giorno
    vix = None
    if not vix_series.empty:
        past_vix = vix_series[vix_series.index <= date]
        if not past_vix.empty:
            vix = float(past_vix.iloc[-1])

    if vix and vix >= VIX_MAX:
        return []

    signals = []

    for ticker in approved_tickers:
        df_t = prices_df[
            (prices_df["ticker"] == ticker) &
            (prices_df["Date"] <= date)
        ].tail(history_days).copy()

        if len(df_t) < 60:
            continue

        ind   = compute_indicators(df_t)
        if not ind:
            continue

        score = calculate_entry_score(ind, vix)
        if score < ENTRY_SCORE_THRESHOLD:
            continue
        if ind.get("momentum_3m") and ind["momentum_3m"] < MOMENTUM_3M_MIN:
            continue

        ai_feats = build_ai_features(
            ticker, ind, vix or 18.0, {}, ticker_stats, 0, 0, date
        )
        ai_prob = score_trade(model, ai_feats, feature_cols)

        if ai_prob < AI_PROB_THRESHOLD:
            continue

        signals.append({
            "ticker":      ticker,
            "date":        date,
            "close":       ind["close"],
            "entry_score": score,
            "ai_prob":     ai_prob,
            "signal":      "FORTE" if ai_prob >= AI_PROB_STRONG else "OK",
            "rsi":         ind.get("rsi"),
            "momentum_3m": ind.get("momentum_3m"),
            "vix":         vix,
        })

    return sorted(signals, key=lambda x: x["ai_prob"], reverse=True)


# =============================================================================
# SIMULAZIONE PORTAFOGLIO
# =============================================================================
class Portfolio:
    def __init__(self, capital: float, max_positions: int):
        self.initial_capital = capital
        self.cash            = capital
        self.max_positions   = max_positions
        self.positions       = {}   # ticker -> {shares, entry_price, stop, target, trailing_active, highest}
        self.trades          = []
        self.daily_values    = []

    def position_size(self) -> float:
        """Capitale da allocare per ogni posizione."""
        return self.initial_capital / self.max_positions

    def can_enter(self, ticker: str) -> bool:
        return (
            ticker not in self.positions and
            len(self.positions) < self.max_positions and
            self.cash >= self.position_size() * 0.5
        )

    def enter(self, ticker: str, price: float, date: datetime, signal: dict):
        size    = self.position_size()
        cost    = size * TRANSACTION_COST
        shares  = (size - cost) / price
        self.cash -= size

        self.positions[ticker] = {
            "shares":           shares,
            "entry_price":      price,
            "entry_date":       date,
            "stop":             price * (1 - STOP_LOSS_PCT),
            "target":           price * (1 + TAKE_PROFIT_PCT),
            "trailing_active":  False,
            "highest":          price,
            "entry_score":      signal.get("entry_score"),
            "ai_prob":          signal.get("ai_prob"),
            "entry_vix":        signal.get("vix"),
        }

    def exit(self, ticker: str, price: float, date: datetime, reason: str):
        pos = self.positions.pop(ticker)
        proceeds = pos["shares"] * price
        cost     = proceeds * TRANSACTION_COST
        net      = proceeds - cost
        self.cash += net

        gross_ret = (price / pos["entry_price"] - 1) * 100
        net_ret   = gross_ret - TRANSACTION_COST * 2 * 100
        holding   = (date - pos["entry_date"]).days

        self.trades.append({
            "ticker":        ticker,
            "entry_date":    pos["entry_date"],
            "exit_date":     date,
            "entry_price":   pos["entry_price"],
            "exit_price":    price,
            "shares":        pos["shares"],
            "holding_days":  holding,
            "exit_reason":   reason,
            "gross_return_pct": round(gross_ret, 2),
            "net_return_pct":   round(net_ret, 2),
            "pnl_eur":          round(net - pos["shares"] * pos["entry_price"], 2),
            "entry_score":   pos.get("entry_score"),
            "ai_prob":       pos.get("ai_prob"),
            "entry_vix":     pos.get("entry_vix"),
        })

    def update_stops(self, ticker: str, current_price: float):
        """Aggiorna trailing stop se attivato."""
        pos = self.positions[ticker]
        if not pos["trailing_active"] and current_price >= pos["target"]:
            pos["trailing_active"] = True
            pos["highest"]         = current_price
            pos["stop"]            = current_price * (1 - TRAILING_STOP_PCT)
        elif pos["trailing_active"] and current_price > pos["highest"]:
            pos["highest"] = current_price
            pos["stop"]    = current_price * (1 - TRAILING_STOP_PCT)

    def total_value(self, prices: dict) -> float:
        """Valore totale portafoglio (cash + posizioni aperte)."""
        val = self.cash
        for ticker, pos in self.positions.items():
            p = prices.get(ticker, pos["entry_price"])
            val += pos["shares"] * p
        return val


# =============================================================================
# LOOP PRINCIPALE
# =============================================================================
def run_simulation(
    capital: float = INITIAL_CAPITAL,
    max_positions: int = MAX_POSITIONS,
    days: int = 365,
) -> dict:

    end_date   = datetime.today()
    start_date = end_date - timedelta(days=days + 300)  # +300 per storia indicatori

    print(f"\n{'='*60}")
    print(f"SIMULAZIONE PORTAFOGLIO — ultimi {days} giorni")
    print(f"Capitale: €{capital:,.0f}  |  Max posizioni: {max_positions}")
    print(f"Periodo:  {(end_date - timedelta(days=days)).strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")
    print(f"{'='*60}")

    # 1. Carica modello
    model        = joblib.load(MODEL_PATH)
    feature_cols = json.load(open(FEAT_PATH))

    # 2. Universe selection
    summary_file = "data/processed/target_stop_trailing_vix_regime_summary.csv"
    if os.path.exists(summary_file):
        summary_df       = pd.read_csv(summary_file)
        approved_tickers = select_universe(summary_df, verbose=False)
    else:
        approved_tickers = FTSE_MIB_TICKERS
    print(f"Ticker universe: {approved_tickers}")

    # 3. Download dati storici
    prices_df  = download_all_history(
        approved_tickers,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )
    vix_series  = download_vix_history(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )
    benchmark   = download_benchmark(
        (end_date - timedelta(days=days)).strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )

    # 4. Ticker stats storiche per AI
    ticker_stats = {}
    tl_file = "data/processed/target_stop_trailing_vix_regime_trade_log.csv"
    if os.path.exists(tl_file):
        tl = pd.read_csv(tl_file)
        for t, g in tl.groupby("ticker"):
            ticker_stats[t] = {
                "win_rate": (g["net_return_pct"] > 0).mean(),
                "count":    len(g),
            }

    # 5. Giorni di borsa del periodo simulato
    sim_start = end_date - timedelta(days=days)
    all_dates = sorted(prices_df["Date"].unique())
    sim_dates = [d for d in all_dates if pd.Timestamp(d) >= pd.Timestamp(sim_start)]

    print(f"\nGiorni di borsa simulati: {len(sim_dates)}")

    # 6. Simulazione giorno per giorno
    portfolio = Portfolio(capital, max_positions)
    daily_log = []

    for i, date in enumerate(sim_dates):
        date_ts = pd.Timestamp(date)

        # Prezzi di chiusura del giorno
        day_prices = prices_df[prices_df["Date"] == date].set_index("ticker")["Adj Close"].to_dict()

        # --- GESTIONE POSIZIONI APERTE ---
        to_exit = []
        for ticker, pos in portfolio.positions.items():
            price = day_prices.get(ticker)
            if price is None:
                continue

            portfolio.update_stops(ticker, price)

            # VIX exit
            vix_today = None
            if not vix_series.empty:
                past = vix_series[vix_series.index <= date_ts]
                if not past.empty:
                    vix_today = float(past.iloc[-1])

            if vix_today and vix_today >= VIX_EXIT:
                to_exit.append((ticker, price, "vix_exit"))
            elif price <= pos["stop"]:
                reason = "trailing_stop" if pos["trailing_active"] else "stop_loss"
                to_exit.append((ticker, price, reason))

        for ticker, price, reason in to_exit:
            portfolio.exit(ticker, price, date_ts, reason)

        # --- NUOVI SEGNALI ---
        if len(portfolio.positions) < max_positions:
            signals = screen_on_date(
                date_ts, prices_df, vix_series,
                model, feature_cols, approved_tickers, ticker_stats
            )
            for sig in signals:
                if portfolio.can_enter(sig["ticker"]):
                    # Ingresso al prezzo del giorno successivo (simulazione realistica)
                    next_dates = [d for d in sim_dates if d > date]
                    if next_dates:
                        next_date   = next_dates[0]
                        next_prices = prices_df[prices_df["Date"] == next_date].set_index("ticker")["Adj Close"]
                        entry_price = next_prices.get(sig["ticker"])
                        if entry_price and not pd.isna(entry_price):
                            portfolio.enter(sig["ticker"], float(entry_price),
                                            pd.Timestamp(next_date), sig)

        # Valore portafoglio fine giornata
        total_val = portfolio.total_value(day_prices)
        daily_log.append({
            "date":           date_ts,
            "total_value":    total_val,
            "cash":           portfolio.cash,
            "n_positions":    len(portfolio.positions),
            "open_positions": list(portfolio.positions.keys()),
        })

        if i % 50 == 0:
            print(f"  {date_ts.strftime('%Y-%m-%d')}  "
                  f"Valore: €{total_val:,.0f}  "
                  f"Posizioni: {list(portfolio.positions.keys())}")

    # Chiudi posizioni aperte alla fine
    last_date = pd.Timestamp(sim_dates[-1])
    for ticker in list(portfolio.positions.keys()):
        price = day_prices.get(ticker, portfolio.positions[ticker]["entry_price"])
        portfolio.exit(ticker, float(price), last_date, "end_of_simulation")

    return {
        "portfolio":  portfolio,
        "daily_log":  pd.DataFrame(daily_log),
        "prices_df":  prices_df,
        "benchmark":  benchmark,
        "sim_start":  sim_start,
        "sim_end":    end_date,
        "capital":    capital,
    }


# =============================================================================
# REPORT E GRAFICI
# =============================================================================
def build_report(result: dict) -> pd.DataFrame:
    portfolio  = result["portfolio"]
    daily_log  = result["daily_log"]
    benchmark  = result["benchmark"]
    capital    = result["capital"]
    sim_start  = result["sim_start"]

    trades_df  = pd.DataFrame(portfolio.trades)
    final_val  = daily_log["total_value"].iloc[-1]
    total_ret  = (final_val / capital - 1) * 100

    # Benchmark return
    bmark_ret = None
    if not benchmark.empty:
        b = benchmark[benchmark.index >= pd.Timestamp(sim_start)]
        if not b.empty:
            bmark_ret = (b.iloc[-1] / b.iloc[0] - 1) * 100

    # Drawdown
    rolling_max = daily_log["total_value"].cummax()
    drawdown    = (daily_log["total_value"] - rolling_max) / rolling_max * 100
    max_dd      = drawdown.min()

    # Sharpe ratio (approssimato, risk-free 3%)
    daily_log["daily_ret"] = daily_log["total_value"].pct_change().fillna(0)
    excess_ret = daily_log["daily_ret"] - 0.03 / 252
    sharpe     = (excess_ret.mean() / excess_ret.std() * np.sqrt(252)) if excess_ret.std() > 0 else 0

    print(f"\n{'='*60}")
    print(f"RISULTATI SIMULAZIONE")
    print(f"{'='*60}")
    print(f"Capitale iniziale:    €{capital:>10,.2f}")
    print(f"Valore finale:        €{final_val:>10,.2f}")
    print(f"Rendimento tot.:      {total_ret:>+.1f}%")
    if bmark_ret is not None:
        print(f"Benchmark (MIB):      {bmark_ret:>+.1f}%")
        print(f"Alpha vs benchmark:   {total_ret - bmark_ret:>+.1f}%")
    print(f"Max drawdown:         {max_dd:.1f}%")
    print(f"Sharpe ratio:         {sharpe:.2f}")

    if not trades_df.empty:
        wins  = trades_df[trades_df["net_return_pct"] > 0]
        loss  = trades_df[trades_df["net_return_pct"] <= 0]
        pf    = wins["net_return_pct"].sum() / abs(loss["net_return_pct"].sum()) if not loss.empty else float("inf")

        print(f"\nTrade totali:         {len(trades_df)}")
        print(f"Win rate:             {len(wins)/len(trades_df)*100:.1f}%  ({len(wins)}W / {len(loss)}L)")
        print(f"Profit factor:        {pf:.2f}")
        print(f"Rendimento medio:     {trades_df['net_return_pct'].mean():.2f}%")
        print(f"Miglior trade:        {trades_df['net_return_pct'].max():.2f}%  ({trades_df.loc[trades_df['net_return_pct'].idxmax(), 'ticker']})")
        print(f"Peggior trade:        {trades_df['net_return_pct'].min():.2f}%  ({trades_df.loc[trades_df['net_return_pct'].idxmin(), 'ticker']})")
        print(f"Holding medio:        {trades_df['holding_days'].mean():.0f} giorni")
        print(f"P&L totale:           €{trades_df['pnl_eur'].sum():,.2f}")

        print(f"\nDettaglio trade:")
        cols = ["ticker","entry_date","exit_date","holding_days",
                "entry_price","exit_price","net_return_pct","pnl_eur","exit_reason","ai_prob"]
        print(trades_df[cols].to_string(index=False))

        print(f"\nExit reasons:")
        print(trades_df["exit_reason"].value_counts().to_string())

    return trades_df


def plot_results(result: dict, trades_df: pd.DataFrame):
    os.makedirs(CHART_DIR, exist_ok=True)

    portfolio  = result["portfolio"]
    daily_log  = result["daily_log"]
    benchmark  = result["benchmark"]
    capital    = result["capital"]
    sim_start  = pd.Timestamp(result["sim_start"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    fig.suptitle(f"Simulazione Portafoglio FTSE MIB — Ultimi 365 giorni\n"
                 f"Capitale iniziale: €{capital:,.0f}",
                 fontsize=13, fontweight="bold", y=0.98)

    # --- Grafico 1: Equity curve vs benchmark ---
    ax1 = axes[0]
    norm_port = daily_log["total_value"] / capital * 100 - 100
    ax1.plot(daily_log["date"], norm_port,
             color="#1D9E75", linewidth=2, label="Portafoglio AI")
    ax1.fill_between(daily_log["date"], norm_port, 0,
                     where=norm_port >= 0, alpha=0.1, color="#1D9E75")
    ax1.fill_between(daily_log["date"], norm_port, 0,
                     where=norm_port < 0, alpha=0.1, color="#D85A30")

    if not benchmark.empty:
        b = benchmark[benchmark.index >= sim_start].copy()
        if not b.empty:
            norm_bmark = (b / b.iloc[0] * 100 - 100)
            ax1.plot(norm_bmark.index, norm_bmark.values,
                     color="#B4B2A9", linewidth=1.5,
                     linestyle="--", label="FTSE MIB (benchmark)")

    # Markers trade
    if not trades_df.empty:
        for _, t in trades_df.iterrows():
            if t["net_return_pct"] > 0:
                ax1.axvline(x=t["exit_date"], color="#1D9E75", alpha=0.2, linewidth=0.8)
            else:
                ax1.axvline(x=t["exit_date"], color="#D85A30", alpha=0.2, linewidth=0.8)

    ax1.axhline(y=0, color="gray", linewidth=0.8, linestyle="-")
    ax1.set_ylabel("Rendimento %")
    ax1.set_title("Equity Curve vs Benchmark")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # --- Grafico 2: Drawdown ---
    ax2 = axes[1]
    rolling_max = daily_log["total_value"].cummax()
    drawdown    = (daily_log["total_value"] - rolling_max) / rolling_max * 100
    ax2.fill_between(daily_log["date"], drawdown, 0,
                     color="#D85A30", alpha=0.5, label="Drawdown")
    ax2.set_ylabel("Drawdown %")
    ax2.set_title("Drawdown")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # --- Grafico 3: Trade results bar chart ---
    ax3 = axes[2]
    if not trades_df.empty:
        colors = ["#1D9E75" if r > 0 else "#D85A30" for r in trades_df["net_return_pct"]]
        bars   = ax3.bar(
            range(len(trades_df)),
            trades_df["net_return_pct"],
            color=colors, alpha=0.8, edgecolor="white", linewidth=0.5
        )
        ax3.set_xticks(range(len(trades_df)))
        ax3.set_xticklabels(
            [f"{r['ticker'].replace('.MI','')}\n{r['entry_date'].strftime('%m/%y')}"
             for _, r in trades_df.iterrows()],
            fontsize=8
        )
        ax3.axhline(y=0, color="gray", linewidth=0.8)
        ax3.set_ylabel("Rendimento netto %")
        ax3.set_title("Rendimento per trade")
        ax3.grid(True, alpha=0.3, axis="y")

        # Etichette sui bar
        for bar, (_, t) in zip(bars, trades_df.iterrows()):
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     h + (0.2 if h >= 0 else -0.5),
                     f"{h:.1f}%", ha="center", va="bottom", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(CHART_DIR, "portfolio_simulation_365d.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Grafico salvato: {out_path}")
    return out_path


def save_outputs(result: dict, trades_df: pd.DataFrame):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    daily_log = result["daily_log"]

    daily_log.to_csv(
        os.path.join(OUTPUT_DIR, "simulation_daily_values.csv"), index=False
    )
    if not trades_df.empty:
        trades_df.to_csv(
            os.path.join(OUTPUT_DIR, "simulation_trades.csv"), index=False
        )
    print(f"✅ File salvati in {OUTPUT_DIR}/")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulazione portafoglio 365 giorni")
    parser.add_argument("--capital",       type=float, default=INITIAL_CAPITAL,
                        help="Capitale iniziale in euro (default: 10000)")
    parser.add_argument("--max-positions", type=int,   default=MAX_POSITIONS,
                        help="Max posizioni contemporanee (default: 3)")
    parser.add_argument("--days",          type=int,   default=365,
                        help="Giorni da simulare (default: 365)")
    args = parser.parse_args()

    result     = run_simulation(args.capital, args.max_positions, args.days)
    trades_df  = build_report(result)
    chart_path = plot_results(result, trades_df)
    save_outputs(result, trades_df)