"""
universe_selection.py  —  Fase 1
Filtra l'universo di titoli prima ancora di cercare segnali di ingresso.

Logica:
  Per ogni ticker calcola metriche di "compatibilità" con la strategia
  basandosi sui summary del backtest storico + caratteristiche strutturali.
  Restituisce solo i ticker che superano tutti i criteri minimi.

Utilizzo:
    from src.universe_selection import select_universe
    tickers_ok = select_universe(summary_df, prices_df)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Criteri di selezione (tutti parametrizzabili)
# ---------------------------------------------------------------------------
MIN_PROFIT_FACTOR      = 1.30   # profit factor minimo storico
MIN_WIN_RATE           = 38.0   # win rate % minimo
MAX_DRAWDOWN           = -55.0  # max drawdown strategia (soglia negativa)
MIN_CLOSED_TRADES      = 20     # esclude titoli con troppo pochi trade
MAX_AVG_HOLDING_DAYS   = 90     # esclude titoli con holding troppo lungo
MIN_AVG_TRADE_RETURN   = 0.0    # rendimento medio per trade > 0

# Criteri su prezzi (calcolati in tempo reale)
MIN_MOMENTUM_3M        = -0.05  # momentum 3 mesi > -5%  (non in caduta libera)
MAX_VOLATILITY_20      = 0.045  # volatilità daily std < 4.5%  (no titoli iper-volatili)
MIN_DAYS_ABOVE_MA200   = 0.50   # almeno 50% dei giorni degli ultimi 252 sopra MA200


# ---------------------------------------------------------------------------
# Selezione da summary storico
# ---------------------------------------------------------------------------
def filter_by_backtest_stats(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prende il summary del backtest e restituisce solo i ticker
    che rispettano i criteri minimi.

    summary_df deve avere le colonne prodotte da build_summary():
        ticker, profit_factor, win_rate_pct, max_drawdown_strategy_net_pct,
        closed_trades, avg_holding_days, avg_trade_return_pct
    """
    df = summary_df.copy()

    mask = (
        (df["profit_factor"]                   >= MIN_PROFIT_FACTOR)    &
        (df["win_rate_pct"]                    >= MIN_WIN_RATE)          &
        (df["max_drawdown_strategy_net_pct"]   >= MAX_DRAWDOWN)          &
        (df["closed_trades"]                   >= MIN_CLOSED_TRADES)     &
        (df["avg_holding_days"]                <= MAX_AVG_HOLDING_DAYS)  &
        (df["avg_trade_return_pct"]            >= MIN_AVG_TRADE_RETURN)
    )

    selected = df[mask].copy()
    selected = selected.sort_values("profit_factor", ascending=False)
    return selected


# ---------------------------------------------------------------------------
# Selezione da dati prezzi recenti
# ---------------------------------------------------------------------------
def filter_by_price_structure(prices_df: pd.DataFrame, lookback_days: int = 252) -> pd.DataFrame:
    """
    Analizza la struttura di prezzo recente e scarta titoli:
      - troppo volatili
      - in downtrend strutturale (sotto MA200 per > 50% dei giorni)
      - con momentum 3 mesi molto negativo

    prices_df: dataframe con colonne Date, ticker, Adj Close, volatility_20,
               momentum_3m, above_ma200  (output di build_features)
    """
    results = []

    for ticker, grp in prices_df.groupby("ticker"):
        grp = grp.sort_values("Date")
        recent = grp.tail(lookback_days)

        if recent.empty:
            continue

        avg_vol = recent["volatility_20"].mean() if "volatility_20" in recent.columns else np.nan
        last_mom3m = recent["momentum_3m"].iloc[-1] if "momentum_3m" in recent.columns else 0
        pct_above_ma200 = recent["above_ma200"].mean() if "above_ma200" in recent.columns else 1.0

        passes = (
            (pd.isna(avg_vol) or avg_vol <= MAX_VOLATILITY_20)  and
            (last_mom3m >= MIN_MOMENTUM_3M)                      and
            (pct_above_ma200 >= MIN_DAYS_ABOVE_MA200)
        )

        results.append({
            "ticker":           ticker,
            "avg_volatility":   round(avg_vol, 4) if not pd.isna(avg_vol) else np.nan,
            "momentum_3m_last": round(last_mom3m, 4),
            "pct_above_ma200":  round(pct_above_ma200, 3),
            "passes_price":     passes,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------
def select_universe(
    summary_df: pd.DataFrame,
    prices_df: pd.DataFrame | None = None,
    verbose: bool = True,
) -> list[str]:
    """
    Combina i due filtri e restituisce la lista dei ticker approvati.

    Args:
        summary_df:  output di build_summary() del backtest
        prices_df:   output di build_features() — opzionale
                     se None, viene saltato il filtro su prezzi
        verbose:     stampa il riepilogo

    Returns:
        lista di ticker string approvati
    """
    # --- Filtro 1: backtest stats ---
    approved_bt = filter_by_backtest_stats(summary_df)

    if verbose:
        print(f"\n{'='*55}")
        print(f"UNIVERSE SELECTION")
        print(f"{'='*55}")
        print(f"Totale ticker in input:        {summary_df['ticker'].nunique()}")
        print(f"Passano filtro backtest stats:  {len(approved_bt)}")
        print(f"\nTicker approvati (backtest):")
        cols = ["ticker", "profit_factor", "win_rate_pct",
                "max_drawdown_strategy_net_pct", "closed_trades", "avg_trade_return_pct"]
        available = [c for c in cols if c in approved_bt.columns]
        print(approved_bt[available].to_string(index=False))

    if prices_df is None:
        final_tickers = approved_bt["ticker"].tolist()
        if verbose:
            print(f"\nFiltro prezzi saltato (prices_df non fornito).")
            print(f"Ticker finali: {final_tickers}")
        return final_tickers

    # --- Filtro 2: struttura prezzi ---
    price_filter = filter_by_price_structure(prices_df)
    approved_price = price_filter[price_filter["passes_price"]]["ticker"].tolist()

    if verbose:
        print(f"\nFiltro struttura prezzi:")
        print(price_filter.to_string(index=False))
        print(f"\nPassano filtro prezzi: {len(approved_price)}")

    # Intersezione
    final_tickers = [t for t in approved_bt["ticker"] if t in approved_price]

    if verbose:
        print(f"\n{'='*55}")
        print(f"UNIVERSE FINALE: {len(final_tickers)} titoli")
        print(f"  {final_tickers}")
        print(f"{'='*55}\n")

    return final_tickers


# ---------------------------------------------------------------------------
# Ranking qualitativo dei titoli approvati
# ---------------------------------------------------------------------------
def rank_universe(summary_df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Dato il summary e la lista dei ticker approvati,
    calcola un rank composito per prioritizzare su quali titoli
    concentrarsi nella strategia.

    Score composito = rank(profit_factor) + rank(win_rate) + rank(-drawdown)
    """
    df = summary_df[summary_df["ticker"].isin(tickers)].copy()

    df["rank_pf"]  = df["profit_factor"].rank(ascending=False)
    df["rank_wr"]  = df["win_rate_pct"].rank(ascending=False)
    df["rank_dd"]  = df["max_drawdown_strategy_net_pct"].rank(ascending=False)  # meno negativo = meglio
    df["rank_ret"] = df["avg_trade_return_pct"].rank(ascending=False)

    df["composite_score"] = df[["rank_pf", "rank_wr", "rank_dd", "rank_ret"]].sum(axis=1)
    df = df.sort_values("composite_score")
    df["universe_rank"] = range(1, len(df) + 1)

    cols = ["universe_rank", "ticker", "profit_factor", "win_rate_pct",
            "max_drawdown_strategy_net_pct", "avg_trade_return_pct", "closed_trades"]
    available = [c for c in cols if c in df.columns]
    return df[available].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Script standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    summary_file = "data/processed/target_stop_trailing_vix_regime_summary.csv"
    if not os.path.exists(summary_file):
        print(f"File non trovato: {summary_file}")
        sys.exit(1)

    summary_df = pd.read_csv(summary_file)

    # Prova anche con summary score se presente
    score_file = "data/processed/target_stop_trailing_vix_score_summary.csv"
    if os.path.exists(score_file):
        score_df = pd.read_csv(score_file)
        # Usa la media tra le due strategie per il profit factor
        merged = summary_df[["ticker", "profit_factor", "win_rate_pct",
                              "max_drawdown_strategy_net_pct", "closed_trades",
                              "avg_holding_days", "avg_trade_return_pct"]].copy()
        print("Usando summary VIX Regime per la selezione.")
    else:
        merged = summary_df

    tickers = select_universe(merged, verbose=True)

    ranked = rank_universe(merged, tickers)
    print("\nRANKING UNIVERSE:")
    print(ranked.to_string(index=False))

    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)
    ranked.to_csv(os.path.join(out_dir, "universe_selected.csv"), index=False)
    print(f"\n✅ Universe salvato in: {out_dir}/universe_selected.csv")