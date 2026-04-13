"""
strategy_precision.py  —  Fase 2
Strategia "pochi trade, alta precisione" con entry score avanzato a 9 punti.

Differenze chiave rispetto alle versioni precedenti:
  1. Entry score da 0 a 9 (era 0-7), soglia minima = 6
  2. Nuovi criteri: ADX, volume ratio, momentum 3M, above_ma200
  3. Filtro universe (da universe_selection.py) applicato prima dell'ingresso
  4. Tutti i parametri sono configurabili in cima al file
  5. Stessi meccanismi di uscita: stop loss, trailing stop, VIX exit

Entry score (massimo 9):
  1. Prezzo > MA50
  2. Prezzo > MA200  (trend di lungo periodo)        ← NUOVO
  3. MA20 > MA50  (golden cross short)
  4. MA50 slope > 0  (trend in accelerazione)
  5. RSI < soglia (non ipercomprato)
  6. ADX > 25  (trend confermato, non laterale)      ← NUOVO
  7. Volume ratio > 1  (volume sopra media)          ← NUOVO
  8. Momentum 3M > 0  (titolo in salita da 3 mesi)  ← NUOVO
  9. VIX < soglia  (regime di bassa volatilità)
  [bonus] SP500 > MA50  (mercato favorevole)        ← conta per il totale
"""

import os
import pandas as pd
import numpy as np
import ta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# PARAMETRI  —  modifica qui per ottimizzare
# =============================================================================
STRATEGY_NAME = "precision_v1"

# Risk management
TRANSACTION_COST  = 0.0015   # 0.15% per lato
STOP_LOSS_PCT     = 0.04     # -4%
TAKE_PROFIT_PCT   = 0.12     # +12%  (alzato da 10%)
TRAILING_STOP_PCT = 0.06     # trailing al 6% dal massimo

# Filtri VIX
VIX_ENTRY_THRESHOLD = 20
VIX_EXIT_THRESHOLD  = 25

# Entry score
ENTRY_SCORE_THRESHOLD = 6    # minimo 6/9 (era 4/7)
RSI_ENTRY_MAX         = 62   # RSI massimo per entrare
ADX_MIN               = 20   # ADX minimo (trend confermato)
MOMENTUM_3M_MIN       = 0.0  # momentum 3 mesi deve essere >= 0


# =============================================================================
# PREPARAZIONE DATI
# =============================================================================
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola tutti gli indicatori necessari per l'entry score avanzato.
    Si aspetta le colonne prodotte da build_features.py.
    Se alcune feature mancano, le ricalcola sul posto.
    """
    df = df.copy()
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    # Returns base
    df["return"] = df.groupby("ticker")["Adj Close"].pct_change()

    # Moving averages (ricalcola se mancano)
    for w in [20, 50, 200]:
        col = f"ma{w}"
        if col not in df.columns:
            df[col] = df.groupby("ticker")["Adj Close"].transform(
                lambda x: x.rolling(w).mean()
            )

    # MA50 slope
    if "ma50_slope" not in df.columns:
        df["ma50_slope"] = df.groupby("ticker")["ma50"].transform(lambda x: x.diff(5))

    # RSI
    if "rsi" not in df.columns:
        df["rsi"] = df.groupby("ticker")["Adj Close"].transform(
            lambda x: ta.momentum.RSIIndicator(close=x, window=14).rsi()
        )

    # ADX
    if "adx" not in df.columns:
        if "High" in df.columns and "Low" in df.columns:
            adx_parts = []
            for ticker, grp in df.groupby("ticker"):
                s = ta.trend.ADXIndicator(
                    high=grp["High"], low=grp["Low"], close=grp["Adj Close"]
                ).adx()
                s.index = grp.index
                adx_parts.append(s)
            df["adx"] = pd.concat(adx_parts).sort_index()
        else:
            df["adx"] = np.nan

    # Momentum 3M
    if "momentum_3m" not in df.columns:
        df["momentum_3m"] = df.groupby("ticker")["Adj Close"].pct_change(63)

    # Volume ratio
    if "volume_ratio" not in df.columns and "Volume" in df.columns:
        vol_ma = df.groupby("ticker")["Volume"].transform(lambda x: x.rolling(20).mean())
        df["volume_ratio"] = df["Volume"] / vol_ma
    elif "volume_ratio" not in df.columns:
        df["volume_ratio"] = np.nan

    # Macro: SP500 e VIX
    if "SP500" in df.columns:
        df["sp500_ma50"] = df["SP500"].rolling(50).mean()
        df["sp500_above_ma50"] = (df["SP500"] > df["sp500_ma50"]).astype(int)
    else:
        df["sp500_above_ma50"] = 0

    if "VIX" in df.columns:
        df["vix_ma10"] = df["VIX"].rolling(10).mean()
    else:
        df["VIX"] = np.nan

    # Calcola entry score
    df["entry_score"] = df.apply(_calculate_entry_score, axis=1)

    return df


def _calculate_entry_score(row: pd.Series) -> int:
    """
    Calcola entry score da 0 a 9 (+ eventuale punto bonus SP500).
    Ogni criterio vale 1 punto.
    """
    score = 0

    # 1. Prezzo > MA50
    if pd.notna(row.get("ma50")) and row["Adj Close"] > row["ma50"]:
        score += 1

    # 2. Prezzo > MA200  (NUOVO — trend di lungo periodo)
    if pd.notna(row.get("ma200")) and row["Adj Close"] > row["ma200"]:
        score += 1

    # 3. MA20 > MA50  (golden cross)
    if pd.notna(row.get("ma20")) and pd.notna(row.get("ma50")) and row["ma20"] > row["ma50"]:
        score += 1

    # 4. MA50 slope positivo
    if pd.notna(row.get("ma50_slope")) and row["ma50_slope"] > 0:
        score += 1

    # 5. RSI non ipercomprato
    if pd.notna(row.get("rsi")) and row["rsi"] < RSI_ENTRY_MAX:
        score += 1

    # 6. ADX > soglia  (NUOVO — trend confermato)
    adx = row.get("adx")
    if pd.notna(adx) and adx >= ADX_MIN:
        score += 1

    # 7. Volume sopra media  (NUOVO)
    vr = row.get("volume_ratio")
    if pd.notna(vr) and vr >= 1.0:
        score += 1

    # 8. Momentum 3M positivo  (NUOVO)
    mom = row.get("momentum_3m")
    if pd.notna(mom) and mom >= MOMENTUM_3M_MIN:
        score += 1

    # 9. VIX sotto soglia
    vix = row.get("VIX")
    if pd.notna(vix) and vix < VIX_ENTRY_THRESHOLD:
        score += 1

    return score


def _entry_condition(row: pd.Series) -> bool:
    """
    Condizione di ingresso:
      - score >= soglia
      - VIX obbligatoriamente sotto la soglia (gate duro)
      - momentum 3M non fortemente negativo
    """
    if pd.isna(row.get("entry_score")):
        return False

    # Gate duro VIX
    vix = row.get("VIX")
    if pd.notna(vix) and vix >= VIX_ENTRY_THRESHOLD:
        return False

    # Gate duro momentum (non entrare su titolo in caduta)
    mom = row.get("momentum_3m")
    if pd.notna(mom) and mom < -0.10:   # caduta > 10% in 3 mesi
        return False

    return row["entry_score"] >= ENTRY_SCORE_THRESHOLD


# =============================================================================
# ESECUZIONE STRATEGIA
# =============================================================================
def run_strategy_for_ticker(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("Date").copy().reset_index(drop=True)

    group["signal"]           = 0
    group["position"]         = 0.0
    group["entry_price"]      = pd.NA
    group["stop_price"]       = pd.NA
    group["target_price"]     = pd.NA
    group["trailing_active"]  = False
    group["highest_price"]    = pd.NA
    group["exit_reason"]      = pd.NA
    group["trade_id"]         = pd.NA
    group["entry_flag_event"] = 0
    group["exit_flag_event"]  = 0
    group["entry_vix_event"]  = pd.NA
    group["exit_vix_event"]   = pd.NA
    group["exit_reason_event"]= pd.NA
    group["entry_price_event"]= pd.NA
    group["exit_price_event"] = pd.NA
    group["entry_score_event"]= pd.NA

    in_position    = False
    entry_price    = None
    stop_price     = None
    target_price   = None
    trailing_active= False
    highest_price  = None
    trade_id       = 0

    for i in range(len(group)):
        row         = group.iloc[i]
        close_price = row["Adj Close"]
        current_vix = row.get("VIX", pd.NA)

        position_today = 0
        signal_today   = 0
        exit_reason    = pd.NA

        if not in_position:
            if _entry_condition(row):
                in_position    = True
                trade_id      += 1
                entry_price    = close_price
                stop_price     = entry_price * (1 - STOP_LOSS_PCT)
                target_price   = entry_price * (1 + TAKE_PROFIT_PCT)
                trailing_active= False
                highest_price  = entry_price
                signal_today   = 1
                position_today = 0

                group.at[i, "entry_flag_event"]  = 1
                group.at[i, "entry_vix_event"]   = current_vix
                group.at[i, "entry_price_event"] = entry_price
                group.at[i, "entry_score_event"] = row.get("entry_score", pd.NA)
                group.at[i, "entry_price"]       = entry_price
                group.at[i, "stop_price"]        = stop_price
                group.at[i, "target_price"]      = target_price
                group.at[i, "trade_id"]          = trade_id
            else:
                group.at[i, "trade_id"] = pd.NA

        else:
            signal_today   = 1
            position_today = 1

            # Exit VIX
            if pd.notna(current_vix) and current_vix >= VIX_EXIT_THRESHOLD:
                exit_reason = "vix_exit"
                group.at[i, "exit_flag_event"]   = 1
                group.at[i, "exit_vix_event"]    = current_vix
                group.at[i, "exit_reason_event"] = exit_reason
                group.at[i, "exit_price_event"]  = close_price
                group.at[i, "trade_id"]          = trade_id
                in_position     = False
                signal_today    = 0
                group.at[i, "signal"]     = signal_today
                group.at[i, "position"]   = position_today
                group.at[i, "exit_reason"]= exit_reason
                entry_price = stop_price = target_price = None
                trailing_active = False; highest_price = None
                group.at[i, "entry_price"] = group.at[i, "stop_price"] = pd.NA
                group.at[i, "target_price"] = group.at[i, "highest_price"] = pd.NA
                continue

            # Attivazione trailing
            if not trailing_active and close_price >= target_price:
                trailing_active = True
                highest_price   = close_price
                stop_price      = highest_price * (1 - TRAILING_STOP_PCT)

            # Aggiornamento trailing
            if trailing_active and close_price > highest_price:
                highest_price = close_price
                stop_price    = highest_price * (1 - TRAILING_STOP_PCT)

            # Uscita stop
            if close_price <= stop_price:
                exit_reason = "trailing_stop" if trailing_active else "stop_loss"
                group.at[i, "exit_flag_event"]   = 1
                group.at[i, "exit_vix_event"]    = current_vix
                group.at[i, "exit_reason_event"] = exit_reason
                group.at[i, "exit_price_event"]  = close_price
                in_position     = False
                signal_today    = 0
                entry_price = stop_price = target_price = None
                trailing_active = False; highest_price = None

            group.at[i, "trade_id"] = trade_id

        group.at[i, "signal"]        = signal_today
        group.at[i, "position"]      = position_today
        group.at[i, "exit_reason"]   = exit_reason
        group.at[i, "entry_price"]   = entry_price   if in_position else pd.NA
        group.at[i, "stop_price"]    = stop_price    if in_position else pd.NA
        group.at[i, "target_price"]  = target_price  if in_position else pd.NA
        group.at[i, "trailing_active"]= trailing_active
        group.at[i, "highest_price"] = highest_price if in_position else pd.NA

    # Fine dati: chiudi posizione aperta
    last_open = group[(group["entry_flag_event"] == 1)].index
    last_exit = group[(group["exit_flag_event"] == 1)].index
    if len(last_open) > len(last_exit):
        idx = group.index[-1]
        group.at[idx, "exit_flag_event"]   = 1
        group.at[idx, "exit_reason_event"] = "end_of_data"
        group.at[idx, "exit_price_event"]  = group.at[idx, "Adj Close"]

    return group


def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for ticker, grp in df.groupby("ticker"):
        parts.append(run_strategy_for_ticker(grp))
    return pd.concat(parts, ignore_index=True)


# =============================================================================
# BACKTEST
# =============================================================================
def _calculate_drawdown(series: pd.Series) -> pd.Series:
    roll_max = series.cummax()
    return (series - roll_max) / roll_max


def backtest(df: pd.DataFrame, transaction_cost: float = TRANSACTION_COST) -> pd.DataFrame:
    df = df.copy()
    df["position_change"] = df.groupby("ticker")["position"].diff().fillna(0).abs()
    df["transaction_cost"] = df["position_change"] * transaction_cost

    df["daily_return_market"]   = df.groupby("ticker")["Adj Close"].pct_change().fillna(0)
    df["daily_return_strategy"] = df["daily_return_market"] * df["position"]

    df["cum_market"] = df.groupby("ticker")["daily_return_market"].transform(
        lambda x: (1 + x).cumprod()
    )
    df["cum_strategy_gross"] = df.groupby("ticker")["daily_return_strategy"].transform(
        lambda x: (1 + x).cumprod()
    )

    costs = df.groupby("ticker")["transaction_cost"].transform("cumsum")
    df["cum_strategy_net"] = df["cum_strategy_gross"] - costs

    return df


# =============================================================================
# TRADE LOG E SUMMARY
# =============================================================================
def build_trade_log(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("Date").reset_index(drop=True)

        entries = grp[grp["entry_flag_event"] == 1]
        exits   = grp[grp["exit_flag_event"]  == 1]

        n = min(len(entries), len(exits))
        for i in range(n):
            e_row = entries.iloc[i]
            x_row = exits.iloc[i]

            entry_price = e_row.get("entry_price_event", pd.NA)
            exit_price  = x_row.get("exit_price_event", pd.NA)
            entry_date  = pd.to_datetime(e_row["Date"])
            exit_date   = pd.to_datetime(x_row["Date"])

            if pd.notna(entry_price) and pd.notna(exit_price) and entry_price != 0:
                gross_return = (exit_price / entry_price) - 1
                net_return   = gross_return - (2 * TRANSACTION_COST)
            else:
                gross_return = net_return = pd.NA

            holding_days = (exit_date - entry_date).days if pd.notna(entry_date) and pd.notna(exit_date) else pd.NA

            records.append({
                "ticker":         ticker,
                "trade_num":      i + 1,
                "entry_date":     entry_date,
                "entry_price":    entry_price,
                "entry_vix":      e_row.get("entry_vix_event", pd.NA),
                "entry_score":    e_row.get("entry_score_event", pd.NA),
                "exit_date":      exit_date,
                "exit_price":     exit_price,
                "exit_vix":       x_row.get("exit_vix_event", pd.NA),
                "exit_reason":    x_row.get("exit_reason_event", pd.NA),
                "holding_days":   holding_days,
                "gross_return_pct": gross_return * 100 if pd.notna(gross_return) else pd.NA,
                "net_return_pct":   net_return   * 100 if pd.notna(net_return)   else pd.NA,
            })

    return pd.DataFrame(records)


def _max_consecutive_losses(returns: pd.Series) -> int:
    max_consec = 0
    current    = 0
    for r in returns:
        if pd.notna(r) and r < 0:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0
    return max_consec


def build_summary(df: pd.DataFrame, trade_log: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("ticker").agg(
        first_date              = ("Date", "min"),
        last_date               = ("Date", "max"),
        rows                    = ("Date", "count"),
        days_in_position        = ("position", "sum"),
        total_cost              = ("transaction_cost", "sum"),
        total_return_market     = ("cum_market", "last"),
        total_return_strategy_net=("cum_strategy_net", "last"),
    ).reset_index()

    summary["market_perf_pct"]      = (summary["total_return_market"]       - 1) * 100
    summary["strategy_net_perf_pct"]= (summary["total_return_strategy_net"] - 1) * 100
    summary["alpha_net_pct"]        = summary["strategy_net_perf_pct"] - summary["market_perf_pct"]

    dd_records = []
    for ticker, grp in df.groupby("ticker"):
        dd_records.append({
            "ticker":                       ticker,
            "max_drawdown_market_pct":      _calculate_drawdown(grp["cum_market"]).min()          * 100,
            "max_drawdown_strategy_net_pct":_calculate_drawdown(grp["cum_strategy_net"]).min()    * 100,
        })
    summary = summary.merge(pd.DataFrame(dd_records), on="ticker", how="left")

    if not trade_log.empty:
        stats = []
        for ticker, trades in trade_log.groupby("ticker"):
            trades = trades.sort_values("entry_date")
            n = len(trades)
            wins   = trades[trades["net_return_pct"] > 0]["net_return_pct"]
            losses = trades[trades["net_return_pct"] < 0]["net_return_pct"]
            pf     = wins.sum() / abs(losses.sum()) if not losses.empty and losses.sum() != 0 else np.nan
            ec     = trades["exit_reason"].value_counts()

            stats.append({
                "ticker":              ticker,
                "closed_trades":       n,
                "avg_trade_return_pct":trades["net_return_pct"].mean(),
                "win_rate_pct":        (trades["net_return_pct"] > 0).mean() * 100 if n > 0 else np.nan,
                "profit_factor":       pf,
                "avg_holding_days":    trades["holding_days"].mean(),
                "avg_entry_score":     trades["entry_score"].mean() if "entry_score" in trades.columns else np.nan,
                "max_consecutive_losses": _max_consecutive_losses(trades["net_return_pct"]),
                "num_stop_loss":       int(ec.get("stop_loss",   0)),
                "num_trailing_stop":   int(ec.get("trailing_stop",0)),
                "num_vix_exit":        int(ec.get("vix_exit",    0)),
                "num_end_of_data":     int(ec.get("end_of_data", 0)),
            })
        summary = summary.merge(pd.DataFrame(stats), on="ticker", how="left")

    return summary


# =============================================================================
# PLOT
# =============================================================================
def plot_equity_curve(df: pd.DataFrame, ticker: str, output_dir: str) -> str | None:
    data = df[df["ticker"] == ticker].sort_values("Date")
    if data.empty:
        return None
    plt.figure(figsize=(12, 5))
    plt.plot(data["Date"], data["cum_market"],        label="Market",        color="#B4B2A9", linewidth=1.5)
    plt.plot(data["Date"], data["cum_strategy_net"],  label="Strategia net", color="#1D9E75", linewidth=2)
    plt.title(f"{STRATEGY_NAME} — Equity Curve — {ticker}")
    plt.xlabel("Data"); plt.ylabel("Rendimento cumulativo")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    path = os.path.join(output_dir, f"{STRATEGY_NAME}_equity_{ticker.replace('.', '_')}.png")
    plt.savefig(path, dpi=120); plt.close()
    return path


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from universe_selection import select_universe, rank_universe

    dataset_file = "data/processed/dataset_extended.csv"
    prices_file  = "data/raw/prices_extended.csv"
    output_dir   = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    input_file = dataset_file if os.path.exists(dataset_file) else prices_file
    if not os.path.exists(input_file):
        print(f"File non trovato: {input_file}")
        sys.exit(1)

    print(f"Lettura dati: {input_file}")
    df = pd.read_csv(input_file)
    df["Date"] = pd.to_datetime(df["Date"])

    # --- Fase 1: Universe selection ---
    summary_file = "data/processed/target_stop_trailing_vix_regime_summary.csv"
    if os.path.exists(summary_file):
        summary_df = pd.read_csv(summary_file)
        approved_tickers = select_universe(summary_df, verbose=True)
        df = df[df["ticker"].isin(approved_tickers)].copy()
        print(f"Titoli dopo universe selection: {df['ticker'].nunique()}")
    else:
        print("Summary non trovato, applico strategia su tutti i ticker.")

    # --- Fase 2: Strategia precision ---
    print("Calcolo feature...")
    df = prepare_data(df)

    print("Eseguo strategia...")
    df = apply_strategy(df)
    df = backtest(df)

    trade_log = build_trade_log(df)
    summary   = build_summary(df, trade_log)

    # Output
    out_bt    = os.path.join(output_dir, f"{STRATEGY_NAME}_backtest.csv")
    out_sum   = os.path.join(output_dir, f"{STRATEGY_NAME}_summary.csv")
    out_trades= os.path.join(output_dir, f"{STRATEGY_NAME}_trade_log.csv")

    df.to_csv(out_bt, index=False)
    summary.to_csv(out_sum, index=False)
    trade_log.to_csv(out_trades, index=False)

    print("\n===== SUMMARY =====")
    cols = ["ticker", "strategy_net_perf_pct", "market_perf_pct", "alpha_net_pct",
            "win_rate_pct", "profit_factor", "closed_trades", "avg_entry_score"]
    available = [c for c in cols if c in summary.columns]
    print(summary[available].sort_values("profit_factor", ascending=False).to_string(index=False))

    if not trade_log.empty:
        n_trades = len(trade_log)
        n_months = (df["Date"].max() - df["Date"].min()).days / 30
        print(f"\nTrade totali: {n_trades}")
        print(f"Media mensile: {n_trades / n_months:.1f} trade/mese")
        print(f"Win rate medio: {trade_log['net_return_pct'].gt(0).mean()*100:.1f}%")

    # Grafico per il primo ticker approvato
    if not summary.empty:
        best = summary.sort_values("profit_factor", ascending=False).iloc[0]["ticker"]
        chart = plot_equity_curve(df, best, output_dir)
        if chart:
            print(f"\n✅ Grafico salvato: {chart}")

    print(f"\n✅ Backtest: {out_bt}")
    print(f"✅ Summary:  {out_sum}")
    print(f"✅ Trade log:{out_trades}")