import os
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt


# =========================
# PARAMETRI STRATEGIA
# =========================
STRATEGY_NAME = "target_stop_trailing_no_vix_exit"

TRANSACTION_COST = 0.0015      # 0.15% per ingresso o uscita
USE_VIX_FILTER = True          # usa il VIX se presente nel dataset
VIX_ENTRY_THRESHOLD = 20       # entro solo se VIX sotto 20
VIX_EXIT_THRESHOLD = 25        # se sono dentro ed il VIX va sopra 25, esco
RSI_ENTRY_THRESHOLD = 65       # entra solo se RSI sotto questa soglia
STOP_LOSS_PCT = 0.04           # stop loss iniziale -4%
TAKE_PROFIT_PCT = 0.10         # target +10%
TRAILING_STOP_PCT = 0.05       # trailing stop al 5% dal massimo
USE_MA50_FILTER = True         # ingresso solo se prezzo sopra MA50


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara il dataset calcolando indicatori tecnici base.

    Colonne create:
    - return: rendimento giornaliero
    - ma50: media mobile a 50 giorni
    - rsi: Relative Strength Index a 14 periodi

    Nota:
    - Adj Close è il prezzo adjusted, corretto per dividendi/split.
    - MA50 serve per identificare un trend di medio periodo.
    - RSI misura il momentum; sotto una certa soglia evitiamo ingressi
      troppo "tirati".
    """
    df = df.copy()
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    df["return"] = df.groupby("ticker")["Adj Close"].pct_change()

    df["ma50"] = df.groupby("ticker")["Adj Close"].transform(
        lambda x: x.rolling(50).mean()
    )

    df["rsi"] = df.groupby("ticker")["Adj Close"].transform(
        lambda x: ta.momentum.RSIIndicator(close=x, window=14).rsi()
    )

    return df


def entry_condition(row: pd.Series) -> bool:
    """
    Regola di ingresso.

    Entra se:
    - abbiamo MA50 e RSI disponibili
    - prezzo sopra MA50 (trend positivo), se attivo il filtro
    - RSI sotto soglia
    - VIX sotto soglia, se richiesto e se presente

    Cos'è il VIX:
    - è un indicatore di volatilità attesa del mercato USA
    - in pratica lo usiamo come proxy di "paura del mercato"
    - VIX alto = mercato nervoso
    - VIX basso = contesto più tranquillo
    """
    if pd.isna(row["rsi"]):
        return False

    if USE_MA50_FILTER:
        if pd.isna(row["ma50"]):
            return False
        if row["Adj Close"] <= row["ma50"]:
            return False

    if row["rsi"] >= RSI_ENTRY_THRESHOLD:
        return False

    if USE_VIX_FILTER and "VIX" in row.index:
        if pd.isna(row["VIX"]):
            return False
        if row["VIX"] >= VIX_ENTRY_THRESHOLD:
            return False

    return True


def run_strategy_for_ticker(group: pd.DataFrame) -> pd.DataFrame:
    """
    Esegue la strategia su un singolo ticker.

    Stato del trade:
    - in_position: siamo dentro o fuori
    - entry_price: prezzo di ingresso
    - stop_price: stop loss corrente
    - target_price: primo target di profitto
    - trailing_active: si attiva dopo aver superato il target
    - highest_price: massimo toccato dopo attivazione trailing

    Uscita:
    - se il VIX supera la soglia => exit di regime
    - se il prezzo scende sotto stop_price => exit
    - se raggiunge target => NON usciamo subito, ma attiviamo trailing
    - una volta attivo il trailing, lo stop sale con i nuovi massimi

    Nota importante:
    - gli eventi di trade vengono salvati in colonne dedicate:
      entry_flag_event, exit_flag_event, exit_reason_event, entry_vix_event, exit_vix_event
    - queste colonne servono poi per costruire un trade log robusto
    """
    group = group.sort_values("Date").copy().reset_index(drop=True)

    # Colonne di stato/output
    group["signal"] = 0
    group["position"] = 0.0
    group["entry_price"] = pd.NA
    group["stop_price"] = pd.NA
    group["target_price"] = pd.NA
    group["trailing_active"] = False
    group["highest_price"] = pd.NA
    group["exit_reason"] = pd.NA
    group["trade_id"] = pd.NA

    # Eventi espliciti trade
    group["entry_flag_event"] = 0
    group["exit_flag_event"] = 0
    group["entry_vix_event"] = pd.NA
    group["exit_vix_event"] = pd.NA
    group["exit_reason_event"] = pd.NA
    group["entry_price_event"] = pd.NA
    group["exit_price_event"] = pd.NA

    in_position = False
    entry_price = None
    stop_price = None
    target_price = None
    trailing_active = False
    highest_price = None
    trade_id = 0

    for i in range(len(group)):
        row = group.iloc[i]
        close_price = row["Adj Close"]
        current_vix = row["VIX"] if "VIX" in group.columns else pd.NA

        position_today = 0
        signal_today = 0
        exit_reason = pd.NA

        if not in_position:
            if entry_condition(row):
                in_position = True
                trade_id += 1

                entry_price = close_price
                stop_price = entry_price * (1 - STOP_LOSS_PCT)
                target_price = entry_price * (1 + TAKE_PROFIT_PCT)
                trailing_active = False
                highest_price = entry_price

                signal_today = 1
                position_today = 0  # ingresso economico dal giorno successivo

                group.at[i, "entry_flag_event"] = 1
                group.at[i, "entry_vix_event"] = current_vix
                group.at[i, "entry_price_event"] = entry_price

                group.at[i, "entry_price"] = entry_price
                group.at[i, "stop_price"] = stop_price
                group.at[i, "target_price"] = target_price
                group.at[i, "trailing_active"] = trailing_active
                group.at[i, "highest_price"] = highest_price
                group.at[i, "trade_id"] = trade_id
            else:
                group.at[i, "trade_id"] = pd.NA

        else:
            signal_today = 1
            position_today = 1

            # Exit di regime VIX
            # Exit di regime VIX DISABILITATA
            if False:
                    exit_reason = "vix_exit"

                    group.at[i, "exit_flag_event"] = 1
                    group.at[i, "exit_vix_event"] = current_vix
                    group.at[i, "exit_reason_event"] = exit_reason
                    group.at[i, "exit_price_event"] = close_price
                    group.at[i, "trade_id"] = trade_id

                    in_position = False
                    signal_today = 0

                    group.at[i, "signal"] = signal_today
                    group.at[i, "position"] = position_today
                    group.at[i, "exit_reason"] = exit_reason

                    entry_price = None
                    stop_price = None
                    target_price = None
                    trailing_active = False
                    highest_price = None

                    group.at[i, "entry_price"] = pd.NA
                    group.at[i, "stop_price"] = pd.NA
                    group.at[i, "target_price"] = pd.NA
                    group.at[i, "trailing_active"] = False
                    group.at[i, "highest_price"] = pd.NA
                    continue

            # Attivazione trailing se si supera il target
            if not trailing_active and close_price >= target_price:
                trailing_active = True
                highest_price = close_price
                stop_price = highest_price * (1 - TRAILING_STOP_PCT)

            # Aggiornamento trailing stop sui nuovi massimi
            if trailing_active:
                if close_price > highest_price:
                    highest_price = close_price
                    stop_price = highest_price * (1 - TRAILING_STOP_PCT)

            # Uscita per stop
            if close_price <= stop_price:
                exit_reason = "trailing_stop" if trailing_active else "stop_loss"

                group.at[i, "exit_flag_event"] = 1
                group.at[i, "exit_vix_event"] = current_vix
                group.at[i, "exit_reason_event"] = exit_reason
                group.at[i, "exit_price_event"] = close_price

                in_position = False
                signal_today = 0

                # economicamente il giorno dell'uscita è ancora position=1
                entry_price = None
                stop_price = None
                target_price = None
                trailing_active = False
                highest_price = None

            group.at[i, "trade_id"] = trade_id
            group.at[i, "entry_price"] = entry_price if entry_price is not None else pd.NA
            group.at[i, "stop_price"] = stop_price if stop_price is not None else pd.NA
            group.at[i, "target_price"] = target_price if target_price is not None else pd.NA
            group.at[i, "trailing_active"] = trailing_active
            group.at[i, "highest_price"] = highest_price if highest_price is not None else pd.NA

        group.at[i, "signal"] = signal_today
        group.at[i, "position"] = position_today
        group.at[i, "exit_reason"] = exit_reason

    # Chiusura forzata a fine serie se rimane una posizione aperta
    if in_position and len(group) > 0:
        last_idx = group.index[-1]
        last_row = group.iloc[-1]
        last_vix = last_row["VIX"] if "VIX" in group.columns else pd.NA
        last_price = last_row["Adj Close"]

        group.at[last_idx, "exit_flag_event"] = 1
        group.at[last_idx, "exit_vix_event"] = last_vix
        group.at[last_idx, "exit_reason_event"] = "end_of_data"
        group.at[last_idx, "exit_price_event"] = last_price

        if pd.isna(group.at[last_idx, "exit_reason"]):
            group.at[last_idx, "exit_reason"] = "end_of_data"

    # Shift posizione reale:
    # il segnale viene deciso a fine giornata, la posizione reale parte dal giorno dopo
    group["position"] = group["signal"].shift(1).fillna(0)

    # Ricalcolo trade_id sugli ingressi effettivi
    group["entry_flag"] = (
        (group["signal"] == 1) & (group["signal"].shift(1).fillna(0) == 0)
    ).astype(int)
    group["trade_id"] = group["entry_flag"].cumsum()
    group.loc[group["signal"] == 0, "trade_id"] = pd.NA

    return group


def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applica la strategia a tutti i ticker.
    """
    parts = []

    for ticker, group in df.groupby("ticker"):
        print(f"Esecuzione strategia su {ticker}...")
        result = run_strategy_for_ticker(group)
        parts.append(result)

    final_df = pd.concat(parts, ignore_index=True)
    final_df = final_df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    return final_df


def backtest(df: pd.DataFrame, transaction_cost: float = TRANSACTION_COST) -> pd.DataFrame:
    """
    Esegue il backtest economico della strategia.

    - strategy_return_gross: rendimento lordo
    - position_change: ingresso/uscita
    - transaction_cost: costo di transazione
    - strategy_return_net: rendimento al netto dei costi
    """
    df = df.copy()

    df["strategy_return_gross"] = df["position"] * df["return"]

    df["position_change"] = df.groupby("ticker")["position"].diff().abs().fillna(df["position"].abs())

    df["transaction_cost"] = df["position_change"] * transaction_cost
    df["strategy_return_net"] = df["strategy_return_gross"] - df["transaction_cost"]

    df["return_filled"] = df["return"].fillna(0)
    df["strategy_return_gross_filled"] = df["strategy_return_gross"].fillna(0)
    df["strategy_return_net_filled"] = df["strategy_return_net"].fillna(0)

    df["cum_market"] = df.groupby("ticker")["return_filled"].transform(
        lambda x: (1 + x).cumprod()
    )

    df["cum_strategy_gross"] = df.groupby("ticker")["strategy_return_gross_filled"].transform(
        lambda x: (1 + x).cumprod()
    )

    df["cum_strategy_net"] = df.groupby("ticker")["strategy_return_net_filled"].transform(
        lambda x: (1 + x).cumprod()
    )

    return df


def calculate_drawdown(equity_series: pd.Series) -> pd.Series:
    """
    Calcola il drawdown:
    quanto siamo sotto il massimo precedente.
    """
    running_max = equity_series.cummax()
    return (equity_series / running_max) - 1


def calculate_max_consecutive_losses(returns: pd.Series) -> int:
    """
    Calcola il massimo numero di trade consecutivi in perdita.
    """
    if returns.empty:
        return 0

    max_streak = 0
    current_streak = 0

    for value in returns:
        if pd.notna(value) and value < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak


def build_trade_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Costruisce un trade log robusto basato sugli eventi espliciti di ingresso/uscita.

    Per ogni trade calcola:
    - ticker
    - trade_num
    - entry_date / entry_price / entry_vix
    - exit_date / exit_price / exit_vix
    - exit_reason
    - holding_days
    - gross_return_pct
    - net_return_pct

    Nota sui costi:
    - il costo totale round-trip è 2 * TRANSACTION_COST
      perché paghiamo una volta in ingresso e una in uscita
    """
    records = []

    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("Date").copy().reset_index(drop=True)

        entry_rows = group[group["entry_flag_event"] == 1].copy().reset_index(drop=True)
        exit_rows = group[group["exit_flag_event"] == 1].copy().reset_index(drop=True)

        n = min(len(entry_rows), len(exit_rows))

        for i in range(n):
            entry_date = pd.to_datetime(entry_rows.loc[i, "Date"])
            exit_date = pd.to_datetime(exit_rows.loc[i, "Date"])

            entry_price = entry_rows.loc[i, "entry_price_event"]
            exit_price = exit_rows.loc[i, "exit_price_event"]

            entry_vix = entry_rows.loc[i, "entry_vix_event"] if "entry_vix_event" in entry_rows.columns else pd.NA
            exit_vix = exit_rows.loc[i, "exit_vix_event"] if "exit_vix_event" in exit_rows.columns else pd.NA
            exit_reason = exit_rows.loc[i, "exit_reason_event"] if "exit_reason_event" in exit_rows.columns else pd.NA

            if pd.notna(entry_price) and entry_price != 0 and pd.notna(exit_price):
                gross_return = (exit_price / entry_price) - 1
                net_return = gross_return - (2 * TRANSACTION_COST)
            else:
                gross_return = pd.NA
                net_return = pd.NA

            holding_days = (exit_date - entry_date).days if pd.notna(entry_date) and pd.notna(exit_date) else pd.NA

            records.append({
                "ticker": ticker,
                "trade_num": i + 1,
                "entry_date": entry_date,
                "entry_price": entry_price,
                "entry_vix": entry_vix,
                "exit_date": exit_date,
                "exit_price": exit_price,
                "exit_vix": exit_vix,
                "exit_reason": exit_reason,
                "holding_days": holding_days,
                "gross_return_pct": gross_return * 100 if pd.notna(gross_return) else pd.NA,
                "net_return_pct": net_return * 100 if pd.notna(net_return) else pd.NA,
            })

    return pd.DataFrame(records)


def build_summary(df: pd.DataFrame, trade_log: pd.DataFrame) -> pd.DataFrame:
    """
    Summary finale per ticker.

    Metriche principali:
    - performance mercato
    - performance strategia lorda/netta
    - alpha netta
    - drawdown
    - statistiche trade
    - metriche avanzate:
      profit_factor, avg_win_pct, avg_loss_pct, avg_holding_days,
      max_consecutive_losses, conteggi e percentuali per motivo di uscita
    """
    summary = df.groupby("ticker").agg(
        first_date=("Date", "min"),
        last_date=("Date", "max"),
        rows=("Date", "count"),
        days_in_position=("position", "sum"),
        position_changes=("position_change", "sum"),
        total_cost=("transaction_cost", "sum"),
        total_return_market=("cum_market", "last"),
        total_return_strategy_gross=("cum_strategy_gross", "last"),
        total_return_strategy_net=("cum_strategy_net", "last"),
    ).reset_index()

    summary["market_perf_pct"] = (summary["total_return_market"] - 1) * 100
    summary["strategy_gross_perf_pct"] = (summary["total_return_strategy_gross"] - 1) * 100
    summary["strategy_net_perf_pct"] = (summary["total_return_strategy_net"] - 1) * 100
    summary["alpha_net_pct"] = summary["strategy_net_perf_pct"] - summary["market_perf_pct"]

    dd_records = []
    for ticker, group in df.groupby("ticker"):
        dd_records.append({
            "ticker": ticker,
            "max_drawdown_market_pct": calculate_drawdown(group["cum_market"]).min() * 100,
            "max_drawdown_strategy_gross_pct": calculate_drawdown(group["cum_strategy_gross"]).min() * 100,
            "max_drawdown_strategy_net_pct": calculate_drawdown(group["cum_strategy_net"]).min() * 100,
        })

    dd_df = pd.DataFrame(dd_records)
    summary = summary.merge(dd_df, on="ticker", how="left")

    if not trade_log.empty:
        trade_stats_records = []

        for ticker, trades_group in trade_log.groupby("ticker"):
            trades_group = trades_group.sort_values("entry_date").copy()

            closed_trades = len(trades_group)
            avg_trade_return_pct = trades_group["net_return_pct"].mean() if "net_return_pct" in trades_group.columns else np.nan
            win_rate_pct = (trades_group["net_return_pct"] > 0).mean() * 100 if closed_trades > 0 else np.nan
            avg_entry_vix = trades_group["entry_vix"].mean() if "entry_vix" in trades_group.columns else np.nan
            avg_exit_vix = trades_group["exit_vix"].mean() if "exit_vix" in trades_group.columns else np.nan
            avg_holding_days = trades_group["holding_days"].mean() if "holding_days" in trades_group.columns else np.nan

            wins = trades_group.loc[trades_group["net_return_pct"] > 0, "net_return_pct"]
            losses = trades_group.loc[trades_group["net_return_pct"] < 0, "net_return_pct"]

            avg_win_pct = wins.mean() if not wins.empty else np.nan
            avg_loss_pct = losses.mean() if not losses.empty else np.nan

            total_wins = wins.sum() if not wins.empty else 0.0
            total_losses_abs = abs(losses.sum()) if not losses.empty else 0.0
            profit_factor = (total_wins / total_losses_abs) if total_losses_abs > 0 else np.nan

            max_consecutive_losses = calculate_max_consecutive_losses(trades_group["net_return_pct"])

            exit_reason_counts = trades_group["exit_reason"].value_counts()

            num_stop_loss = int(exit_reason_counts.get("stop_loss", 0))
            num_trailing_stop = int(exit_reason_counts.get("trailing_stop", 0))
            num_vix_exit = int(exit_reason_counts.get("vix_exit", 0))
            num_end_of_data = int(exit_reason_counts.get("end_of_data", 0))
            num_take_profit = int(exit_reason_counts.get("take_profit", 0))  # attualmente probabilmente 0

            pct_stop_loss = (num_stop_loss / closed_trades * 100) if closed_trades > 0 else np.nan
            pct_trailing_stop = (num_trailing_stop / closed_trades * 100) if closed_trades > 0 else np.nan
            pct_vix_exit = (num_vix_exit / closed_trades * 100) if closed_trades > 0 else np.nan
            pct_end_of_data = (num_end_of_data / closed_trades * 100) if closed_trades > 0 else np.nan
            pct_take_profit = (num_take_profit / closed_trades * 100) if closed_trades > 0 else np.nan

            trade_stats_records.append({
                "ticker": ticker,
                "closed_trades": closed_trades,
                "avg_trade_return_pct": avg_trade_return_pct,
                "win_rate_pct": win_rate_pct,
                "avg_entry_vix": avg_entry_vix,
                "avg_exit_vix": avg_exit_vix,
                "avg_holding_days": avg_holding_days,
                "profit_factor": profit_factor,
                "avg_win_pct": avg_win_pct,
                "avg_loss_pct": avg_loss_pct,
                "max_consecutive_losses": max_consecutive_losses,
                "num_stop_loss": num_stop_loss,
                "num_take_profit": num_take_profit,
                "num_trailing_stop": num_trailing_stop,
                "num_vix_exit": num_vix_exit,
                "num_end_of_data": num_end_of_data,
                "pct_stop_loss": pct_stop_loss,
                "pct_take_profit": pct_take_profit,
                "pct_trailing_stop": pct_trailing_stop,
                "pct_vix_exit": pct_vix_exit,
                "pct_end_of_data": pct_end_of_data,
            })

        trade_stats = pd.DataFrame(trade_stats_records)
        summary = summary.merge(trade_stats, on="ticker", how="left")

    return summary


def build_entries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elenco ingressi.
    """
    cols = ["Date", "ticker", "Adj Close", "ma50", "rsi"]
    if "VIX" in df.columns:
        cols.append("VIX")

    entries = df[(df["signal"] == 1) & (df["signal"].shift(1).fillna(0) == 0)][cols].copy()
    return entries


def plot_equity_curve(df: pd.DataFrame, ticker: str, output_dir: str) -> str | None:
    """
    Grafico market vs strategia per un ticker.
    """
    data = df[df["ticker"] == ticker].copy()
    if data.empty:
        return None

    data = data.sort_values("Date")

    plt.figure(figsize=(12, 6))
    plt.plot(data["Date"], data["cum_market"], label="Market")
    plt.plot(data["Date"], data["cum_strategy_gross"], label="Strategy Gross")
    plt.plot(data["Date"], data["cum_strategy_net"], label="Strategy Net")
    plt.title(f"{STRATEGY_NAME} - Equity Curve - {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_file = os.path.join(output_dir, f"{STRATEGY_NAME}_equity_curve_{ticker.replace('.', '_')}.png")
    plt.savefig(output_file)
    plt.close()

    return output_file


if __name__ == "__main__":
    dataset_file = "data/processed/dataset.csv"
    prices_file = "data/raw/prices.csv"
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    input_file = dataset_file if os.path.exists(dataset_file) else prices_file

    output_backtest_file = os.path.join(output_dir, f"{STRATEGY_NAME}_backtest.csv")
    output_summary_file = os.path.join(output_dir, f"{STRATEGY_NAME}_summary.csv")
    output_entries_file = os.path.join(output_dir, f"{STRATEGY_NAME}_entries.csv")
    output_trade_log_file = os.path.join(output_dir, f"{STRATEGY_NAME}_trade_log.csv")

    print(f"Lettura file input: {input_file}")
    df = pd.read_csv(input_file)
    df["Date"] = pd.to_datetime(df["Date"])

    df = prepare_data(df)
    df = apply_strategy(df)
    df = backtest(df, transaction_cost=TRANSACTION_COST)

    trade_log = build_trade_log(df)
    summary = build_summary(df, trade_log)
    entries = build_entries(df)

    df.to_csv(output_backtest_file, index=False)
    summary.to_csv(output_summary_file, index=False)
    entries.to_csv(output_entries_file, index=False)
    trade_log.to_csv(output_trade_log_file, index=False)

    print("\n===== SUMMARY PER TICKER =====")
    print(summary.to_string(index=False))

    print("\n===== PRIME 20 ENTRY =====")
    if entries.empty:
        print("Nessuna entry trovata.")
    else:
        print(entries.head(20).to_string(index=False))

    if not trade_log.empty:
        print("\n===== PRIME 20 TRADE CHIUSI =====")
        print(trade_log.head(20).to_string(index=False))

    first_ticker = df["ticker"].dropna().iloc[0]
    chart_file = plot_equity_curve(df, first_ticker, output_dir)

    if chart_file:
        print(f"\n✅ Grafico equity salvato in: {chart_file}")

    print(f"\n✅ Backtest salvato in: {output_backtest_file}")
    print(f"✅ Summary salvato in: {output_summary_file}")
    print(f"✅ Entries salvate in: {output_entries_file}")
    print(f"✅ Trade log salvato in: {output_trade_log_file}")