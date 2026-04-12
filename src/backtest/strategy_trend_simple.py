import os
import pandas as pd
import ta
import matplotlib.pyplot as plt


TRANSACTION_COST = 0.0015   # 0.15% a ogni cambio posizione
USE_VIX_FILTER = True       # Se True usa il VIX quando disponibile
VIX_THRESHOLD = 25          # Soglia oltre la quale consideriamo il mercato "stressato"


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola le feature tecniche di base per ticker.

    Colonne create:
    - return: rendimento giornaliero
    - ma50: media mobile a 50 giorni
    - rsi: Relative Strength Index a 14 periodi
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


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regole base:
    - BUY se prezzo > MA50 e RSI < 70
    - SELL se prezzo < MA50

    Se USE_VIX_FILTER=True e la colonna VIX esiste:
    - BUY solo anche con VIX < VIX_THRESHOLD

    Cos'è il VIX, in pratica:
    - VIX basso = mercato tranquillo
    - VIX alto = mercato nervoso / rischioso
    """
    df = df.copy()
    df["signal"] = 0

    base_buy_condition = (
        df["ma50"].notna() &
        df["rsi"].notna() &
        (df["Adj Close"] > df["ma50"]) &
        (df["rsi"] < 70)
    )

    if USE_VIX_FILTER and "VIX" in df.columns:
        buy_condition = base_buy_condition & df["VIX"].notna() & (df["VIX"] < VIX_THRESHOLD)
    else:
        buy_condition = base_buy_condition

    sell_condition = (
        df["ma50"].notna() &
        (df["Adj Close"] < df["ma50"])
    )

    df.loc[buy_condition, "signal"] = 1
    df.loc[sell_condition, "signal"] = 0

    # Posizione reale dal giorno dopo per evitare look-ahead bias
    df["position"] = df.groupby("ticker")["signal"].shift(1).fillna(0)

    return df


def backtest(df: pd.DataFrame, transaction_cost: float = TRANSACTION_COST) -> pd.DataFrame:
    """
    Esegue il backtest con costi di transazione.

    - strategy_return_gross: rendimento lordo
    - transaction_cost: costo quando la posizione cambia
    - strategy_return_net: rendimento netto
    """
    df = df.copy()

    df["strategy_return_gross"] = df["position"] * df["return"]

    # position_change vale 1 quando entri o esci
    df["position_change"] = df.groupby("ticker")["position"].diff().abs().fillna(0)

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


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("ticker").agg(
        first_date=("Date", "min"),
        last_date=("Date", "max"),
        rows=("Date", "count"),
        buy_signals=("signal", "sum"),
        days_in_position=("position", "sum"),
        trades=("position_change", "sum"),
        total_cost=("transaction_cost", "sum"),
        total_return_market=("cum_market", "last"),
        total_return_strategy_gross=("cum_strategy_gross", "last"),
        total_return_strategy_net=("cum_strategy_net", "last"),
    ).reset_index()

    summary["market_perf_pct"] = (summary["total_return_market"] - 1) * 100
    summary["strategy_gross_perf_pct"] = (summary["total_return_strategy_gross"] - 1) * 100
    summary["strategy_net_perf_pct"] = (summary["total_return_strategy_net"] - 1) * 100
    summary["alpha_net_pct"] = summary["strategy_net_perf_pct"] - summary["market_perf_pct"]

    return summary


def build_trade_entries(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Date", "ticker", "Adj Close", "ma50", "rsi"]
    if "VIX" in df.columns:
        cols.append("VIX")

    entries = df[
        (df["signal"] == 1) &
        (df["position"] == 0)
    ][cols].copy()

    return entries


def calculate_drawdown(equity_series: pd.Series) -> pd.Series:
    running_max = equity_series.cummax()
    drawdown = (equity_series / running_max) - 1
    return drawdown


def build_drawdown_summary(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for ticker, group in df.groupby("ticker"):
        market_dd = calculate_drawdown(group["cum_market"]).min()
        gross_dd = calculate_drawdown(group["cum_strategy_gross"]).min()
        net_dd = calculate_drawdown(group["cum_strategy_net"]).min()

        records.append({
            "ticker": ticker,
            "max_drawdown_market_pct": market_dd * 100,
            "max_drawdown_strategy_gross_pct": gross_dd * 100,
            "max_drawdown_strategy_net_pct": net_dd * 100,
        })

    return pd.DataFrame(records)


def plot_equity_curve(df: pd.DataFrame, ticker: str, output_dir: str) -> str | None:
    data = df[df["ticker"] == ticker].copy()

    if data.empty:
        print(f"Nessun dato trovato per {ticker}")
        return None

    data = data.sort_values("Date")

    plt.figure(figsize=(12, 6))
    plt.plot(data["Date"], data["cum_market"], label="Market")
    plt.plot(data["Date"], data["cum_strategy_gross"], label="Strategy Gross")
    plt.plot(data["Date"], data["cum_strategy_net"], label="Strategy Net")
    plt.title(f"Equity Curve - {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)

    output_file = os.path.join(output_dir, f"equity_curve_{ticker.replace('.', '_')}.png")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    return output_file


if __name__ == "__main__":

    dataset_file = "data/processed/dataset.csv"
    prices_file = "data/raw/prices.csv"
    output_dir = "data/processed"

    os.makedirs(output_dir, exist_ok=True)

    input_file = dataset_file if os.path.exists(dataset_file) else prices_file

    output_backtest_file = os.path.join(output_dir, "backtest.csv")
    output_summary_file = os.path.join(output_dir, "backtest_summary.csv")
    output_entries_file = os.path.join(output_dir, "backtest_entries.csv")
    output_drawdown_file = os.path.join(output_dir, "backtest_drawdown_summary.csv")

    print(f"Lettura file input: {input_file}")
    df = pd.read_csv(input_file)
    df["Date"] = pd.to_datetime(df["Date"])

    df = prepare_data(df)
    df = generate_signals(df)
    df = backtest(df, transaction_cost=TRANSACTION_COST)

    summary = build_summary(df)
    drawdown_summary = build_drawdown_summary(df)
    summary = summary.merge(drawdown_summary, on="ticker", how="left")

    entries = build_trade_entries(df)

    df.to_csv(output_backtest_file, index=False)
    summary.to_csv(output_summary_file, index=False)
    entries.to_csv(output_entries_file, index=False)
    drawdown_summary.to_csv(output_drawdown_file, index=False)

    print("\n===== SUMMARY PER TICKER =====")
    print(summary.to_string(index=False))

    print("\n===== PRIME 20 ENTRY =====")
    if entries.empty:
        print("Nessuna entry trovata.")
    else:
        print(entries.head(20).to_string(index=False))

    first_ticker = df["ticker"].dropna().iloc[0]
    chart_file = plot_equity_curve(df, first_ticker, output_dir)

    if chart_file:
        print(f"\n✅ Grafico equity salvato in: {chart_file}")

    print(f"\n✅ Backtest salvato in: {output_backtest_file}")
    print(f"✅ Summary salvato in: {output_summary_file}")
    print(f"✅ Entries salvate in: {output_entries_file}")
    print(f"✅ Drawdown summary salvato in: {output_drawdown_file}")