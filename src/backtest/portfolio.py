import os
import pandas as pd
import matplotlib.pyplot as plt


def build_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Costruisce un portafoglio equal-weight sui titoli attivi.
    """

    df = df.copy()

    # Conta quanti titoli sono attivi ogni giorno
    df["active_positions"] = df.groupby("Date")["position"].transform("sum")

    # Peso per titolo (equal weight)
    df["weight"] = 0.0
    df.loc[df["active_positions"] > 0, "weight"] = (
        df["position"] / df["active_positions"]
    )

    # Rendimento del portafoglio per riga
    df["weighted_return"] = df["weight"] * df["return"]

    # Rendimento giornaliero aggregato
    portfolio = df.groupby("Date").agg(
        portfolio_return=("weighted_return", "sum"),
        active_positions=("active_positions", "first")
    ).reset_index()

    # Equity curve
    portfolio["portfolio_return_filled"] = portfolio["portfolio_return"].fillna(0)
    portfolio["cum_portfolio"] = (1 + portfolio["portfolio_return_filled"]).cumprod()

    return portfolio


def plot_portfolio(portfolio: pd.DataFrame, output_dir: str):
    """
    Grafico equity del portafoglio
    """

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio["Date"], portfolio["cum_portfolio"], label="Portfolio")
    plt.title("Portfolio Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()

    output_file = os.path.join(output_dir, "portfolio_equity.png")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    return output_file


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    input_file = os.path.join(base_dir, "data", "processed", "backtest.csv")
    output_dir = os.path.join(base_dir, "data", "processed")

    df = pd.read_csv(input_file)
    df["Date"] = pd.to_datetime(df["Date"])

    portfolio = build_portfolio(df)

    output_csv = os.path.join(output_dir, "portfolio.csv")
    portfolio.to_csv(output_csv, index=False)

    chart_file = plot_portfolio(portfolio, output_dir)

    print("\n===== PORTFOLIO SUMMARY =====")
    print(f"Return totale: {(portfolio['cum_portfolio'].iloc[-1] - 1) * 100:.2f}%")
    print(f"Giorni totali: {len(portfolio)}")

    print(f"\n✅ Portfolio salvato in: {output_csv}")
    print(f"✅ Grafico salvato in: {chart_file}")