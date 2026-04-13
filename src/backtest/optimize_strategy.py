import os
import pandas as pd
from itertools import product

# Import della strategia
from src.backtest.strategy_target_stop_trailing import (
    prepare_data,
    apply_strategy,
    backtest,
    build_summary
)


# =========================
# CONFIG
# =========================
DATA_FILE = "data/processed/dataset.csv"

TRAIN_END = "2021-12-31"

# Griglia parametri
STOP_LOSS_LIST = [0.02, 0.03, 0.04]
TAKE_PROFIT_LIST = [0.08, 0.10, 0.12]
TRAILING_LIST = [0.03, 0.05, 0.07]

RESULT_FILE = "data/processed/optimization_results.csv"


def run_single_test(df, stop, take, trailing):
    """
    Esegue un test con parametri specifici
    """

    # Import dinamico dei parametri
    import src.backtest.strategy_target_stop_trailing as strat

    strat.STOP_LOSS_PCT = stop
    strat.TAKE_PROFIT_PCT = take
    strat.TRAILING_STOP_PCT = trailing

    df_proc = prepare_data(df)
    df_proc = apply_strategy(df_proc)
    df_proc = backtest(df_proc)

    summary = build_summary(df_proc, pd.DataFrame())

    avg_return = summary["strategy_net_perf_pct"].mean()
    avg_drawdown = summary["max_drawdown_strategy_net_pct"].mean()

    return avg_return, avg_drawdown


def split_train_test(df):
    train = df[df["Date"] <= TRAIN_END].copy()
    test = df[df["Date"] > TRAIN_END].copy()
    return train, test


if __name__ == "__main__":
    print("Caricamento dataset...")
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])

    train_df, test_df = split_train_test(df)

    print(f"TRAIN size: {len(train_df)}")
    print(f"TEST size: {len(test_df)}")

    results = []

    combinations = list(product(STOP_LOSS_LIST, TAKE_PROFIT_LIST, TRAILING_LIST))

    print(f"Test combinazioni: {len(combinations)}")

    for stop, take, trailing in combinations:
        print(f"\nTest: SL={stop}, TP={take}, TR={trailing}")

        train_return, train_dd = run_single_test(train_df, stop, take, trailing)
        test_return, test_dd = run_single_test(test_df, stop, take, trailing)

        results.append({
            "stop_loss": stop,
            "take_profit": take,
            "trailing": trailing,
            "train_return": train_return,
            "train_drawdown": train_dd,
            "test_return": test_return,
            "test_drawdown": test_dd,
        })

    results_df = pd.DataFrame(results)

    # Ordina per performance TRAIN
    results_df = results_df.sort_values(by="train_return", ascending=False)

    os.makedirs("data/processed", exist_ok=True)
    results_df.to_csv(RESULT_FILE, index=False)

    print("\n===== TOP 10 STRATEGIE (TRAIN) =====")
    print(results_df.head(10).round(2).to_string(index=False))

    print(f"\n✅ Risultati salvati in: {RESULT_FILE}")