import os
import glob
import pandas as pd
import numpy as np


INPUT_DIR = "data/processed"
OUTPUT_DIR = "data/processed"

COMBINED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "strategies_comparison_all.csv")
RANKED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "strategies_comparison_ranked.csv")
BEST_PER_STRATEGY_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "strategies_best_tickers_per_strategy.csv")


def load_all_summaries(input_dir: str) -> pd.DataFrame:
    """
    Carica tutti i file *_summary.csv presenti nella cartella
    e aggiunge una colonna 'strategy_name' derivata dal nome file.
    """
    pattern = os.path.join(input_dir, "*_summary.csv")
    files = glob.glob(pattern)
    files = [
        f for f in files
        if os.path.basename(f) in [
            "target_stop_trailing_no_vix_exit_summary.csv",
            "target_stop_trailing_vix_regime_summary.csv"
        ]
    ]
    if not files:
        print("Nessun file *_summary.csv trovato.")
        return pd.DataFrame()

    parts = []

    for file_path in files:
        try:
            df = pd.read_csv(file_path)

            file_name = os.path.basename(file_path)
            strategy_name = file_name.replace("_summary.csv", "")

            df["strategy_name"] = strategy_name
            parts.append(df)

            print(f"Caricato: {file_name} ({len(df)} righe)")
        except Exception as e:
            print(f"Errore lettura file {file_path}: {e}")

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    return combined


def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge ranking per ciascun ticker confrontando le strategie tra loro.

    Rank migliori:
    - strategy_net_perf_pct: più alto è meglio
    - alpha_net_pct: più alto è meglio
    - profit_factor: più alto è meglio
    - max_drawdown_strategy_net_pct: più alto è meglio
      perché essendo negativo, -20 è meglio di -50
    """
    df = df.copy()

    rank_groups = []

    for ticker, group in df.groupby("ticker"):
        group = group.copy()

        group["rank_net_perf"] = group["strategy_net_perf_pct"].rank(ascending=False, method="min")
        group["rank_alpha"] = group["alpha_net_pct"].rank(ascending=False, method="min")

        if "profit_factor" in group.columns:
            group["profit_factor_filled"] = group["profit_factor"].fillna(-999999)
            group["rank_profit_factor"] = group["profit_factor_filled"].rank(ascending=False, method="min")
        else:
            group["rank_profit_factor"] = np.nan

        group["rank_drawdown"] = group["max_drawdown_strategy_net_pct"].rank(ascending=False, method="min")

        rank_cols = ["rank_net_perf", "rank_alpha", "rank_drawdown"]

        if group["rank_profit_factor"].notna().any():
            rank_cols.append("rank_profit_factor")

        group["rank_total"] = group[rank_cols].sum(axis=1)

        rank_groups.append(group)

    ranked = pd.concat(rank_groups, ignore_index=True)
    return ranked


def build_best_per_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per ogni strategia prende i ticker migliori ordinati per rank combinato.
    """
    if df.empty:
        return pd.DataFrame()

    parts = []

    for strategy_name, group in df.groupby("strategy_name"):
        group = group.sort_values(
            by=["rank_total", "strategy_net_perf_pct", "profit_factor"],
            ascending=[True, False, False]
        ).copy()

        group["strategy_rank_within_strategy"] = range(1, len(group) + 1)
        parts.append(group)

    return pd.concat(parts, ignore_index=True)


def build_best_strategy_per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per ogni ticker trova la strategia migliore.
    """
    if df.empty:
        return pd.DataFrame()

    best_rows = []

    for ticker, group in df.groupby("ticker"):
        group = group.sort_values(
            by=["rank_total", "strategy_net_perf_pct", "profit_factor"],
            ascending=[True, False, False]
        ).copy()

        best_rows.append(group.iloc[0])

    best_df = pd.DataFrame(best_rows).reset_index(drop=True)
    best_df = best_df.sort_values(
        by=["rank_total", "strategy_net_perf_pct", "profit_factor"],
        ascending=[True, False, False]
    )

    return best_df


def print_top_results(best_per_ticker: pd.DataFrame, best_per_strategy: pd.DataFrame):
    """
    Stampa una sintesi a video.
    """
    print("\n===== MIGLIOR STRATEGIA PER TICKER =====")
    if best_per_ticker.empty:
        print("Nessun dato disponibile.")
    else:
        cols = [
            "ticker",
            "strategy_name",
            "strategy_net_perf_pct",
            "alpha_net_pct",
            "profit_factor",
            "max_drawdown_strategy_net_pct",
            "rank_total"
        ]
        available_cols = [c for c in cols if c in best_per_ticker.columns]
        print(best_per_ticker[available_cols].head(30).to_string(index=False))

    print("\n===== TOP TICKER PER STRATEGIA =====")
    if best_per_strategy.empty:
        print("Nessun dato disponibile.")
    else:
        cols = [
            "strategy_name",
            "strategy_rank_within_strategy",
            "ticker",
            "strategy_net_perf_pct",
            "alpha_net_pct",
            "profit_factor",
            "max_drawdown_strategy_net_pct",
            "rank_total"
        ]
        available_cols = [c for c in cols if c in best_per_strategy.columns]
        print(best_per_strategy[available_cols].head(50).to_string(index=False))


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Lettura summary da: {INPUT_DIR}")
    combined = load_all_summaries(INPUT_DIR)

    if combined.empty:
        print("Nessun summary disponibile, termina.")
        raise SystemExit(0)

    ranked = add_ranks(combined)
    best_per_strategy = build_best_per_strategy(ranked)
    best_per_ticker = build_best_strategy_per_ticker(ranked)

    combined.to_csv(COMBINED_OUTPUT_FILE, index=False)
    ranked.to_csv(RANKED_OUTPUT_FILE, index=False)
    best_per_strategy.to_csv(BEST_PER_STRATEGY_OUTPUT_FILE, index=False)

    print_top_results(best_per_ticker, best_per_strategy)

    print(f"\n✅ File completo salvato in: {COMBINED_OUTPUT_FILE}")
    print(f"✅ File ranking salvato in: {RANKED_OUTPUT_FILE}")
    print(f"✅ File top ticker per strategia salvato in: {BEST_PER_STRATEGY_OUTPUT_FILE}")