"""
optimizer.py  —  Ottimizzatore parametri
Esegue una grid search sui parametri della strategia usando
la simulazione portafoglio come funzione obiettivo.

Metriche ottimizzate (in ordine di priorità):
  1. Sharpe ratio  (rischio/rendimento)
  2. Profit factor
  3. Win rate

Utilizzo:
    python src/optimizer.py
    python src/optimizer.py --metric sharpe --days 365
    python src/optimizer.py --quick   (grid ridotta, ~5 minuti)
"""

import os, sys, json, itertools, warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT       = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = OUTPUT_DIR / "optimizer_results.csv"
BEST_FILE    = OUTPUT_DIR / "optimizer_best.json"


# =============================================================================
# GRIGLIA PARAMETRI
# =============================================================================
PARAM_GRID_FULL = {
    "stop_loss_pct":          [0.03, 0.04, 0.05],
    "take_profit_pct":        [0.10, 0.12, 0.15],
    "trailing_stop_pct":      [0.05, 0.06, 0.08],
    "vix_entry":              [18, 20, 22],
    "entry_score_threshold":  [5, 6, 7],
    "ai_prob_threshold":      [0.58, 0.62, 0.65],
}

PARAM_GRID_QUICK = {
    "stop_loss_pct":          [0.03, 0.05],
    "take_profit_pct":        [0.10, 0.15],
    "trailing_stop_pct":      [0.05, 0.08],
    "vix_entry":              [18, 22],
    "entry_score_threshold":  [5, 7],
    "ai_prob_threshold":      [0.58, 0.65],
}


# =============================================================================
# SIMULAZIONE LEGGERA (senza download — usa dati già scaricati)
# =============================================================================
def _run_single(params: dict, prices_df: pd.DataFrame, vix_series,
                model, feature_cols: list, approved_tickers: list,
                ticker_stats: dict, sim_days: int = 365) -> dict:
    """
    Esegue una simulazione con i parametri dati.
    Ritorna le metriche principali.
    """
    from daily_screener import (
        compute_indicators, calculate_entry_score,
        build_ai_features, VIX_MAX, MOMENTUM_3M_MIN,
    )
    from train_model import score_trade

    INITIAL_CAPITAL = 10_000.0
    MAX_POSITIONS   = 3
    TC              = 0.0015

    sl  = params["stop_loss_pct"]
    tp  = params["take_profit_pct"]
    tr  = params["trailing_stop_pct"]
    vix_thr   = params["vix_entry"]
    score_thr = params["entry_score_threshold"]
    ai_thr    = params["ai_prob_threshold"]

    end_date   = datetime.today()
    sim_start  = end_date - timedelta(days=sim_days)
    prices_df["Date"] = pd.to_datetime(prices_df["Date"]).dt.tz_localize(None)
    all_dates  = sorted(prices_df["Date"].unique())
    sim_dates  = [d for d in all_dates if pd.Timestamp(d) >= pd.Timestamp(sim_start)]

    if len(sim_dates) < 20:
        return {}

    cash      = INITIAL_CAPITAL
    positions = {}
    trades    = []
    daily_vals= []

    for i, date in enumerate(sim_dates):
        date_ts    = pd.Timestamp(date)
        day_prices = prices_df[prices_df["Date"] == date].set_index("ticker")["Adj Close"].to_dict()

        # gestione posizioni aperte
        to_exit = []
        for ticker, pos in positions.items():
            price = day_prices.get(ticker)
            if price is None:
                continue

            # trailing activation
            if not pos["trailing_active"] and price >= pos["target"]:
                pos["trailing_active"] = True
                pos["highest"]         = price
                pos["stop"]            = price * (1 - tr)
            elif pos["trailing_active"] and price > pos["highest"]:
                pos["highest"] = price
                pos["stop"]    = price * (1 - tr)

            # vix exit
            past_vix = vix_series[vix_series.index <= date_ts] if hasattr(vix_series, 'index') else pd.Series()
            vix_today = float(past_vix.iloc[-1]) if not past_vix.empty else None
            if vix_today and vix_today >= 25:
                to_exit.append((ticker, price, "vix_exit"))
            elif price <= pos["stop"]:
                reason = "trailing_stop" if pos["trailing_active"] else "stop_loss"
                to_exit.append((ticker, price, reason))

        for ticker, price, reason in to_exit:
            pos     = positions.pop(ticker)
            net     = pos["shares"] * price * (1 - TC)
            cash   += net
            gr      = (price / pos["entry_price"] - 1) * 100
            nr      = gr - TC * 2 * 100
            trades.append({"net_return_pct": nr, "exit_reason": reason})

        # nuovi ingressi
        if len(positions) < MAX_POSITIONS:
            vix_now = None
            past_vix2 = vix_series[vix_series.index <= date_ts] if hasattr(vix_series, 'index') else pd.Series()
            if not past_vix2.empty:
                vix_now = float(past_vix2.iloc[-1])

            if not vix_now or vix_now < vix_thr:
                for ticker in approved_tickers:
                    if ticker in positions or len(positions) >= MAX_POSITIONS:
                        continue
                    df_t = prices_df[
                        (prices_df["ticker"] == ticker) &
                        (prices_df["Date"] <= date)
                    ].tail(252)
                    if len(df_t) < 60:
                        continue

                    ind = compute_indicators(df_t)
                    if not ind:
                        continue

                    # score con parametri custom
                    score = 0
                    if ind.get("ma50")  and ind["close"] > ind["ma50"]:            score += 1
                    if ind.get("ma200") and ind["close"] > ind["ma200"]:           score += 1
                    if ind.get("ma20")  and ind.get("ma50") and ind["ma20"] > ind["ma50"]: score += 1
                    if ind.get("ma50_slope") and ind["ma50_slope"] > 0:            score += 1
                    if ind.get("rsi")   and ind["rsi"] < 62:                       score += 1
                    if ind.get("adx")   and ind["adx"] >= 20:                      score += 1
                    if ind.get("volume_ratio") and ind["volume_ratio"] >= 1.0:     score += 1
                    if ind.get("momentum_3m") and ind["momentum_3m"] >= 0:         score += 1
                    if vix_now and vix_now < vix_thr:                              score += 1

                    if score < score_thr:
                        continue
                    if ind.get("momentum_3m") and ind["momentum_3m"] < -0.05:
                        continue

                    ai_feats = build_ai_features(
                        ticker, ind, vix_now or 18.0, {}, ticker_stats, 0, 0, date_ts.to_pydatetime()
                    )
                    ai_prob = score_trade(model, ai_feats, feature_cols)
                    if ai_prob < ai_thr:
                        continue

                    # ingresso al giorno dopo
                    next_dates = [d for d in sim_dates if d > date]
                    if not next_dates:
                        continue
                    next_prices = prices_df[prices_df["Date"] == next_dates[0]].set_index("ticker")["Adj Close"]
                    ep = next_prices.get(ticker)
                    if ep is None or pd.isna(ep):
                        continue

                    ep = float(ep)
                    size  = INITIAL_CAPITAL / MAX_POSITIONS
                    cash -= size
                    positions[ticker] = {
                        "shares":          (size * (1 - TC)) / ep,
                        "entry_price":     ep,
                        "stop":            ep * (1 - sl),
                        "target":          ep * (1 + tp),
                        "trailing_active": False,
                        "highest":         ep,
                    }

        total_val = cash + sum(
            pos["shares"] * day_prices.get(t, pos["entry_price"])
            for t, pos in positions.items()
        )
        daily_vals.append(total_val)

    if not daily_vals or not trades:
        return {}

    vals   = pd.Series(daily_vals)
    ret    = (vals.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    dd     = ((vals - vals.cummax()) / vals.cummax() * 100).min()
    dr     = vals.pct_change().fillna(0)
    sharpe = (dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0

    df_t = pd.DataFrame(trades)
    wins  = df_t[df_t["net_return_pct"] > 0]
    loss  = df_t[df_t["net_return_pct"] <= 0]
    pf    = wins["net_return_pct"].sum() / abs(loss["net_return_pct"].sum()) \
            if not loss.empty and loss["net_return_pct"].sum() != 0 else 0.0

    return {
        "total_return_pct": round(ret, 2),
        "max_drawdown_pct": round(dd, 2),
        "sharpe":           round(sharpe, 3),
        "profit_factor":    round(pf, 3),
        "win_rate":         round(len(wins) / len(df_t) * 100, 1) if df_t.shape[0] > 0 else 0,
        "n_trades":         len(df_t),
    }


# =============================================================================
# GRID SEARCH
# =============================================================================
def run_optimization(
    quick:      bool  = False,
    metric:     str   = "sharpe",
    sim_days:   int   = 365,
    progress_cb = None,
) -> pd.DataFrame:
    """
    Esegue la grid search e ritorna un DataFrame con tutte le combinazioni
    ordinate per la metrica scelta.

    progress_cb: funzione(done, total) chiamata ad ogni iterazione (per la UI)
    """
    import joblib
    from universe_selection import select_universe

    # Carica modello e dati
    model_path = ROOT / "models" / "trade_scorer_v2.joblib"
    feat_path  = ROOT / "models" / "feature_names_v2.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_path}")

    model        = joblib.load(model_path)
    feature_cols = json.load(open(feat_path))

    # Prezzi
    prices_file = OUTPUT_DIR / "dataset_extended.csv"
    if not prices_file.exists():
        prices_file = ROOT / "data" / "raw" / "prices_extended.csv"
    if not prices_file.exists():
        raise FileNotFoundError("Dataset prezzi non trovato. Esegui prima build_dataset.py")

    print(f"Caricamento dati: {prices_file}")
    prices_df = pd.read_csv(prices_file, usecols=["Date","ticker","Adj Close",
                                                    "High","Low","Volume"])
    prices_df["Date"] = pd.to_datetime(prices_df["Date"]).dt.tz_localize(None)

    # VIX
    import yfinance as yf
    end   = datetime.today()
    start = end - timedelta(days=sim_days + 300)
    try:
        vix_df = yf.download("^VIX", start=start.strftime("%Y-%m-%d"),
                              auto_adjust=True, progress=False)
        vix_df.columns = [c[0] if isinstance(c, tuple) else c for c in vix_df.columns]
        vix_series = vix_df["Close"]
        vix_series.index = pd.to_datetime(vix_series.index).tz_localize(None)
    except Exception:
        vix_series = pd.Series(dtype=float)

    # Universe
    summary_file = OUTPUT_DIR / "target_stop_trailing_vix_regime_summary.csv"
    if summary_file.exists():
        summary_df       = pd.read_csv(summary_file)
        approved_tickers = select_universe(summary_df, verbose=False)
    else:
        from tickers import FTSE_MIB_TICKERS
        approved_tickers = FTSE_MIB_TICKERS

    # Ticker stats
    ticker_stats = {}
    tl_file = OUTPUT_DIR / "target_stop_trailing_vix_regime_trade_log.csv"
    if tl_file.exists():
        tl = pd.read_csv(tl_file)
        for t, g in tl.groupby("ticker"):
            ticker_stats[t] = {"win_rate": (g["net_return_pct"] > 0).mean(), "count": len(g)}

    # Grid
    grid = PARAM_GRID_QUICK if quick else PARAM_GRID_FULL
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    total  = len(combos)
    print(f"Grid search: {total} combinazioni (metric={metric}, days={sim_days})")

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        metrics = _run_single(
            params, prices_df.copy(), vix_series,
            model, feature_cols, approved_tickers, ticker_stats, sim_days
        )
        if metrics:
            row = {**params, **metrics}
            results.append(row)

        if progress_cb:
            progress_cb(i + 1, total)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{total} completate...")

    if not results:
        print("[WARN] Nessun risultato valido.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    metric_col = {"sharpe": "sharpe", "pf": "profit_factor",
                  "return": "total_return_pct", "winrate": "win_rate"}.get(metric, "sharpe")
    df = df.sort_values(metric_col, ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"

    # Salva
    df.to_csv(RESULTS_FILE)
    best = df.iloc[0].to_dict()
    with open(BEST_FILE, "w") as f:
        json.dump(best, f, indent=2)

    print(f"\n=== TOP 5 COMBINAZIONI ({metric_col}) ===")
    cols_show = list(keys) + [metric_col, "n_trades", "max_drawdown_pct"]
    print(df[[c for c in cols_show if c in df.columns]].head(5).to_string())
    print(f"\n[OK] Risultati salvati: {RESULTS_FILE}")
    print(f"[OK] Migliori parametri: {BEST_FILE}")
    return df


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="sharpe",
                        choices=["sharpe","pf","return","winrate"])
    parser.add_argument("--days",   type=int, default=365)
    parser.add_argument("--quick",  action="store_true")
    args = parser.parse_args()

    run_optimization(quick=args.quick, metric=args.metric, sim_days=args.days)
