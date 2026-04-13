"""
strategy_ai_scored.py  —  Fase 3 integrata
Strategia completa che combina:
  - Universe selection  (Fase 1)
  - Entry score avanzato (Fase 2)
  - Filtro AI probabilistico  (Fase 3)  ← NUOVO

Il modello AI viene interrogato PRIMA dell'ingresso:
solo se la probabilità stimata di vincita >= SCORE_THRESHOLD
il trade viene aperto.

Flusso decisionale per ogni segnale candidato:
  1. Ticker nell'universo approvato?          → NO: skip
  2. Entry score >= 6/9?                      → NO: skip
  3. VIX < 20 e momentum3M >= -10%?           → NO: skip
  4. AI prob_win >= 0.60?                     → NO: skip  ← NUOVO
  5. Apri trade con stop/trailing/VIX exit.

Utilizzo:
    python strategy_ai_scored.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import ta
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_model import (
    build_features as build_ml_features,
    score_trade,
    FEATURE_COLS,
    SECTOR_MAP,
    SCORE_THRESHOLD,
)

# =============================================================================
# PARAMETRI
# =============================================================================
STRATEGY_NAME = "ai_scored_v2"

# Modello v2 — addestrato con dataset esteso + feature macro (Fase 4)
# Per tornare al v1: "models/trade_scorer.joblib" / "models/feature_names.json"
MODEL_PATH = "models/trade_scorer_v2.joblib"
FEAT_PATH  = "models/feature_names_v2.json"

TRANSACTION_COST  = 0.0015
STOP_LOSS_PCT     = 0.04
TAKE_PROFIT_PCT   = 0.12
TRAILING_STOP_PCT = 0.06
VIX_ENTRY_THRESHOLD = 20
VIX_EXIT_THRESHOLD  = 25
ENTRY_SCORE_THRESHOLD = 6
RSI_ENTRY_MAX     = 62
AI_PROB_THRESHOLD = SCORE_THRESHOLD   # 0.60


# =============================================================================
# PREPARAZIONE DATI (stessa di strategy_precision.py)
# =============================================================================
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)
    df["return"] = df.groupby("ticker")["Adj Close"].pct_change()

    for w in [20, 50, 200]:
        col = f"ma{w}"
        if col not in df.columns:
            df[col] = df.groupby("ticker")["Adj Close"].transform(
                lambda x: x.rolling(w).mean()
            )

    if "ma50_slope" not in df.columns:
        df["ma50_slope"] = df.groupby("ticker")["ma50"].transform(lambda x: x.diff(5))

    if "rsi" not in df.columns:
        df["rsi"] = df.groupby("ticker")["Adj Close"].transform(
            lambda x: ta.momentum.RSIIndicator(close=x, window=14).rsi()
        )

    if "adx" not in df.columns and "High" in df.columns:
        adx_parts = []
        for ticker, grp in df.groupby("ticker"):
            s = ta.trend.ADXIndicator(
                high=grp["High"], low=grp["Low"], close=grp["Adj Close"]
            ).adx()
            s.index = grp.index
            adx_parts.append(s)
        df["adx"] = pd.concat(adx_parts).sort_index()

    if "momentum_3m" not in df.columns:
        df["momentum_3m"] = df.groupby("ticker")["Adj Close"].pct_change(63)

    if "volume_ratio" not in df.columns and "Volume" in df.columns:
        vol_ma = df.groupby("ticker")["Volume"].transform(lambda x: x.rolling(20).mean())
        df["volume_ratio"] = df["Volume"] / vol_ma

    if "SP500" in df.columns:
        df["sp500_ma50"]     = df["SP500"].rolling(50).mean()
        df["sp500_above_ma50"] = (df["SP500"] > df["sp500_ma50"]).astype(int)
    else:
        df["sp500_above_ma50"] = 0

    if "VIX" not in df.columns:
        df["VIX"] = np.nan

    df["entry_score"] = df.apply(_calculate_entry_score, axis=1)
    return df


def _calculate_entry_score(row: pd.Series) -> int:
    score = 0
    if pd.notna(row.get("ma50"))  and row["Adj Close"] > row["ma50"]:       score += 1
    if pd.notna(row.get("ma200")) and row["Adj Close"] > row["ma200"]:      score += 1
    if pd.notna(row.get("ma20"))  and pd.notna(row.get("ma50")) and row["ma20"] > row["ma50"]: score += 1
    if pd.notna(row.get("ma50_slope")) and row["ma50_slope"] > 0:           score += 1
    if pd.notna(row.get("rsi"))   and row["rsi"] < RSI_ENTRY_MAX:           score += 1
    if pd.notna(row.get("adx"))   and row["adx"] >= 20:                     score += 1
    if pd.notna(row.get("volume_ratio")) and row["volume_ratio"] >= 1.0:    score += 1
    if pd.notna(row.get("momentum_3m")) and row["momentum_3m"] >= 0:        score += 1
    if pd.notna(row.get("VIX"))   and row["VIX"] < VIX_ENTRY_THRESHOLD:     score += 1
    return score


# =============================================================================
# AI SCORER — costruisce il feature dict per il modello
# =============================================================================
def _build_ai_features(
    row: pd.Series,
    ticker: str,
    ticker_stats: dict,
    trade_count_so_far: int,
    streak_so_far: int,
    strategy_name: str,
) -> dict:
    """
    Costruisce il dizionario di feature per il modello AI
    a partire dalla riga del dataset e dal contesto del ticker.
    """
    entry_date = pd.to_datetime(row["Date"])
    vix = row.get("VIX", 18.0)
    if pd.isna(vix):
        vix = 18.0

    sector     = SECTOR_MAP.get(ticker, "other")
    sector_enc = {
        "consumer":0, "energy":1, "financials":2, "healthcare":3,
        "industrials":4, "technology":5, "telecom":6, "utilities":7, "other":8
    }.get(sector, 8)

    hist_wr = ticker_stats.get(ticker, {}).get("win_rate", 0.43)
    hist_n  = ticker_stats.get(ticker, {}).get("count",    30)

    year    = entry_date.year
    month   = entry_date.month
    quarter = (month - 1) // 3 + 1
    dow     = entry_date.dayofweek

    return {
        "entry_vix":            vix,
        "entry_vix_bucket":     min(4, int(vix // 5) - 1) if vix >= 5 else 0,
        "vix_low":              int(vix < 15),
        "vix_medium":           int(15 <= vix < 20),
        "vix_high":             int(vix >= 20),
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
        "ticker_prev_trades":   trade_count_so_far,
        "ticker_streak":        streak_so_far,
        "entry_score":          row.get("entry_score", 4),
        "is_score_strategy":    int("score" in strategy_name),
        "bull_period":          int((2019 <= year <= 2021) or year >= 2023),
        "covid_period":         int(pd.Timestamp("2020-02-01") <= entry_date <= pd.Timestamp("2020-06-30")),
        "entry_year":           year,
    }


# =============================================================================
# ESECUZIONE STRATEGIA CON AI FILTER
# =============================================================================
def run_strategy_for_ticker(
    group: pd.DataFrame,
    ticker: str,
    model,
    feature_cols: list,
    ticker_stats: dict,
) -> pd.DataFrame:
    group = group.sort_values("Date").copy().reset_index(drop=True)

    for col in ["signal","position","entry_price","stop_price","target_price",
                "trailing_active","highest_price","exit_reason","trade_id",
                "entry_flag_event","exit_flag_event","entry_vix_event",
                "exit_vix_event","exit_reason_event","entry_price_event",
                "exit_price_event","entry_score_event","ai_prob_event"]:
        group[col] = pd.NA if col not in ["signal","position","trailing_active","entry_flag_event","exit_flag_event"] else 0

    in_position    = False
    entry_price    = None
    stop_price     = None
    target_price   = None
    trailing_active= False
    highest_price  = None
    trade_id       = 0
    trade_count    = 0
    streak         = 0
    last_win       = None

    for i in range(len(group)):
        row         = group.iloc[i]
        close_price = row["Adj Close"]
        current_vix = row.get("VIX", pd.NA)

        position_today = 0
        signal_today   = 0
        exit_reason    = pd.NA

        if not in_position:
            # Gate 1: entry score
            score = row.get("entry_score", 0)
            if pd.isna(score) or score < ENTRY_SCORE_THRESHOLD:
                group.at[i, "trade_id"] = pd.NA
                group.at[i, "signal"]   = 0
                group.at[i, "position"] = 0
                continue

            # Gate 2: VIX e momentum
            vix = current_vix
            if pd.notna(vix) and vix >= VIX_ENTRY_THRESHOLD:
                group.at[i, "trade_id"] = pd.NA
                group.at[i, "signal"]   = 0
                group.at[i, "position"] = 0
                continue

            mom3m = row.get("momentum_3m", 0)
            if pd.notna(mom3m) and mom3m < -0.10:
                group.at[i, "trade_id"] = pd.NA
                group.at[i, "signal"]   = 0
                group.at[i, "position"] = 0
                continue

            # Gate 3: AI score  ← NUOVO
            ai_features = _build_ai_features(
                row, ticker, ticker_stats, trade_count, streak, STRATEGY_NAME
            )
            ai_prob = score_trade(model, ai_features, feature_cols)

            if ai_prob < AI_PROB_THRESHOLD:
                group.at[i, "trade_id"] = pd.NA
                group.at[i, "signal"]   = 0
                group.at[i, "position"] = 0
                continue

            # Tutti i gate superati → ingresso
            in_position    = True
            trade_id      += 1
            trade_count   += 1
            entry_price    = close_price
            stop_price     = entry_price * (1 - STOP_LOSS_PCT)
            target_price   = entry_price * (1 + TAKE_PROFIT_PCT)
            trailing_active= False
            highest_price  = entry_price
            signal_today   = 1

            group.at[i, "entry_flag_event"]  = 1
            group.at[i, "entry_vix_event"]   = current_vix
            group.at[i, "entry_price_event"] = entry_price
            group.at[i, "entry_score_event"] = score
            group.at[i, "ai_prob_event"]     = round(ai_prob, 3)
            group.at[i, "entry_price"]       = entry_price
            group.at[i, "stop_price"]        = stop_price
            group.at[i, "target_price"]      = target_price
            group.at[i, "trade_id"]          = trade_id

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
                in_position = False
                signal_today = 0
                # aggiorna streak (vix_exit conta come perdita di momentum)
                streak = max(streak, 0) - 1
                entry_price = stop_price = target_price = None
                trailing_active = False; highest_price = None
                group.at[i, "signal"]     = signal_today
                group.at[i, "position"]   = position_today
                group.at[i, "exit_reason"]= exit_reason
                group.at[i, "entry_price"]= group.at[i, "stop_price"] = pd.NA
                continue

            # Trailing
            if not trailing_active and close_price >= target_price:
                trailing_active = True
                highest_price   = close_price
                stop_price      = highest_price * (1 - TRAILING_STOP_PCT)

            if trailing_active and close_price > highest_price:
                highest_price = close_price
                stop_price    = highest_price * (1 - TRAILING_STOP_PCT)

            # Stop
            if close_price <= stop_price:
                exit_reason = "trailing_stop" if trailing_active else "stop_loss"
                group.at[i, "exit_flag_event"]   = 1
                group.at[i, "exit_vix_event"]    = current_vix
                group.at[i, "exit_reason_event"] = exit_reason
                group.at[i, "exit_price_event"]  = close_price
                in_position     = False
                signal_today    = 0
                # aggiorna streak
                streak = max(streak, 0) - 1
                last_win = False
                entry_price = stop_price = target_price = None
                trailing_active = False; highest_price = None
            # ancora in posizione — nessuna azione

            group.at[i, "trade_id"] = trade_id

        group.at[i, "signal"]         = signal_today
        group.at[i, "position"]       = position_today
        group.at[i, "exit_reason"]    = exit_reason
        group.at[i, "trailing_active"]= trailing_active
        group.at[i, "entry_price"]    = entry_price    if in_position else pd.NA
        group.at[i, "stop_price"]     = stop_price     if in_position else pd.NA
        group.at[i, "target_price"]   = target_price   if in_position else pd.NA
        group.at[i, "highest_price"]  = highest_price  if in_position else pd.NA

    return group


def apply_strategy(df: pd.DataFrame, model, feature_cols: list, ticker_stats: dict) -> pd.DataFrame:
    parts = []
    for ticker, grp in df.groupby("ticker"):
        parts.append(run_strategy_for_ticker(grp, ticker, model, feature_cols, ticker_stats))
    return pd.concat(parts, ignore_index=True)


# =============================================================================
# BACKTEST, TRADE LOG, SUMMARY  (riutilizza da strategy_precision)
# =============================================================================
def _drawdown(series):
    rm = series.cummax()
    return (series - rm) / rm

def backtest(df):
    df = df.copy()
    df["position_change"]  = df.groupby("ticker")["position"].diff().fillna(0).abs()
    df["transaction_cost"] = df["position_change"] * TRANSACTION_COST
    df["daily_return_market"]   = df.groupby("ticker")["Adj Close"].pct_change().fillna(0)
    df["daily_return_strategy"] = df["daily_return_market"] * df["position"]
    df["cum_market"]         = df.groupby("ticker")["daily_return_market"].transform(lambda x: (1+x).cumprod())
    df["cum_strategy_gross"] = df.groupby("ticker")["daily_return_strategy"].transform(lambda x: (1+x).cumprod())
    costs = df.groupby("ticker")["transaction_cost"].transform("cumsum")
    df["cum_strategy_net"]   = df["cum_strategy_gross"] - costs
    return df

def build_trade_log(df):
    records = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("Date").reset_index(drop=True)
        entries = grp[grp["entry_flag_event"] == 1]
        exits   = grp[grp["exit_flag_event"]  == 1]
        n = min(len(entries), len(exits))
        for i in range(n):
            e, x = entries.iloc[i], exits.iloc[i]
            ep, xp = e.get("entry_price_event", pd.NA), x.get("exit_price_event", pd.NA)
            ed, xd = pd.to_datetime(e["Date"]), pd.to_datetime(x["Date"])
            if pd.notna(ep) and pd.notna(xp) and ep != 0:
                gr = (xp / ep) - 1
                nr = gr - 2 * TRANSACTION_COST
            else:
                gr = nr = pd.NA
            records.append({
                "ticker":        ticker,
                "trade_num":     i + 1,
                "entry_date":    ed,
                "entry_price":   ep,
                "entry_vix":     e.get("entry_vix_event", pd.NA),
                "entry_score":   e.get("entry_score_event", pd.NA),
                "ai_prob":       e.get("ai_prob_event", pd.NA),
                "exit_date":     xd,
                "exit_price":    xp,
                "exit_vix":      x.get("exit_vix_event", pd.NA),
                "exit_reason":   x.get("exit_reason_event", pd.NA),
                "holding_days":  (xd - ed).days if pd.notna(ed) and pd.notna(xd) else pd.NA,
                "gross_return_pct": gr * 100 if pd.notna(gr) else pd.NA,
                "net_return_pct":   nr * 100 if pd.notna(nr) else pd.NA,
            })
    return pd.DataFrame(records)

def build_summary(df, trade_log):
    s = df.groupby("ticker").agg(
        first_date               = ("Date", "min"),
        last_date                = ("Date", "max"),
        days_in_position         = ("position", "sum"),
        total_cost               = ("transaction_cost", "sum"),
        total_return_market      = ("cum_market", "last"),
        total_return_strategy_net= ("cum_strategy_net", "last"),
    ).reset_index()
    s["market_perf_pct"]       = (s["total_return_market"]       - 1) * 100
    s["strategy_net_perf_pct"] = (s["total_return_strategy_net"] - 1) * 100
    s["alpha_net_pct"]         = s["strategy_net_perf_pct"] - s["market_perf_pct"]

    dd = []
    for t, g in df.groupby("ticker"):
        dd.append({"ticker": t,
                   "max_drawdown_strategy_net_pct": _drawdown(g["cum_strategy_net"]).min() * 100})
    s = s.merge(pd.DataFrame(dd), on="ticker", how="left")

    if not trade_log.empty:
        stats = []
        for t, trd in trade_log.groupby("ticker"):
            wins   = trd[trd["net_return_pct"] > 0]["net_return_pct"]
            losses = trd[trd["net_return_pct"] < 0]["net_return_pct"]
            pf     = wins.sum() / abs(losses.sum()) if not losses.empty else np.nan
            stats.append({
                "ticker":          t,
                "closed_trades":   len(trd),
                "win_rate_pct":    (trd["net_return_pct"] > 0).mean() * 100,
                "profit_factor":   pf,
                "avg_holding_days":trd["holding_days"].mean(),
                "avg_ai_prob":     trd["ai_prob"].mean() if "ai_prob" in trd.columns else np.nan,
                "avg_entry_score": trd["entry_score"].mean() if "entry_score" in trd.columns else np.nan,
            })
        s = s.merge(pd.DataFrame(stats), on="ticker", how="left")
    return s


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import sys
    from universe_selection import select_universe

    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Carica modello AI
    if not os.path.exists(MODEL_PATH):
        print(f"Modello non trovato: {MODEL_PATH}")
        print("Esegui prima: python train_model.py")
        sys.exit(1)

    print("Caricamento modello AI...")
    model, feature_cols = joblib.load(MODEL_PATH), json.load(open(FEAT_PATH))

    # 2. Carica dati prezzi
    #dataset_file = "data/processed/dataset.csv"
    #prices_file  = "data/raw/prices.csv"
    dataset_file = "data/processed/dataset_extended.csv"
    prices_file  = "data/raw/prices_extended.csv"
    input_file   = dataset_file if os.path.exists(dataset_file) else prices_file
    if not os.path.exists(input_file):
        print(f"File non trovato: {input_file}")
        sys.exit(1)

    print(f"Lettura dati: {input_file}")
    df = pd.read_csv(input_file)
    df["Date"] = pd.to_datetime(df["Date"])

    # 3. Universe selection
    summary_file = "data/processed/target_stop_trailing_vix_regime_summary.csv"
    if os.path.exists(summary_file):
        summary_df = pd.read_csv(summary_file)
        approved   = select_universe(summary_df, verbose=True)
        df = df[df["ticker"].isin(approved)].copy()
        print(f"Titoli approvati: {df['ticker'].nunique()}")

    # 4. Calcola ticker stats storiche (per il feature engineering AI)
    tl_file = "data/processed/target_stop_trailing_vix_regime_trade_log.csv"
    ticker_stats = {}
    if os.path.exists(tl_file):
        tl = pd.read_csv(tl_file)
        for ticker, grp in tl.groupby("ticker"):
            ticker_stats[ticker] = {
                "win_rate": (grp["net_return_pct"] > 0).mean(),
                "count":    len(grp),
            }

    # 5. Prepara feature e applica strategia
    print("Calcolo feature...")
    df = prepare_data(df)

    print("Eseguo strategia con filtro AI...")
    df = apply_strategy(df, model, feature_cols, ticker_stats)
    df = backtest(df)

    trade_log = build_trade_log(df)
    summary   = build_summary(df, trade_log)

    # 6. Output
    df.to_csv(os.path.join(output_dir, f"{STRATEGY_NAME}_backtest.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, f"{STRATEGY_NAME}_summary.csv"), index=False)
    trade_log.to_csv(os.path.join(output_dir, f"{STRATEGY_NAME}_trade_log.csv"), index=False)

    # 7. Report finale
    print("\n===== SUMMARY AI SCORED =====")
    cols = ["ticker", "closed_trades", "win_rate_pct", "profit_factor",
            "strategy_net_perf_pct", "alpha_net_pct", "avg_ai_prob", "avg_entry_score"]
    available = [c for c in cols if c in summary.columns]
    print(summary[available].sort_values("profit_factor", ascending=False).to_string(index=False))

    if not trade_log.empty:
        months = (df["Date"].max() - df["Date"].min()).days / 30
        print(f"\nTrade totali: {len(trade_log)}")
        print(f"Media mensile: {len(trade_log)/months:.1f} trade/mese")
        print(f"Win rate medio: {(trade_log.net_return_pct > 0).mean()*100:.1f}%")
        print(f"AI prob media sui trade approvati: {trade_log.ai_prob.mean():.3f}")

    print(f"\n✅ File salvati in {output_dir}/")