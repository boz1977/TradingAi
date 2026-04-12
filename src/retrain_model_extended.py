"""
retrain_model_extended.py  —  Fase 4
Ri-addestra il modello AI con il dataset esteso:
  - Più storia (dal 2000 invece che 2015)
  - Nuove feature macro: BTP/Bund spread, FTSEMIB trend, EURUSD
  - Feature settore più ricche
  - Ricalcola il trade log sul dataset esteso

Da eseguire DOPO:
    1. download_prices_extended.py
    2. download_macro_extended.py
    3. build_dataset.py
    4. (ri-esegui backtest su dataset_extended.csv con strategy_precision.py)

Output:
    models/trade_scorer_v2.joblib
    models/feature_names_v2.json
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import joblib

from train_model import (
    load_trade_logs,
    build_features as build_ml_features_base,
    evaluate_by_ticker,
    explain_model,
    SECTOR_MAP,
    SCORE_THRESHOLD,
    XGB_PARAMS,
)

MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "trade_scorer_v2.joblib")
FEAT_PATH   = os.path.join(MODEL_DIR, "feature_names_v2.json")


# =============================================================================
# Feature aggiuntive disponibili nel dataset esteso
# =============================================================================
EXTRA_FEATURE_COLS = [
    # Macro Italia
    "btp_bund_spread_at_entry",
    "spread_rising_at_entry",
    "spread_high_at_entry",
    # Mercato globale
    "sp500_ret_1m_at_entry",
    "sp500_ret_3m_at_entry",
    "sp500_above_ma200_at_entry",
    # FTSE MIB
    "ftsemib_ret_1m_at_entry",
    "ftsemib_above_ma50_at_entry",
    # Euro/Dollaro
    "eurusd_trend_at_entry",
    # Titolo
    "beta",
    "is_defensive",
    "is_financial",
    "is_cyclical",
    # Periodo storico
    "pre_2008_crisis",
    "gfc_period",
    "eurozone_crisis",
    "post_qe_period",
]


def enrich_trade_log_with_dataset(
    trade_log: pd.DataFrame,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per ogni trade nel log, cerca nel dataset i valori delle feature macro
    alla data di ingresso.
    Questo permette al modello di usare il contesto macro al momento dell'ingresso.
    """
    dataset = dataset.copy()
    dataset["Date"] = pd.to_datetime(dataset["Date"])

    # Colonne macro da aggiungere al trade log
    macro_cols = [c for c in dataset.columns if c in [
        "VIX", "SP500", "BRENT", "BTP_BUND_SPREAD", "FTSEMIB", "EURUSD",
        "sp500_ret_1m", "sp500_ret_3m", "sp500_above_ma200",
        "ftsemib_ret_1m", "ftsemib_above_ma50",
        "spread_rising", "spread_high",
        "eurusd_trend", "sector", "beta",
        "is_defensive", "is_financial", "is_cyclical",
        "vix_vs_avg",
    ]]

    # Crea un dataset "snapshot" per data + ticker
    snap = dataset[["Date", "ticker"] + macro_cols].copy()
    snap = snap.sort_values(["ticker", "Date"])

    trade_log = trade_log.copy()
    trade_log["entry_date"] = pd.to_datetime(trade_log["entry_date"])

    # Merge per ticker + data più vicina
    enriched = trade_log.merge(
        snap.rename(columns={"Date": "entry_date"}),
        on=["ticker", "entry_date"],
        how="left",
    )

    # Rinomina per chiarezza
    rename = {c: f"{c}_at_entry" for c in macro_cols}
    enriched = enriched.rename(columns=rename)

    return enriched


def build_features_extended(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estende le feature base di train_model.build_features()
    con le feature macro aggiuntive disponibili nel dataset esteso.
    """
    # Prima applica le feature base
    df = build_ml_features_base(df)

    # Periodi storici aggiuntivi
    df["pre_2008_crisis"]  = (df["entry_year"] < 2008).astype(int)
    df["gfc_period"]       = (
        (df["entry_date"] >= "2008-09-01") & (df["entry_date"] <= "2009-06-30")
    ).astype(int)
    df["eurozone_crisis"]  = (
        (df["entry_date"] >= "2011-06-01") & (df["entry_date"] <= "2012-12-31")
    ).astype(int)
    df["post_qe_period"]   = (
        (df["entry_date"] >= "2015-01-01") & (df["entry_date"] <= "2018-12-31")
    ).astype(int)

    # Rinomina colonne arricchite al momento ingresso
    for col in [
        "BTP_BUND_SPREAD", "sp500_ret_1m", "sp500_ret_3m", "sp500_above_ma200",
        "ftsemib_ret_1m", "ftsemib_above_ma50", "spread_rising", "spread_high",
        "eurusd_trend", "beta", "is_defensive", "is_financial", "is_cyclical",
    ]:
        src = f"{col}_at_entry"
        dst = col.lower() + "_at_entry"
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
        elif col in df.columns and dst not in df.columns:
            df[dst] = df[col]

    if "btp_bund_spread_at_entry" not in df.columns and "BTP_BUND_SPREAD_at_entry" in df.columns:
        df["btp_bund_spread_at_entry"] = df["BTP_BUND_SPREAD_at_entry"]

    return df


BASE_FEATURE_COLS = [
    "entry_vix", "entry_vix_bucket", "vix_low", "vix_medium", "vix_high",
    "entry_month", "entry_quarter", "entry_dayofweek",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "sector_enc", "ticker_win_rate_hist", "ticker_trade_count",
    "ticker_prev_trades", "ticker_streak",
    "entry_score", "is_score_strategy",
    "bull_period", "covid_period", "entry_year",
]


def train_extended_model(df: pd.DataFrame) -> tuple:
    df_feat = build_features_extended(df)

    all_feature_cols = BASE_FEATURE_COLS + [
        c for c in EXTRA_FEATURE_COLS if c in df_feat.columns
    ]

    # Rimuovi NA
    df_clean = df_feat.dropna(subset=all_feature_cols + ["win"]).copy()

    print(f"\nDataset training esteso: {len(df_clean)} trade")
    print(f"Feature base:      {len(BASE_FEATURE_COLS)}")
    print(f"Feature aggiuntive:{len([c for c in EXTRA_FEATURE_COLS if c in df_feat.columns])}")
    print(f"Win rate:          {df_clean['win'].mean()*100:.1f}%")

    X = df_clean[all_feature_cols].values
    y = df_clean["win"].values

    cv = StratifiedKFold(n_splits=5, shuffle=False)
    model = xgb.XGBClassifier(**XGB_PARAMS)
    cv_aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    print(f"\nCV AUC (5-fold): {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")
    for i, a in enumerate(cv_aucs, 1):
        print(f"  Fold {i}: {a:.4f}")

    model.fit(X, y)

    importance = pd.DataFrame({
        "feature":    all_feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\nTop 15 feature:")
    print(importance.head(15).to_string(index=False))

    return model, importance, cv_aucs, all_feature_cols, df_clean


if __name__ == "__main__":
    import sys

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("RE-TRAINING MODELLO — Dataset esteso  (Fase 4)")
    print("=" * 60)

    # 1. Trade log (cerca prima quelli della strategia precision/ai_scored)
    trade_log_files = []
    candidates = [
        "data/processed/ai_scored_v1_trade_log.csv",
        "data/processed/precision_v1_trade_log.csv",
        "data/processed/target_stop_trailing_vix_regime_trade_log.csv",
        "data/processed/target_stop_trailing_vix_score_trade_log.csv",
    ]
    for f in candidates:
        if os.path.exists(f):
            trade_log_files.append(f)
            print(f"  Trade log trovato: {f}")

    if not trade_log_files:
        print("Nessun trade log trovato.")
        sys.exit(1)

    print("\n[1/4] Caricamento trade log...")
    df_raw = load_trade_logs(trade_log_files)

    # 2. Arricchisci con dataset esteso se disponibile
    dataset_path = "data/processed/dataset_extended.csv"
    if os.path.exists(dataset_path):
        print(f"\n[2/4] Arricchimento con dataset esteso ({dataset_path})...")
        dataset = pd.read_csv(dataset_path)
        dataset["Date"] = pd.to_datetime(dataset["Date"])
        df_raw = enrich_trade_log_with_dataset(df_raw, dataset)
        print(f"  Colonne dopo arricchimento: {len(df_raw.columns)}")
    else:
        print(f"\n[2/4] dataset_extended non trovato — uso solo feature base.")

    # 3. Training
    print("\n[3/4] Training modello esteso...")
    model, importance, cv_scores, feat_cols, df_clean = train_extended_model(df_raw)

    # 4. Salva
    print("\n[4/4] Salvataggio...")
    joblib.dump(model, MODEL_PATH)
    with open(FEAT_PATH, "w") as f:
        json.dump(feat_cols, f, indent=2)

    print(f"\n✅ Modello v2:      {MODEL_PATH}")
    print(f"✅ Feature list v2: {FEAT_PATH}")
    print(f"\nPer usare il modello v2 in strategy_ai_scored.py:")
    print(f"  MODEL_PATH = '{MODEL_PATH}'")
    print(f"  FEAT_PATH  = '{FEAT_PATH}'")