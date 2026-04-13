"""
train_model.py  —  Fase 3
Addestra un modello XGBoost sui trade storici per predire
la probabilità che un trade sia vincente.

Input:
    data/processed/target_stop_trailing_vix_regime_trade_log.csv
    data/processed/target_stop_trailing_vix_score_trade_log.csv
    (opzionale) data/processed/precision_v1_trade_log.csv

Output:
    models/trade_scorer.joblib   — modello salvato
    models/feature_names.json    — lista feature usate
    data/processed/model_report.csv — performance per ticker

Utilizzo successivo (live scoring):
    from train_model import score_trade
    prob = score_trade(model, feature_dict)
    if prob >= 0.60: entra nel trade
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Dipendenze — installa con:  pip install xgboost scikit-learn joblib shap
# ---------------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# =============================================================================
# CONFIGURAZIONE
# =============================================================================
TRADE_LOG_FILES = [
    "data/processed/target_stop_trailing_vix_regime_trade_log.csv",
    "data/processed/target_stop_trailing_vix_score_trade_log.csv",
]

MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "trade_scorer.joblib")
FEAT_PATH   = os.path.join(MODEL_DIR, "feature_names.json")
REPORT_PATH = "data/processed/model_report.csv"

# Soglia probabilità per ingresso
SCORE_THRESHOLD = 0.60

# Settori FTSE MIB
SECTOR_MAP = {
    "A2A.MI":   "utilities",   "AMP.MI":   "financials", "AZM.MI":   "industrials",
    "BAMI.MI":  "financials",  "BMED.MI":  "healthcare",  "BPE.MI":   "financials",
    "CPR.MI":   "financials",  "DIA.MI":   "consumer",    "ENEL.MI":  "utilities",
    "ENI.MI":   "energy",      "ERG.MI":   "utilities",   "FBK.MI":   "financials",
    "G.MI":     "financials",  "HER.MI":   "energy",      "IG.MI":    "financials",
    "INW.MI":   "telecom",     "ISP.MI":   "financials",  "LDO.MI":   "industrials",
    "MB.MI":    "financials",  "MONC.MI":  "consumer",    "NEXI.MI":  "technology",
    "PIRC.MI":  "industrials", "PRY.MI":   "industrials", "PST.MI":   "telecom",
    "REC.MI":   "industrials", "SRG.MI":   "utilities",   "STLAM.MI": "consumer",
    "TEN.MI":   "industrials", "TRN.MI":   "utilities",   "UCG.MI":   "financials",
    "UNI.MI":   "financials",
}

# Parametri XGBoost
XGB_PARAMS = {
    "n_estimators":     400,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "scale_pos_weight": 1.3,
    "eval_metric":      "auc",
    "random_state":     42,
    "n_jobs":           -1,
}


# =============================================================================
# 1. CARICAMENTO E UNIONE TRADE LOG
# =============================================================================
def load_trade_logs(files: list[str]) -> pd.DataFrame:
    parts = []
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            strategy_name = os.path.basename(f).replace("_trade_log.csv", "")
            df["strategy"] = strategy_name
            parts.append(df)
            print(f"  Caricato: {f}  ({len(df)} trade)")
        else:
            print(f"  Non trovato: {f}")

    if not parts:
        raise FileNotFoundError("Nessun trade log trovato.")

    combined = pd.concat(parts, ignore_index=True)
    combined["entry_date"] = pd.to_datetime(combined["entry_date"])
    combined["exit_date"]  = pd.to_datetime(combined["exit_date"])
    return combined


# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Costruisce le feature per il modello a partire dal trade log.
    Ogni riga = un trade. Feature disponibili al momento dell'ingresso.
    """
    df = df.copy()

    # --- Target ---
    df["win"] = (df["net_return_pct"] > 0).astype(int)

    # --- Feature temporali (disponibili all'ingresso) ---
    df["entry_month"]     = df["entry_date"].dt.month
    df["entry_dayofweek"] = df["entry_date"].dt.dayofweek   # 0=lunedì
    df["entry_quarter"]   = df["entry_date"].dt.quarter
    df["entry_year"]      = df["entry_date"].dt.year

    # --- Stagionalità ciclica (sin/cos per evitare discontinuità) ---
    df["month_sin"] = np.sin(2 * np.pi * df["entry_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["entry_month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["entry_dayofweek"] / 5)
    df["dow_cos"]   = np.cos(2 * np.pi * df["entry_dayofweek"] / 5)

    # --- VIX all'ingresso ---
    df["entry_vix_bucket"] = pd.cut(
        df["entry_vix"],
        bins=[0, 13, 16, 20, 25, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)

    df["vix_low"]    = (df["entry_vix"] < 15).astype(int)
    df["vix_medium"] = ((df["entry_vix"] >= 15) & (df["entry_vix"] < 20)).astype(int)
    df["vix_high"]   = (df["entry_vix"] >= 20).astype(int)

    # --- Settore (label encoded) ---
    df["sector"] = df["ticker"].map(SECTOR_MAP).fillna("other")
    le = LabelEncoder()
    df["sector_enc"] = le.fit_transform(df["sector"])

    # --- Feature ticker (win rate storico per ticker) ---
    ticker_stats = (
        df.groupby("ticker")["win"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "ticker_win_rate_hist", "count": "ticker_trade_count"})
    )
    df = df.merge(ticker_stats, on="ticker", how="left")

    # --- Numero trade precedenti per lo stesso ticker (esperienza) ---
    df = df.sort_values(["ticker", "entry_date"]).reset_index(drop=True)
    df["ticker_prev_trades"] = df.groupby("ticker").cumcount()

    # --- Streak: quanti trade consecutivi ha vinto/perso il ticker ---
    def _streak(wins: pd.Series) -> pd.Series:
        streak = []
        current = 0
        for w in wins:
            if w == 1:
                current = max(current, 0) + 1
            else:
                current = min(current, 0) - 1
            streak.append(current)
        return pd.Series(streak, index=wins.index)

    df["ticker_streak"] = df.groupby("ticker")["win"].transform(_streak)
    df["ticker_streak"] = df.groupby("ticker")["ticker_streak"].shift(1).fillna(0)

    # --- Strategia (regime vs score) ---
    df["is_score_strategy"] = df["strategy"].str.contains("score").astype(int)

    # --- Entry score (se presente) ---
    if "entry_score" in df.columns:
        df["entry_score"] = df["entry_score"].fillna(df["entry_score"].median())
    else:
        df["entry_score"] = 4.0   # valore default

    # --- Anno come proxy del regime di mercato ---
    df["bull_period"] = (
        ((df["entry_year"] >= 2019) & (df["entry_year"] <= 2021)) |
        (df["entry_year"] >= 2023)
    ).astype(int)

    df["covid_period"] = (
        (df["entry_date"] >= "2020-02-01") & (df["entry_date"] <= "2020-06-30")
    ).astype(int)

    return df


# =============================================================================
# 3. LISTA FEATURE PER IL MODELLO
# =============================================================================
FEATURE_COLS = [
    # VIX
    "entry_vix", "entry_vix_bucket", "vix_low", "vix_medium", "vix_high",
    # Temporali
    "entry_month", "entry_quarter", "entry_dayofweek",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    # Ticker / settore
    "sector_enc", "ticker_win_rate_hist", "ticker_trade_count",
    "ticker_prev_trades", "ticker_streak",
    # Score ingresso
    "entry_score",
    # Strategia
    "is_score_strategy",
    # Regime mercato
    "bull_period", "covid_period", "entry_year",
]


# =============================================================================
# 4. TRAINING E VALUTAZIONE
# =============================================================================
def train_model(df: pd.DataFrame) -> tuple:
    """
    Addestra XGBoost con cross-validation temporale.
    Ritorna (model, feature_importance_df, cv_scores).
    """
    df_feat = build_features(df)

    # Rimuovi righe con NA nelle feature
    available_feats = [c for c in FEATURE_COLS if c in df_feat.columns]
    df_clean = df_feat.dropna(subset=available_feats + ["win"]).copy()

    print(f"\nDataset training: {len(df_clean)} trade (da {len(df_feat)} totali)")
    print(f"Feature usate:    {len(available_feats)}")
    print(f"Win rate:         {df_clean['win'].mean()*100:.1f}%  "
          f"({df_clean['win'].sum()} win / {(df_clean['win']==0).sum()} loss)")

    X = df_clean[available_feats].values
    y = df_clean["win"].values

    # Cross-validation temporale (5 fold stratificati)
    cv = StratifiedKFold(n_splits=5, shuffle=False)
    model = xgb.XGBClassifier(**XGB_PARAMS)
    cv_aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    print(f"\nCross-validation AUC (5-fold):")
    for i, auc in enumerate(cv_aucs, 1):
        print(f"  Fold {i}: {auc:.4f}")
    print(f"  Media: {cv_aucs.mean():.4f}  ±  {cv_aucs.std():.4f}")

    # Training finale su tutti i dati
    model.fit(X, y)

    # Feature importance
    importance = pd.DataFrame({
        "feature":   available_feats,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\nTop 10 feature importanti:")
    print(importance.head(10).to_string(index=False))

    return model, importance, cv_aucs, available_feats, df_clean


def evaluate_by_ticker(model, df_clean: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Valuta la performance del modello ticker per ticker.
    """
    records = []
    X = df_clean[feature_cols].values
    df_clean = df_clean.copy()
    df_clean["prob_win"] = model.predict_proba(X)[:, 1]
    df_clean["model_signal"] = (df_clean["prob_win"] >= SCORE_THRESHOLD).astype(int)

    for ticker, grp in df_clean.groupby("ticker"):
        filtered = grp[grp["model_signal"] == 1]
        if filtered.empty:
            continue

        records.append({
            "ticker":             ticker,
            "total_trades":       len(grp),
            "model_approved":     len(filtered),
            "pct_approved":       round(len(filtered) / len(grp) * 100, 1),
            "win_rate_all":       round(grp["win"].mean() * 100, 1),
            "win_rate_filtered":  round(filtered["win"].mean() * 100, 1),
            "win_rate_delta":     round((filtered["win"].mean() - grp["win"].mean()) * 100, 1),
            "avg_return_all":     round(grp["net_return_pct"].mean(), 2),
            "avg_return_filtered":round(filtered["net_return_pct"].mean(), 2),
        })

    return pd.DataFrame(records).sort_values("win_rate_delta", ascending=False)


# =============================================================================
# 5. FUNZIONE DI SCORING LIVE (usata dalla strategia)
# =============================================================================
def score_trade(model, features: dict, feature_cols: list) -> float:
    """
    Dato un dizionario di feature per un singolo trade candidato,
    ritorna la probabilità stimata che sia vincente.

    Esempio:
        features = {
            "entry_vix": 16.5,
            "entry_month": 4,
            "sector_enc": 3,
            "ticker_win_rate_hist": 0.51,
            ...
        }
        prob = score_trade(model, features, feature_cols)
        if prob >= 0.60:
            # entra nel trade
    """
    row = pd.DataFrame([features])
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0.0
    X = row[feature_cols].values
    return float(model.predict_proba(X)[0, 1])


def load_model(model_path: str = MODEL_PATH, feat_path: str = FEAT_PATH):
    """Carica modello e lista feature dal disco."""
    model = joblib.load(model_path)
    with open(feat_path) as f:
        feature_cols = json.load(f)
    return model, feature_cols


# =============================================================================
# 6. SHAP — interpretabilità (opzionale)
# =============================================================================
def explain_model(model, df_clean: pd.DataFrame, feature_cols: list, output_dir: str):
    """Genera plot SHAP per capire quali feature guidano le predizioni."""
    if not HAS_SHAP:
        print("shap non installato. Salta spiegazione modello.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X = df_clean[feature_cols].values
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
    plt.tight_layout()
    out = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✅ SHAP summary salvato: {out}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("=" * 60)
    print("FASE 3 — TRAINING MODELLO AI TRADE SCORER")
    print("=" * 60)

    # 1. Carica dati
    print("\n[1/5] Caricamento trade log...")
    df_raw = load_trade_logs(TRADE_LOG_FILES)
    print(f"      Totale trade caricati: {len(df_raw)}")

    # 2. Addestra
    print("\n[2/5] Training XGBoost con cross-validation...")
    model, importance, cv_scores, feature_cols, df_clean = train_model(df_raw)

    # 3. Report per ticker
    print("\n[3/5] Valutazione per ticker...")
    report = evaluate_by_ticker(model, df_clean, feature_cols)

    print("\n===== REPORT: IMPATTO FILTRO AI PER TICKER =====")
    print(report.to_string(index=False))

    # Statistiche aggregate
    all_wr     = df_clean["win"].mean() * 100
    filtered   = df_clean[df_clean["prob_win"] >= SCORE_THRESHOLD] if "prob_win" in df_clean.columns else pd.DataFrame()

    # Ricalcola prob su tutto il dataset
    X_all = df_clean[feature_cols].values
    df_clean["prob_win"] = model.predict_proba(X_all)[:, 1]
    filtered = df_clean[df_clean["prob_win"] >= SCORE_THRESHOLD]

    print(f"\n===== RISULTATI AGGREGATI (soglia {SCORE_THRESHOLD}) =====")
    print(f"Trade totali nel training:      {len(df_clean)}")
    print(f"Trade approvati dal modello:    {len(filtered)}  "
          f"({len(filtered)/len(df_clean)*100:.1f}%)")
    print(f"Win rate SENZA filtro AI:       {all_wr:.1f}%")
    print(f"Win rate CON filtro AI:         {filtered['win'].mean()*100:.1f}%")
    print(f"Rendimento medio SENZA filtro:  {df_clean['net_return_pct'].mean():.2f}%")
    print(f"Rendimento medio CON filtro:    {filtered['net_return_pct'].mean():.2f}%")
    print(f"\nCV AUC medio: {cv_scores.mean():.4f}")

    # 4. Salva modello
    print("\n[4/5] Salvataggio modello...")
    joblib.dump(model, MODEL_PATH)
    with open(FEAT_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    report.to_csv(REPORT_PATH, index=False)

    print(f"  ✅ Modello:      {MODEL_PATH}")
    print(f"  ✅ Feature list: {FEAT_PATH}")
    print(f"  ✅ Report:       {REPORT_PATH}")

    # 5. SHAP (se disponibile)
    print("\n[5/5] Interpretabilità modello...")
    explain_model(model, df_clean, feature_cols, "data/processed")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETATO")
    print(f"Soglia consigliata per ingresso: prob_win >= {SCORE_THRESHOLD}")
    print("=" * 60)