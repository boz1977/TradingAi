"""
build_dataset.py  —  Fase 4
Costruisce il dataset finale unendo:
  - Prezzi storici estesi (data/raw/prices_extended.csv)
  - Dati macro giornalieri (data/raw/macro_extended.csv)
  - Feature tecniche (build_features.py)
  - Metadati settore per ticker

Output:
  data/processed/dataset_extended.csv  — dataset completo
  data/processed/dataset_extended_stats.csv — statistiche per ticker

Utilizzo:
    python src/build_dataset.py
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_features import build_features


# ---------------------------------------------------------------------------
# Settori FTSE MIB
# ---------------------------------------------------------------------------
SECTOR_MAP = {
    "A2A.MI":   "utilities",    "AMP.MI":   "financials",  "AZM.MI":   "industrials",
    "BAMI.MI":  "financials",   "BMED.MI":  "healthcare",  "BPE.MI":   "financials",
    "CPR.MI":   "financials",   "DIA.MI":   "consumer",    "ENEL.MI":  "utilities",
    "ENI.MI":   "energy",       "ERG.MI":   "utilities",   "FBK.MI":   "financials",
    "G.MI":     "financials",   "HER.MI":   "energy",      "IG.MI":    "financials",
    "INW.MI":   "telecom",      "ISP.MI":   "financials",  "LDO.MI":   "industrials",
    "MB.MI":    "financials",   "MONC.MI":  "consumer",    "NEXI.MI":  "technology",
    "PIRC.MI":  "industrials",  "PRY.MI":   "industrials", "PST.MI":   "telecom",
    "REC.MI":   "industrials",  "SRG.MI":   "utilities",   "STLAM.MI": "consumer",
    "TEN.MI":   "industrials",  "TRN.MI":   "utilities",   "UCG.MI":   "financials",
    "UNI.MI":   "financials",
}

# Beta approssimativo vs FTSE MIB (da calcolare idealmente sui dati,
# questi sono valori storici indicativi)
BETA_MAP = {
    "A2A.MI":0.7,  "AMP.MI":1.1,  "AZM.MI":0.9,  "BAMI.MI":1.4, "BMED.MI":0.8,
    "BPE.MI":1.3,  "CPR.MI":0.9,  "DIA.MI":0.8,  "ENEL.MI":0.7, "ENI.MI":0.9,
    "ERG.MI":0.8,  "FBK.MI":1.1,  "G.MI":1.0,    "HER.MI":0.7,  "IG.MI":0.9,
    "INW.MI":0.9,  "ISP.MI":1.3,  "LDO.MI":1.2,  "MB.MI":1.2,   "MONC.MI":0.9,
    "NEXI.MI":1.3, "PIRC.MI":0.8, "PRY.MI":1.1,  "PST.MI":0.6,  "REC.MI":0.9,
    "SRG.MI":0.6,  "STLAM.MI":1.5,"TEN.MI":1.0,  "TRN.MI":0.6,  "UCG.MI":1.4,
    "UNI.MI":1.3,
}

OUTPUT_DIR = "data/processed"


def load_prices(path: str) -> pd.DataFrame:
    print(f"Caricamento prezzi: {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Rimuovi timezone se presente
    if df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)

    print(f"  {len(df)} righe, {df['ticker'].nunique()} ticker, "
          f"{df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


def load_macro(path: str) -> pd.DataFrame:
    print(f"Caricamento macro: {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    if df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)
    print(f"  {len(df)} righe, colonne: {list(df.columns)}")
    return df


def merge_prices_macro(prices_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Unisce prezzi e macro su Date con left join.
    I dati macro vengono forward-filled per coprire i giorni mancanti.
    """
    # Seleziona solo le colonne macro utili (esclude Date)
    macro_cols = [c for c in macro_df.columns if c != "Date"]

    merged = prices_df.merge(macro_df[["Date"] + macro_cols], on="Date", how="left")

    # Forward fill per i gap (weekend, festivi)
    merged = merged.sort_values(["ticker", "Date"]).reset_index(drop=True)
    merged[macro_cols] = merged.groupby("ticker")[macro_cols].transform(lambda x: x.ffill())

    # Quanti NA rimangono dopo ffill?
    for col in macro_cols:
        n_na = merged[col].isna().sum()
        if n_na > 0:
            pct = n_na / len(merged) * 100
            if pct > 5:
                print(f"  ATTENZIONE: {col} ha {n_na} NA ({pct:.1f}%) dopo ffill")

    return merged


def add_sector_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge settore, beta e feature derivate dal settore."""
    df = df.copy()
    df["sector"] = df["ticker"].map(SECTOR_MAP).fillna("other")
    df["beta"]   = df["ticker"].map(BETA_MAP).fillna(1.0)

    # Flag settori difensivi (utile come feature per il modello AI)
    df["is_defensive"] = df["sector"].isin(["utilities", "telecom", "healthcare"]).astype(int)
    df["is_financial"]  = (df["sector"] == "financials").astype(int)
    df["is_cyclical"]   = df["sector"].isin(["consumer", "industrials", "energy"]).astype(int)

    return df


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola feature derivate dai dati macro:
      - VIX regime (basso/medio/alto)
      - Trend SP500
      - BTP/Bund spread livello
      - Momentum FTSEMIB
    """
    df = df.copy()

    # VIX regime
    if "VIX" in df.columns:
        df["vix_regime"] = pd.cut(
            df["VIX"],
            bins=[0, 15, 20, 25, 100],
            labels=["low", "medium", "high", "extreme"]
        )
        df["vix_rolling_20"] = df["VIX"].rolling(20).mean()
        df["vix_vs_avg"]     = df["VIX"] / df["vix_rolling_20"] - 1  # VIX sopra/sotto la sua media

    # SP500 trend
    if "SP500" in df.columns:
        df["sp500_ret_1m"]  = df["SP500"].pct_change(21)
        df["sp500_ret_3m"]  = df["SP500"].pct_change(63)
        df["sp500_ma200"]   = df["SP500"].rolling(200).mean()
        df["sp500_above_ma200"] = (df["SP500"] > df["sp500_ma200"]).astype(int)

    # BTP/Bund spread (rischio Italia)
    if "BTP_BUND_SPREAD" in df.columns:
        df["spread_ma20"]    = df["BTP_BUND_SPREAD"].rolling(20).mean()
        df["spread_rising"]  = (df["BTP_BUND_SPREAD"] > df["spread_ma20"]).astype(int)
        df["spread_high"]    = (df["BTP_BUND_SPREAD"] > 2.0).astype(int)   # spread > 200bp = rischio elevato

    # FTSE MIB momentum
    if "FTSEMIB" in df.columns:
        df["ftsemib_ret_1m"] = df["FTSEMIB"].pct_change(21)
        df["ftsemib_ret_3m"] = df["FTSEMIB"].pct_change(63)
        df["ftsemib_ma50"]   = df["FTSEMIB"].rolling(50).mean()
        df["ftsemib_above_ma50"] = (df["FTSEMIB"] > df["ftsemib_ma50"]).astype(int)

    # Euro/Dollaro (impatta aziende esportatrici)
    if "EURUSD" in df.columns:
        df["eurusd_ret_1m"]  = df["EURUSD"].pct_change(21)
        df["eurusd_ma50"]    = df["EURUSD"].rolling(50).mean()
        df["eurusd_trend"]   = (df["EURUSD"] > df["eurusd_ma50"]).astype(int)

    return df


def build_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Statistiche descrittive per ticker."""
    stats = df.groupby("ticker").agg(
        first_date  = ("Date", "min"),
        last_date   = ("Date", "max"),
        total_rows  = ("Date", "count"),
        sector      = ("sector", "first"),
        beta        = ("beta", "first"),
        avg_volume  = ("Volume", "mean"),
        avg_close   = ("Adj Close", "mean"),
    ).reset_index()

    stats["years_of_data"] = (
        (pd.to_datetime(stats["last_date"]) - pd.to_datetime(stats["first_date"])).dt.days / 365
    ).round(1)

    return stats.sort_values("sector")


if __name__ == "__main__":
    import sys

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    prices_file = "data/raw/prices_extended.csv"
    macro_file  = "data/raw/macro_extended.csv"

    # Fallback: se non esiste prices_extended usa prices standard
    if not os.path.exists(prices_file):
        prices_file = "data/raw/prices.csv"
        print(f"prices_extended non trovato, uso: {prices_file}")

    if not os.path.exists(prices_file):
        print(f"Nessun file prezzi trovato. Esegui prima download_prices_extended.py")
        sys.exit(1)

    print("=" * 55)
    print("BUILD DATASET ESTESO  —  Fase 4")
    print("=" * 55)

    # 1. Carica dati
    prices_df = load_prices(prices_file)

    if os.path.exists(macro_file):
        macro_df = load_macro(macro_file)
        print("\nUnione prezzi + macro...")
        df = merge_prices_macro(prices_df, macro_df)
    else:
        print(f"\nmacro_extended non trovato ({macro_file}), procedo senza macro.")
        df = prices_df.copy()

    # 2. Aggiungi settore e beta
    print("\nAggiunta feature settore...")
    df = add_sector_features(df)

    # 3. Feature tecniche (build_features.py)
    print("Calcolo feature tecniche...")
    df = build_features(df)

    # 4. Feature macro derivate
    print("Calcolo feature macro derivate...")
    df = add_macro_features(df)

    # 5. Salva
    out_path   = os.path.join(OUTPUT_DIR, "dataset_extended.csv")
    stats_path = os.path.join(OUTPUT_DIR, "dataset_extended_stats.csv")

    df.to_csv(out_path, index=False)

    stats = build_stats(df)
    stats.to_csv(stats_path, index=False)

    print(f"\n{'='*55}")
    print(f"DATASET COMPLETATO")
    print(f"  File:          {out_path}")
    print(f"  Righe totali:  {len(df)}")
    print(f"  Ticker:        {df['ticker'].nunique()}")
    print(f"  Periodo:       {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"  Colonne:       {len(df.columns)}")
    print(f"\nColonne disponibili:")
    for i, col in enumerate(sorted(df.columns), 1):
        print(f"  {i:2d}. {col}")
    print(f"\nStatistiche per ticker:")
    print(stats.to_string(index=False))
    print(f"\n✅ Stats: {stats_path}")