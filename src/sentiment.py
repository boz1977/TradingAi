"""
sentiment.py  —  Fase 2
Analisi sentiment delle news finanziarie per i titoli italiani.

Fonti (tutte gratuite):
  1. NewsAPI    — notizie in italiano/inglese (chiave gratuita: 100 req/giorno)
  2. GDELT      — database globale notizie, completamente gratuito
  3. yfinance   — news integrate nel ticker (fallback)

Score sentiment:
  - Analisi keyword-based (veloce, no ML, funziona offline)
  - Score da -1 (molto negativo) a +1 (molto positivo)
  - Decay temporale: news più vecchie pesano meno

Configurazione NewsAPI (opzionale):
  Registrati su newsapi.org — piano gratuito: 100 req/giorno
  Imposta variabile d'ambiente: NEWSAPI_KEY=la_tua_chiave

Utilizzo:
    python src/sentiment.py
    python src/sentiment.py --ticker ENI.MI --days 7
"""

import os, sys, re, json, time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import DB
from tickers import ITALY_LIQUID_TICKERS, SECTOR_MAP

ROOT       = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SENTIMENT_FILE = OUTPUT_DIR / "sentiment.csv"

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")

# ── Dizionario keyword sentiment (italiano + inglese) ──────────────────────
POSITIVE_WORDS = {
    # Italiano
    "utile", "profitto", "crescita", "rialzo", "aumento", "record",
    "dividendo", "acquisto", "upgrade", "positivo", "forte", "solido",
    "superato", "battuto", "raccomandata", "buy", "ottimismo", "ripresa",
    "espansione", "accordo", "partnership", "investimento", "innovazione",
    "contratto", "ordine", "vincita", "approvazione", "superare", "migliorato",
    # Inglese
    "profit", "growth", "rise", "increase", "record", "dividend", "upgrade",
    "positive", "strong", "solid", "beat", "exceeded", "recommend", "optimism",
    "recovery", "expansion", "agreement", "investment", "innovation", "contract",
    "win", "approval", "improved", "outperform", "buy", "bullish",
}

NEGATIVE_WORDS = {
    # Italiano
    "perdita", "calo", "ribasso", "diminuzione", "crisi", "debito", "taglio",
    "licenziamento", "downgrade", "negativo", "debole", "mancato", "deluso",
    "sell", "pessimismo", "recessione", "contrazione", "rottura", "indagine",
    "multa", "sanzione", "rischio", "pericolo", "difficoltà", "perdere",
    # Inglese
    "loss", "decline", "fall", "decrease", "crisis", "debt", "cut",
    "layoff", "downgrade", "negative", "weak", "missed", "disappointed",
    "sell", "pessimism", "recession", "contraction", "breach", "investigation",
    "fine", "sanction", "risk", "danger", "difficulty", "bearish", "underperform",
}

# Nome completo → ticker (per ricerche news)
COMPANY_NAMES = {
    "ENI.MI":      ["ENI", "Eni spa"],
    "ENEL.MI":     ["ENEL", "Enel spa"],
    "ISP.MI":      ["Intesa Sanpaolo", "Intesa"],
    "UCG.MI":      ["UniCredit", "Unicredit"],
    "TRN.MI":      ["Terna"],
    "SRG.MI":      ["Snam"],
    "A2A.MI":      ["A2A"],
    "ERG.MI":      ["ERG", "Erg spa"],
    "LDO.MI":      ["Leonardo", "Leonardo spa"],
    "MONC.MI":     ["Moncler"],
    "STLAM.MI":    ["Stellantis"],
    "RACE.MI":     ["Ferrari", "Ferrari NV"],
    "BC.MI":       ["Brunello Cucinelli"],
    "NEXI.MI":     ["Nexi"],
    "PRY.MI":      ["Prysmian"],
    "PST.MI":      ["Poste Italiane"],
    "POSTE.MI":    ["Poste Italiane"],
    "WEBUILD.MI":  ["Webuild", "Salini"],
    "BZU.MI":      ["Buzzi Unicem", "Buzzi"],
    "CNHI.MI":     ["CNH Industrial", "CNH"],
    "MB.MI":       ["Mediobanca"],
    "FBK.MI":      ["FinecoBank", "Fineco"],
    "BAMI.MI":     ["Banco BPM"],
    "G.MI":        ["Generali", "Assicurazioni Generali"],
    "AZM.MI":      ["Azimut"],
    "IG.MI":       ["Italgas"],
    "ITMG.MI":     ["Italgas"],
    "HER.MI":      ["Hera"],
    "TEN.MI":      ["Tenaris"],
    "INW.MI":      ["Inwit"],
    "REC.MI":      ["Recordati"],
    "BMED.MI":     ["Biomedical"],
    "MTS.MI":      ["Maire Tecnimont", "Maire"],
    "SALC.MI":     ["Salcef"],
    "BGN.MI":      ["Banca Generali"],
    "ANIM.MI":     ["Anima Holding", "Anima"],
    "SOL.MI":      ["SOL Group", "SOL spa"],
    "SAVE.MI":     ["Save Aeroporti", "Aeroporto Venezia"],
    "TOD.MI":      ["Tod's", "Tods"],
    "GVS.MI":      ["GVS"],
    "TIP.MI":      ["Tamburi Investment"],
    "SFER.MI":     ["Saipem"],
    "OVS.MI":      ["OVS"],
    "ENAV.MI":     ["ENAV"],
    "IVG.MI":      ["Iveco Group", "Iveco"],
    "ILLIMITY.MI": ["Illimity Bank", "Illimity"],
    "DEDA.MI":     ["Dedagroup", "Deda"],
    "SIT.MI":      ["SIT Group"],
    "ELEN.MI":     ["El.En", "Elen"],
}


# =============================================================================
# ANALISI SENTIMENT KEYWORD-BASED
# =============================================================================
def _score_text(text: str) -> float:
    """
    Calcola lo score sentiment di un testo con keyword matching.
    Score: -1 (molto neg) → 0 (neutro) → +1 (molto pos)
    """
    if not text:
        return 0.0

    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)

    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos + neg

    if total == 0:
        return 0.0

    return round((pos - neg) / total, 3)


def _decay_weight(published_at: str, decay_days: int = 7) -> float:
    """Peso esponenziale: news più recenti pesano di più."""
    try:
        pub = datetime.fromisoformat(published_at.replace("Z", ""))
        age = (datetime.now() - pub).days
        return max(0.1, np.exp(-age / decay_days))
    except Exception:
        return 0.5


# =============================================================================
# FONTE 1: YFINANCE NEWS (fallback sempre disponibile)
# =============================================================================
def _fetch_yfinance_news(ticker: str, max_news: int = 10) -> list:
    """Scarica news da yfinance (sempre disponibile, no key richiesta)."""
    import yfinance as yf
    articles = []
    try:
        t    = yf.Ticker(ticker)
        news = t.news or []
        for item in news[:max_news]:
            articles.append({
                "title":        item.get("title", ""),
                "description":  item.get("summary", ""),
                "published_at": datetime.fromtimestamp(
                    item.get("providerPublishTime", time.time())
                ).isoformat(),
                "source":       item.get("publisher", "yfinance"),
                "url":          item.get("link", ""),
            })
    except Exception as e:
        print(f"  [WARN] yfinance news {ticker}: {e}")
    return articles


# =============================================================================
# FONTE 2: NEWSAPI (richiede chiave gratuita)
# =============================================================================
def _fetch_newsapi(query: str, days_back: int = 7) -> list:
    """Scarica notizie da NewsAPI."""
    if not NEWSAPI_KEY:
        return []

    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        query,
        "from":     from_date,
        "language": "it",
        "sortBy":   "publishedAt",
        "pageSize": 20,
        "apiKey":   NEWSAPI_KEY,
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data     = r.json()
        articles = data.get("articles", [])
        return [{
            "title":        a.get("title", ""),
            "description":  a.get("description", ""),
            "published_at": a.get("publishedAt", ""),
            "source":       a.get("source", {}).get("name", "newsapi"),
            "url":          a.get("url", ""),
        } for a in articles]
    except Exception as e:
        print(f"  [WARN] NewsAPI: {e}")
        return []


# =============================================================================
# FONTE 3: GDELT (completamente gratuito)
# =============================================================================
def _fetch_gdelt(query: str, days_back: int = 7) -> list:
    """Scarica notizie da GDELT (gratuito, no key)."""
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query":      f"{query} sourcelang:italian OR sourcelang:english",
        "mode":       "artlist",
        "maxrecords": 20,
        "timespan":   f"{days_back}d",
        "format":     "json",
    }

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data     = r.json()
        articles = data.get("articles", [])
        return [{
            "title":        a.get("title", ""),
            "description":  a.get("title", ""),  # GDELT non ha description
            "published_at": a.get("seendate", ""),
            "source":       a.get("domain", "gdelt"),
            "url":          a.get("url", ""),
        } for a in articles]
    except Exception as e:
        print(f"  [WARN] GDELT: {e}")
        return []


# =============================================================================
# CALCOLO SCORE AGGREGATO
# =============================================================================
def compute_sentiment(
    ticker:    str,
    days_back: int = 7,
    verbose:   bool = False,
) -> dict:
    """
    Calcola il sentiment aggregato per un ticker dalle ultime N notizie.

    Ritorna:
        {
            "ticker":        "ENI.MI"
            "score":         0.35   (-1 a +1)
            "n_articles":    8
            "n_positive":    5
            "n_negative":    1
            "n_neutral":     2
            "confidence":    0.7    (0-1, bassa con pochi articoli)
            "date":          "2026-04-17"
            "signal":        "positive" / "negative" / "neutral"
        }
    """
    company_names = COMPANY_NAMES.get(ticker, [ticker.replace(".MI", "")])
    query         = " OR ".join(f'"{name}"' for name in company_names[:2])

    # Raccoglie da tutte le fonti
    articles = []
    articles += _fetch_yfinance_news(ticker)
    articles += _fetch_newsapi(query, days_back)
    articles += _fetch_gdelt(company_names[0], days_back)

    if not articles:
        return {
            "ticker": ticker, "score": 0.0, "n_articles": 0,
            "n_positive": 0, "n_negative": 0, "n_neutral": 0,
            "confidence": 0.0, "date": datetime.now().strftime("%Y-%m-%d"),
            "signal": "neutral",
        }

    # Calcola score pesato per ogni articolo
    weighted_scores = []
    n_pos = n_neg = n_neu = 0

    for art in articles:
        text  = f"{art.get('title', '')} {art.get('description', '')}"
        score = _score_text(text)
        weight= _decay_weight(art.get("published_at", ""), decay_days=days_back)
        weighted_scores.append(score * weight)

        if score > 0.1:    n_pos += 1
        elif score < -0.1: n_neg += 1
        else:               n_neu += 1

    total_weight  = sum(abs(s) for s in weighted_scores) or 1
    avg_score     = sum(weighted_scores) / len(weighted_scores)
    confidence    = min(1.0, len(articles) / 10)

    signal = "positive" if avg_score > 0.1 else "negative" if avg_score < -0.1 else "neutral"

    result = {
        "ticker":      ticker,
        "score":       round(avg_score, 3),
        "n_articles":  len(articles),
        "n_positive":  n_pos,
        "n_negative":  n_neg,
        "n_neutral":   n_neu,
        "confidence":  round(confidence, 2),
        "date":        datetime.now().strftime("%Y-%m-%d"),
        "signal":      signal,
    }

    if verbose:
        print(f"  {ticker}: score={avg_score:+.2f} ({signal}) "
              f"articles={len(articles)} (+{n_pos}/-{n_neg}/={n_neu})")

    return result


def compute_all_sentiment(
    tickers:   list = None,
    days_back: int = 7,
    verbose:   bool = True,
) -> pd.DataFrame:
    """Calcola sentiment per tutti i ticker e salva nel CSV."""
    tickers = tickers or ITALY_LIQUID_TICKERS
    results = []

    if verbose:
        print(f"Calcolo sentiment per {len(tickers)} titoli (ultimi {days_back} giorni)...")

    for ticker in tickers:
        result = compute_sentiment(ticker, days_back, verbose=verbose)
        results.append(result)
        time.sleep(0.3)   # rate limiting gentile

    df = pd.DataFrame(results)
    df.to_csv(SENTIMENT_FILE, index=False)
    if verbose:
        print(f"\n[OK] Sentiment salvato: {SENTIMENT_FILE}")
        pos = (df["signal"] == "positive").sum()
        neg = (df["signal"] == "negative").sum()
        print(f"  Positivi: {pos}  Negativi: {neg}  Neutri: {len(df)-pos-neg}")
    return df


def get_sentiment_feature(ticker: str, sent_df: pd.DataFrame = None) -> dict:
    """Feature sentiment per il modello AI."""
    defaults = {
        "sentiment_score":    0.0,
        "sentiment_positive": 0,
        "sentiment_negative": 0,
        "sentiment_confidence": 0.0,
        "n_news_articles":    0,
    }

    if sent_df is None:
        if SENTIMENT_FILE.exists():
            sent_df = pd.read_csv(SENTIMENT_FILE)
        else:
            return defaults

    row = sent_df[sent_df["ticker"] == ticker]
    if row.empty:
        return defaults

    row = row.iloc[0]
    return {
        "sentiment_score":      float(row.get("score", 0)),
        "sentiment_positive":   int(row.get("signal") == "positive"),
        "sentiment_negative":   int(row.get("signal") == "negative"),
        "sentiment_confidence": float(row.get("confidence", 0)),
        "n_news_articles":      int(row.get("n_articles", 0)),
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--days",   type=int, default=7)
    parser.add_argument("--quick",  action="store_true",
                        help="Solo top 10 titoli per FTSE MIB")
    args = parser.parse_args()

    print("=" * 55)
    print("SENTIMENT ANALYSIS — Fase 2")
    print("=" * 55)

    if not NEWSAPI_KEY:
        print("\n[INFO] NEWSAPI_KEY non configurata.")
        print("  Registrati su newsapi.org (gratuito) e imposta:")
        print("  Windows: set NEWSAPI_KEY=la_tua_chiave")
        print("  Continuo con yfinance news + GDELT...")

    if args.ticker:
        result = compute_sentiment(args.ticker, args.days, verbose=True)
        print(json.dumps(result, indent=2))
    else:
        from tickers import FTSE_MIB_TICKERS
        tickers = FTSE_MIB_TICKERS[:10] if args.quick else ITALY_LIQUID_TICKERS
        df = compute_all_sentiment(tickers, args.days)
        print(f"\nTop 5 positivi:")
        print(df.nlargest(5, "score")[["ticker","score","signal","n_articles"]].to_string(index=False))
        print(f"\nTop 5 negativi:")
        print(df.nsmallest(5, "score")[["ticker","score","signal","n_articles"]].to_string(index=False))
