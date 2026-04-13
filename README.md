# FTSE MIB Strategy Lab

Dashboard Streamlit per analisi quantitativa e trading algoritmico sui titoli FTSE MIB.
Pipeline completa: download dati → feature engineering → backtest → AI scoring → screener giornaliero.

---

## Setup locale

```bash
# 1. Clona il repo
git clone https://github.com/TUO-USERNAME/trading-ai.git
cd trading-ai

# 2. Installa dipendenze
pip install -r requirements.txt

# 3. Genera i dati (prima volta — ~10 minuti)
python src/download_prices_extended.py
python src/download_macro_extended.py
python src/build_dataset.py

# 4. Esegui il backtest e addestra il modello
python src/strategy_precision.py
python src/train_model.py
python src/retrain_model_extended.py

# 5. Avvia l'app
streamlit run app.py
```

---

## Deploy su Streamlit Cloud

1. Crea un repo GitHub con tutti i file (i dati grandi sono esclusi dal `.gitignore`)
2. Vai su [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Seleziona il tuo repo e come file principale: `app.py`
4. **Importante**: aggiungi i secrets in Streamlit Cloud:
   - Vai su **Settings → Secrets**
   - Aggiungi: `GMAIL_APP_PASSWORD = "la-tua-app-password"`
5. Dopo il primo deploy, lancia gli script dalla sezione **"Lancia script"** dell'app
   per scaricare i dati e addestrare il modello direttamente nel cloud

---

## Struttura progetto

```
trading-ai/
├── app.py                          ← App Streamlit principale
├── requirements.txt
├── .gitignore
├── config.json                     ← Parametri strategia (generato dall'app)
├── .streamlit/
│   └── config.toml                 ← Tema e configurazione Streamlit
│
├── src/                            ← Pipeline Python
│   ├── tickers.py                  ← Lista ticker FTSE MIB
│   ├── build_features.py           ← Feature engineering
│   ├── universe_selection.py       ← Fase 1: filtra titoli
│   ├── strategy_precision.py       ← Fase 2: entry score avanzato
│   ├── train_model.py              ← Fase 3: training XGBoost
│   ├── strategy_ai_scored.py       ← Fase 3: strategia con AI filter
│   ├── download_prices_extended.py ← Fase 4: prezzi dal 2000
│   ├── download_macro_extended.py  ← Fase 4: VIX, SP500, BTP/Bund
│   ├── build_dataset.py            ← Fase 4: dataset unificato
│   ├── retrain_model_extended.py   ← Fase 4: re-training con più dati
│   ├── daily_screener.py           ← Fase 5: screener giornaliero
│   ├── portfolio_simulation.py     ← Simulazione portafoglio 365 giorni
│   └── notify_email.py             ← Notifiche email Gmail
│
├── models/                         ← Modelli salvati (esclusi da git tranne .json)
│   ├── feature_names.json
│   └── feature_names_v2.json
│
└── data/                           ← Dati (esclusi da git — troppo grandi)
    ├── raw/                        ← Prezzi e macro scaricati
    └── processed/                  ← Dataset, summary, trade log, screener
```

---

## Notifiche email

Per ricevere i segnali via email ogni sera:

1. Attiva la [verifica in 2 passaggi](https://myaccount.google.com/security) su Gmail
2. Crea una [App Password](https://myaccount.google.com/apppasswords) → Mail
3. Imposta la variabile d'ambiente:
   ```powershell
   # Windows PowerShell
   [System.Environment]::SetEnvironmentVariable("GMAIL_APP_PASSWORD","xxxx-xxxx-xxxx-xxxx","User")
   ```
4. Testa: `python src/notify_email.py --test`

Per inviare automaticamente ogni sera, crea un'attività pianificata Windows:
```
Programma: python
Argomenti: C:\path\to\trading-ai\src\daily_screener.py
Orario: 18:00 ogni giorno lavorativo
```

---

## Parametri principali

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| Stop loss | 4% | Uscita automatica in perdita |
| Take profit | 12% | Target prima dell'attivazione trailing |
| Trailing stop | 6% | Stop mobile dal massimo |
| VIX entry | < 20 | Non entrare con volatilità alta |
| VIX exit | > 25 | Uscita d'emergenza |
| Entry score | ≥ 6/9 | Soglia indicatori tecnici |
| AI prob | ≥ 60% | Soglia modello XGBoost |
| Max posizioni | 3 | Posizioni contemporanee nel portafoglio |
