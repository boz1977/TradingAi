"""
notify_email.py  —  Notifiche email giornaliere
Invia una email con i segnali del daily screener.
Può essere lanciato da solo o importato da daily_screener.py.

Setup (una volta sola):
  1. Vai su https://myaccount.google.com/apppasswords
  2. Crea una "App password" per "Mail" (serve 2FA attiva sul tuo Gmail)
  3. Copia la password di 16 caratteri in EMAIL_PASSWORD qui sotto
     (o meglio: mettila come variabile d'ambiente GMAIL_APP_PASSWORD)

Utilizzo:
    python src/notify_email.py                     # usa screener_latest.csv
    python src/notify_email.py --file screener_20260412.csv
    python src/notify_email.py --test              # invia email di test
"""

import os
import sys
import json
import argparse
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.base      import MIMEBase
from email                import encoders
from datetime             import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# CONFIGURAZIONE EMAIL
# =============================================================================
EMAIL_SENDER   = "boz1977@gmail.com"
EMAIL_RECEIVER = "boz1977@gmail.com"
EMAIL_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "rukryclqjnybwxhw")

# SMTP Gmail
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


# =============================================================================
# TEMPLATE HTML EMAIL
# =============================================================================
def build_html(signals_df: pd.DataFrame, date_str: str, vix: float | None) -> str:
    """Costruisce il corpo HTML dell'email."""

    def signal_rows(df):
        rows = ""
        for _, r in df.iterrows():
            color      = "#1D9E75" if r.get("signal") == "FORTE" else "#378ADD"
            label      = "🟢 FORTE" if r.get("signal") == "FORTE" else "🔵 OK"
            mom1m      = f"{r['momentum_1m']:+.1f}%" if pd.notna(r.get("momentum_1m")) else "—"
            mom3m      = f"{r['momentum_3m']:+.1f}%" if pd.notna(r.get("momentum_3m")) else "—"
            dist_ma50  = f"{r['dist_ma50_pct']:+.1f}%" if pd.notna(r.get("dist_ma50_pct")) else "—"
            rows += f"""
            <tr>
              <td style="padding:10px 8px;font-weight:600;color:{color}">{label}</td>
              <td style="padding:10px 8px;font-weight:700;font-size:15px">{r['ticker']}</td>
              <td style="padding:10px 8px;color:#666">{r.get('sector','')}</td>
              <td style="padding:10px 8px;text-align:right">€{r['close']:.3f}</td>
              <td style="padding:10px 8px;text-align:center;font-weight:600;color:{color}">{r['ai_prob']:.1%}</td>
              <td style="padding:10px 8px;text-align:center">{int(r['entry_score'])}/9</td>
              <td style="padding:10px 8px;text-align:center">{r.get('rsi', '—'):.0f}" if pd.notna(r.get('rsi')) else "—"</td>
              <td style="padding:10px 8px;text-align:center">{mom1m}</td>
              <td style="padding:10px 8px;text-align:center">{mom3m}</td>
              <td style="padding:10px 8px;text-align:center">{dist_ma50}</td>
            </tr>"""
        return rows

    n_signals = len(signals_df)
    n_strong  = len(signals_df[signals_df.get("signal", "") == "FORTE"]) if not signals_df.empty else 0
    vix_str   = f"{vix:.1f}" if vix else "N/D"
    vix_color = "#1D9E75" if (vix and vix < 16) else "#BA7517" if (vix and vix < 20) else "#D85A30"

    table_content = signal_rows(signals_df) if not signals_df.empty else """
        <tr><td colspan="10" style="text-align:center;padding:20px;color:#999">
            Nessun segnale oggi
        </td></tr>"""

    html = f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
             background:#f5f5f5;margin:0;padding:0">

  <div style="max-width:700px;margin:20px auto;background:white;
              border-radius:10px;overflow:hidden;box-shadow:0 2px 10px rgba(0,0,0,0.1)">

    <!-- Header -->
    <div style="background:#1a1a1a;padding:24px 28px;color:white">
      <div style="font-size:11px;letter-spacing:2px;color:#888;text-transform:uppercase;margin-bottom:6px">
        FTSE MIB Strategy Lab
      </div>
      <div style="font-size:22px;font-weight:600">
        Daily Screener — {date_str}
      </div>
      <div style="margin-top:10px;display:flex;gap:20px">
        <span style="font-size:13px;color:#aaa">
          Segnali trovati: <strong style="color:white">{n_signals}</strong>
        </span>
        <span style="font-size:13px;color:#aaa">
          Segnali forti: <strong style="color:#1D9E75">{n_strong}</strong>
        </span>
        <span style="font-size:13px;color:#aaa">
          VIX: <strong style="color:{vix_color}">{vix_str}</strong>
        </span>
      </div>
    </div>

    <!-- Segnali -->
    <div style="padding:20px 28px">
      {"<p style='color:#D85A30;font-weight:500'>⛔ VIX sopra soglia — nessun ingresso consigliato oggi.</p>" if n_signals == 0 else ""}

      <table style="width:100%;border-collapse:collapse;font-size:13px">
        <thead>
          <tr style="background:#f8f8f8;border-bottom:2px solid #eee">
            <th style="padding:8px;text-align:left;color:#666">Segnale</th>
            <th style="padding:8px;text-align:left;color:#666">Ticker</th>
            <th style="padding:8px;text-align:left;color:#666">Settore</th>
            <th style="padding:8px;text-align:right;color:#666">Prezzo</th>
            <th style="padding:8px;text-align:center;color:#666">AI prob</th>
            <th style="padding:8px;text-align:center;color:#666">Score</th>
            <th style="padding:8px;text-align:center;color:#666">RSI</th>
            <th style="padding:8px;text-align:center;color:#666">Mom 1M</th>
            <th style="padding:8px;text-align:center;color:#666">Mom 3M</th>
            <th style="padding:8px;text-align:center;color:#666">Dist MA50</th>
          </tr>
        </thead>
        <tbody>
          {table_content}
        </tbody>
      </table>
    </div>

    <!-- Parametri -->
    <div style="padding:16px 28px;background:#fafafa;border-top:1px solid #eee">
      <div style="font-size:11px;color:#999;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">
        Parametri strategia attivi
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:#555">
        <span>Stop loss: <strong>-4%</strong></span>
        <span>Take profit: <strong>+12%</strong></span>
        <span>Trailing stop: <strong>6%</strong></span>
        <span>VIX entry: <strong>&lt;20</strong></span>
        <span>VIX exit: <strong>&gt;25</strong></span>
        <span>Score min: <strong>6/9</strong></span>
        <span>AI prob min: <strong>60%</strong></span>
      </div>
    </div>

    <!-- Disclaimer -->
    <div style="padding:14px 28px;background:#fff8f0;border-top:1px solid #ffe0b2">
      <p style="font-size:11px;color:#999;margin:0">
        ⚠️ Questo è un sistema automatico a scopo informativo e non costituisce consulenza finanziaria.
        Verifica sempre i segnali prima di operare. Past performance non garantisce risultati futuri.
      </p>
    </div>

    <!-- Footer -->
    <div style="padding:14px 28px;background:#1a1a1a;text-align:center">
      <p style="font-size:11px;color:#555;margin:0">
        FTSE MIB Strategy Lab · Generato il {datetime.now().strftime('%d/%m/%Y %H:%M')}
      </p>
    </div>

  </div>
</body>
</html>"""
    return html


def build_text(signals_df: pd.DataFrame, date_str: str, vix: float | None) -> str:
    """Versione plain-text dell'email (fallback)."""
    lines = [
        f"FTSE MIB Daily Screener — {date_str}",
        f"VIX: {vix:.1f if vix else 'N/D'}",
        "=" * 50,
    ]
    if signals_df.empty:
        lines.append("Nessun segnale oggi.")
    else:
        for _, r in signals_df.iterrows():
            lines.append(
                f"{'[FORTE]' if r.get('signal')=='FORTE' else '[OK]   '} "
                f"{r['ticker']:<10} AI:{r['ai_prob']:.1%}  "
                f"Score:{int(r['entry_score'])}/9  "
                f"RSI:{r.get('rsi', 0):.0f}  "
                f"Mom3M:{r.get('momentum_3m', 0):+.1f}%"
            )
    lines += ["", "Stop loss:-4%  Take profit:+12%  Trailing:6%"]
    return "\n".join(lines)


# =============================================================================
# INVIO EMAIL
# =============================================================================
def send_email(
    signals_df: pd.DataFrame,
    date_str: str | None = None,
    vix: float | None = None,
    attach_csv: str | None = None,
    test_mode: bool = False,
) -> bool:
    """
    Invia l'email con i segnali.

    Args:
        signals_df:  DataFrame dei segnali (può essere vuoto)
        date_str:    Data da mostrare nell'oggetto
        vix:         Valore VIX attuale
        attach_csv:  Path del CSV da allegare (opzionale)
        test_mode:   Se True, invia una email di test senza segnali reali

    Returns:
        True se inviata con successo, False altrimenti
    """
    if EMAIL_PASSWORD == "METTI_QUI_LA_APP_PASSWORD":
        print("⚠️  Email non configurata.")
        print("    1. Vai su https://myaccount.google.com/apppasswords")
        print("    2. Crea App Password per 'Mail'")
        print("    3. Imposta la variabile d'ambiente GMAIL_APP_PASSWORD=<password>")
        print("       oppure modifica EMAIL_PASSWORD in notify_email.py")
        return False

    date_str = date_str or datetime.now().strftime("%d/%m/%Y")
    n = len(signals_df)

    if test_mode:
        subject = f"[TEST] FTSE MIB Screener — {date_str}"
    elif n == 0:
        subject = f"FTSE MIB Screener {date_str} — Nessun segnale"
    elif n == 1:
        ticker = signals_df.iloc[0]["ticker"]
        subject = f"FTSE MIB Screener {date_str} — 1 segnale: {ticker}"
    else:
        tickers = ", ".join(signals_df["ticker"].tolist())
        subject = f"FTSE MIB Screener {date_str} — {n} segnali: {tickers}"

    # Costruisci messaggio
    msg = MIMEMultipart("alternative")
    msg["From"]    = EMAIL_SENDER
    msg["To"]      = EMAIL_RECEIVER
    msg["Subject"] = subject

    text_body = build_text(signals_df, date_str, vix) if not test_mode else "Email di test — sistema funzionante."
    html_body = build_html(signals_df, date_str, vix) if not test_mode else "<h2>Test email OK</h2>"

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    # Allegato CSV
    if attach_csv and os.path.exists(attach_csv):
        with open(attach_csv, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f'attachment; filename="{os.path.basename(attach_csv)}"'
        )
        msg.attach(part)

    # Invio
    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.ehlo()
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print(f"✅ Email inviata a {EMAIL_RECEIVER}  (oggetto: {subject})")
        return True
    except smtplib.SMTPAuthenticationError:
        print("❌ Errore autenticazione Gmail.")
        print("   Assicurati di usare una App Password (non la password normale).")
        print("   https://myaccount.google.com/apppasswords")
        return False
    except Exception as e:
        print(f"❌ Errore invio email: {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Notifica email screener FTSE MIB")
    parser.add_argument("--file",  type=str, default=None,
                        help="CSV segnali da inviare (default: screener_latest.csv)")
    parser.add_argument("--test",  action="store_true",
                        help="Invia email di test")
    args = parser.parse_args()

    if args.test:
        send_email(pd.DataFrame(), test_mode=True)
        sys.exit(0)

    csv_file = args.file or "data/processed/screener_latest.csv"
    if not os.path.exists(csv_file):
        print(f"File non trovato: {csv_file}")
        print("Esegui prima: python src/daily_screener.py")
        sys.exit(1)

    df = pd.read_csv(csv_file)
    date_str = df["date"].iloc[0] if "date" in df.columns and not df.empty else datetime.now().strftime("%d/%m/%Y")
    vix      = df["vix"].iloc[0]  if "vix"  in df.columns and not df.empty else None

    send_email(df, date_str, vix, attach_csv=csv_file)