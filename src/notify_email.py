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
              <td style="padding:10px 8px;text-align:center">{f"{r.get('rsi',0):.0f}" if pd.notna(r.get('rsi')) else "—"}</td>
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


# =============================================================================
# EMAIL GIORNALIERA COMPLETA — posizioni reali + alert + segnali
# =============================================================================

def build_daily_html(signals_df, open_positions, alerts, portfolio_summary, date_str, vix=None):
    """Costruisce HTML email giornaliera."""
    import pandas as pd

    vix_str   = f"{vix:.1f}" if vix else "N/D"
    vix_color = "#1D9E75" if (vix and vix < 16) else "#BA7517" if (vix and vix < 20) else "#D85A30"
    n_open    = portfolio_summary.get("n_open", 0)
    total_pnl = portfolio_summary.get("total_pnl_eur", 0)
    pnl_color = "#1D9E75" if total_pnl >= 0 else "#D85A30"

    # ── Posizioni aperte ──
    pos_rows = ""
    has_pos = (open_positions is not None and
               hasattr(open_positions, "empty") and
               not open_positions.empty)
    if has_pos:
        for _, row in open_positions.iterrows():
            pnl_pct   = float(row.get("current_pnl_pct", 0) or 0)
            pnl_eur   = float(row.get("current_pnl_eur", 0) or 0)
            pc        = "#1D9E75" if pnl_pct >= 0 else "#D85A30"
            alert_val = row.get("alert")
            has_alert = alert_val and str(alert_val) not in ("nan","None","")
            bg        = "#fff8f0" if has_alert else "#ffffff"
            alert_td  = f'<br><small style="color:#A32D2D">{alert_val}</small>' if has_alert else ""
            trailing  = " (trailing)" if row.get("trailing_active") else ""
            ep = float(row.get("entry_price", 0))
            cp = float(row.get("current_price", 0))
            sl = float(row.get("stop_loss_price", 0))
            days = int(row.get("days_open", 0))
            ticker = row.get("ticker", "")
            edate  = row.get("entry_date", "")
            pos_rows += (
                '<tr style="background:' + bg + '">'
                '<td style="padding:10px 8px;font-weight:600">' + ticker + '</td>'
                '<td style="padding:10px 8px">' + edate + '</td>'
                '<td style="padding:10px 8px">&euro;' + f'{ep:.3f}' + '</td>'
                '<td style="padding:10px 8px">&euro;' + f'{cp:.3f}' + '</td>'
                '<td style="padding:10px 8px;color:' + pc + ';font-weight:500">'
                + f'{pnl_pct:+.1f}%<br><small>&euro;{pnl_eur:+.0f}</small></td>'
                '<td style="padding:10px 8px">&euro;' + f'{sl:.3f}' + '<small>' + trailing + '</small></td>'
                '<td style="padding:10px 8px">' + str(days) + ' gg' + alert_td + '</td>'
                '</tr>'
            )
    else:
        pos_rows = '<tr><td colspan="7" style="padding:16px;text-align:center;color:#999">Nessuna posizione aperta</td></tr>'

    # ── Alert ──
    alert_section = ""
    if alerts:
        items = ""
        for a in alerts:
            items += (
                '<div style="background:#fff8f0;border-left:3px solid #D85A30;'
                'padding:10px 14px;margin-bottom:8px;border-radius:0 8px 8px 0">'
                '<strong>' + a["ticker"] + '</strong> &mdash; ' + a["alert"] + ' '
                '<small style="color:#888">&euro;' + f'{a["price"]:.3f}' + ' | '
                + f'{a["pnl_pct"]:+.1f}%</small></div>'
            )
        alert_section = (
            '<div style="padding:16px 24px;background:#fff5f5">'
            '<div style="font-size:11px;font-weight:500;text-transform:uppercase;'
            'color:#A32D2D;margin-bottom:10px">Azione richiesta</div>'
            + items + '</div>'
        )

    # ── Segnali ──
    n_segnali = 0
    sig_rows  = ""
    has_sig = (signals_df is not None and
               hasattr(signals_df, "empty") and
               not signals_df.empty)
    if has_sig:
        n_segnali = len(signals_df)
        for _, row in signals_df.iterrows():
            is_forte = row.get("signal") == "FORTE"
            prob     = float(row.get("ai_prob", 0)) * 100
            score    = int(row.get("entry_score", 0))
            bc = "#E1F5EE" if is_forte else "#E6F1FB"
            bt = "#0F6E56" if is_forte else "#185FA5"
            label = "FORTE" if is_forte else "OK"
            mom = f'{float(row.get("momentum_3m",0)):+.1f}%' if row.get("momentum_3m") else "&#8212;"
            ticker  = row.get("ticker","")
            sector  = str(row.get("sector","")).capitalize()
            close   = float(row.get("close",0))
            sig_rows += (
                '<tr>'
                '<td style="padding:10px 8px"><span style="background:' + bc + ';color:' + bt + ';'
                'padding:2px 8px;border-radius:20px;font-size:11px;font-weight:500">' + label + '</span></td>'
                '<td style="padding:10px 8px;font-weight:600">' + ticker + '</td>'
                '<td style="padding:10px 8px;color:#888">' + sector + '</td>'
                '<td style="padding:10px 8px">&euro;' + f'{close:.3f}' + '</td>'
                '<td style="padding:10px 8px;font-weight:500;color:#1D9E75">' + f'{prob:.0f}%' + '</td>'
                '<td style="padding:10px 8px">' + f'{score}/9' + '</td>'
                '<td style="padding:10px 8px">' + mom + '</td>'
                '</tr>'
            )
    if not sig_rows:
        sig_rows = '<tr><td colspan="7" style="padding:16px;text-align:center;color:#999">Nessun segnale oggi</td></tr>'

    tip = ('<p style="margin:12px 0 0;font-size:12px;color:#888">Entra domani al prezzo di apertura '
           'se il segnale ti convince.</p>' if n_segnali > 0 else "")

    return (
        '<!DOCTYPE html><html><head><meta charset="UTF-8"></head>'
        '<body style="font-family:sans-serif;background:#f5f5f5;margin:0;padding:0">'
        '<div style="max-width:680px;margin:20px auto;background:white;border-radius:12px;overflow:hidden">'
        '<div style="background:#0f0f0f;padding:22px 26px;color:white">'
        '<div style="font-size:11px;color:#666;margin-bottom:4px">FTSE MIB Strategy Lab</div>'
        '<div style="font-size:20px;font-weight:500">Riepilogo serale &mdash; ' + date_str + '</div>'
        '<div style="margin-top:10px;font-size:13px;color:#aaa">'
        'Posizioni: <strong style="color:white">' + str(n_open) + '</strong> &nbsp;'
        'P&L: <strong style="color:' + pnl_color + '">&euro;' + f'{total_pnl:+,.0f}' + '</strong> &nbsp;'
        'VIX: <strong style="color:' + vix_color + '">' + vix_str + '</strong> &nbsp;'
        'Segnali: <strong style="color:white">' + str(n_segnali) + '</strong>'
        '</div></div>'
        + alert_section +
        '<div style="padding:18px 24px">'
        '<div style="font-size:11px;text-transform:uppercase;color:#888;margin-bottom:12px">Portafoglio reale</div>'
        '<table style="width:100%;border-collapse:collapse;font-size:13px">'
        '<thead><tr style="background:#f8f8f6;border-bottom:2px solid #eee">'
        '<th style="padding:8px;text-align:left;color:#666">Ticker</th>'
        '<th style="padding:8px;text-align:left;color:#666">Ingresso</th>'
        '<th style="padding:8px;text-align:left;color:#666">Prezzo entr.</th>'
        '<th style="padding:8px;text-align:left;color:#666">Prezzo att.</th>'
        '<th style="padding:8px;text-align:left;color:#666">P&L</th>'
        '<th style="padding:8px;text-align:left;color:#666">Stop loss</th>'
        '<th style="padding:8px;text-align:left;color:#666">Giorni</th>'
        '</tr></thead><tbody>' + pos_rows + '</tbody></table></div>'
        '<div style="padding:18px 24px;border-top:1px solid #f0f0ee">'
        '<div style="font-size:11px;text-transform:uppercase;color:#888;margin-bottom:12px">Segnali per domani</div>'
        '<table style="width:100%;border-collapse:collapse;font-size:13px">'
        '<thead><tr style="background:#f8f8f6;border-bottom:2px solid #eee">'
        '<th style="padding:8px;text-align:left;color:#666">Tipo</th>'
        '<th style="padding:8px;text-align:left;color:#666">Ticker</th>'
        '<th style="padding:8px;text-align:left;color:#666">Settore</th>'
        '<th style="padding:8px;text-align:left;color:#666">Prezzo</th>'
        '<th style="padding:8px;text-align:left;color:#666">AI prob</th>'
        '<th style="padding:8px;text-align:left;color:#666">Score</th>'
        '<th style="padding:8px;text-align:left;color:#666">Mom 3M</th>'
        '</tr></thead><tbody>' + sig_rows + '</tbody></table>' + tip + '</div>'
        '<div style="padding:14px 24px;background:#f8f8f6;border-top:1px solid #eee">'
        '<p style="font-size:11px;color:#aaa;margin:0">Apri l\'app per registrare operazioni.</p>'
        '</div></div></body></html>'
    )


def send_daily_email(signals_df=None, open_positions=None, alerts=None, portfolio_summary=None):
    """Manda l'email serale completa."""
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    if EMAIL_PASSWORD == "METTI_QUI_LA_APP_PASSWORD":
        print("  [WARN] Email non configurata")
        return False

    alerts            = alerts or []
    portfolio_summary = portfolio_summary or {}
    date_str = datetime.now().strftime("%d/%m/%Y")

    vix = None
    if signals_df is not None and hasattr(signals_df, "empty") and not signals_df.empty:
        if "vix" in signals_df.columns:
            try:
                vix = float(signals_df["vix"].iloc[0])
            except Exception:
                pass

    n_alerts  = len(alerts)
    n_sig     = len(signals_df) if (signals_df is not None and hasattr(signals_df,"empty") and not signals_df.empty) else 0
    total_pnl = portfolio_summary.get("total_pnl_eur", 0)
    n_open    = portfolio_summary.get("n_open", 0)

    if n_alerts > 0:
        subject = f"[ALERT] FTSE MIB {date_str} - {n_alerts} azione richiesta"
    elif n_sig > 0:
        subject = f"FTSE MIB {date_str} - {n_sig} segnali | P&L euro{total_pnl:+,.0f}"
    else:
        subject = f"FTSE MIB {date_str} - {n_open} posizioni | P&L euro{total_pnl:+,.0f}"

    html = build_daily_html(signals_df, open_positions, alerts, portfolio_summary, date_str, vix)

    msg = MIMEMultipart("alternative")
    msg["From"]    = EMAIL_SENDER
    msg["To"]      = EMAIL_RECEIVER
    msg["Subject"] = subject
    msg.attach(MIMEText(html, "html"))

    try:
        srv = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        srv.ehlo()
        srv.starttls()
        srv.login(EMAIL_SENDER, EMAIL_PASSWORD)
        srv.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        srv.quit()
        print(f"  [OK] Email inviata: {subject}")
        return True
    except Exception as e:
        print(f"  [ERRORE] {e}")
        return False