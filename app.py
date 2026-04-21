"""
app.py — FTSE MIB Strategy Lab (versione semplificata)

4 sezioni:
  1. Oggi       — segnali di oggi, cosa fare
  2. Storico    — come sta andando la strategia
  3. Portafoglio — simulazione e performance
  4. Impostazioni — parametri e aggiornamento dati

Avvio: streamlit run app.py
"""

import os, sys, json, subprocess
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="FTSE MIB Strategy Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main-title {
    font-size: 28px; font-weight: 500; margin-bottom: 2px;
}
.subtitle {
    font-size: 14px; color: #888; margin-bottom: 28px;
}
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px; letter-spacing: .12em;
    text-transform: uppercase; color: #888; margin-bottom: 10px;
}
.signal-box {
    border: 0.5px solid #e0e0d8;
    border-radius: 14px; padding: 20px 24px;
    margin-bottom: 12px; background: #fff;
}
.signal-box.forte { border-left: 4px solid #1D9E75; }
.signal-box.ok    { border-left: 4px solid #378ADD; }
.signal-ticker  { font-size: 22px; font-weight: 500; }
.signal-sector  { font-size: 12px; color: #888; margin-bottom: 10px; }
.signal-prob    { font-size: 32px; font-weight: 500; color: #1D9E75; }
.signal-label   { font-size: 11px; color: #888; }
.tag { display:inline-block; padding:3px 10px; border-radius:20px;
       font-size:11px; font-weight:500; margin-right:6px; }
.tag-green  { background:#E1F5EE; color:#0F6E56; }
.tag-blue   { background:#E6F1FB; color:#185FA5; }
.tag-red    { background:#FCEBEB; color:#A32D2D; }
.tag-gray   { background:#f0f0ee; color:#555; }
.step-box {
    background: #f8f8f6; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 10px;
    border-left: 3px solid #e0e0d8;
}
.step-box.done  { border-left-color: #1D9E75; }
.step-box.ready { border-left-color: #378ADD; }
.step-box.wait  { border-left-color: #e0e0d8; opacity: .6; }
.log-area {
    background:#0f0f0f; color:#00ff88;
    font-family:'DM Mono',monospace; font-size:12px;
    padding:14px; border-radius:10px;
    max-height:280px; overflow-y:auto;
    white-space:pre-wrap;
}
.stButton > button {
    background: #0f0f0f; color: #fff; border: none;
    border-radius: 10px; font-family: 'DM Mono', monospace;
    font-size: 12px; padding: 10px 20px; transition: opacity .2s;
    width: 100%;
}
.stButton > button:hover { opacity: .8; }
.stButton > button[kind="secondary"] {
    background: transparent; color: #333;
    border: 0.5px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
DATA = ROOT / "data" / "processed"
DATA.mkdir(parents=True, exist_ok=True)

def load(name):
    p = DATA / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{'+' if v >= 0 else ''}{v:.1f}%"

def run(script, args=""):
    """Esegue uno script e ritorna (output, return_code)."""
    path = SRC / script
    cmd  = f'"{sys.executable}" -u "{path}" {args}'
    lines = []
    try:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", cwd=str(ROOT)
        )
        for line in proc.stdout:
            lines.append(line.rstrip())
        proc.wait()
        return "\n".join(lines), proc.returncode
    except Exception as e:
        return str(e), -1

def system_ready():
    """Controlla se il sistema è pronto per generare segnali."""
    model  = (ROOT / "models" / "trade_scorer_v2.joblib").exists()
    data   = (DATA / "target_stop_trailing_vix_regime_summary.csv").exists()
    return model and data

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGAZIONE — 4 tab in alto, niente sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">FTSE MIB Strategy Lab</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">Ultimo aggiornamento: {datetime.now().strftime("%d/%m/%Y %H:%M")}</div>', unsafe_allow_html=True)

tab_oggi, tab_storico, tab_portfolio, tab_impostazioni = st.tabs([
    "📡  Oggi",
    "📋  Storico",
    "💼  Portafoglio",
    "⚙️  Impostazioni",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — OGGI
# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — OGGI
# ═════════════════════════════════════════════════════════════════════════════
with tab_oggi:
    ready = system_ready()
    if not ready:
        st.warning("Il sistema non è ancora configurato. Vai su **Impostazioni** e clicca **Configura sistema**.")
    else:
        col_btn, col_info = st.columns([2, 5])
        with col_btn:
            run_screener = st.button("Aggiorna segnali ora")
        with col_info:
            screen_df = load("screener_latest.csv")
            if not screen_df.empty:
                data_run = screen_df["date"].iloc[0] if "date" in screen_df.columns else "—"
                vix_val  = screen_df["vix"].iloc[0]  if "vix"  in screen_df.columns else None
                st.markdown(f"**Ultimo aggiornamento:** {data_run}")
                if vix_val:
                    vix_color = "tag-green" if vix_val < 16 else "tag-blue" if vix_val < 20 else "tag-red"
                    st.markdown(f'<span class="tag {vix_color}">VIX {vix_val:.1f}</span>', unsafe_allow_html=True)
            else:
                st.markdown("Nessun dato — clicca **Aggiorna segnali ora**")

        if run_screener:
            with st.spinner("Analisi in corso..."):
                out, rc = run("daily_screener.py")
                st.code(out[-2000:] if len(out) > 2000 else out, language=None)
            if rc == 0:
                st.success("Aggiornamento completato")
                st.rerun()
            else:
                st.error("Errore — vedi log sopra")

        screen_df = load("screener_latest.csv")
        if screen_df.empty:
            st.info("Nessun segnale disponibile. Clicca **Aggiorna segnali ora**.")
        else:
            vix_val = float(screen_df["vix"].iloc[0]) if "vix" in screen_df.columns and pd.notna(screen_df["vix"].iloc[0]) else None
            if vix_val and vix_val >= 20:
                st.error(f"VIX = {vix_val:.1f} — volatilita alta. Nessun ingresso consigliato oggi.")
            else:
                n_forte = len(screen_df[screen_df["signal"] == "FORTE"]) if "signal" in screen_df.columns else 0
                st.markdown(f'<div class="section-label">{len(screen_df)} segnali trovati — {n_forte} forti</div>', unsafe_allow_html=True)

                for _, row in screen_df.iterrows():
                    is_forte = row.get("signal") == "FORTE"
                    cls      = "forte" if is_forte else "ok"
                    prob     = float(row.get("ai_prob", 0)) * 100
                    score    = int(row.get("entry_score", 0))
                    rsi      = row.get("rsi")
                    mom3m    = row.get("momentum_3m")
                    dist50   = row.get("dist_ma50_pct")
                    earn_warn= row.get("earnings_warning", None)
                    label_tag = '<span class="tag tag-green">FORTE</span>' if is_forte else '<span class="tag tag-blue">OK</span>'

                    tags_html = f'<span class="tag tag-gray">Score {score}/9</span>'
                    if rsi and not (isinstance(rsi, float) and np.isnan(rsi)):
                        tags_html += f'<span class="tag tag-gray">RSI {float(rsi):.0f}</span>'
                    if mom3m and not (isinstance(mom3m, float) and np.isnan(mom3m)):
                        tags_html += f'<span class="tag tag-gray">Mom 3M {fmt_pct(float(mom3m))}</span>'
                    if dist50 and not (isinstance(dist50, float) and np.isnan(dist50)):
                        tags_html += f'<span class="tag tag-gray">Dist MA50 {fmt_pct(float(dist50))}</span>'
                    if earn_warn and str(earn_warn) not in ("nan", "None", ""):
                        tags_html += f'<span class="tag tag-red">Earnings: {earn_warn}</span>'

                    ticker_str = str(row["ticker"])
                    sector_str = str(row.get("sector", "")).capitalize()
                    close_str  = f"{float(row.get('close', 0)):.3f}"

                    st.markdown(f"""
                    <div class="signal-box {cls}">
                      <div style="display:flex;justify-content:space-between;align-items:flex-start">
                        <div>
                          {label_tag}
                          <div class="signal-ticker" style="margin-top:6px">{ticker_str}</div>
                          <div class="signal-sector">{sector_str} &nbsp;&middot;&nbsp; Prezzo &euro;{close_str}</div>
                          <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px">{tags_html}</div>
                        </div>
                        <div style="text-align:right;min-width:80px">
                          <div class="signal-prob">{prob:.0f}%</div>
                          <div class="signal-label">probabilit&agrave; AI</div>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<div class="section-label">Cosa fare con questi segnali</div>', unsafe_allow_html=True)
            st.markdown("""
**FORTE (prob >= 70%)** — tutti i criteri in accordo. Considera l'ingresso.
Stop loss: **-4%** · Trailing stop: si attiva dopo **+12%** · Esci se VIX > 25.

**OK (prob 60-70%)** — segnale valido ma con meno convinzione. Attendi conferma o riduci la size.

**Nessun segnale** — il mercato non offre condizioni favorevoli. Stai fermo.
            """)

        # Prossimi earnings
        try:
            db_path = ROOT / "trading.db"
            if db_path.exists():
                import sqlite3
                conn    = sqlite3.connect(db_path)
                today   = datetime.now().strftime("%Y-%m-%d")
                end_e   = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
                earn_df = pd.read_sql(
                    "SELECT ticker, report_date FROM earnings_calendar WHERE report_date BETWEEN ? AND ? ORDER BY report_date",
                    conn, params=(today, end_e)
                )
                conn.close()
                if not earn_df.empty:
                    st.markdown("---")
                    st.markdown('<div class="section-label">Earnings nei prossimi 14 giorni</div>', unsafe_allow_html=True)
                    for _, r in earn_df.iterrows():
                        days = (datetime.strptime(r["report_date"], "%Y-%m-%d") - datetime.now()).days
                        tag_cls = "tag-red" if days <= 3 else "tag-gray"
                        st.markdown(f'<span class="tag {tag_cls}">{r["ticker"]} — {r["report_date"]} (tra {days}gg)</span>', unsafe_allow_html=True)
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — STORICO
# ═════════════════════════════════════════════════════════════════════════════
with tab_storico:
    col_upd, _ = st.columns([2, 6])
    if col_upd.button("Aggiorna risultati segnali passati"):
        with st.spinner("Aggiornamento in corso..."):
            out, rc = run("signal_history.py", "--update")
        if rc == 0:
            st.success("Risultati aggiornati")
        else:
            st.error(out[-500:])

    # Carica storico
    try:
        db_path = ROOT / "trading.db"
        if db_path.exists():
            import sqlite3
            conn    = sqlite3.connect(db_path)
            hist_df = pd.read_sql("SELECT * FROM signals ORDER BY signal_date DESC", conn)
            conn.close()
        else:
            hist_df = load("signal_history.csv")
    except Exception:
        hist_df = load("signal_history.csv")

    if hist_df.empty:
        st.info("Nessuno storico disponibile. I segnali vengono salvati automaticamente ogni volta che aggiorni la sezione **Oggi**.")
    else:
        closed = hist_df[hist_df["status"].isin(["win","loss"])] if "status" in hist_df.columns else pd.DataFrame()
        open_s = hist_df[hist_df["status"] == "open"]            if "status" in hist_df.columns else pd.DataFrame()
        wins   = hist_df[hist_df["status"] == "win"]             if "status" in hist_df.columns else pd.DataFrame()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Segnali generati", len(hist_df))
        c2.metric("Ancora aperti",    len(open_s))
        c3.metric("Chiusi",           len(closed))
        if not closed.empty:
            wr  = len(wins) / len(closed) * 100
            avg = closed["net_return_pct"].mean() if "net_return_pct" in closed.columns else 0
            c4.metric("Win rate reale",        f"{wr:.1f}%")
            c5.metric("Rend. medio per trade", fmt_pct(avg))

        st.markdown("---")

        if not closed.empty and "net_return_pct" in closed.columns and "signal_date" in closed.columns:
            closed_sorted = closed.sort_values("signal_date").copy()
            closed_sorted["cum_return"] = closed_sorted["net_return_pct"].cumsum()

            col_a, col_b = st.columns([3, 2])
            with col_a:
                st.markdown('<div class="section-label">Rendimento cumulativo segnali chiusi</div>', unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_scatter(
                    x=pd.to_datetime(closed_sorted["signal_date"]),
                    y=closed_sorted["cum_return"],
                    mode="lines+markers",
                    line=dict(color="#1D9E75", width=2),
                    marker=dict(color=["#1D9E75" if v >= 0 else "#D85A30" for v in closed_sorted["net_return_pct"]], size=7),
                    fill="tozeroy", fillcolor="rgba(29,158,117,0.07)",
                )
                fig.add_hline(y=0, line_dash="dot", line_color="#ccc")
                fig.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white",
                                  margin=dict(l=0,r=0,t=0,b=0), showlegend=False,
                                  yaxis=dict(ticksuffix="%", gridcolor="#f0f0ee"),
                                  font_family="DM Sans")
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                st.markdown('<div class="section-label">Win rate per ticker</div>', unsafe_allow_html=True)
                by_tk = closed.groupby("ticker").agg(n=("status","count"), wins=("status", lambda x: (x=="win").sum()))
                by_tk["wr"] = by_tk["wins"] / by_tk["n"] * 100
                by_tk = by_tk[by_tk["n"] >= 2].sort_values("wr", ascending=True)
                if not by_tk.empty:
                    fig2 = go.Figure(go.Bar(
                        x=by_tk["wr"], y=by_tk.index, orientation="h",
                        marker_color=["#1D9E75" if v >= 55 else "#D85A30" for v in by_tk["wr"]],
                        text=[f"{v:.0f}% ({n})" for v, n in zip(by_tk["wr"], by_tk["n"])],
                        textposition="outside",
                    ))
                    fig2.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white",
                                       margin=dict(l=0,r=60,t=0,b=0), showlegend=False,
                                       xaxis=dict(ticksuffix="%", range=[0,115]),
                                       font_family="DM Sans")
                    st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-label">Tutti i segnali</div>', unsafe_allow_html=True)
        stato = st.multiselect("Filtra per stato", ["open","win","loss","expired"],
                               default=["open","win","loss","expired"])
        display_df = hist_df[hist_df["status"].isin(stato)] if stato and "status" in hist_df.columns else hist_df

        def color_status(v):
            m = {"win":"background-color:#E1F5EE","loss":"background-color:#FCEBEB",
                 "open":"background-color:#E6F1FB","expired":"background-color:#f8f8f6"}
            return m.get(v,"")

        cols_show = ["signal_date","ticker","signal_type","entry_score","ai_prob","status","net_return_pct","exit_reason","holding_days"]
        cols_ok   = [c for c in cols_show if c in display_df.columns]
        fmt_h     = {}
        if "ai_prob"        in cols_ok: fmt_h["ai_prob"]        = "{:.0%}"
        if "net_return_pct" in cols_ok: fmt_h["net_return_pct"] = "{:+.2f}%"

        st.dataframe(
            display_df[cols_ok].style.format(fmt_h, na_rep="—")
                .map(color_status, subset=["status"] if "status" in cols_ok else []),
            use_container_width=True, height=380
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — PORTAFOGLIO
# ═════════════════════════════════════════════════════════════════════════════
with tab_portfolio:

    # Sotto-tab: Reale | Simulazione
    sub_reale, sub_sim = st.tabs(["💰 Portafoglio reale", "🔬 Simulazione storica"])

    # ── SUB-TAB: PORTAFOGLIO REALE ──
    with sub_reale:
        try:
            from real_portfolio import RealPortfolio
            rp = RealPortfolio()
            _RP_OK = True
        except Exception as e:
            st.error(f"Errore caricamento portafoglio: {e}")
            _RP_OK = False

        if _RP_OK:
            # Pulsante aggiorna
            col_upd, col_info = st.columns([2, 6])
            if col_upd.button("Aggiorna prezzi e alert"):
                with st.spinner("Aggiornamento posizioni..."):
                    alerts = rp.update_all()
                if alerts:
                    for a in alerts:
                        st.warning(f"{a['ticker']}: {a['alert']}")
                else:
                    st.success("Tutte le posizioni nella norma")

            summary = rp.get_summary()
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Posizioni aperte",   summary.get("n_open",0))
            c2.metric("Capitale investito", f"€{summary.get('invested_eur',0):,.0f}")
            c3.metric("P&L aperto",         f"€{summary.get('open_pnl_eur',0):+,.2f}")
            c4.metric("P&L totale",         f"€{summary.get('total_pnl_eur',0):+,.2f}")

            st.markdown("---")

            # Posizioni aperte
            open_pos = rp.get_open_positions()
            st.markdown('<div class="section-label">Posizioni aperte</div>', unsafe_allow_html=True)

            if open_pos.empty:
                st.info("Nessuna posizione aperta. Usa il form sotto per registrare un ingresso.")
            else:
                for _, row in open_pos.iterrows():
                    pnl_pct   = float(row.get("current_pnl_pct", 0) or 0)
                    pnl_eur   = float(row.get("current_pnl_eur", 0) or 0)
                    pnl_color = "#1D9E75" if pnl_pct >= 0 else "#D85A30"
                    alert_val = row.get("alert")
                    has_alert = alert_val and str(alert_val) not in ("nan","None","")

                    with st.container():
                        ca, cb, cc, cd, ce = st.columns([2,2,2,2,2])
                        ca.markdown(f"**{row['ticker']}**<br><small>{row.get('entry_date','')}</small>", unsafe_allow_html=True)
                        cb.metric("Prezzo ingresso", f"€{float(row.get('entry_price',0)):.3f}")
                        cc.metric("Prezzo attuale",  f"€{float(row.get('current_price',0) or 0):.3f}")
                        cd.metric("P&L", f"€{pnl_eur:+.2f}", delta=f"{pnl_pct:+.1f}%")
                        ce.metric("Stop loss", f"€{float(row.get('stop_loss_price',0)):.3f}",
                                  delta="trailing" if row.get("trailing_active") else None)

                        if has_alert:
                            st.warning(f"**{row['ticker']}** — {alert_val}")

                        # Pulsante chiudi posizione
                        with st.expander(f"Chiudi posizione {row['ticker']}"):
                            col_xp, col_xd, col_xr, col_xb = st.columns([2,2,2,1])
                            xprice = col_xp.number_input("Prezzo uscita", value=float(row.get("current_price",0) or 0),
                                                          step=0.01, key=f"xp_{row['id']}")
                            xdate  = col_xd.date_input("Data uscita", value=datetime.now().date(),
                                                        key=f"xd_{row['id']}")
                            xreason= col_xr.selectbox("Motivo", ["manuale","stop_loss","trailing_stop","vix_exit","target"],
                                                       key=f"xr_{row['id']}")
                            col_xb.write("")
                            col_xb.write("")
                            if col_xb.button("Chiudi", key=f"xbtn_{row['id']}"):
                                result = rp.close_position(row["ticker"], xprice,
                                                           xdate.strftime("%Y-%m-%d"), xreason,
                                                           position_id=int(row["id"]))
                                st.success(f"Posizione chiusa: P&L €{result.get('net_pnl_eur',0):+.2f}")
                                st.rerun()

            st.markdown("---")

            # Form nuovo ingresso
            st.markdown('<div class="section-label">Registra nuovo ingresso</div>', unsafe_allow_html=True)

            # Suggerisce i ticker dai segnali recenti
            screen_df = load("screener_latest.csv")
            suggested = screen_df["ticker"].tolist() if not screen_df.empty else []

            with st.form("form_open_position"):
                col1, col2, col3 = st.columns(3)
                ticker_input = col1.selectbox("Ticker",
                    options=suggested + ["Altro..."],
                    index=0 if suggested else 0
                )
                if ticker_input == "Altro...":
                    ticker_input = col1.text_input("Inserisci ticker (es. ENI.MI)")

                entry_price = col2.number_input("Prezzo ingresso €", min_value=0.01, step=0.01)
                size_eur    = col3.number_input("Importo investito €", min_value=100, value=3000, step=100)

                col4, col5, col6 = st.columns(3)
                entry_date  = col4.date_input("Data ingresso", value=datetime.now().date())
                custom_sl   = col5.number_input("Stop loss % personalizzato (0=default)",
                                                  min_value=0.0, max_value=20.0, value=0.0, step=0.5)
                notes       = col6.text_input("Note (opzionale)")

                # Recupera ai_prob dal segnale se disponibile
                ai_prob = None
                if not screen_df.empty and ticker_input in screen_df["ticker"].values:
                    ai_prob = float(screen_df[screen_df["ticker"]==ticker_input]["ai_prob"].iloc[0])

                submitted = st.form_submit_button("Registra ingresso")
                if submitted and ticker_input and entry_price > 0:
                    sl_pct = custom_sl / 100 if custom_sl > 0 else None
                    pos_id = rp.open_position(
                        ticker      = ticker_input,
                        entry_price = entry_price,
                        size_eur    = size_eur,
                        entry_date  = entry_date.strftime("%Y-%m-%d"),
                        notes       = notes,
                        ai_prob     = ai_prob,
                        stop_loss_pct = sl_pct,
                    )
                    sl_price = entry_price * (1 - (sl_pct or 0.04))
                    tp_price = entry_price * 1.12
                    st.success(f"Posizione aperta! ID={pos_id} | Stop: €{sl_price:.3f} | Target: €{tp_price:.3f}")
                    st.rerun()

            # Storico chiuse
            closed_pos = rp.get_closed_positions()
            if not closed_pos.empty:
                st.markdown("---")
                st.markdown('<div class="section-label">Posizioni chiuse</div>', unsafe_allow_html=True)
                cols_c = ["ticker","entry_date","exit_date","entry_price","exit_price",
                          "net_pnl_pct","net_pnl_eur","exit_reason","days_open"]
                cols_ok = [c for c in cols_c if c in closed_pos.columns]
                fmt_c = {}
                if "entry_price" in cols_ok:  fmt_c["entry_price"]  = "€{:.3f}"
                if "exit_price" in cols_ok:   fmt_c["exit_price"]   = "€{:.3f}"
                if "net_pnl_pct" in cols_ok:  fmt_c["net_pnl_pct"]  = "{:+.2f}%"
                if "net_pnl_eur" in cols_ok:  fmt_c["net_pnl_eur"]  = "€{:+.0f}"
                st.dataframe(
                    closed_pos[cols_ok].style.format(fmt_c, na_rep="—")
                        .background_gradient(subset=["net_pnl_pct"] if "net_pnl_pct" in cols_ok else [], cmap="RdYlGn"),
                    use_container_width=True, height=280
                )

    # ── SUB-TAB: SIMULAZIONE ──
    with sub_sim:
        st.markdown('<div class="section-label">Simulazione portafoglio — ultimi 365 giorni</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        capital  = c1.number_input("Capitale (€)", 1000, 500000, 10000, step=1000)
        max_pos  = c2.number_input("Max posizioni", 1, 10, 3, key="sim_max_pos")
        sim_days = c3.number_input("Giorni", 90, 730, 365)
        c4.write("")
        c4.write("")
        run_sim = c4.button("Avvia simulazione")

        if run_sim:
            args = f"--capital {capital} --max-positions {max_pos} --days {sim_days}"
            with st.spinner(f"Simulazione {sim_days} giorni in corso..."):
                out, rc = run("portfolio_simulation.py", args)
                st.code(out[-3000:] if len(out) > 3000 else out, language=None)
            if rc == 0:
                st.success("Simulazione completata")
                st.rerun()
            else:
                st.error("Errore — vedi log sopra")

        trades_df = load("simulation_trades.csv")
        daily_df  = load("simulation_daily_values.csv")

        if trades_df.empty:
            st.info("Nessuna simulazione disponibile. Imposta i parametri e clicca **Avvia simulazione**.")
        else:
            wins   = trades_df[trades_df["net_return_pct"] > 0]
        losses = trades_df[trades_df["net_return_pct"] <= 0]
        pf     = wins["net_return_pct"].sum() / abs(losses["net_return_pct"].sum()) if not losses.empty and losses["net_return_pct"].sum() != 0 else 0
        pnl    = trades_df["pnl_eur"].sum() if "pnl_eur" in trades_df.columns else 0

        if not daily_df.empty:
            cap_init = daily_df["total_value"].iloc[0]
            cap_fin  = daily_df["total_value"].iloc[-1]
            tot_ret  = (cap_fin / cap_init - 1) * 100
            roll_max = daily_df["total_value"].cummax()
            max_dd   = ((daily_df["total_value"] - roll_max) / roll_max * 100).min()
            dr       = daily_df["total_value"].pct_change().fillna(0)
            sharpe   = (dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0
        else:
            tot_ret = max_dd = sharpe = 0

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Rendimento totale", fmt_pct(tot_ret))
        c2.metric("P&L",               f"€{pnl:+,.0f}")
        c3.metric("Max drawdown",      fmt_pct(max_dd))
        c4.metric("Sharpe ratio",      f"{sharpe:.2f}")
        c5.metric("Win rate",          f"{len(wins)/len(trades_df)*100:.1f}%" if len(trades_df) > 0 else "—")
        c6.metric("Profit factor",     f"{pf:.2f}")

        if not daily_df.empty:
            from plotly.subplots import make_subplots
            daily_df["date"] = pd.to_datetime(daily_df["date"])
            fig = make_subplots(rows=2, cols=1, row_heights=[0.68, 0.32],
                                shared_xaxes=True, vertical_spacing=0.04)
            fig.add_scatter(x=daily_df["date"], y=daily_df["total_value"],
                            mode="lines", line=dict(color="#1D9E75", width=2),
                            fill="tozeroy", fillcolor="rgba(29,158,117,0.06)",
                            row=1, col=1, name="Portafoglio")
            fig.add_hline(y=cap_init, line_dash="dot", line_color="#ccc", row=1, col=1)
            roll_max2 = daily_df["total_value"].cummax()
            drawdown  = (daily_df["total_value"] - roll_max2) / roll_max2 * 100
            fig.add_scatter(x=daily_df["date"], y=drawdown, fill="tozeroy",
                            fillcolor="rgba(216,90,48,0.2)", line=dict(color="#D85A30", width=1),
                            row=2, col=1, name="Drawdown")
            fig.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white",
                              margin=dict(l=0,r=0,t=0,b=0), showlegend=False, font_family="DM Sans")
            fig.update_yaxes(tickprefix="€", gridcolor="#f0f0ee", row=1, col=1)
            fig.update_yaxes(ticksuffix="%", gridcolor="#f0f0ee", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-label">Trade eseguiti nella simulazione</div>', unsafe_allow_html=True)
        cols_t  = ["ticker","entry_date","exit_date","holding_days","entry_price","exit_price","net_return_pct","pnl_eur","exit_reason","ai_prob"]
        cols_ok = [c for c in cols_t if c in trades_df.columns]
        fmt_t   = {}
        if "entry_price"    in cols_ok: fmt_t["entry_price"]    = "€{:.3f}"
        if "exit_price"     in cols_ok: fmt_t["exit_price"]     = "€{:.3f}"
        if "net_return_pct" in cols_ok: fmt_t["net_return_pct"] = "{:+.2f}%"
        if "pnl_eur"        in cols_ok: fmt_t["pnl_eur"]        = "€{:+.0f}"
        if "ai_prob"        in cols_ok: fmt_t["ai_prob"]        = "{:.0%}"
        st.dataframe(
            trades_df[cols_ok].style.format(fmt_t, na_rep="—")
                .background_gradient(subset=["net_return_pct"] if "net_return_pct" in cols_ok else [], cmap="RdYlGn"),
            use_container_width=True, height=350
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — IMPOSTAZIONI
# ═════════════════════════════════════════════════════════════════════════════
with tab_impostazioni:
    st.markdown('<div class="section-label">Setup e aggiornamento dati</div>', unsafe_allow_html=True)

    model_ok   = (ROOT / "models" / "trade_scorer_v2.joblib").exists()
    dataset_ok = (DATA / "dataset_extended.csv").exists()
    db_ok      = (ROOT / "trading.db").exists()

    steps = [
        ("Prezzi storici dal 2000",    dataset_ok or (ROOT/"data"/"raw"/"prices_extended.csv").exists(), "download_prices_extended.py", ""),
        ("Dati macro (VIX, SP500...)", (ROOT/"data"/"raw"/"macro_extended.csv").exists(),                "download_macro_extended.py",  ""),
        ("Dataset con indicatori",     dataset_ok,                                                        "build_dataset.py",            ""),
        ("Training modello AI",        model_ok,                                                          "retrain_model_extended.py",   ""),
        ("Database SQLite",            db_ok,                                                             "database.py",                 ""),
        ("Earnings calendar",          db_ok,                                                             "earnings_calendar.py",        ""),
        ("Fondamentali",               (DATA/"fundamentals.csv").exists(),                                "fundamentals.py",             "--quick"),
    ]

    all_done = all(s[1] for s in steps)
    if not all_done:
        st.info("Alcuni componenti non sono ancora configurati.")
        if st.button("Configura tutto (prima installazione — ~20 min)"):
            for name, done, script, args in steps:
                if not done:
                    st.write(f"Esecuzione: {name}...")
                    out, rc = run(script, args)
                    st.code(out[-1000:], language=None)
                    if rc != 0:
                        st.error(f"Errore in: {name}")
                        break
            st.success("Setup completato")
            st.rerun()
        st.markdown("---")

    for name, done, script, args in steps:
        icon  = "✓" if done else "○"
        color = "#1D9E75" if done else "#378ADD"
        stato_label = "completato" if done else script
        st.markdown(f"""
        <div class="step-box {'done' if done else 'ready'}">
          <span style="color:{color};font-weight:500;margin-right:8px">{icon}</span>
          <strong>{name}</strong>
          &nbsp;<span style="color:{'#1D9E75' if done else '#888'};font-size:12px">{stato_label}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Aggiornamento settimanale (ogni lunedi)</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    if col1.button("Aggiorna earnings"):
        with st.spinner("Download earnings..."):
            out, rc = run("earnings_calendar.py", "")
        st.success("[OK]") if rc == 0 else st.error(out[-500:])
    if col2.button("Aggiorna fondamentali"):
        with st.spinner("Download fondamentali FTSE MIB..."):
            out, rc = run("fundamentals.py", "--quick")
        st.success("[OK]") if rc == 0 else st.error(out[-500:])

    st.markdown("---")
    st.markdown('<div class="section-label">Aggiornamento mensile (primo del mese)</div>', unsafe_allow_html=True)
    if st.button("Re-addestra modello AI con nuovi dati"):
        with st.spinner("Training in corso (~3 min)..."):
            out, rc = run("retrain_model_extended.py", "")
            st.code(out[-2000:], language=None)
        st.success("[OK] Modello aggiornato") if rc == 0 else st.error("Errore")

    st.markdown("---")
    st.markdown('<div class="section-label">Parametri strategia</div>', unsafe_allow_html=True)
    cfg_path = ROOT / "config.json"
    defaults = {"stop_loss_pct":0.04,"take_profit_pct":0.12,"trailing_stop_pct":0.06,
                "vix_entry":20,"vix_exit":25,"entry_score_threshold":6,
                "ai_prob_threshold":0.60,"max_positions":3}
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else defaults

    col1, col2, col3, col4 = st.columns(4)
    cfg["stop_loss_pct"]         = col1.slider("Stop loss %",      1,  10, int(cfg["stop_loss_pct"]*100))      / 100
    cfg["take_profit_pct"]       = col2.slider("Take profit %",    5,  30, int(cfg["take_profit_pct"]*100))    / 100
    cfg["trailing_stop_pct"]     = col3.slider("Trailing stop %",  2,  15, int(cfg["trailing_stop_pct"]*100))  / 100
    cfg["vix_entry"]             = col4.slider("VIX max ingresso", 14, 28, cfg["vix_entry"])
    col5, col6, col7, col8 = st.columns(4)
    cfg["vix_exit"]              = col5.slider("VIX uscita",       18, 40, cfg["vix_exit"])
    cfg["entry_score_threshold"] = col6.slider("Score minimo",      4,  9, cfg["entry_score_threshold"])
    cfg["ai_prob_threshold"]     = col7.slider("AI prob min %",    50, 80, int(cfg["ai_prob_threshold"]*100))  / 100
    cfg["max_positions"]         = col8.number_input("Max posizioni", 1, 10, cfg["max_positions"], key="cfg_max_pos")

    rr = cfg["take_profit_pct"] / cfg["stop_loss_pct"]
    st.info(f"Risk/Reward: **{rr:.1f}x** — per ogni euro rischiato, ne guadagni in media {rr:.1f} se il trade va bene.")
    if st.button("Salva parametri"):
        cfg_path.write_text(json.dumps(cfg, indent=2))
        st.success("[OK] Parametri salvati")

    st.markdown("---")
    st.markdown('<div class="section-label">Notifiche email</div>', unsafe_allow_html=True)
    email_ok = bool(os.environ.get("GMAIL_APP_PASSWORD",""))
    if email_ok:
        st.success("Email configurata — riceverai i segnali ogni sera")
    else:
        st.warning('Email non configurata. Su Streamlit Cloud: Settings → Secrets → aggiungi GMAIL_APP_PASSWORD = "xxxx-xxxx-xxxx-xxxx"')
    if st.button("Invia email di test"):
        out, rc = run("notify_email.py", "--test")
        st.success("[OK] Email inviata") if rc == 0 else st.error(out[-300:])