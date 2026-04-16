"""
app.py — FTSE MIB Strategy Lab
Dashboard Streamlit completa per gestire l'intera pipeline di trading.

Installazione:
    pip install streamlit plotly pandas yfinance joblib xgboost scikit-learn ta

Avvio locale:
    streamlit run app.py

Deploy su Streamlit Cloud:
    1. Crea repo GitHub con tutti i file del progetto
    2. Vai su share.streamlit.io → New app → punta al repo
    3. File principale: app.py
"""

import os, sys, json, subprocess, threading, time
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FTSE MIB Strategy Lab",
    page_icon="[CHART]",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f0f !important;
    border-right: 1px solid #1e1e1e;
}
[data-testid="stSidebar"] * { color: #aaa !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] .sidebar-title { color: #fff !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #f8f8f6;
    border: 0.5px solid #e8e6e0;
    border-radius: 10px;
    padding: 16px !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Buttons */
.stButton > button {
    background: #0f0f0f;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    padding: 8px 18px;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: 0.8; }

/* Section headers */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 12px;
}

/* Signal cards */
.signal-card {
    border: 0.5px solid #e0e0d8;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
    background: #fff;
}
.signal-card.forte { border-left: 3px solid #1D9E75; }
.signal-card.ok    { border-left: 3px solid #378ADD; }
.signal-ticker { font-size: 18px; font-weight: 500; margin-bottom: 2px; }
.signal-meta   { font-size: 12px; color: #888; }
.signal-prob   { font-family: 'DM Mono', monospace; font-size: 24px; font-weight: 500; color: #1D9E75; }

/* Log output */
.log-box {
    background: #0f0f0f;
    color: #00ff88;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    padding: 16px;
    border-radius: 10px;
    height: 320px;
    overflow-y: auto;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# ── helpers ───────────────────────────────────────────────────────────────────
DATA = ROOT / "data" / "processed"
RAW  = ROOT / "data" / "raw"

@st.cache_resource
def get_db():
    try:
        sys.path.insert(0, str(SRC))
        from database import DB
        return DB()
    except Exception:
        return None

def load_csv(name, folder=DATA):
    p = folder / name
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()

def color_val(v, good="pos"):
    if good == "pos":
        return "#1D9E75" if v >= 0 else "#D85A30"
    return "#1D9E75" if v <= 0 else "#D85A30"

def fmt_pct(v):
    return f"{'+' if v>=0 else ''}{v:.1f}%"

def run_script(script_name, args="", placeholder=None):
    """Esegue uno script src/ — usa sys.executable per garantire stesso ambiente."""
    script_path = SRC / script_name
    # sys.executable = stesso python che sta girando Streamlit (con tutti i pacchetti)
    cmd = f'"{sys.executable}" -u "{script_path}" {args}'
    output_lines = []
    try:
        proc = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            cwd=str(ROOT)
        )
        for line in proc.stdout:
            clean = line.rstrip()
            output_lines.append(clean)
            if placeholder:
                display = "\n".join(output_lines[-80:])
                placeholder.code(display, language=None)
        proc.wait()
        rc = proc.returncode
    except Exception as e:
        output_lines.append(f"[ERRORE AVVIO] {e}")
        rc = -1
    return "\n".join(output_lines), rc

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### [CHART] Strategy Lab")
    st.markdown('<div class="section-label">FTSE MIB · AI Scoring</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigazione",
        ["🏠 Dashboard", "📡 Screener", "📋 Storico segnali",
         "📅 Earnings", "⚙️ Backtest", "💼 Portafoglio",
         "🔬 Ottimizzatore", "🔔 Alert", "🔧 Parametri", "🚀 Lancia script"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<div class="section-label">Stato sistema</div>', unsafe_allow_html=True)

    model_ok  = (ROOT / "models" / "trade_scorer_v2.joblib").exists()
    data_ok   = (DATA / "target_stop_trailing_vix_regime_summary.csv").exists()
    screen_ok = (DATA / "screener_latest.csv").exists()
    sim_ok    = (DATA / "simulation_trades.csv").exists()

    db_ok = (ROOT / "trading.db").exists()
    st.markdown(f"{'[OK]' if db_ok     else '[X]'} Database SQLite")
    st.markdown(f"{'[OK]' if model_ok  else '[X]'} Modello AI v2")
    st.markdown(f"{'[OK]' if data_ok   else '[X]'} Summary backtest")
    st.markdown(f"{'[OK]' if screen_ok else '[X]'} Screener (ultimo run)")
    st.markdown(f"{'[OK]' if sim_ok    else '[X]'} Simulazione portafoglio")

    st.markdown("---")
    st.markdown('<div class="section-label">Aggiornamento rapido</div>', unsafe_allow_html=True)
    if st.button("Lancia screener ora"):
        with st.spinner("Screener in corso..."):
            out, rc = run_script("daily_screener.py")
        if rc == 0:
            st.success("Screener completato!")
        else:
            st.error("Errore screener")
        st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# PAGINA: DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("## Dashboard")
    st.markdown('<div class="section-label">FTSE MIB Strategy Lab · panoramica</div>', unsafe_allow_html=True)

    # Metriche principali
    regime_sum  = load_csv("target_stop_trailing_vix_regime_summary.csv")
    sim_trades  = load_csv("simulation_trades.csv")
    screener_df = load_csv("screener_latest.csv")

    c1, c2, c3, c4, c5 = st.columns(5)
    if not regime_sum.empty:
        c1.metric("Ticker universe", f"{regime_sum['ticker'].nunique()}")
        c2.metric("Win rate medio", f"{regime_sum['win_rate_pct'].mean():.1f}%")
        c3.metric("Profit factor medio", f"{regime_sum['profit_factor'].mean():.2f}")
        avg_dd = regime_sum['max_drawdown_strategy_net_pct'].mean()
        c4.metric("Max DD medio", f"{avg_dd:.1f}%")
    if not sim_trades.empty:
        wins = (sim_trades["net_return_pct"] > 0).mean() * 100
        c5.metric("Win rate simulazione", f"{wins:.1f}%", delta="365 giorni")

    st.markdown("---")

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-label">Performance strategia per ticker (VIX Regime)</div>', unsafe_allow_html=True)
        if not regime_sum.empty:
            df_plot = regime_sum.sort_values("strategy_net_perf_pct", ascending=True).tail(15)
            fig = go.Figure()
            fig.add_bar(
                x=df_plot["strategy_net_perf_pct"],
                y=df_plot["ticker"],
                orientation="h",
                marker_color=["#1D9E75" if v >= 0 else "#D85A30" for v in df_plot["strategy_net_perf_pct"]],
                name="Strategia net"
            )
            fig.update_layout(height=380, margin=dict(l=0,r=0,t=0,b=0),
                              plot_bgcolor="white", paper_bgcolor="white",
                              xaxis=dict(ticksuffix="%", gridcolor="#f0f0ee"),
                              showlegend=False, font_family="DM Sans")
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-label">Segnali attivi (ultimo screener)</div>', unsafe_allow_html=True)
        if not screener_df.empty:
            for _, row in screener_df.iterrows():
                cls = "forte" if row.get("signal") == "FORTE" else "ok"
                prob = row.get("ai_prob", 0) * 100
                rsi = f"{row.get('rsi', 0):.0f}" if pd.notna(row.get("rsi")) else "—"
                st.markdown(f"""
                <div class="signal-card {cls}">
                  <div style="display:flex;justify-content:space-between;align-items:start">
                    <div>
                      <div class="signal-ticker">{row['ticker']}</div>
                      <div class="signal-meta">{row.get('sector','')} · Score {int(row.get('entry_score',0))}/9 · RSI {rsi}</div>
                      <div class="signal-meta" style="margin-top:4px">
                        Mom 1M: {row.get('momentum_1m',0):+.1f}% · Mom 3M: {row.get('momentum_3m',0):+.1f}%
                      </div>
                    </div>
                    <div style="text-align:right">
                      <div class="signal-prob">{prob:.0f}%</div>
                      <div class="signal-meta">AI prob</div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Nessun segnale. Lancia lo screener dalla sidebar.")

    # Simulazione recap
    if not sim_trades.empty:
        st.markdown("---")
        st.markdown('<div class="section-label">Simulazione portafoglio — ultimi 365 giorni</div>', unsafe_allow_html=True)

        daily = load_csv("simulation_daily_values.csv")
        col1, col2 = st.columns([3, 1])

        with col1:
            if not daily.empty:
                daily["date"] = pd.to_datetime(daily["date"])
                fig2 = go.Figure()
                fig2.add_scatter(x=daily["date"], y=daily["total_value"],
                                 mode="lines", name="Portafoglio",
                                 line=dict(color="#1D9E75", width=2))
                fig2.add_hline(y=daily["total_value"].iloc[0], line_dash="dot",
                               line_color="#ccc", annotation_text="Capitale iniziale")
                fig2.update_layout(height=260, margin=dict(l=0,r=0,t=0,b=0),
                                   plot_bgcolor="white", paper_bgcolor="white",
                                   yaxis=dict(tickprefix="€", gridcolor="#f0f0ee"),
                                   showlegend=False, font_family="DM Sans")
                st.plotly_chart(fig2, use_container_width=True)

        with col2:
            wins = sim_trades[sim_trades["net_return_pct"] > 0]
            losses = sim_trades[sim_trades["net_return_pct"] <= 0]
            total_pnl = sim_trades["pnl_eur"].sum() if "pnl_eur" in sim_trades.columns else 0
            st.metric("Trade totali", len(sim_trades))
            st.metric("Win rate", f"{len(wins)/len(sim_trades)*100:.1f}%")
            st.metric("P&L totale", f"€{total_pnl:+,.0f}",
                      delta=f"{total_pnl/10000*100:+.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# PAGINA: SCREENER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📡 Screener":
    st.markdown("## Screener segnali")
    st.markdown('<div class="section-label">Segnali di ingresso · modello AI v2</div>', unsafe_allow_html=True)

    col_run, col_date, _ = st.columns([2, 2, 4])
    with col_run:
        run_now = st.button("Esegui screener oggi")
    with col_date:
        replay_date = st.date_input("Oppure replay su data:", value=None)

    if run_now or replay_date:
        args = f"--date {replay_date}" if replay_date else ""
        log_ph = st.empty()
        with st.spinner("Screener in corso..."):
            out, rc = run_script("daily_screener.py", args, placeholder=log_ph)
        if rc == 0:
            st.success("[OK] Screener completato — aggiorna la pagina per vedere i segnali")
        else:
            st.error("[ERRORE] — vedi log sopra")

    screener_df = load_csv("screener_latest.csv")
    if screener_df.empty:
        st.info("Nessun segnale disponibile. Lancia lo screener.")
        st.stop()

    # Tabella segnali
    st.markdown("---")
    date_run = screener_df["date"].iloc[0] if "date" in screener_df.columns else "—"
    st.markdown(f'<div class="section-label">Segnali del {date_run} — {len(screener_df)} trovati</div>', unsafe_allow_html=True)

    col_f1, col_f2 = st.columns(2)
    min_prob = col_f1.slider("AI prob minima", 0.60, 0.90, 0.60, 0.01, format="%.2f")
    min_score = col_f2.slider("Score minimo", 4, 9, 6)

    filtered = screener_df[
        (screener_df["ai_prob"] >= min_prob) &
        (screener_df["entry_score"] >= min_score)
    ].sort_values("ai_prob", ascending=False)

    for _, row in filtered.iterrows():
        cls   = "forte" if row.get("signal") == "FORTE" else "ok"
        prob  = row.get("ai_prob", 0) * 100
        badge = "🟢 FORTE" if cls == "forte" else "🔵 OK"
        with st.expander(f"{badge}  **{row['ticker']}** · {prob:.1f}% · Score {int(row.get('entry_score',0))}/9 · {row.get('sector','')}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Prezzo", f"€{row.get('close', 0):.3f}")
            c2.metric("RSI", f"{row.get('rsi', 0):.1f}")
            c3.metric("ADX", f"{row.get('adx', 0):.1f}" if pd.notna(row.get('adx')) else "—")
            c4.metric("Vol ratio", f"{row.get('volume_ratio', 0):.2f}" if pd.notna(row.get('volume_ratio')) else "—")
            c1.metric("Mom 1M", fmt_pct(row.get("momentum_1m", 0)))
            c2.metric("Mom 3M", fmt_pct(row.get("momentum_3m", 0)))
            c3.metric("Dist MA50", fmt_pct(row.get("dist_ma50_pct", 0)))
            c4.metric("Dist MA200", fmt_pct(row.get("dist_ma200_pct", 0)))


# ═════════════════════════════════════════════════════════════════════════════
# PAGINA: BACKTEST
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Backtest":
    st.markdown("## Risultati backtest")

    tab1, tab2 = st.tabs(["VIX Regime", "VIX Score"])

    for tab, fname in [(tab1, "target_stop_trailing_vix_regime_summary.csv"),
                       (tab2, "target_stop_trailing_vix_score_summary.csv")]:
        with tab:
            df = load_csv(fname)
            if df.empty:
                st.info("Dati non disponibili.")
                continue

            # Metriche aggregate
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Win rate medio", f"{df['win_rate_pct'].mean():.1f}%")
            c2.metric("Profit factor medio", f"{df['profit_factor'].mean():.2f}")
            c3.metric("Alpha medio", fmt_pct(df['alpha_net_pct'].mean()))
            c4.metric("Trade totali", f"{df['closed_trades'].sum():,}")

            # Grafico alpha per ticker
            df_s = df.sort_values("alpha_net_pct")
            fig = go.Figure()
            fig.add_bar(
                x=df_s["ticker"], y=df_s["alpha_net_pct"],
                marker_color=["#1D9E75" if v >= 0 else "#D85A30" for v in df_s["alpha_net_pct"]],
            )
            fig.update_layout(height=300, margin=dict(l=0,r=0,t=24,b=0),
                              title="Alpha netto per ticker",
                              plot_bgcolor="white", paper_bgcolor="white",
                              yaxis=dict(ticksuffix="%", gridcolor="#f0f0ee"),
                              showlegend=False, font_family="DM Sans")
            st.plotly_chart(fig, use_container_width=True)

            # Tabella completa
            cols_show = ["ticker","strategy_net_perf_pct","market_perf_pct","alpha_net_pct",
                         "win_rate_pct","profit_factor","max_drawdown_strategy_net_pct",
                         "closed_trades","avg_holding_days"]
            cols_ok = [c for c in cols_show if c in df.columns]
            st.dataframe(
                df[cols_ok].sort_values("profit_factor", ascending=False)
                  .style.format({
                      "strategy_net_perf_pct": "{:+.1f}%",
                      "market_perf_pct": "{:.1f}%",
                      "alpha_net_pct": "{:+.1f}%",
                      "win_rate_pct": "{:.1f}%",
                      "profit_factor": "{:.2f}",
                      "max_drawdown_strategy_net_pct": "{:.1f}%",
                      "avg_holding_days": "{:.0f}",
                  })
                  .background_gradient(subset=["profit_factor"], cmap="Greens")
                  .background_gradient(subset=["alpha_net_pct"], cmap="RdYlGn"),
                use_container_width=True, height=400
            )


# ═════════════════════════════════════════════════════════════════════════════
# PAGINA: PORTAFOGLIO
# ═════════════════════════════════════════════════════════════════════════════
elif page == "💼 Portafoglio":
    st.markdown("## Simulazione portafoglio")

    # Parametri simulazione
    with st.expander("⚙️ Parametri simulazione", expanded=False):
        col1, col2, col3 = st.columns(3)
        cap  = col1.number_input("Capitale (€)", 1000, 500000, 10000, step=1000)
        mpos = col2.number_input("Max posizioni", 1, 10, 3)
        days = col3.number_input("Giorni", 90, 730, 365)

        if st.button("Lancia simulazione"):
            log_ph = st.empty()
            with st.spinner(f"Simulazione {days} giorni in corso..."):
                args = f"--capital {cap} --max-positions {mpos} --days {days}"
                out, rc = run_script("portfolio_simulation.py", args, placeholder=log_ph)
            if rc == 0:
                st.success("[OK] Simulazione completata — aggiorna la pagina per vedere i risultati")
            else:
                st.error("[ERRORE] — vedi log sopra")

    # Risultati
    trades_df = load_csv("simulation_trades.csv")
    daily_df  = load_csv("simulation_daily_values.csv")

    if trades_df.empty:
        st.info("Nessuna simulazione disponibile. Lancia la simulazione con i parametri sopra.")
        st.stop()

    # KPI
    wins   = trades_df[trades_df["net_return_pct"] > 0]
    losses = trades_df[trades_df["net_return_pct"] <= 0]
    pf     = wins["net_return_pct"].sum() / abs(losses["net_return_pct"].sum()) if not losses.empty else float("inf")
    total_pnl = trades_df["pnl_eur"].sum() if "pnl_eur" in trades_df.columns else 0

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Trade totali", len(trades_df))
    c2.metric("Win rate", f"{len(wins)/len(trades_df)*100:.1f}%")
    c3.metric("Profit factor", f"{pf:.2f}")
    c4.metric("P&L totale", f"€{total_pnl:+,.0f}")
    c5.metric("Rend. medio trade", fmt_pct(trades_df["net_return_pct"].mean()))
    c6.metric("Holding medio", f"{trades_df['holding_days'].mean():.0f} gg")

    # Equity curve
    if not daily_df.empty:
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        cap_init = daily_df["total_value"].iloc[0]

        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                            shared_xaxes=True, vertical_spacing=0.05)

        fig.add_scatter(x=daily_df["date"], y=daily_df["total_value"],
                        mode="lines", name="Valore portafoglio",
                        line=dict(color="#1D9E75", width=2), row=1, col=1)
        fig.add_hline(y=cap_init, line_dash="dot", line_color="#ccc", row=1, col=1)

        # Drawdown
        roll_max = daily_df["total_value"].cummax()
        dd = (daily_df["total_value"] - roll_max) / roll_max * 100
        fig.add_scatter(x=daily_df["date"], y=dd, fill="tozeroy",
                        name="Drawdown", line=dict(color="#D85A30", width=1),
                        fillcolor="rgba(216,90,48,0.15)", row=2, col=1)

        fig.update_layout(height=480, plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=0,r=0,t=0,b=0), showlegend=False,
                          font_family="DM Sans")
        fig.update_yaxes(tickprefix="€", gridcolor="#f0f0ee", row=1, col=1)
        fig.update_yaxes(ticksuffix="%", gridcolor="#f0f0ee", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    # Tabella trade
    st.markdown('<div class="section-label">Dettaglio trade</div>', unsafe_allow_html=True)
    cols_t = ["ticker","entry_date","exit_date","holding_days","entry_price",
              "exit_price","net_return_pct","pnl_eur","exit_reason","ai_prob"]
    cols_ok = [c for c in cols_t if c in trades_df.columns]
    st.dataframe(
        trades_df[cols_ok].style.format({
            "entry_price": "€{:.3f}", "exit_price": "€{:.3f}",
            "net_return_pct": "{:+.2f}%",
            "pnl_eur": "€{:+.0f}",
            "ai_prob": "{:.1%}",
        }).background_gradient(subset=["net_return_pct"], cmap="RdYlGn"),
        use_container_width=True
    )

    # Exit reasons chart
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-label">Exit reasons</div>', unsafe_allow_html=True)
        er = trades_df["exit_reason"].value_counts()
        fig_er = go.Figure(go.Pie(
            labels=er.index, values=er.values,
            marker_colors=["#378ADD","#1D9E75","#D85A30","#B4B2A9"],
            hole=0.55
        ))
        fig_er.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0),
                             showlegend=True, font_family="DM Sans",
                             paper_bgcolor="white")
        st.plotly_chart(fig_er, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-label">P&L per ticker</div>', unsafe_allow_html=True)
        if "pnl_eur" in trades_df.columns:
            by_tk = trades_df.groupby("ticker")["pnl_eur"].sum().sort_values()
            fig_tk = go.Figure(go.Bar(
                x=by_tk.values, y=by_tk.index, orientation="h",
                marker_color=["#1D9E75" if v >= 0 else "#D85A30" for v in by_tk.values]
            ))
            fig_tk.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0),
                                 plot_bgcolor="white", paper_bgcolor="white",
                                 xaxis=dict(tickprefix="€", gridcolor="#f0f0ee"),
                                 showlegend=False, font_family="DM Sans")
            st.plotly_chart(fig_tk, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGINA: PARAMETRI
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔧 Parametri":
    st.markdown("## Parametri strategia")
    st.markdown('<div class="section-label">Modifica i parametri e salva il file di configurazione</div>', unsafe_allow_html=True)

    config_path = ROOT / "config.json"
    defaults = {
        "stop_loss_pct": 0.04, "take_profit_pct": 0.12, "trailing_stop_pct": 0.06,
        "transaction_cost": 0.0015, "vix_entry": 20, "vix_exit": 25,
        "rsi_max": 62, "entry_score_threshold": 6, "ai_prob_threshold": 0.60,
        "max_positions": 3, "initial_capital": 10000,
        "momentum_3m_min": -0.05, "adx_min": 20,
    }
    cfg = json.loads(config_path.read_text()) if config_path.exists() else defaults

    st.markdown("#### Risk management")
    col1, col2, col3, col4 = st.columns(4)
    cfg["stop_loss_pct"]     = col1.slider("Stop loss %",     1, 10, int(cfg["stop_loss_pct"]*100),     1) / 100
    cfg["take_profit_pct"]   = col2.slider("Take profit %",   5, 30, int(cfg["take_profit_pct"]*100),   1) / 100
    cfg["trailing_stop_pct"] = col3.slider("Trailing stop %", 2, 15, int(cfg["trailing_stop_pct"]*100), 1) / 100
    cfg["transaction_cost"]  = col4.slider("Costo trans. bp", 5, 50, int(cfg["transaction_cost"]*10000),5) / 10000

    rr = cfg["take_profit_pct"] / cfg["stop_loss_pct"]
    st.info(f"**Risk/Reward ratio: {rr:.2f}x** — Stop loss: {cfg['stop_loss_pct']*100:.1f}% · Take profit: {cfg['take_profit_pct']*100:.1f}%")

    st.markdown("#### Filtri VIX ed entry score")
    col1, col2, col3, col4 = st.columns(4)
    cfg["vix_entry"]              = col1.slider("VIX entry max",   12, 30, cfg["vix_entry"])
    cfg["vix_exit"]               = col2.slider("VIX exit",        15, 40, cfg["vix_exit"])
    cfg["entry_score_threshold"]  = col3.slider("Score minimo",    4,  9,  cfg["entry_score_threshold"])
    cfg["ai_prob_threshold"]      = col4.slider("AI prob min",     0.50, 0.80, cfg["ai_prob_threshold"], 0.01)

    st.markdown("#### Portafoglio")
    col1, col2, col3 = st.columns(3)
    cfg["initial_capital"] = col1.number_input("Capitale (€)", 1000, 500000, int(cfg["initial_capital"]), 1000)
    cfg["max_positions"]   = col2.number_input("Max posizioni", 1, 10, cfg["max_positions"])
    cfg["rsi_max"]         = col3.slider("RSI entry max", 40, 80, cfg["rsi_max"])

    if st.button("Salva configurazione"):
        config_path.write_text(json.dumps(cfg, indent=2))
        st.success(f"Configurazione salvata in `config.json`")
        st.json(cfg)


# ═════════════════════════════════════════════════════════════════════════════
# PAGINA: LANCIA SCRIPT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🚀 Lancia script":
    st.markdown("## Lancia script")
    st.markdown('<div class="section-label">Esegui i componenti della pipeline direttamente dall\'app</div>', unsafe_allow_html=True)

    scripts = {
        "Download prezzi estesi":   ("download_prices_extended.py", ""),
        "Download macro estesi":    ("download_macro_extended.py",  ""),
        "Build dataset":            ("build_dataset.py",            ""),
        "Daily screener":           ("daily_screener.py",           ""),
        "Training modello AI":      ("train_model.py",              ""),
        "Re-training v2":           ("retrain_model_extended.py",   ""),
        "Simulazione portafoglio":  ("portfolio_simulation.py",     "--days 365"),
        "Setup database SQLite":    ("database.py",                 ""),
        "Aggiorna earnings":        ("earnings_calendar.py",        ""),
        "Aggiorna risultati segnali":("signal_history.py",          "--update"),
    }

    selected = st.selectbox("Scegli script da eseguire", list(scripts.keys()))
    script_file, default_args = scripts[selected]

    extra_args = st.text_input("Argomenti aggiuntivi (opzionale)", value=default_args,
                               placeholder="es. --days 180 --capital 20000")

    col1, col2 = st.columns([2, 6])
    run_btn = col1.button("Esegui")

    if "last_log"    not in st.session_state: st.session_state.last_log    = ""
    if "last_rc"     not in st.session_state: st.session_state.last_rc     = None
    if "last_script" not in st.session_state: st.session_state.last_script = ""

    if run_btn:
        st.session_state.last_script = script_file
        log_ph = st.empty()
        st.markdown(f"**Output: {script_file}**")
        with st.spinner(f"Eseguendo {script_file}..."):
            out, rc = run_script(script_file, extra_args, placeholder=log_ph)
        st.session_state.last_log = out
        st.session_state.last_rc  = rc
        if rc == 0:
            st.success("[OK] Completato con successo")
        else:
            st.error(f"[ERRORE] Codice uscita: {rc} — leggi il log sopra")

    elif st.session_state.last_log:
        st.markdown(f"**Ultimo output: {st.session_state.last_script}**")
        if st.session_state.last_rc == 0:
            st.success("[OK] Completato")
        elif st.session_state.last_rc is not None:
            st.error(f"[ERRORE] Codice: {st.session_state.last_rc}")
        st.code(st.session_state.last_log, language=None)

    # Status file di output
    st.markdown("---")
    st.markdown('<div class="section-label">File di output disponibili</div>', unsafe_allow_html=True)
    output_files = {
        "data/raw/prices_extended.csv":        "Prezzi storici estesi (dal 2000)",
        "data/raw/macro_extended.csv":         "Dati macro (VIX, SP500, BTP spread...)",
        "data/processed/dataset_extended.csv": "Dataset completo con feature",
        "models/trade_scorer_v2.joblib":       "Modello AI v2",
        "data/processed/screener_latest.csv":  "Ultimo screener",
        "data/processed/simulation_trades.csv":"Trade simulazione portafoglio",
    }
    for fpath, desc in output_files.items():
        p = ROOT / fpath
        if p.exists():
            size = p.stat().st_size / 1024
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%d/%m/%Y %H:%M")
            st.markdown(f"[OK] `{fpath}` — {desc} ·  {size:.0f} KB · {mtime}")
        else:
            st.markdown(f"[ ] `{fpath}` — {desc}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGINA: STORICO SEGNALI
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📋 Storico segnali":
    st.markdown("## Storico segnali")
    st.markdown('<div class="section-label">Tutti i segnali generati con risultato finale</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 6])
    if col1.button("Aggiorna risultati"):
        log_ph = st.empty()
        with st.spinner("Aggiornamento risultati in corso..."):
            out, rc = run_script("signal_history.py", "--update", placeholder=log_ph)
        if rc == 0:
            st.success("[OK] Risultati aggiornati")
        else:
            st.error("[ERRORE] vedi log")

    # Usa DB se disponibile, altrimenti fallback su CSV
    db = get_db()
    if db:
        history_df = db.get_signal_history(days_back=365)
        if history_df.empty:
            history_df = load_csv("signals.csv")
    else:
        history_df = load_csv("signal_history.csv")
        if history_df.empty:
            history_df = load_csv("signals.csv")

    if history_df.empty:
        st.info("Nessuno storico disponibile. Lo storico si costruisce automaticamente ogni volta che lanci lo screener.")
        st.stop()

    # KPI
    closed = history_df[history_df["status"].isin(["win","loss"])]
    open_s = history_df[history_df["status"] == "open"]
    wins   = history_df[history_df["status"] == "win"]
    losses = history_df[history_df["status"] == "loss"]

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Segnali totali",  len(history_df))
    c2.metric("Aperti",          len(open_s))
    c3.metric("Chiusi",          len(closed))
    if not closed.empty:
        wr = len(wins)/len(closed)*100
        c4.metric("Win rate storico", f"{wr:.1f}%",
                  delta="Forte" if history_df[history_df["signal_type"]=="FORTE"]["status"].isin(["win","loss"]).any() else None)
        avg_ret = closed["net_return_pct"].mean()
        c5.metric("Rend. medio chiusi", fmt_pct(avg_ret))

    st.markdown("---")

    # Filtri
    col_f1, col_f2, col_f3 = st.columns(3)
    status_filter = col_f1.multiselect("Stato", ["open","win","loss","expired"],
                                        default=["open","win","loss","expired"])
    sector_filter = col_f2.multiselect("Settore",
                                        history_df["sector"].dropna().unique().tolist(),
                                        default=[])
    signal_filter = col_f3.multiselect("Tipo segnale", ["FORTE","OK"], default=["FORTE","OK"])

    filtered = history_df[history_df["status"].isin(status_filter)]
    if sector_filter:
        filtered = filtered[filtered["sector"].isin(sector_filter)]
    if signal_filter:
        filtered = filtered[filtered["signal_type"].isin(signal_filter)]

    filtered = filtered.sort_values("signal_date", ascending=False)

    # Tabella
    def style_status(v):
        colors = {"win":"#E1F5EE","loss":"#FCEBEB","open":"#E6F1FB","expired":"#f8f8f6"}
        return f"background-color:{colors.get(v,'')};"

    cols_show = ["signal_date","ticker","sector","signal_type","entry_score",
                 "ai_prob","entry_price","status","exit_date","exit_price",
                 "net_return_pct","holding_days","exit_reason"]
    cols_ok = [c for c in cols_show if c in filtered.columns]

    fmt_dict = {}
    if "ai_prob" in cols_ok:         fmt_dict["ai_prob"]         = "{:.1%}"
    if "entry_price" in cols_ok:     fmt_dict["entry_price"]     = "€{:.3f}"
    if "exit_price" in cols_ok:      fmt_dict["exit_price"]      = "€{:.3f}"
    if "net_return_pct" in cols_ok:  fmt_dict["net_return_pct"]  = "{:+.2f}%"

    st.dataframe(
        filtered[cols_ok].style
            .format(fmt_dict)
            .applymap(style_status, subset=["status"])
            .background_gradient(subset=["net_return_pct"] if "net_return_pct" in cols_ok else [], cmap="RdYlGn"),
        use_container_width=True, height=450
    )

    # Grafico win rate per ticker
    if not closed.empty and len(closed) >= 3:
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-label">Win rate per ticker (segnali chiusi)</div>', unsafe_allow_html=True)
            by_ticker = closed.groupby("ticker").agg(
                n=("status","count"),
                wins=("status", lambda x: (x=="win").sum())
            )
            by_ticker["win_rate"] = by_ticker["wins"]/by_ticker["n"]*100
            by_ticker = by_ticker[by_ticker["n"] >= 2].sort_values("win_rate")
            fig = go.Figure(go.Bar(
                x=by_ticker["win_rate"], y=by_ticker.index, orientation="h",
                marker_color=["#1D9E75" if v>=50 else "#D85A30" for v in by_ticker["win_rate"]],
                text=[f"{v:.0f}% ({n})" for v,n in zip(by_ticker["win_rate"],by_ticker["n"])],
                textposition="outside"
            ))
            fig.update_layout(height=280,margin=dict(l=0,r=60,t=0,b=0),
                              plot_bgcolor="white",paper_bgcolor="white",
                              xaxis=dict(ticksuffix="%",range=[0,110]),
                              showlegend=False,font_family="DM Sans")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-label">Rendimento per tipo segnale</div>', unsafe_allow_html=True)
            by_type = closed.groupby("signal_type")["net_return_pct"].agg(["mean","count","std"])
            fig2 = go.Figure(go.Bar(
                x=by_type.index, y=by_type["mean"],
                marker_color=["#1D9E75" if v>=0 else "#D85A30" for v in by_type["mean"]],
                text=[f"{v:+.1f}%" for v in by_type["mean"]],
                textposition="outside"
            ))
            fig2.update_layout(height=280,margin=dict(l=0,r=0,t=0,b=0),
                               plot_bgcolor="white",paper_bgcolor="white",
                               yaxis=dict(ticksuffix="%",gridcolor="#f0f0ee"),
                               showlegend=False,font_family="DM Sans")
            st.plotly_chart(fig2, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGINA: OTTIMIZZATORE
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# PAGINA: EARNINGS CALENDAR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📅 Earnings":
    st.markdown("## Earnings calendar")
    st.markdown('<div class="section-label">Date risultati trimestrali — evita ingressi rischiosi</div>', unsafe_allow_html=True)

    col1, col2, _ = st.columns([2, 2, 4])
    if col1.button("Aggiorna calendario"):
        log_ph = st.empty()
        with st.spinner("Download earnings in corso..."):
            out, rc = run_script("earnings_calendar.py", "", placeholder=log_ph)
        if rc == 0:
            st.success("[OK] Calendario aggiornato")
        else:
            st.error("[ERRORE] vedi log")

    days_ahead = col2.slider("Mostra prossimi N giorni", 7, 60, 30)

    db = get_db()
    if db is None:
        st.warning("Database non disponibile. Lancia prima \'Setup database SQLite\'.")
        st.stop()

    # Prossimi earnings
    try:
        from earnings_calendar import get_upcoming_earnings
        upcoming = get_upcoming_earnings(days_ahead=days_ahead, db=db)
    except Exception as e:
        upcoming = pd.DataFrame()
        st.error(f"Errore: {e}")

    if upcoming.empty:
        st.info(f"Nessun earnings nei prossimi {days_ahead} giorni — o calendario vuoto. Clicca \'Aggiorna calendario\'.")
    else:
        st.markdown(f'<div class="section-label">{len(upcoming)} earnings nei prossimi {days_ahead} giorni</div>', unsafe_allow_html=True)

        for _, row in upcoming.iterrows():
            days = int(row.get("days_to_earnings", 0))
            if days <= 3:
                color = "#FCEBEB"
                label = "BLOCCO"
                border = "#A32D2D"
            elif days <= 7:
                color = "#FAEEDA"
                label = "WARN"
                border = "#BA7517"
            else:
                color = "#f8f8f6"
                label = "OK"
                border = "#e0e0d8"

            st.markdown(f"""
            <div style="border:0.5px solid {border};border-left:3px solid {border};
                        border-radius:8px;padding:10px 16px;margin-bottom:8px;
                        background:{color};display:flex;justify-content:space-between;align-items:center">
              <div>
                <strong>{row['ticker']}</strong>
                <span style="color:#888;font-size:12px;margin-left:8px">{row.get('period','Q')}</span>
              </div>
              <div style="text-align:right">
                <div style="font-size:13px;font-weight:500">{row['report_date']}</div>
                <div style="font-size:11px;color:#888">tra {days} giorni — <strong>{label}</strong></div>
              </div>
            </div>""", unsafe_allow_html=True)

    # Spiegazione logica
    st.markdown("---")
    st.markdown('<div class="section-label">Come funziona il filtro earnings</div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Blocco segnale", "entro 3 giorni", delta="segnale non inviato")
    col_b.metric("Warning", "entro 7 giorni", delta="segnale con avviso")
    col_c.metric("Libero", "oltre 7 giorni", delta="nessuna azione")
    st.caption("Fonte: yfinance calendar API. Le date possono variare — verifica sempre prima di operare.")

elif page == "🔬 Ottimizzatore":
    st.markdown("## Ottimizzatore parametri")
    st.markdown('<div class="section-label">Trova i parametri ottimali con grid search sulla simulazione</div>', unsafe_allow_html=True)

    st.info("L'ottimizzatore esegue la simulazione portafoglio su tutte le combinazioni di parametri e trova quella con il miglior risultato. Modalita rapida: ~5-10 min. Modalita completa: ~30-60 min.")

    col1, col2, col3 = st.columns(3)
    opt_metric = col1.selectbox("Metrica da ottimizzare",
                                 ["sharpe","profit_factor","total_return","win_rate"],
                                 index=0)
    opt_days   = col2.number_input("Giorni simulazione", 90, 730, 365)
    opt_mode   = col3.radio("Modalita", ["Rapida (64 combo)", "Completa (729 combo)"], index=0)
    quick_mode = "Rapida" in opt_mode

    metric_map = {"sharpe":"sharpe","profit_factor":"pf",
                  "total_return":"return","win_rate":"winrate"}

    if st.button("Avvia ottimizzazione"):
        args = f"--metric {metric_map[opt_metric]} --days {opt_days}"
        if quick_mode:
            args += " --quick"
        log_ph = st.empty()
        st.markdown(f"**Ottimizzazione in corso... ({opt_mode})**")
        with st.spinner("Grid search in corso — potrebbe richiedere alcuni minuti..."):
            out, rc = run_script("optimizer.py", args, placeholder=log_ph)
        if rc == 0:
            st.success("[OK] Ottimizzazione completata")
        else:
            st.error("[ERRORE] vedi log")

    # Risultati
    results_df = load_csv("optimizer_results.csv")
    best_file  = ROOT / "data" / "processed" / "optimizer_best.json"

    if not results_df.empty:
        st.markdown("---")

        # Migliori parametri
        if best_file.exists():
            best = json.loads(best_file.read_text())
            st.markdown('<div class="section-label">Parametri ottimali trovati</div>', unsafe_allow_html=True)
            param_cols = ["stop_loss_pct","take_profit_pct","trailing_stop_pct",
                          "vix_entry","entry_score_threshold","ai_prob_threshold"]
            metric_cols = ["sharpe","profit_factor","total_return_pct","win_rate","n_trades","max_drawdown_pct"]

            c1,c2,c3 = st.columns(3)
            c1.metric("Stop loss",    f"{best.get('stop_loss_pct',0)*100:.0f}%")
            c1.metric("Take profit",  f"{best.get('take_profit_pct',0)*100:.0f}%")
            c2.metric("Trailing",     f"{best.get('trailing_stop_pct',0)*100:.0f}%")
            c2.metric("VIX entry",    f"{best.get('vix_entry',20)}")
            c3.metric("Score min",    f"{best.get('entry_score_threshold',6)}/9")
            c3.metric("AI prob min",  f"{best.get('ai_prob_threshold',0.60):.0%}")

            st.markdown("---")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Sharpe",         f"{best.get('sharpe',0):.2f}")
            c2.metric("Profit factor",  f"{best.get('profit_factor',0):.2f}")
            c3.metric("Rendimento",     fmt_pct(best.get('total_return_pct',0)))
            c4.metric("Max drawdown",   fmt_pct(best.get('max_drawdown_pct',0)))

            if st.button("Applica parametri ottimali alla strategia"):
                cfg = {
                    "stop_loss_pct":        best.get("stop_loss_pct", 0.04),
                    "take_profit_pct":      best.get("take_profit_pct", 0.12),
                    "trailing_stop_pct":    best.get("trailing_stop_pct", 0.06),
                    "transaction_cost":     0.0015,
                    "vix_entry":            int(best.get("vix_entry", 20)),
                    "vix_exit":             25,
                    "rsi_max":              62,
                    "entry_score_threshold":int(best.get("entry_score_threshold", 6)),
                    "ai_prob_threshold":    best.get("ai_prob_threshold", 0.60),
                    "max_positions":        3,
                    "initial_capital":      10000,
                }
                (ROOT / "config.json").write_text(json.dumps(cfg, indent=2))
                st.success("[OK] config.json aggiornato con i parametri ottimali")

        # Tabella completa risultati
        st.markdown('<div class="section-label">Tutte le combinazioni testate</div>', unsafe_allow_html=True)
        cols_show = ["stop_loss_pct","take_profit_pct","trailing_stop_pct",
                     "vix_entry","entry_score_threshold","ai_prob_threshold",
                     "sharpe","profit_factor","total_return_pct","win_rate","n_trades","max_drawdown_pct"]
        cols_ok = [c for c in cols_show if c in results_df.columns]
        st.dataframe(
            results_df[cols_ok].head(20).style
                .format({
                    "stop_loss_pct": "{:.0%}", "take_profit_pct": "{:.0%}",
                    "trailing_stop_pct": "{:.0%}", "ai_prob_threshold": "{:.0%}",
                    "sharpe": "{:.3f}", "profit_factor": "{:.3f}",
                    "total_return_pct": "{:+.1f}%", "win_rate": "{:.1f}%",
                    "max_drawdown_pct": "{:.1f}%",
                })
                .background_gradient(subset=["sharpe"] if "sharpe" in cols_ok else [], cmap="Greens"),
            use_container_width=True
        )
    else:
        st.info("Nessun risultato di ottimizzazione. Avvia la grid search.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGINA: ALERT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔔 Alert":
    st.markdown("## Configurazione alert")
    st.markdown('<div class="section-label">Notifiche email automatiche quando escono segnali forti</div>', unsafe_allow_html=True)

    # Status configurazione email
    import smtplib
    email_configured = bool(os.environ.get("GMAIL_APP_PASSWORD",""))
    if email_configured:
        st.success("[OK] App Password Gmail configurata — email attive")
    else:
        st.warning("[WARN] GMAIL_APP_PASSWORD non configurata. Vai su Streamlit Cloud → Settings → Secrets e aggiungi: GMAIL_APP_PASSWORD = \"xxxx-xxxx-xxxx-xxxx\"")

    st.markdown("---")

    # Test email
    st.markdown('<div class="section-label">Test notifica</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,6])
    if col1.button("Invia email di test"):
        out, rc = run_script("notify_email.py", "--test")
        if rc == 0:
            st.success("[OK] Email di test inviata a boz1977@gmail.com")
        else:
            st.error(f"[ERRORE] {out[-200:]}")

    st.markdown("---")

    # Configurazione soglie alert
    st.markdown('<div class="section-label">Soglie per gli alert</div>', unsafe_allow_html=True)

    alert_cfg_file = ROOT / "alert_config.json"
    alert_defaults = {
        "min_prob_for_email": 0.65,
        "only_forte":         False,
        "max_vix":            20,
        "send_empty":         False,
    }
    alert_cfg = json.loads(alert_cfg_file.read_text()) if alert_cfg_file.exists() else alert_defaults

    col1, col2 = st.columns(2)
    alert_cfg["min_prob_for_email"] = col1.slider(
        "AI prob minima per ricevere email", 0.60, 0.85,
        float(alert_cfg["min_prob_for_email"]), 0.01
    )
    alert_cfg["only_forte"] = col2.checkbox(
        "Invia solo se ci sono segnali FORTE",
        value=alert_cfg["only_forte"]
    )
    alert_cfg["send_empty"] = st.checkbox(
        "Invia email anche quando non ci sono segnali",
        value=alert_cfg["send_empty"]
    )

    if st.button("Salva configurazione alert"):
        alert_cfg_file.write_text(json.dumps(alert_cfg, indent=2))
        st.success("[OK] Configurazione alert salvata")

    st.markdown("---")

    # Automazione
    st.markdown('<div class="section-label">Automazione — scheduler</div>', unsafe_allow_html=True)
    st.markdown("""
Per ricevere l'email ogni sera automaticamente **senza aprire l'app**, hai due opzioni:

**Opzione A — Windows Task Scheduler (consigliata per uso locale)**
""")
    st.code("""# Nel PowerShell (copia e incolla)
$action  = New-ScheduledTaskAction -Execute "python" -Argument "src\\daily_screener.py" -WorkingDirectory "C:\\path\\to\\trading-ai"
$trigger = New-ScheduledTaskTrigger -Daily -At 18:00
$settings= New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
Register-ScheduledTask -TaskName "FTSE MIB Screener" -Action $action -Trigger $trigger -Settings $settings
""", language="powershell")

    st.markdown("**Opzione B — GitHub Actions (per Streamlit Cloud, gira anche senza PC acceso)**")
    st.code("""# .github/workflows/daily_screener.yml
name: Daily Screener
on:
  schedule:
    - cron: '30 16 * * 1-5'  # 18:30 ora italiana, solo lun-ven
  workflow_dispatch:           # permette lancio manuale

jobs:
  screener:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: python src/daily_screener.py
        env:
          GMAIL_APP_PASSWORD: ${{ secrets.GMAIL_APP_PASSWORD }}
""", language="yaml")
    st.info("Con GitHub Actions lo screener gira ogni giorno lavorativo alle 18:30 direttamente su GitHub, ti manda la email, e non richiede che il tuo PC sia acceso.")

    # Ultimo screener
    screener_df = load_csv("screener_latest.csv")
    if not screener_df.empty:
        st.markdown("---")
        st.markdown("""<div class="section-label">Invia l'ultimo screener ora</div>""", unsafe_allow_html=True)
        if st.button("Invia screener attuale via email"):
            out, rc = run_script("notify_email.py")
            if rc == 0:
                st.success(f"[OK] Email inviata con {len(screener_df)} segnali")
            else:
                st.error(f"[ERRORE] {out[-200:]}")