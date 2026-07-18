"""Streamlit dashboard for the Financial News Radar.

Consumes the FastAPI backend (F6.1). Reads the API base URL from the
environment (API_BASE_URL) or falls back to the dev default.

Run with:
    streamlit run src/app/dashboard.py
(FastAPI must be running in a separate terminal)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import plotly.graph_objects as go
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Financial News Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .alert-card   { background:#3d0000; border:1px solid #ff4b4b;
                    border-radius:8px; padding:14px 18px; margin-bottom:8px; }
    .ok-card      { background:#0d1117; border:1px solid #30363d;
                    border-radius:8px; padding:14px 18px; margin-bottom:8px; }
    .ticker-big   { font-size:1.25rem; font-weight:700; letter-spacing:.05em; }
    .company-name { font-size:.85rem; color:#8b949e; margin-bottom:4px; }
    .metric-label { font-size:.75rem; color:#8b949e; }
    .metric-val   { font-size:1.1rem; font-weight:600; }
    .alert-badge  { color:#ff4b4b; font-weight:700; font-size:.9rem; }
    .ok-badge     { color:#3fb950; font-weight:700; font-size:.9rem; }
    .reason-txt   { font-size:.78rem; color:#e3b341; margin-top:4px; }
    .sent-pos     { color:#3fb950; font-weight:600; }
    .sent-neg     { color:#ff4b4b; font-weight:600; }
    .sent-neu     { color:#8b949e; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── API helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=120)
def fetch_catalog() -> list[dict]:
    resp = requests.get(f"{API_BASE}/catalog", timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=60)
def fetch_portfolio(tickers: str) -> list[dict]:
    resp = requests.get(f"{API_BASE}/portfolio", params={"tickers": tickers}, timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=60)
def fetch_company(ticker: str) -> dict:
    resp = requests.get(f"{API_BASE}/company/{ticker}", timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_briefing(ticker: str, days: int = 7) -> dict:
    resp = requests.get(
        f"{API_BASE}/company/{ticker}/briefing",
        params={"days": days},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ── Shared helpers ────────────────────────────────────────────────────────────

def _sentiment_color(label: str) -> str:
    return {"positive": "#3fb950", "negative": "#ff4b4b"}.get(label, "#8b949e")


def _net_bar(net: float) -> str:
    """Small ASCII progress-bar-style indicator for net sentiment."""
    filled = int(round((net + 1) / 2 * 10))
    filled = max(0, min(10, filled))
    bar = "█" * filled + "░" * (10 - filled)
    color = "#3fb950" if net > 0.1 else "#ff4b4b" if net < -0.1 else "#8b949e"
    return f'<span style="color:{color};font-family:monospace">{bar}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📡 Financial News Radar")
    st.caption("Monitorización de cartera · UCM TFM")
    st.divider()

    try:
        catalog = fetch_catalog()
    except Exception as exc:
        st.error(f"Cannot reach API at {API_BASE}\n\n{exc}")
        st.stop()

    all_tickers = [c["ticker"] for c in catalog]

    selected = st.multiselect(
        "Selecciona tu cartera",
        options=all_tickers,
        default=all_tickers,
        help="Tickers disponibles en la BD",
    )

    if not selected:
        st.warning("Selecciona al menos un ticker.")
        st.stop()

    st.divider()
    st.caption(f"API: `{API_BASE}`")
    refresh = st.button("🔄 Actualizar datos", use_container_width=True)
    if refresh:
        st.cache_data.clear()
        st.rerun()

    st.divider()
    # Drill-down selector (populated after portfolio render)
    drill_ticker = st.selectbox(
        "🔍 Detalle de empresa",
        options=["(ninguna)"] + selected,
        index=0,
    )


# ── Portfolio view ────────────────────────────────────────────────────────────

tickers_param = ",".join(selected)
try:
    portfolio = fetch_portfolio(tickers_param)
except Exception as exc:
    st.error(f"Error cargando cartera: {exc}")
    st.stop()

# Sort: alerts first, then by net sentiment ascending (worst first within alerts)
portfolio_sorted = sorted(portfolio, key=lambda r: (not r["alert"], r["net_sentiment_7d"]))

alert_count = sum(1 for r in portfolio_sorted if r["alert"])

st.header("Vista de cartera", divider="gray")

col_meta1, col_meta2, col_meta3 = st.columns(3)
col_meta1.metric("Empresas monitorizadas", len(portfolio_sorted))
col_meta2.metric("🚨 Alertas activas", alert_count, delta=None)
col_meta3.metric(
    "Sentimiento medio",
    f"{sum(r['net_sentiment_7d'] for r in portfolio_sorted)/max(len(portfolio_sorted),1):.3f}",
)

st.divider()

# Render cards in a 2-column grid
left_items = portfolio_sorted[::2]
right_items = portfolio_sorted[1::2]

for pair in zip(left_items, right_items + [None]):
    col_l, col_r = st.columns(2)
    for col, item in [(col_l, pair[0]), (col_r, pair[1])]:
        if item is None:
            continue
        with col:
            card_cls = "alert-card" if item["alert"] else "ok-card"
            badge = (
                '<span class="alert-badge">🚨 ALERTA</span>'
                if item["alert"]
                else '<span class="ok-badge">✓ OK</span>'
            )
            net = item["net_sentiment_7d"]
            net_str = f"{net:+.3f}"
            reason_html = (
                f'<div class="reason-txt">⚠ {item["alert_reason"]}</div>'
                if item.get("alert_reason")
                else ""
            )
            st.markdown(
                f"""
                <div class="{card_cls}">
                  <div class="company-name">{item["name"]}</div>
                  <div style="display:flex; justify-content:space-between; align-items:center">
                    <span class="ticker-big">{item["ticker"]}</span>
                    {badge}
                  </div>
                  <div style="margin-top:10px; display:flex; gap:28px;">
                    <div>
                      <div class="metric-label">Sentimiento neto 7d</div>
                      <div class="metric-val">{net_str} &nbsp; {_net_bar(net)}</div>
                    </div>
                    <div>
                      <div class="metric-label">Noticias 7d</div>
                      <div class="metric-val">{item["news_count_7d"]}</div>
                    </div>
                  </div>
                  {reason_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Drill-down ────────────────────────────────────────────────────────────────

if drill_ticker != "(ninguna)":
    st.divider()
    st.header(f"Detalle: {drill_ticker}", divider="gray")

    try:
        detail: dict[str, Any] = fetch_company(drill_ticker)
    except Exception as exc:
        st.error(f"Error cargando {drill_ticker}: {exc}")
        st.stop()

    company_name = detail.get("name", drill_ticker)
    st.subheader(company_name)

    tab_sent, tab_news, tab_events, tab_briefing = st.tabs(
        ["📈 Sentimiento", "📰 Noticias", "⚡ Eventos", "📄 Informe"]
    )

    # ── Tab 1: Sentiment series ───────────────────────────────────────────────
    with tab_sent:
        series = detail.get("sentiment_series", [])
        if not series:
            st.info("Sin datos de sentimiento en los últimos 30 días.")
        else:
            dates = [s["date"] for s in series]
            nets = [s["net_sentiment"] for s in series]
            counts = [s["n_articles"] for s in series]

            colors = ["#ff4b4b" if v < -0.1 else "#3fb950" if v > 0.1 else "#8b949e" for v in nets]

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=nets,
                    marker_color=colors,
                    name="Sentimiento neto",
                    hovertemplate="<b>%{x}</b><br>Net: %{y:.3f}<extra></extra>",
                )
            )
            fig.add_hline(y=0, line_color="#444", line_width=1)
            fig.add_hline(y=-0.3, line_color="#ff4b4b", line_dash="dot",
                          line_width=1, annotation_text="Umbral alerta",
                          annotation_position="bottom right")
            fig.update_layout(
                title=f"Sentimiento diario neto — {drill_ticker} (últimos 30d)",
                xaxis_title=None,
                yaxis_title="Net sentiment",
                yaxis=dict(range=[-1.1, 1.1]),
                plot_bgcolor="#0d1117",
                paper_bgcolor="#0d1117",
                font_color="#c9d1d9",
                height=380,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Volume bar below
            fig2 = go.Figure()
            fig2.add_trace(
                go.Bar(
                    x=dates, y=counts, marker_color="#388bfd",
                    name="Noticias",
                    hovertemplate="<b>%{x}</b><br>Artículos: %{y}<extra></extra>",
                )
            )
            fig2.update_layout(
                title="Volumen de noticias relevantes por día",
                xaxis_title=None, yaxis_title="Artículos",
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                font_color="#c9d1d9", height=220,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 2: Recent news ────────────────────────────────────────────────────
    with tab_news:
        news = detail.get("recent_news", [])
        if not news:
            st.info("Sin noticias recientes.")
        else:
            st.caption(f"{len(news)} artículos recientes (is_relevant=1)")
            for n in news:
                lbl = n["sentiment_label"]
                color = _sentiment_color(lbl)
                url = n.get("url") or "#"
                score = n.get("sentiment_score", 0.5)
                headline_html = (
                    f'<a href="{url}" target="_blank" '
                    f'style="color:#58a6ff;text-decoration:none;">{n["headline"]}</a>'
                    if url != "#"
                    else n["headline"]
                )
                st.markdown(
                    f"""
                    <div style="padding:8px 0; border-bottom:1px solid #21262d;">
                      <span style="color:#8b949e;font-size:.78rem;">{n["date"]}</span>
                      &nbsp;&nbsp;
                      <span style="color:{color};font-size:.78rem;font-weight:600;">
                        {lbl.upper()} ({score:.2f})
                      </span><br/>
                      {headline_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Tab 3: Events ─────────────────────────────────────────────────────────
    with tab_events:
        events = detail.get("events", [])
        material = [e for e in events if e["is_material"]]
        other = [e for e in events if not e["is_material"]]

        if not events:
            st.info("Sin eventos detectados.")
        else:
            st.caption(
                f"{len(events)} eventos totales · {len(material)} materiales · {len(other)} no materiales"
            )

            TYPE_COLORS: dict[str, str] = {
                "legal": "#ff4b4b",
                "earnings": "#e3b341",
                "guidance": "#e3b341",
                "m_and_a": "#79c0ff",
                "leadership": "#d2a8ff",
                "product": "#56d364",
                "analyst": "#58a6ff",
                "macro": "#8b949e",
                "other": "#8b949e",
            }

            for ev in events:
                etype = ev["event_type"]
                color = TYPE_COLORS.get(etype, "#8b949e")
                mat_icon = "●" if ev["is_material"] else "○"
                dt = ev["datetime"][:10]
                st.markdown(
                    f"""
                    <div style="padding:7px 0; border-bottom:1px solid #21262d;">
                      <span style="color:#8b949e;font-size:.76rem;">{dt}</span>
                      &nbsp;
                      <span style="background:{color}22;color:{color};border:1px solid {color}44;
                        border-radius:4px;padding:1px 7px;font-size:.75rem;font-weight:600;">
                        {etype}
                      </span>
                      &nbsp;<span style="color:{color};font-size:.75rem;">{mat_icon} material</span><br/>
                      <span style="font-size:.9rem;">{ev["description"]}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Tab 4: Briefing ───────────────────────────────────────────────────────
    with tab_briefing:
        days_input = st.slider("Ventana del informe (días)", min_value=3, max_value=30, value=7)
        if st.button(f"📄 Generar informe para {drill_ticker}", type="primary"):
            with st.spinner("Llamando a gpt-4o-mini… puede tardar ~30 s"):
                try:
                    brief = fetch_briefing(drill_ticker, days=days_input)
                    st.success(f"Informe generado — {brief['generated_at']}")
                    st.markdown(brief["markdown"])
                except Exception as exc:
                    st.error(f"Error generando informe: {exc}")
        else:
            st.info(
                "Haz clic en el botón para generar el informe. "
                "Llamará a la API de OpenAI (gpt-4o-mini, ~$0.01)."
            )
