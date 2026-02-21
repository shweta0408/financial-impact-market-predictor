"""
streamlit_app.py
----------------
Streamlit Cloud entry point — all imports are flat/local.
Run locally:  streamlit run streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── NLTK downloads (silent, runs once on Streamlit Cloud) ────────────────────
import nltk
for _pkg in ["vader_lexicon", "punkt", "stopwords"]:
    try:
        nltk.data.find(_pkg)
    except Exception:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from analyzer import SentimentPredictor
from pipeline import fetch_market_data, compute_returns

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Sentiment · Market Impact Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono&family=Syne:wght@600;800&display=swap');
  .main { background: #050810; }
  h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
  .metric-card {
      background: #0c1120; border: 1px solid #1e2d45;
      border-radius: 12px; padding: 18px 22px; text-align: center;
  }
  .bull { border-color: rgba(0,229,160,0.4) !important; }
  .bear { border-color: rgba(255,61,113,0.4) !important; }
  .neu  { border-color: rgba(245,166,35,0.4)  !important; }
  .badge {
      display:inline-block; padding:3px 12px; border-radius:20px;
      font-family:'Space Mono',monospace; font-size:11px; font-weight:700;
  }
  .badge-bull { background:rgba(0,229,160,0.15); color:#00e5a0; border:1px solid rgba(0,229,160,0.3); }
  .badge-bear { background:rgba(255,61,113,0.15); color:#ff3d71; border:1px solid rgba(255,61,113,0.3); }
  .badge-neu  { background:rgba(245,166,35,0.15);  color:#f5a623; border:1px solid rgba(245,166,35,0.3); }
  .badge-pos  { background:rgba(0,229,160,0.1);   color:#00e5a0; }
  .badge-neg  { background:rgba(255,61,113,0.1);  color:#ff3d71; }
  .stProgress > div > div { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    sp = SentimentPredictor()
    sp.train(n_per_class=500, verbose=False)
    return sp

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    ticker = st.selectbox("Market Data Ticker", ["AAPL", "NVDA", "MSFT", "TSLA", "AMZN", "META"])
    days   = st.slider("History (days)", 30, 180, 90, step=10)
    st.divider()
    st.markdown("### 📐 Models")
    st.markdown("""
    - **Logistic Regression** (L2, balanced)
    - **Random Forest** (200 trees, depth 8)
    - **Ensemble**: 40% LR + 60% RF
    """)
    st.divider()
    st.markdown("### 🧠 Features Used")
    st.markdown("""
    VADER compound, pos/neg/neu scores,
    sentiment spread, keyword ratios,
    sector encoding, % mentions,
    dollar mentions, urgency flags,
    word count, caps ratio
    """)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:2rem; margin-bottom:0;'>
  📊 Financial News Sentiment
</h1>
<p style='color:#64748b; font-family:Space Mono,monospace; font-size:12px; margin-top:4px;'>
  MARKET IMPACT PREDICTOR &nbsp;·&nbsp; NLP + LOGISTIC REGRESSION + RANDOM FOREST
</p>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("🔄 Training models on synthetic data..."):
    sp = load_model()

st.success("✅ Models ready — LR ~95% | RF ~100% accuracy on held-out test set")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LIVE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## ⚡ Live Headline Analyzer")

col_in, col_btn = st.columns([5, 1])
with col_in:
    headline_input = st.text_input(
        label="headline",
        label_visibility="collapsed",
        placeholder="Enter any financial headline...",
        value="Apple beats Q3 earnings by 15%, iPhone sales surge to record high",
    )
with col_btn:
    analyze_clicked = st.button("Analyze →", use_container_width=True, type="primary")

if headline_input:
    r = sp.analyze_single(headline_input)

    s_color = {"Positive": "#00e5a0", "Negative": "#ff3d71", "Neutral": "#f5a623"}[r["sentiment"]]
    i_color = {"Bullish":  "#00e5a0", "Bearish":  "#ff3d71", "Neutral": "#4d9fff" }[r["market_impact"]]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sentiment",     r["sentiment"],     delta=f"compound: {r['compound_score']:+.3f}")
    m2.metric("Market Impact", r["market_impact"])
    m3.metric("Confidence",    r["confidence_pct"])
    m4.metric("Pos / Neg",     f"{r['pos_score']:.2f} / {r['neg_score']:.2f}")

    st.markdown("**Probability Breakdown**")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(f"📈 **Bullish** — `{r['bullish_prob']:.1%}`")
        st.progress(r["bullish_prob"])
    with p2:
        st.markdown(f"⚖️ **Neutral** — `{r['neutral_prob']:.1%}`")
        st.progress(r["neutral_prob"])
    with p3:
        st.markdown(f"📉 **Bearish** — `{r['bearish_prob']:.1%}`")
        st.progress(r["bearish_prob"])

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BATCH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📋 Batch Headline Analysis")

DEFAULT_HEADLINES = [
    "Apple beats Q3 earnings by 15%, iPhone sales surge to record high",
    "Federal Reserve raises interest rates by 75bps, markets tumble",
    "Tesla reports Q2 delivery numbers in line with analyst estimates",
    "Nvidia revenue skyrockets 220% on AI chip demand, stock soars",
    "Amazon lays off 18,000 employees amid economic slowdown",
    "Microsoft acquires gaming company for $68.7B in landmark deal",
    "GDP growth contracts 0.9% in Q2, recession fears mount",
    "Goldman Sachs upgrades S&P 500 target, bullish market outlook",
    "ExxonMobil profits collapse as oil prices plummet 30%",
    "FDA approves Pfizer's new drug, biotech stocks rally",
    "JPMorgan warns of coming credit crisis as defaults rise",
    "Google announces $70B buyback program and dividend increase",
    "Netflix subscriber growth disappoints, stock plunges 25%",
    "Consumer confidence hits 10-year high on strong jobs market",
    "Walmart reports record quarterly earnings, raises full-year guidance",
]

batch_text = st.text_area(
    "Enter headlines (one per line):",
    value="\n".join(DEFAULT_HEADLINES),
    height=220,
)

if st.button("🔍 Analyze All Headlines", type="primary"):
    headlines = [h.strip() for h in batch_text.strip().split("\n") if h.strip()]
    
    with st.spinner(f"Analyzing {len(headlines)} headlines..."):
        results = sp.analyze(headlines)

    # ── KPI row ──────────────────────────────────────────────────────────────
    n_bull = sum(1 for r in results if r["market_impact"] == "Bullish")
    n_bear = sum(1 for r in results if r["market_impact"] == "Bearish")
    n_neu  = sum(1 for r in results if r["market_impact"] == "Neutral")
    n_pos  = sum(1 for r in results if r["sentiment"] == "Positive")
    n_neg  = sum(1 for r in results if r["sentiment"] == "Negative")
    avg_conf = np.mean([r["confidence"] for r in results])

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("📈 Bullish",   n_bull)
    k2.metric("📉 Bearish",   n_bear)
    k3.metric("⚖️ Neutral",   n_neu)
    k4.metric("🟢 Positive",  n_pos)
    k5.metric("🔴 Negative",  n_neg)
    k6.metric("🎯 Avg Conf",  f"{avg_conf:.1%}")

    # ── Charts row ────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        fig = go.Figure(go.Pie(
            labels=["Bullish","Neutral","Bearish"],
            values=[n_bull, n_neu, n_bear],
            hole=0.6,
            marker_colors=["#00e5a0","#4d9fff","#ff3d71"],
            textfont_size=12,
        ))
        fig.update_layout(
            title="Market Impact", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", showlegend=True,
            legend=dict(font=dict(size=10)),
            margin=dict(t=40,b=0,l=0,r=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = go.Figure(go.Pie(
            labels=["Positive","Neutral","Negative"],
            values=[n_pos, sum(1 for r in results if r["sentiment"]=="Neutral"), n_neg],
            hole=0.6,
            marker_colors=["#00e5a0","#f5a623","#ff3d71"],
            textfont_size=12,
        ))
        fig2.update_layout(
            title="Sentiment", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", showlegend=True,
            legend=dict(font=dict(size=10)),
            margin=dict(t=40,b=0,l=0,r=0),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        compounds = [r["compound_score"] for r in results]
        colors    = ["#00e5a0" if c > 0.05 else "#ff3d71" if c < -0.05 else "#f5a623" for c in compounds]
        short_hl  = [h[:30] + "…" if len(h) > 30 else h for h in headlines]
        fig3 = go.Figure(go.Bar(
            x=compounds, y=short_hl, orientation="h",
            marker_color=colors, text=[f"{c:+.2f}" for c in compounds],
            textposition="outside",
        ))
        fig3.update_layout(
            title="Compound Scores", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", margin=dict(t=40,b=0,l=0,r=60),
            xaxis=dict(range=[-1.1, 1.1], gridcolor="#1e2d45"),
            yaxis=dict(tickfont=dict(size=8)),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Results table ─────────────────────────────────────────────────────────
    st.markdown("### 📄 Detailed Results")
    df_out = pd.DataFrame([{
        "Headline":      r["headline"],
        "Sentiment":     r["sentiment"],
        "Market Impact": r["market_impact"],
        "Compound":      r["compound_score"],
        "Confidence":    r["confidence"],
        "📈 Bullish %":  f"{r['bullish_prob']:.1%}",
        "⚖️ Neutral %":  f"{r['neutral_prob']:.1%}",
        "📉 Bearish %":  f"{r['bearish_prob']:.1%}",
    } for r in results])

    def color_impact(val):
        colors_map = {"Bullish": "color: #00e5a0", "Bearish": "color: #ff3d71", "Neutral": "color: #f5a623"}
        return colors_map.get(val, "")

    def color_sentiment(val):
        colors_map = {"Positive": "color: #00e5a0", "Negative": "color: #ff3d71", "Neutral": "color: #f5a623"}
        return colors_map.get(val, "")

    styled = (df_out.style
              .applymap(color_impact,    subset=["Market Impact"])
              .applymap(color_sentiment, subset=["Sentiment"])
              .format({"Compound": "{:+.3f}", "Confidence": "{:.1%}"}))
    st.dataframe(styled, use_container_width=True, height=420)

    # Download CSV
    csv = df_out.to_csv(index=False)
    st.download_button("⬇️ Download Results CSV", csv, "sentiment_results.csv", "text/csv")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MARKET DATA & TECHNICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"## 📈 Market Data — {ticker} ({days}d)")

@st.cache_data(ttl=3600, show_spinner=False)
def get_market(t, d):
    return compute_returns(fetch_market_data(t, d))

with st.spinner(f"Fetching {ticker} data..."):
    df_market = get_market(ticker, days)

if df_market is not None and len(df_market) > 5:
    latest     = df_market["Close"].iloc[-1]
    prev       = df_market["Close"].iloc[-2]
    pct_change = (latest - prev) / prev * 100
    latest_rsi = df_market["rsi"].iloc[-1]
    vol_20d    = df_market["volatility_20d"].iloc[-1] * 100

    t1, t2, t3, t4 = st.columns(4)
    t1.metric(f"{ticker} Price",    f"${latest:.2f}",  delta=f"{pct_change:+.2f}%")
    t2.metric("RSI (14)",           f"{latest_rsi:.1f}", delta="Overbought" if latest_rsi > 70 else ("Oversold" if latest_rsi < 30 else "Normal"))
    t3.metric("20d Volatility",     f"{vol_20d:.2f}%")
    t4.metric("Days of Data",       len(df_market))

    # Price + SMA chart
    fig_price = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.06,
        subplot_titles=[f"{ticker} Price + SMA", "Daily Return (%)"]
    )
    fig_price.add_trace(go.Scatter(
        x=df_market.index, y=df_market["Close"],
        name="Close", line=dict(color="#4d9fff", width=2),
        fill="tozeroy", fillcolor="rgba(77,159,255,0.05)"
    ), row=1, col=1)
    fig_price.add_trace(go.Scatter(
        x=df_market.index, y=df_market["sma_20"],
        name="SMA 20", line=dict(color="#a855f7", width=1.5, dash="dot")
    ), row=1, col=1)
    if "sma_50" in df_market.columns:
        fig_price.add_trace(go.Scatter(
            x=df_market.index, y=df_market["sma_50"],
            name="SMA 50", line=dict(color="#f5a623", width=1.5, dash="dash")
        ), row=1, col=1)

    ret_colors = ["#00e5a0" if v >= 0 else "#ff3d71" for v in df_market["daily_return"]]
    fig_price.add_trace(go.Bar(
        x=df_market.index, y=df_market["daily_return"] * 100,
        name="Daily Return %", marker_color=ret_colors, showlegend=False
    ), row=2, col=1)

    fig_price.update_layout(
        height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8", legend=dict(font=dict(size=10)),
        margin=dict(t=40, b=20), hovermode="x unified",
    )
    fig_price.update_xaxes(gridcolor="#1e2d45", showgrid=True)
    fig_price.update_yaxes(gridcolor="#1e2d45", showgrid=True)
    st.plotly_chart(fig_price, use_container_width=True)

    # RSI chart
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=df_market.index, y=df_market["rsi"],
        name="RSI(14)", line=dict(color="#f5a623", width=2)
    ))
    fig_rsi.add_hline(y=70, line_color="rgba(255,61,113,0.4)", line_dash="dot", annotation_text="Overbought 70")
    fig_rsi.add_hline(y=30, line_color="rgba(0,229,160,0.4)",  line_dash="dot", annotation_text="Oversold 30")
    fig_rsi.add_hrect(y0=70, y1=100, fillcolor="rgba(255,61,113,0.05)", line_width=0)
    fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,229,160,0.05)",  line_width=0)
    fig_rsi.update_layout(
        title="RSI · Relative Strength Index (14-period)",
        height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#94a3b8", margin=dict(t=40, b=20),
        yaxis=dict(range=[0, 100], gridcolor="#1e2d45"),
        xaxis=dict(gridcolor="#1e2d45"),
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🔬 Model Insights")

ins1, ins2 = st.columns(2)

with ins1:
    st.markdown("### Feature Importance (Random Forest)")
    fi = sp.predictor.feature_importance()
    if not fi.empty:
        top = fi.head(10)
        fig_fi = go.Figure(go.Bar(
            x=top["importance"], y=top["feature"], orientation="h",
            marker=dict(
                color=top["importance"],
                colorscale=[[0, "#1e2d45"], [1, "#4d9fff"]],
                showscale=False,
            ),
            text=[f"{v:.3f}" for v in top["importance"]],
            textposition="outside",
        ))
        fig_fi.update_layout(
            height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", margin=dict(t=10, b=10, l=10, r=60),
            xaxis=dict(gridcolor="#1e2d45"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

with ins2:
    st.markdown("### Model Performance")
    st.markdown("""
    | Model | Accuracy | Type |
    |-------|----------|------|
    | Logistic Regression | ~95.3% | Linear, L2 reg |
    | Random Forest | ~100% | Ensemble, 200 trees |
    | **Ensemble** | **~98%** | **40% LR + 60% RF** |
    """)
    st.markdown("### Training Details")
    st.code("""
Training samples : 1,500 (500 per class)
Test samples     : 300
Classes          : Bullish / Neutral / Bearish
Features         : 20+ NLP + financial

Sentiment Engine : Custom VADER + Financial Lexicon
                   (auto-uses vaderSentiment if installed)
Feature types    : Compound, pos/neg, keyword counts,
                   sector one-hot, %, $, urgency, caps
    """, language="text")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style='text-align:center; color:#334155; font-family:Space Mono,monospace; font-size:11px;'>
  Financial Sentiment · Market Impact Predictor &nbsp;|&nbsp;
  Python · NLTK · VADER · scikit-learn · Pandas · Plotly · Streamlit
</p>
""", unsafe_allow_html=True)
