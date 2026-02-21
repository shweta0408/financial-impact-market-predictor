# 📊 Financial News Sentiment: Market Impact Predictor

> NLP + Machine Learning pipeline that analyzes financial headlines and predicts market impact in real-time.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://financial-impact-market-predictor.streamlit.app/)

🔗 **Live Demo → [financial-impact-market-predictor.streamlit.app](https://financial-impact-market-predictor.streamlit.app/)**

---

## 🎯 What It Does

Input a financial news headline → get back:

- ✅ **Sentiment Score** — Positive / Neutral / Negative + compound score
- ✅ **Market Impact** — Bullish / Bearish / Neutral prediction
- ✅ **Confidence Score** — probability breakdown across all 3 classes
- ✅ **Visualization Dashboard** — charts, RSI, price data, feature importance

---

## 🗂️ Project Structure

```
├── streamlit_app.py       ← Streamlit entry point  ⬅ main file
├── analyzer.py            ← Main orchestrator (SentimentPredictor class)
├── sentiment_engine.py    ← VADER-style NLP engine with financial lexicon
├── feature_engineering.py ← 20+ NLP & financial feature extractor
├── predictor.py           ← Logistic Regression + Random Forest ensemble
├── pipeline.py            ← Synthetic data generator + yfinance market data
├── requirements.txt       ← Python dependencies
└── README.md
```

> All files are in the **same flat folder** — no subfolders, no package imports. Works perfectly on Streamlit Cloud.

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/financial-impact-market-predictor.git
cd financial-impact-market-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run streamlit_app.py
```

---

## 🧠 Python API Usage

```python
from analyzer import SentimentPredictor

sp = SentimentPredictor()
sp.train()

# Single headline
r = sp.analyze_single("Apple beats Q3 earnings, stock surges 15%")
print(r["sentiment"])       # → Positive
print(r["market_impact"])   # → Bullish
print(r["confidence_pct"])  # → 95.8%
print(r["bullish_prob"])    # → 0.958

# Batch analysis
results = sp.analyze([
    "Apple beats Q3 earnings, stock surges 15%",
    "Fed raises rates 75bps, markets tumble",
    "Tesla Q2 deliveries in line with estimates",
])
sp.print_results(results)
```

---

## 🤖 Models & Performance

| Model | Accuracy | Config |
|-------|----------|--------|
| Logistic Regression | ~95.3% | L2 regularized, balanced class weights |
| Random Forest | ~100% | 200 trees, max depth 8 |
| **Ensemble (final)** | **~98%** | **40% LR + 60% RF** |

- Trained on **1,500 synthetic headlines** (500 per class)
- Tested on **300 held-out samples**
- Classes: **Bullish · Neutral · Bearish**

---

## 📐 Feature Engineering (20+ features)

| Category | Features |
|----------|---------|
| Sentiment | Compound score, pos/neg/neu, sentiment spread, absolute value |
| Text | Word count, char count, caps ratio, punctuation flags |
| Financial | % mentions, $ mentions, ticker presence |
| Domain | Bullish/bearish keyword counts & ratio, urgency flags |
| Sector | Tech, Finance, Energy, Health, Macro (one-hot encoded) |

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| NLP | NLTK · VADER Sentiment · Custom Financial Lexicon |
| ML Models | Logistic Regression · Random Forest (scikit-learn) |
| Data | pandas · numpy · yfinance |
| Visualization | Plotly · Streamlit |
| Deployment | Streamlit Cloud |

---

## 📸 App Sections

1. **⚡ Live Headline Analyzer** — type any headline, get instant results
2. **📋 Batch Analysis** — analyze multiple headlines at once, download CSV
3. **📈 Market Data** — 90-day price chart, SMA, daily returns, RSI indicator
4. **🔬 Model Insights** — feature importance bar chart, model accuracy details

---

## ☁️ Deploy Your Own

1. Fork this repo
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → **New app**
3. Select your fork → set **Main file path** to `streamlit_app.py`
4. Click **Deploy**


---

## Connect Me!
[Linkedin](https://www.linkedin.com/in/shweta-mishra-4777681a4)
[Github](https://github.com/shweta0408)
---
