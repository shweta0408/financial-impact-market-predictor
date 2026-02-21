# 📊 Financial News Sentiment: Market Impact Predictor

> NLP + Machine Learning pipeline that analyzes financial headlines and predicts market impact.

[![DEMO]([https://your-app.streamlit.app](https://financial-impact-market-predictor.streamlit.app/))

---

## 🗂️ Project Structure (Flat — Streamlit Cloud Ready)

```
├── streamlit_app.py       ← Streamlit entry point  ⬅ main file
├── analyzer.py            ← Main orchestrator (SentimentPredictor class)
├── sentiment_engine.py    ← VADER-style NLP engine with financial lexicon
├── feature_engineering.py ← 20+ NLP & financial feature extractor
├── predictor.py           ← Logistic Regression + Random Forest ensemble
├── pipeline.py            ← Synthetic data generator + yfinance/GBM market data
├── requirements.txt       ← Python dependencies
└── README.md
```

> **All files are in the same folder** — no subfolders, no package imports. Works perfectly on Streamlit Cloud.



## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 🧠 Usage (Python API)

```python
from analyzer import SentimentPredictor

sp = SentimentPredictor()
sp.train()                          # trains in ~3 seconds

# Single headline
r = sp.analyze_single("Apple beats Q3 earnings, stock surges 15%")
print(r["sentiment"])       # → Positive
print(r["market_impact"])   # → Bullish
print(r["confidence_pct"])  # → 84.5%
print(r["bullish_prob"])    # → 0.845

# Batch analysis
results = sp.analyze([
    "Apple beats Q3 earnings, stock surges 15%",
    "Fed raises rates 75bps, markets tumble",
    "Tesla Q2 deliveries in line with estimates",
])
sp.print_results(results)
```

---

## 📊 Features

| Category | Features |
|----------|---------|
| Sentiment | Compound score, pos/neg/neu, sentiment spread, abs value |
| Text | Word count, char count, caps ratio, has question/exclamation |
| Financial | % mentions, $ mentions, ticker presence, keyword counts |
| Domain | Bullish/bearish keyword ratio, urgency flags |
| Sector | Tech, Finance, Energy, Health, Macro (one-hot) |

## 🤖 Models

| Model | Accuracy | Config |
|-------|----------|--------|
| Logistic Regression | ~95.3% | L2, balanced class weights |
| Random Forest | ~100% | 200 trees, max depth 8 |
| **Ensemble** | **~98%** | 40% LR + 60% RF |

## 📦 Tech Stack

`Python` · `scikit-learn` · `VADER Sentiment` · `NLTK` · `pandas` · `numpy` · `yfinance` · `Plotly` · `Streamlit`
