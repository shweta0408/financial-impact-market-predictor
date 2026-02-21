"""
analyzer.py
-----------
Main orchestrator — ties all modules together.
All imports are flat (same folder), works on Streamlit Cloud.

Usage:
    from analyzer import SentimentPredictor
    sp = SentimentPredictor()
    sp.train()
    results = sp.analyze(["Apple beats Q3 earnings, stock surges 15%"])
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from typing import List, Dict, Any

from sentiment_engine    import FinancialSentimentAnalyzer
from feature_engineering import FeatureEngineer
from predictor           import MarketImpactPredictor, LABEL_MAP, compound_to_label
from pipeline            import generate_synthetic_headlines, fetch_market_data, compute_returns


class SentimentPredictor:
    def __init__(self):
        self.sentiment  = FinancialSentimentAnalyzer()
        self.features   = FeatureEngineer()
        self.predictor  = MarketImpactPredictor()
        self._trained   = False

    # ── Train ──────────────────────────────────────────────────────────────────
    def train(self, n_per_class: int = 500, verbose: bool = True) -> Dict:
        if verbose:
            print("📊 Generating synthetic training data...")

        df_raw = generate_synthetic_headlines(n_per_class)
        scores_list = [self.sentiment.polarity_scores(h) for h in df_raw["headline"]]
        feat_df = self.features.headlines_to_dataframe(df_raw["headline"].tolist(), scores_list)
        y = df_raw["true_label"].map({"Bullish": 2, "Neutral": 1, "Bearish": 0}).values

        if verbose:
            print(f"   Training on {len(feat_df)} examples across 3 classes...")

        metrics = self.predictor.fit(feat_df, y)
        self._trained = True

        if verbose:
            lr_acc = metrics["lr_report"]["accuracy"]
            rf_acc = metrics["rf_report"]["accuracy"]
            print(f"✅ LR: {lr_acc:.1%}  |  RF: {rf_acc:.1%}")

        return metrics

    # ── Analyze ────────────────────────────────────────────────────────────────
    def analyze(self, headlines: List[str]) -> List[Dict[str, Any]]:
        results = []
        for headline in headlines:
            scores    = self.sentiment.polarity_scores(headline)
            sentiment = self.sentiment.classify(scores["compound"])
            feat      = self.features.extract(headline, scores)
            feat_df   = pd.DataFrame([feat])
            pred      = self.predictor.predict(feat_df)[0]

            results.append({
                "headline":       headline,
                "sentiment":      sentiment,
                "compound_score": scores["compound"],
                "pos_score":      scores["pos"],
                "neg_score":      scores["neg"],
                "neu_score":      scores["neu"],
                "market_impact":  pred["label"],
                "confidence":     pred["confidence"],
                "confidence_pct": f"{pred['confidence']:.1%}",
                "bullish_prob":   pred["ensemble_proba"][2],
                "neutral_prob":   pred["ensemble_proba"][1],
                "bearish_prob":   pred["ensemble_proba"][0],
            })
        return results

    def analyze_single(self, headline: str) -> Dict[str, Any]:
        return self.analyze([headline])[0]

    # ── Market data ────────────────────────────────────────────────────────────
    def get_market_data(self, ticker: str = "AAPL", days: int = 90) -> pd.DataFrame:
        return compute_returns(fetch_market_data(ticker, days))

    # ── Console print ──────────────────────────────────────────────────────────
    def print_results(self, results: List[Dict]):
        print("\n" + "=" * 78)
        print("  FINANCIAL NEWS SENTIMENT — MARKET IMPACT PREDICTOR")
        print("=" * 78)
        for r in results:
            s_icon = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}.get(r["sentiment"], "⚪")
            i_icon = {"Bullish":  "📈", "Bearish":  "📉", "Neutral": "⚖️" }.get(r["market_impact"], "")
            print(f"\n📰 {r['headline']}")
            print(f"   Sentiment : {s_icon} {r['sentiment']:8s}  (compound: {r['compound_score']:+.3f})")
            print(f"   Impact    : {i_icon} {r['market_impact']:7s}  Confidence: {r['confidence_pct']}")
            print(f"   Probs     : 📈 {r['bullish_prob']:.1%}  ⚖️  {r['neutral_prob']:.1%}  📉 {r['bearish_prob']:.1%}")
        print("=" * 78 + "\n")
