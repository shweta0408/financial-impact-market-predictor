"""
sentiment_engine.py
-------------------
VADER-inspired sentiment analyzer with financial domain lexicon.
Works without any external NLP packages installed.
If vaderSentiment IS installed, uses it automatically.
"""

import re
import math
from typing import Dict, List

FINANCIAL_LEXICON = {
    # Strong Positive
    "surge": 3.5, "soar": 3.5, "skyrocket": 4.0, "rally": 3.0, "boom": 3.2,
    "breakthrough": 3.4, "outperform": 3.0, "beat": 2.5, "exceed": 2.8,
    "strong": 2.5, "profit": 2.8, "gain": 2.5, "growth": 2.6, "upgrade": 3.0,
    "bullish": 3.5, "opportunity": 2.2, "positive": 2.0, "dividend": 2.0,
    "buyback": 2.5, "acquisition": 1.8, "partnership": 2.0, "approval": 2.5,
    "innovation": 2.3, "expansion": 2.2, "accelerate": 2.0, "recover": 2.5,
    "rebound": 2.8, "upside": 2.2, "overweight": 2.5, "momentum": 2.0,
    "record": 2.5, "high": 1.5, "launch": 1.8, "buy": 2.0,
    # Mild Positive
    "stable": 1.0, "steady": 1.0, "improve": 1.5, "rise": 1.5,
    "increase": 1.5, "up": 1.0, "higher": 1.3, "better": 1.5,
    "optimistic": 2.0, "confident": 1.8, "support": 1.2, "benefit": 1.5,
    "tops": 2.0, "raises": 1.5, "lifts": 1.5, "climbs": 1.5,
    # Neutral
    "announce": 0.0, "report": 0.0, "plan": 0.0, "expect": 0.0,
    "forecast": 0.0, "guidance": 0.0, "update": 0.0, "change": 0.0,
    "quarter": 0.0, "fiscal": 0.0, "market": 0.0, "new": 0.2,
    # Mild Negative
    "concern": -1.5, "risk": -1.2, "uncertainty": -1.5, "volatile": -1.3,
    "pressure": -1.2, "slow": -1.2, "weak": -1.5, "miss": -2.0,
    "decline": -1.8, "drop": -1.8, "fall": -1.5, "lower": -1.3,
    "below": -1.5, "cut": -2.0, "reduce": -1.3, "downgrade": -2.8,
    "warns": -1.8, "caution": -1.5, "headwinds": -1.5,
    # Strong Negative
    "crash": -4.0, "collapse": -3.8, "plunge": -3.5, "plummet": -3.5,
    "tumble": -3.0, "tank": -3.2, "crisis": -3.5, "recession": -3.0,
    "bankrupt": -4.0, "default": -3.5, "loss": -2.5, "losses": -2.8,
    "fraud": -3.8, "scandal": -3.5, "lawsuit": -2.5, "investigation": -2.2,
    "bearish": -3.5, "sell": -2.0, "downside": -2.2, "underweight": -2.5,
    "layoff": -2.8, "layoffs": -3.0, "recall": -2.5, "warning": -2.5,
    "shortfall": -2.8, "disappointing": -2.8, "disappoint": -2.5,
    "negative": -2.0, "worst": -3.5, "poor": -2.0, "slump": -2.5,
    "plunges": -3.5, "crashes": -4.0, "fears": -2.0, "mounts": -1.5,
}

BOOSTERS = {
    "very": 0.293, "really": 0.293, "extremely": 0.733,
    "hugely": 0.5, "massively": 0.6, "significantly": 0.4,
    "drastically": 0.6, "slightly": -0.293, "somewhat": -0.2,
}

NEGATIONS = {"not", "never", "no", "neither", "nor", "barely", "hardly"}


class FinancialSentimentAnalyzer:
    def __init__(self):
        self._vader = None
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
        except ImportError:
            pass

    def polarity_scores(self, text: str) -> Dict[str, float]:
        if self._vader:
            return self._vader.polarity_scores(text)
        return self._rule_based(text)

    def classify(self, compound: float) -> str:
        if compound >= 0.05:
            return "Positive"
        elif compound <= -0.05:
            return "Negative"
        return "Neutral"

    def _rule_based(self, text: str) -> Dict[str, float]:
        tokens = re.findall(r"[\w\-]+", text.lower())
        valence_list = []
        for i, token in enumerate(tokens):
            val = 0.0
            negated = any(tokens[max(0, i - j)] in NEGATIONS for j in range(1, 4))
            clean = re.sub(r"[^a-z\-]", "", token)
            if clean in FINANCIAL_LEXICON:
                val = FINANCIAL_LEXICON[clean]
                if i > 0 and tokens[i - 1] in BOOSTERS:
                    val += BOOSTERS[tokens[i - 1]] * abs(val) / 4
                if negated:
                    val *= -0.74
            if val != 0:
                valence_list.append(val)

        if not valence_list:
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

        raw = sum(valence_list)
        compound = raw / math.sqrt(raw * raw + 15)
        pos = sum(v for v in valence_list if v > 0) / (len(valence_list) + 1e-9)
        neg = abs(sum(v for v in valence_list if v < 0)) / (len(valence_list) + 1e-9)
        neu = max(0.0, 1.0 - pos - neg)
        total = pos + neg + neu
        return {
            "neg": round(neg / total, 3),
            "neu": round(neu / total, 3),
            "pos": round(pos / total, 3),
            "compound": round(compound, 4),
        }
