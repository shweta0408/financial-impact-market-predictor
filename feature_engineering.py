"""
feature_engineering.py
-----------------------
Extracts 20+ NLP and financial features from headlines + sentiment scores.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any

PERCENTAGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
DOLLAR_RE     = re.compile(r"\$(\d+(?:\.\d+)?)")
TICKER_RE     = re.compile(r"\b[A-Z]{2,5}\b")

SECTOR_KEYWORDS = {
    "tech":    ["apple", "microsoft", "google", "meta", "nvidia", "semiconductor",
                "software", "ai", "cloud", "chip", "tech", "technology", "amazon"],
    "finance": ["bank", "fed", "federal reserve", "interest rate", "treasury",
                "goldman", "jpmorgan", "lending", "credit", "mortgage", "rate"],
    "energy":  ["oil", "gas", "opec", "exxon", "chevron", "renewable", "solar",
                "wind", "energy", "petroleum"],
    "health":  ["fda", "drug", "pharma", "vaccine", "clinical", "biotech",
                "hospital", "healthcare", "medical", "pfizer"],
    "macro":   ["gdp", "inflation", "cpi", "unemployment", "recession",
                "economy", "growth", "policy", "tariff", "trade", "consumer"],
}

BULLISH_KWS = [
    "beat", "beats", "surge", "surges", "soar", "soars", "record", "profit",
    "growth", "upgrade", "rally", "boom", "outperform", "bullish", "buyback",
    "dividend", "exceed", "strong", "approval", "approved", "launches",
    "skyrocket", "rebound", "raises", "lifts", "tops",
]

BEARISH_KWS = [
    "miss", "misses", "crash", "crashes", "plunge", "plunges", "loss", "losses",
    "decline", "downgrade", "layoff", "layoffs", "bankrupt", "fraud", "warning",
    "warns", "weak", "disappoint", "disappoints", "recall", "investigation",
    "bearish", "crisis", "collapse", "collapses", "fears", "recession", "tumble",
]

URGENCY_WORDS = [
    "breaking", "urgent", "alert", "flash", "just in",
    "q1", "q2", "q3", "q4", "quarterly", "earnings",
]


class FeatureEngineer:
    def extract(self, headline: str, scores: Dict[str, float]) -> Dict[str, Any]:
        text_lower = headline.lower()
        f = {}

        # Sentiment scores
        f["compound"]         = scores["compound"]
        f["pos"]              = scores["pos"]
        f["neg"]              = scores["neg"]
        f["neu"]              = scores["neu"]
        f["sentiment_spread"] = scores["pos"] - scores["neg"]
        f["sentiment_abs"]    = abs(scores["compound"])

        # Text structure
        words = text_lower.split()
        f["word_count"]       = len(words)
        f["char_count"]       = len(headline)
        f["has_question"]     = int("?" in headline)
        f["has_exclamation"]  = int("!" in headline)
        f["caps_ratio"]       = sum(1 for c in headline if c.isupper()) / max(len(headline), 1)

        # Financial numerics
        pcts = PERCENTAGE_RE.findall(text_lower)
        f["pct_mentioned"]    = int(bool(pcts))
        f["max_pct"]          = max((float(p) for p in pcts), default=0.0)
        f["dollar_mentioned"] = int(bool(DOLLAR_RE.search(headline)))
        f["has_ticker"]       = int(bool(TICKER_RE.search(headline)))

        # Urgency
        f["urgency"] = int(any(w in text_lower for w in URGENCY_WORDS))

        # Sector one-hot
        for sector, kws in SECTOR_KEYWORDS.items():
            f[f"sector_{sector}"] = int(any(kw in text_lower for kw in kws))

        # Keyword counts
        f["bullish_kw_count"] = sum(kw in text_lower for kw in BULLISH_KWS)
        f["bearish_kw_count"] = sum(kw in text_lower for kw in BEARISH_KWS)
        denom = max(f["bullish_kw_count"] + f["bearish_kw_count"], 1)
        f["kw_ratio"]         = (f["bullish_kw_count"] - f["bearish_kw_count"]) / denom

        return f

    def headlines_to_dataframe(self, headlines: List[str], scores_list: List[Dict]) -> pd.DataFrame:
        rows = [self.extract(h, s) for h, s in zip(headlines, scores_list)]
        return pd.DataFrame(rows)
