"""
pipeline.py
-----------
Synthetic training data generator + market data fetcher.
Falls back to Geometric Brownian Motion simulation if yfinance unavailable.
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

random.seed(42)
np.random.seed(42)

# ── Headline templates ────────────────────────────────────────────────────────
BULLISH = [
    "{co} beats Q{q} earnings by {pct}%, stock surges",
    "{co} revenue surges {pct}% year-over-year, analysts upgrade to buy",
    "{co} announces record quarterly profit, shares soar",
    "Fed signals rate cuts as inflation eases, markets rally strongly",
    "{co} secures ${amt}B government contract, shares soar",
    "{co} AI breakthrough drives {pct}% stock surge",
    "Strong jobs report boosts market confidence, indices climb {pct}%",
    "{co} buyback program of ${amt}B announced, bullish signal",
    "{co} FDA approval granted for blockbuster drug, stock rallies",
    "GDP growth exceeds expectations at {pct}%, economy booming",
    "{co} acquires rival for ${amt}B, synergies expected to boost profit",
    "{co} raises full-year guidance, outperforming all estimates",
    "Consumer confidence index hits {val}-year high on strong jobs data",
    "{co} dividend increased by {pct}%, investors cheer",
    "{co} reports {pct}% revenue growth, beats on every metric",
]

BEARISH = [
    "{co} misses Q{q} earnings estimates, shares plunge {pct}%",
    "{co} lays off {val}% of workforce amid restructuring",
    "{co} faces ${amt}B fraud investigation, stock crashes",
    "Fed raises rates by {pct}bps, markets tumble on inflation fears",
    "{co} Q{q} revenue disappoints, downgraded to sell by analysts",
    "Recession fears mount as GDP contracts {pct}%",
    "{co} loses ${amt}B lawsuit, shares collapse",
    "Inflation surges to {val}-year high, bearish market outlook",
    "{co} guidance cut sharply, stock plummets {pct}%",
    "Banking crisis deepens, {co} shares crash amid contagion fears",
    "{co} CEO resigns amid scandal, investors flee the stock",
    "Oil prices collapse {pct}% on demand destruction concerns",
    "{co} recalls {val}M products over safety concerns, stock drops",
    "Trade war escalates, new tariffs on ${amt}B of goods announced",
    "{co} reports massive loss of ${amt}B, worst quarter in history",
]

NEUTRAL = [
    "{co} announces Q{q} earnings in line with expectations",
    "{co} reports flat revenue growth, guidance maintained for year",
    "Federal Reserve holds rates steady, markets largely unmoved",
    "{co} completes merger review, transition expected to be smooth",
    "Market trading range-bound ahead of key jobs report",
    "{co} updates product roadmap at annual investor day event",
    "Q{q} GDP growth meets consensus estimate of {pct}%",
    "{co} appoints new CFO, smooth leadership transition expected",
    "Oil prices stable as supply and demand remain in balance",
    "{co} files quarterly 10-Q with no material changes disclosed",
    "Sector rotation continues as investors reassess portfolio weights",
    "{co} maintains annual guidance despite macro headwinds noted",
    "{co} reports mixed results: revenue up, margins slightly down",
    "Analysts issue mixed views on {co} ahead of earnings report",
    "Market awaits Fed decision, {co} trades near 52-week average",
]

COMPANIES = [
    "Apple", "Microsoft", "Tesla", "Amazon", "Google", "Meta",
    "Nvidia", "JPMorgan", "Goldman Sachs", "ExxonMobil",
    "Pfizer", "Boeing", "Netflix", "Walmart", "Berkshire Hathaway",
    "Visa", "UnitedHealth", "Johnson & Johnson", "Chevron", "Salesforce",
]


def _fill(tmpl: str) -> str:
    return tmpl.format(
        co=random.choice(COMPANIES),
        q=random.randint(1, 4),
        pct=round(random.uniform(2, 48), 1),
        amt=round(random.uniform(0.5, 75), 1),
        val=random.randint(2, 35),
    )


def generate_synthetic_headlines(n_per_class: int = 500) -> pd.DataFrame:
    rows = []
    for _ in range(n_per_class):
        rows.append({"headline": _fill(random.choice(BULLISH)),  "true_label": "Bullish"})
    for _ in range(n_per_class):
        rows.append({"headline": _fill(random.choice(BEARISH)),  "true_label": "Bearish"})
    for _ in range(n_per_class):
        rows.append({"headline": _fill(random.choice(NEUTRAL)),  "true_label": "Neutral"})
    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


# ── Market data ───────────────────────────────────────────────────────────────

def fetch_market_data(ticker: str, days: int = 90) -> pd.DataFrame:
    try:
        import yfinance as yf
        end   = datetime.today()
        start = end - timedelta(days=days)
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index   = pd.to_datetime(df.index)
        if len(df) > 10:
            return df
    except Exception:
        pass
    return _gbm_price_data(ticker, days)


def _gbm_price_data(ticker: str, days: int) -> pd.DataFrame:
    seed = abs(hash(ticker)) % (2 ** 31)
    rng  = np.random.RandomState(seed)
    start_prices = {"AAPL": 185, "NVDA": 140, "MSFT": 380, "TSLA": 230,
                    "AMZN": 195, "META": 510, "GOOG": 170}
    price  = start_prices.get(ticker, 150.0)
    mu, sigma = 0.0003, 0.018
    prices = [price]
    for _ in range(days - 1):
        prices.append(prices[-1] * (1 + rng.normal(mu, sigma)))

    dates   = pd.bdate_range(end=datetime.today(), periods=min(days, len(prices)))
    closes  = np.array(prices[:len(dates)])
    opens   = closes * (1 + rng.normal(0, 0.005, len(dates)))
    highs   = np.maximum(closes, opens) * (1 + np.abs(rng.normal(0, 0.008, len(dates))))
    lows    = np.minimum(closes, opens) * (1 - np.abs(rng.normal(0, 0.008, len(dates))))
    volumes = rng.randint(20_000_000, 120_000_000, len(dates))

    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=dates
    )


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["daily_return"]   = df["Close"].pct_change()
    df["log_return"]     = np.log(df["Close"] / df["Close"].shift(1))
    df["volatility_5d"]  = df["daily_return"].rolling(5).std()
    df["volatility_20d"] = df["daily_return"].rolling(20).std()
    df["sma_20"]         = df["Close"].rolling(20).mean()
    df["sma_50"]         = df["Close"].rolling(50).mean()
    df["rsi"]            = _rsi(df["Close"])
    return df.dropna()


def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).round(2)
