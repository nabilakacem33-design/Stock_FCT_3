"""News sentiment scoring with VADER (recency-weighted)."""
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_an = SentimentIntensityAnalyzer()


def score_news(items: list[dict]) -> tuple[float, list[dict]]:
    """Return (overall score in [-1, 1], items with per-article scores)."""
    if not items:
        return 0.0, []
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    scored, weights = [], []
    for it in items:
        s = _an.polarity_scores(f"{it['title']}. {it['summary']}")["compound"]
        d = it["date"]
        if pd.notna(d):
            d = d.tz_localize(None) if d.tzinfo else d
            age = max((now - d).total_seconds() / 86400, 0)
        else:
            age = 7
        weights.append(np.exp(-age / 5))
        scored.append({**it, "score": s})
    overall = float(np.average([x["score"] for x in scored], weights=weights))
    return overall, scored
