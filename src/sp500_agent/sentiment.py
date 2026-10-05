"""Headline sentiment scores in [-1, 1].

Three scorers, chosen with the SENTIMENT_MODEL environment variable:

- "lexicon" (default): a finance word list in the spirit of Loughran and McDonald, who showed that
  general-purpose sentiment words mislead on financial text ("liability", "tax" and "cost" are not
  negative in a 10-K). Handles simple negation ("did not miss").
- "lm": the full Loughran-McDonald master dictionary. It is free for academic use but not
  redistributed here: download the CSV from https://sraf.nd.edu and set LM_DICTIONARY_PATH to it.
- "finbert": the ProsusAI/finbert transformer (needs `pip install transformers torch`); scores are
  P(positive) - P(negative).

Historical headlines from free sources are sparse, so sentiment rarely has enough history to become a
model input; it matters most for the agent's commentary on current news.
"""

from __future__ import annotations

import csv
import os
import re
from functools import lru_cache
from pathlib import Path

POSITIVE_WORDS = frozenset("""
beat beats exceeded exceeds exceeding outperform outperformed outperforms strong stronger strongest growth grew gain gains
gained raises raised raise upgrade upgraded upgrades bullish record surge surged surges rally rallied rallies jump jumped
soar soared soars profitable profitability improve improved improvement improves improving expand expands expanded expansion
boost boosted boosts approval approved wins won award awarded breakthrough robust rebound rebounded rebounds optimistic upbeat
accelerate accelerated accelerating tops topped favorable success successful efficiency upside momentum outpaced
""".split())
NEGATIVE_WORDS = frozenset("""
miss misses missed weak weaker weakness pressure pressured contract contracts contraction downgrade downgraded downgrades
bearish lawsuit lawsuits sued sues decline declined declines declining fall fell falls drop dropped drops plunge plunged
plunges slump slumped slumps loss losses losing cut cuts layoff layoffs recall recalls investigation probe fraud bankruptcy
bankrupt default defaults warning warns warned disappointing disappoint disappointed concern concerns headwind headwinds
slowdown slowing underperform underperformed underperforms delay delayed delays fined penalty penalties impairment writedown
restatement restate resign resigns resigned halt halted shortfall downturn deficit litigation subpoena breach tumble tumbled
tumbles sink sank sinks crash crashed
""".split())
NEGATORS = frozenset({"not", "no", "never", "without", "didn't", "doesn't", "isn't", "wasn't", "won't", "cannot", "can't"})
WORD = re.compile(r"[a-z']+")


def lexicon_score(text: str, positive=POSITIVE_WORDS, negative=NEGATIVE_WORDS) -> float:
    """(positive - negative) / (positive + negative) over the words found; a negator within three words flips a hit."""
    words = WORD.findall(str(text).lower())
    pos = neg = 0
    for i, word in enumerate(words):
        hit = 1 if word in positive else -1 if word in negative else 0
        if hit and NEGATORS.intersection(words[max(0, i - 3) : i]):
            hit = -hit
        pos += hit > 0
        neg += hit < 0
    return 0.0 if pos == neg == 0 else (pos - neg) / (pos + neg)


@lru_cache(maxsize=2)
def load_lm_dictionary(path: str) -> tuple[frozenset, frozenset]:
    """Positive and negative words from the Loughran-McDonald master dictionary CSV (non-zero = in the list)."""
    positive, negative = set(), set()
    with Path(path).open(newline="", encoding="utf-8", errors="ignore") as handle:
        for row in csv.DictReader(handle):
            word = row.get("Word", "").lower()
            if row.get("Positive", "0") not in ("0", ""):
                positive.add(word)
            if row.get("Negative", "0") not in ("0", ""):
                negative.add(word)
    return frozenset(positive), frozenset(negative)


@lru_cache(maxsize=1)
def _finbert():
    from transformers import pipeline  # optional dependency

    return pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)


def score_texts(texts: list[str], method: str | None = None) -> list[float]:
    method = (method or os.environ.get("SENTIMENT_MODEL") or "lexicon").lower()
    if method == "finbert":
        results = _finbert()([str(t) for t in texts], batch_size=32)
        return [sum(r["score"] * {"positive": 1, "negative": -1}.get(r["label"].lower(), 0) for r in result) for result in results]
    if method == "lm":
        path = os.environ.get("LM_DICTIONARY_PATH")
        if not path or not Path(path).exists():
            raise RuntimeError("SENTIMENT_MODEL=lm needs LM_DICTIONARY_PATH pointing to the Loughran-McDonald master dictionary CSV.")
        positive, negative = load_lm_dictionary(path)
        return [lexicon_score(t, positive, negative) for t in texts]
    return [lexicon_score(t) for t in texts]
