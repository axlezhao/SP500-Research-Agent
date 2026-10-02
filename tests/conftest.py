from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sp500_agent.data_loader import load_fundamentals, load_news, load_prices
from sp500_agent.features import build_research_features
from sp500_agent.sample_data import make_sample_data


def build_sample_features(raw_dir: Path, seed: int = 7) -> pd.DataFrame:
    make_sample_data(seed=seed, raw_dir=raw_dir)
    return build_research_features(load_prices(raw_dir), load_fundamentals(raw_dir), load_news(raw_dir))


@pytest.fixture(scope="session")
def sample_raw_dir(tmp_path_factory) -> Path:
    raw_dir = tmp_path_factory.mktemp("raw")
    make_sample_data(raw_dir=raw_dir)
    return raw_dir


@pytest.fixture(scope="session")
def sample_features(sample_raw_dir) -> pd.DataFrame:
    return build_research_features(load_prices(sample_raw_dir), load_fundamentals(sample_raw_dir), load_news(sample_raw_dir))



@pytest.fixture(scope="session")
def research_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("research")


@pytest.fixture(scope="session")
def research_run(sample_features, research_dir):
    from sp500_agent.research import run_research

    return run_research(sample_features, output_dir=research_dir, model_path=None)


@pytest.fixture(scope="session")
def research_data(sample_features, sample_raw_dir, research_run, research_dir):
    """Loaded the way the app loads it, so saving and reading the artifacts is exercised too."""
    from sp500_agent.features import _sentiment_scores
    from sp500_agent.research import load_artifacts
    from sp500_agent.research_tools import ResearchData

    news = load_news(sample_raw_dir)
    news = news.assign(sentiment_score=_sentiment_scores(news))
    return ResearchData(sample_features, research_run.bundle, news, load_artifacts(research_dir))


def make_live_world(seed: int = 3):
    """A small synthetic 'live' dataset: prices with a split, point-in-time fundamentals, macro and membership."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=420)
    tickers = [f"T{i:02d}" for i in range(14)]
    rows = []
    for i, ticker in enumerate(tickers):
        close = 50 * np.cumprod(1 + rng.normal(0.0003, 0.015, len(dates)))
        rows.append(pd.DataFrame({"ticker": ticker, "date": dates, "close": close, "raw_close": close, "volume": rng.integers(1e5, 1e6, len(dates))}))
    prices = pd.concat(rows, ignore_index=True)
    companies = pd.DataFrame({"ticker": tickers, "company_name": [f"{t} Inc" for t in tickers], "sector": ["Tech", "Health", "Energy"] * 4 + ["Tech", "Health"], "cik": range(1, 15)})
    filings = []
    for i, ticker in enumerate(tickers):
        for q, filed in enumerate(pd.date_range("2019-11-01", periods=7, freq="91D")):
            filings.append({
                "ticker": ticker, "available_date": filed, "period_end": filed - pd.Timedelta(days=35),
                "revenue_ttm": 1e9 * (1 + i) * (1 + 0.02 * q), "net_income_ttm": 1e8 * (1 + i % 5) - 2e7 * (i == 3),
                "revenue_growth_yoy": 0.05 * (i % 4) - 0.05, "equity": 5e8 * (1 + i), "liabilities": 4e8 * (1 + i % 3),
                "shares_outstanding": 1e7 * (1 + i),
            })
    fundamentals = pd.DataFrame(filings)
    macro = pd.DataFrame({"date": dates, "tbill_3m": np.linspace(1.5, 4.5, len(dates)), "treasury_10y": 3.0, "vix": 15 + 5 * np.sin(np.arange(len(dates)) / 30)})
    macro["term_spread"] = macro["treasury_10y"] - macro["tbill_3m"]
    # T13 joined the index part-way through; T12 left it.
    membership = pd.DataFrame({
        "ticker": tickers,
        "start": [pd.NaT] * 13 + [dates[200]],
        "end": [pd.NaT] * 12 + [dates[300], pd.NaT],
    })
    changes = pd.DataFrame({"date": [dates[300], dates[200]], "added": [None, "T13"], "added_name": [None, "T13 Inc"], "removed": ["T12", None], "removed_name": ["T12 Inc", None], "reason": ["Cap", "Cap"]})
    return SimpleNamespace(prices=prices, companies=companies, fundamentals=fundamentals, macro=macro, membership=membership, index_changes=changes)


@pytest.fixture(scope="session")
def live_world():
    return make_live_world()


@pytest.fixture(scope="session")
def live_features(live_world):
    return build_research_features(
        live_world.prices, live_world.companies, pd.DataFrame(columns=["ticker", "date", "title"]),
        pit_fundamentals=live_world.fundamentals, macro=live_world.macro, membership=live_world.membership,
    )


class FakeNewsFetcher:
    source = "Fake RSS"

    def fetch(self, ticker):
        return pd.DataFrame({
            "ticker": ticker,
            "date": pd.to_datetime(["2021-08-02 12:00", "2021-08-01 09:00"], utc=True),
            "title": [f"{ticker} beats estimates", f"{ticker} faces lawsuit"],
            "summary": [None, None], "source": ["Wire", "Wire"], "url": ["u1", "u2"],
        })


class FakeFilingsFetcher:
    def fetch(self, ticker, forms=("10-K", "10-Q", "8-K")):
        filings = pd.DataFrame({
            "ticker": ticker, "form": ["8-K", "10-Q", "10-K"], "filed": pd.to_datetime(["2021-07-30", "2021-07-28", "2021-02-10"]),
            "report_date": pd.to_datetime(["2021-07-30", "2021-06-30", "2020-12-31"]), "items": ["2.02", None, None], "url": ["a", "b", "c"],
        })
        return filings[filings["form"].isin(forms)]


@pytest.fixture(scope="session")
def live_research_data(live_world, live_features, tmp_path_factory):
    from sp500_agent.research import load_artifacts, run_research
    from sp500_agent.research_tools import ResearchData

    output = tmp_path_factory.mktemp("live_research")
    run = run_research(live_features, n_splits=3, output_dir=output, model_path=None, quality={"source": "live", "prices": {"tickers": 14, "end": "2021-08-11"}})
    return ResearchData(
        live_features, run.bundle, artifacts=load_artifacts(output),
        fundamentals=live_world.fundamentals, macro=live_world.macro, membership=live_world.membership,
        index_changes=live_world.index_changes, companies=live_world.companies, quality={"source": "live"},
        news_fetcher=FakeNewsFetcher(), filings_fetcher=FakeFilingsFetcher(),
    )
