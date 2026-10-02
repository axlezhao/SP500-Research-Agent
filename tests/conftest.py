from __future__ import annotations

from pathlib import Path

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
