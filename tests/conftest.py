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
