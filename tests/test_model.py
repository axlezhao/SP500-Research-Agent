import pandas as pd

from conftest import build_sample_features
from sp500_agent.config import HORIZON_DAYS
from sp500_agent.model import NUMERIC_FEATURES, score_latest, time_split, train_return_direction_model


def test_time_split_orders_dates_with_a_gap(sample_features):
    labelled = sample_features.dropna(subset=["target_up_5d"])
    train, test = time_split(labelled)
    dates = sorted(labelled["date"].unique())
    gap = dates.index(test["date"].min()) - dates.index(train["date"].max())
    assert gap == HORIZON_DAYS + 1


def test_random_walk_has_no_skill_on_the_holdout(tmp_path):
    # Sample prices are a random walk, so any real edge here would mean leakage.
    # The old random train/test split scored about 0.69 on this data.
    aucs = []
    for seed in range(1, 5):
        features = build_sample_features(tmp_path / str(seed), seed=seed)
        bundle, _ = train_return_direction_model(features, model_path=None)
        aucs.append(bundle["metrics"]["auc"])
    assert sum(aucs) / len(aucs) < 0.56


def test_snapshot_fundamentals_are_not_model_inputs():
    assert not {"market_cap", "pe_ratio", "revenue", "profit_margin", "debt_to_equity", "roe"} & set(NUMERIC_FEATURES)


def test_report_includes_baseline(sample_features):
    _, report = train_return_direction_model(sample_features, model_path=None)
    assert "always predicting" in report and "ROC AUC" in report


def test_score_latest_skips_stale_tickers_and_ranks(sample_features):
    bundle, _ = train_return_direction_model(sample_features, model_path=None)
    cutoff = sample_features["date"].max() - pd.Timedelta(days=30)
    stale = sample_features[(sample_features["ticker"] == "XOM") & (sample_features["date"] <= cutoff)]
    features = pd.concat([sample_features[sample_features["ticker"] != "XOM"], stale])
    scored = score_latest(features, bundle)
    assert "XOM" not in set(scored["ticker"])
    assert (scored["date"] == sample_features["date"].max()).all()
    assert scored["rank"].tolist() == list(range(1, len(scored) + 1))
    assert scored["up_probability_5d"].is_monotonic_decreasing
