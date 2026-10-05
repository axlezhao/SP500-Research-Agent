import pandas as pd
import pytest

from sp500_agent.config import ResearchSpec
from sp500_agent.model import NUMERIC_FEATURES, make_pipeline, models_for, score_latest, train_final_model


def test_snapshot_fundamentals_are_not_model_inputs():
    assert not {"market_cap", "pe_ratio", "revenue", "profit_margin", "debt_to_equity", "roe"} & set(NUMERIC_FEATURES)


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        make_pipeline("svm", ["return_5d"], [])


def test_models_match_the_target_kind():
    assert "logistic_regression" in models_for(ResearchSpec(5, "market")) and "ridge" not in models_for(ResearchSpec(5, "market"))
    assert "ridge" in models_for(ResearchSpec(5, "market", "rank")) and "logistic_regression" not in models_for(ResearchSpec(5, "market", "rank"))
    assert "composite" in models_for(ResearchSpec(5, "market")) and "composite" in models_for(ResearchSpec(5, "market", "rank"))
    with pytest.raises(ValueError, match="can't learn a rank target"):
        make_pipeline("logistic_regression", ["return_5d_z"], [], kind="rank")


def test_final_model_records_what_it_was_trained_on(sample_features):
    bundle = train_final_model(sample_features, "ridge", validation={"auc_mean": 0.5}, model_path=None)
    assert bundle["model_name"] == "ridge" and bundle["spec"]["target"] == "rank"
    assert bundle["params"] in ({"alpha": 10.0}, {"alpha": 1e3}, {"alpha": 1e5})  # chosen inside the training window
    assert bundle["metrics"] == {"auc_mean": 0.5}
    binary = train_final_model(sample_features, "logistic_regression", model_path=None, spec=ResearchSpec(5, "market"))
    assert binary["model_name"] == "logistic_regression"
    # Trained through the last date with a known 5-day outcome, not the last price date.
    assert bundle["trained_through"] < sample_features["date"].max()


def test_score_latest_skips_stale_tickers_and_ranks(sample_features):
    bundle = train_final_model(sample_features, "ridge", model_path=None)
    cutoff = sample_features["date"].max() - pd.Timedelta(days=30)
    stale = sample_features[(sample_features["ticker"] == "XOM") & (sample_features["date"] <= cutoff)]
    features = pd.concat([sample_features[sample_features["ticker"] != "XOM"], stale])
    scored = score_latest(features, bundle)
    assert "XOM" not in set(scored["ticker"])
    assert (scored["date"] == sample_features["date"].max()).all()
    assert scored["rank"].tolist() == list(range(1, len(scored) + 1))
    assert scored["model_probability"].is_monotonic_decreasing
