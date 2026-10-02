import numpy as np
import pandas as pd

from conftest import build_sample_features
from sp500_agent.config import HORIZON_DAYS
from sp500_agent.model import TARGET
from sp500_agent.validation import (
    calibration_table,
    ic_summary,
    information_coefficients,
    walk_forward,
    walk_forward_folds,
)


def test_folds_are_ordered_contiguous_and_embargoed():
    dates = pd.bdate_range("2024-01-01", periods=200)
    folds = walk_forward_folds(dates, n_splits=4)
    position = {d: i for i, d in enumerate(dates)}
    for fold in folds:
        # Training labels look HORIZON_DAYS ahead, so the last training date sits HORIZON_DAYS + 1 sessions before the test.
        assert position[fold.test_start] - position[fold.train_end] == HORIZON_DAYS + 1
    for earlier, later in zip(folds, folds[1:]):
        assert position[later.test_start] == position[earlier.test_end] + 1
    assert folds[-1].test_end == dates[-1]


def test_predictions_are_out_of_sample(sample_features):
    result = walk_forward(sample_features, "logistic_regression", n_splits=3)
    folds = result.fold_metrics.set_index("fold")
    merged = result.predictions.join(folds[["train_end", "test_start"]], on="fold")
    assert (merged["date"] > merged["train_end"]).all()
    assert result.predictions["probability"].between(0, 1).all()
    assert not result.predictions.duplicated(["date", "ticker"]).any()


def test_random_walk_has_no_skill_out_of_sample(tmp_path):
    # Sample prices are a random walk, so a real edge here would mean leakage.
    # The original random train/test split scored about 0.69 on this data.
    aucs = [walk_forward(build_sample_features(tmp_path / str(s), seed=s), "random_forest").fold_metrics["auc"].mean() for s in range(1, 5)]
    assert np.mean(aucs) < 0.56


def test_information_coefficient_of_a_perfect_signal_is_one():
    df = pd.DataFrame({"date": np.repeat(pd.bdate_range("2024-01-01", periods=10), 6), "ticker": list("ABCDEF") * 10})
    df["future_return_5d"] = np.random.default_rng(0).normal(size=len(df))
    ic = information_coefficients(df.assign(score=df["future_return_5d"] * 3), "score")
    assert np.allclose(ic, 1.0) and len(ic) == 10


def test_ic_tstat_uses_non_overlapping_dates():
    ic = pd.Series(np.r_[np.full(50, 0.1), np.full(50, 0.3)] + np.tile([0.01, -0.01], 50))
    summary = ic_summary(ic, step=5)
    assert summary["ic_days"] == 100
    independent = ic.iloc[::5]
    assert np.isclose(summary["ic_tstat"], independent.mean() / independent.std() * np.sqrt(len(independent)))


def test_calibration_buckets_are_sorted_by_prediction():
    rng = np.random.default_rng(1)
    preds = pd.DataFrame({"probability": rng.uniform(size=1000)})
    preds[TARGET] = (rng.uniform(size=1000) < preds["probability"]).astype(float)
    table = calibration_table(preds)
    assert len(table) == 10 and table["rows"].sum() == 1000
    assert table["mean_predicted"].is_monotonic_increasing
    # Well-calibrated synthetic data: actual rate tracks prediction.
    assert np.corrcoef(table["mean_predicted"], table["actual_up_rate"])[0, 1] > 0.9
