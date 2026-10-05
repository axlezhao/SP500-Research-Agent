"""Inference tools: Newey-West, HAC regression, bootstrap, deflated Sharpe and backtest-overfitting probability."""

import numpy as np
import pandas as pd
import pytest

from sp500_agent.stats import (
    bootstrap_alpha,
    bootstrap_performance,
    deflated_sharpe,
    expected_max_sharpe,
    hac_regression,
    newey_west_tstat,
    probabilistic_sharpe,
    probability_of_backtest_overfitting,
    stationary_bootstrap_indices,
)


def test_newey_west_matches_plain_t_for_independent_data_and_shrinks_for_overlap():
    rng = np.random.default_rng(1)
    iid = rng.normal(0.05, 1, 5000)
    plain = iid.mean() / iid.std(ddof=1) * np.sqrt(len(iid))
    assert newey_west_tstat(iid, lags=0) == pytest.approx(plain, rel=0.01)
    overlapping = np.convolve(rng.normal(0.05, 1, 5000), np.ones(10), mode="valid")
    naive = overlapping.mean() / overlapping.std(ddof=1) * np.sqrt(len(overlapping))
    assert abs(newey_west_tstat(overlapping, lags=10)) < abs(naive) / 2


def test_hac_regression_recovers_coefficients():
    rng = np.random.default_rng(2)
    x = rng.normal(size=500)
    y = 0.5 + 2 * x + rng.normal(size=500)
    coef, se_ols, se_hac = hac_regression(y, np.column_stack([np.ones(500), x]))
    assert coef == pytest.approx([0.5, 2.0], abs=0.15)
    assert se_hac[1] == pytest.approx(se_ols[1], rel=0.3)


def test_stationary_bootstrap_keeps_runs_together():
    idx = stationary_bootstrap_indices(200, 50, mean_block=20, seed=0)
    steps = np.diff(idx, axis=1)
    assert idx.shape == (50, 200) and idx.min() >= 0 and idx.max() < 200
    assert ((steps == 1) | (steps == -199)).mean() > 0.9  # mostly consecutive


def test_bootstrap_intervals_cover_the_truth():
    rng = np.random.default_rng(3)
    returns = pd.DataFrame({"good": rng.normal(0.01, 0.02, 400), "flat": rng.normal(0.0, 0.02, 400)})
    table = bootstrap_performance(returns, ["good", "flat"], periods_per_year=50, reps=500).set_index("strategy")
    assert table.loc["good", "sharpe_low"] > 0 and table.loc["good", "prob_sharpe_not_positive"] < 0.05
    assert table.loc["flat", "sharpe_low"] < 0 < table.loc["flat", "sharpe_high"]
    factors = pd.DataFrame({"MKT": rng.normal(0, 0.02, 400)})
    y = pd.Series(0.002 + factors["MKT"] + rng.normal(0, 0.005, 400))
    ci = bootstrap_alpha(y, factors, periods_per_year=50, reps=300)
    assert ci["alpha_low"] < 0.1 < ci["alpha_high"]
    assert bootstrap_alpha(y, factors.assign(SMB=np.nan), periods_per_year=50, reps=50)  # an unbuildable factor is skipped


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.normal(0.003, 0.02, 300))
    sr = returns.mean() / returns.std()
    assert probabilistic_sharpe(sr, 300, 0.0, 3.0) > probabilistic_sharpe(sr, 300, 0.0, 3.0, benchmark=0.1)
    few = deflated_sharpe(returns, rng.normal(0, 0.05, 3), n_trials=3)
    many = deflated_sharpe(returns, rng.normal(0, 0.05, 200), n_trials=200)
    assert many["deflated_sharpe"] < few["deflated_sharpe"] < few["probabilistic_sharpe"]
    assert expected_max_sharpe([0.1, 0.1], 2) == 0.0  # no dispersion, no luck


def test_overfitting_probability_is_high_for_noise_and_low_for_a_real_edge():
    # For pure noise, picking the best backtest is a coin flip out of sample: PBO about 0.5 on average.
    noise_pbo = [probability_of_backtest_overfitting(pd.DataFrame(np.random.default_rng(seed).normal(0, 0.01, (480, 12))))["pbo"] for seed in range(12)]
    assert 0.3 < np.mean(noise_pbo) < 0.65
    rng = np.random.default_rng(5)
    noise = pd.DataFrame(rng.normal(0, 0.01, (480, 12)))
    edge = noise.copy()
    edge[0] = rng.normal(0.004, 0.01, 480)  # one strategy really is better
    edge[1] = rng.normal(0.002, 0.01, 480)
    assert probability_of_backtest_overfitting(edge)["pbo"] < 0.1
    assert probability_of_backtest_overfitting(noise[[0]]) == {}  # one strategy: nothing to choose between
