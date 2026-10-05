"""Research process: training on independent rows, tuning inside the window, the baseline, holdout and forward test."""

import json

import numpy as np
import pandas as pd
import pytest

from sp500_agent.config import ResearchSpec
from sp500_agent.model import COMPOSITE_SIGNS, make_pipeline, non_overlapping, predict_score, tune_params
from sp500_agent.research import evaluate_holdout, research_window
from sp500_agent.sentiment import lexicon_score
from sp500_agent.validation import walk_forward, walk_forward_folds


def test_training_uses_one_date_per_forward_window():
    rows = pd.DataFrame({"date": np.repeat(pd.bdate_range("2024-01-01", periods=23), 2), "ticker": ["A", "B"] * 23})
    thinned = non_overlapping(rows, 5)
    dates = np.sort(thinned["date"].unique())
    assert len(dates) == 5 and dates[-1] == rows["date"].max()  # anchored at the latest date
    assert (np.diff(dates).astype("timedelta64[D]").astype(int) >= 5).all()


def test_composite_is_a_fixed_sign_average():
    numeric = ["momentum_12_1_z", "volatility_20d_z", "ma_gap_50d_z"]
    X = pd.DataFrame({"momentum_12_1_z": [2.0, -2.0, 0.0], "volatility_20d_z": [-1.0, 1.0, 0.0], "ma_gap_50d_z": [5.0, 5.0, -5.0]})
    pipeline = make_pipeline("composite", numeric, [], kind="rank").fit(X, np.zeros(3))
    scores = predict_score(pipeline, X)
    assert scores[0] > scores[2] > scores[1]  # strong momentum and low volatility first; ma_gap has no sign and is ignored
    assert COMPOSITE_SIGNS["volatility_20d_z"] == -1 and "ma_gap_50d_z" not in COMPOSITE_SIGNS


def test_hyperparameters_are_chosen_inside_the_training_window(live_features):
    spec = ResearchSpec(5, "market", "rank")
    labelled = live_features[live_features["in_index"]].dropna(subset=[spec.target_column])
    numeric = [c for c in ["momentum_60d_z", "return_20d_z", "volatility_20d_z"] if c in labelled]
    assert tune_params("ridge", labelled, numeric, [], spec) in ({"alpha": 10.0}, {"alpha": 1e3}, {"alpha": 1e5})
    assert tune_params("composite", labelled, numeric, [], spec) == {}


def test_quarterly_refits_and_fixed_test_start():
    dates = pd.bdate_range("2020-01-01", periods=400)
    folds = walk_forward_folds(dates, block_size=63)
    assert len(folds) == int(np.ceil(200 / 63)) and folds[0].test_start == dates[200]
    later = walk_forward_folds(dates, block_size=63, test_start=pd.Timestamp("2021-03-01"))
    assert later[0].test_start == dates[dates >= "2021-03-01"][0]


def test_research_window_hides_the_holdout_and_labels_that_reach_into_it(live_features):
    cut = live_features["date"].sort_values().unique()[350]
    research = research_window(live_features, cut)
    assert research["date"].max() < cut
    sessions = pd.DatetimeIndex(live_features["date"].unique()).sort_values()
    last_known = sessions[sessions.searchsorted(cut) - 5 - 1 - 1]  # a 5-day label plus the 1-session lag must end before the cut
    known = research.dropna(subset=["tradable_return_5d"])["date"].max()
    assert known <= last_known
    assert research.dropna(subset=["target_rank_market_21d"])["date"].max() < known
    assert research_window(live_features, "none") is live_features


def test_holdout_evaluation_is_logged(live_features, tmp_path):
    log = tmp_path / "holdout.jsonl"
    cut = live_features["date"].sort_values().unique()[330]
    first = evaluate_holdout(live_features, "ridge", {"alpha": 1000.0}, holdout_start=cut, retrain_every=40, log_path=log)
    assert first["previous_evaluations"] == 0 and pd.Timestamp(first["holdout_start"]) == pd.Timestamp(cut)
    second = evaluate_holdout(live_features, "ridge", {"alpha": 1000.0}, holdout_start=cut, retrain_every=40, log_path=log)
    assert second["previous_evaluations"] == 1
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(lines) == 2 and lines[0]["model"] == "ridge" and "ic_mean" in lines[0]


def test_walk_forward_with_fixed_params_starts_at_the_holdout(live_features):
    cut = live_features["date"].sort_values().unique()[330]
    result = walk_forward(live_features, "ridge", spec=ResearchSpec(5, "market", "rank"), retrain_every=40, params={"alpha": 10.0}, test_start=cut)
    assert result.predictions["date"].min() >= cut
    assert all(p == {"alpha": 10.0} for p in result.params)


def test_forward_test_records_and_scores_rankings(live_features, tmp_path):
    from sp500_agent.forward import evaluate_signals, record_signals
    from sp500_agent.model import score_latest, train_final_model

    spec = ResearchSpec(5, "market", "rank")
    past = research_window(live_features, live_features["date"].sort_values().unique()[-29])  # as if run 29 sessions ago
    bundle = train_final_model(past, "ridge", model_path=None, spec=spec, tune=False)
    path = record_signals(score_latest(past, bundle), bundle, tmp_path)
    assert path.exists()
    assert evaluate_signals(past, spec, tmp_path).empty  # the future isn't known yet
    scored = evaluate_signals(live_features, spec, tmp_path)
    assert len(scored) == 1 and scored["ic"].between(-1, 1).all()


@pytest.mark.parametrize("text,expected", [
    ("Company beats estimates and raises guidance", 1.0),
    ("Shares plunge after earnings miss", -1.0),
    ("Company did not miss estimates", 1.0),
    ("Annual meeting scheduled", 0.0),
])
def test_finance_lexicon_with_negation(text, expected):
    assert lexicon_score(text) == expected
