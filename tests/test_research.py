import json

from sp500_agent import charts
from sp500_agent.research import ARTIFACT_FILES, load_artifacts


# Attribution needs 30+ names per date (the 8-stock sample has fewer); French factors need the live source.
OPTIONAL_ARTIFACTS = {"attribution", "factor_returns", "french_attribution"}


def test_research_run_saves_every_artifact(research_run, research_dir):
    for key, name in ARTIFACT_FILES.items():
        assert key in OPTIONAL_ARTIFACTS or (research_dir / name).exists(), name
    loaded = load_artifacts(research_dir)
    assert len(loaded["model_comparison"]) == 4
    assert set(loaded["constructions"]["construction"]) == {"Top 20%, no buffer (original)", "Buffer: sell below top 40%", "Buffer + 3-day score average", "Optimiser: cost-aware, beta/size/value neutral"}
    assert loaded["constructions"]["production"].sum() == 1
    assert loaded["robustness"]["trials_counted"] >= 4 and "long_short" in loaded["robustness"]
    assert set(loaded["backtest_summary"]["strategy"]) == {"long_only", "long_short", "benchmark"}
    assert json.loads((research_dir / "eda.json").read_text())["tickers"] == 8


def test_best_model_is_the_one_saved(research_run):
    assert research_run.bundle["model_name"] == research_run.best_model == research_run.comparison.iloc[0]["model"]
    assert research_run.bundle["metrics"]["auc_mean"] == research_run.comparison.iloc[0]["auc_mean"]


def test_report_covers_data_models_backtest_and_caveats(research_run):
    for heading in ["## 1. Data", "## 2. Model comparison", "## 3. Backtest", "Where the returns come from", "## 4. Experiments", "## 5. Is it luck?", "## 6. Caveats", "Survivorship bias", "Portfolio construction", "Does the rebalance day matter?", "IC decay"]:
        assert heading in research_run.report


def test_charts_build_from_artifacts(research_run):
    specs = [
        charts.equity_curve(research_run.backtest.returns),
        charts.drawdown_chart(research_run.backtest.returns, mode="dark"),
        charts.quintile_chart(research_run.backtest.quantile_returns),
        charts.fold_auc_chart(research_run.fold_metrics),
        charts.importance_chart(research_run.importance),
        charts.calibration_chart(research_run.calibration),
    ]
    for chart in specs:
        assert chart.to_dict()


def test_experiments_compare_setups_and_mark_production(research_run):
    experiments = research_run.experiments
    assert experiments["production"].sum() == 1
    assert set(experiments["horizon"]) == {5, 21}
    # The 8-stock sample has no sector with 4+ members, so sector-relative setups are reported as not evaluated.
    skipped = experiments[experiments["relative_to"] == "sector"]
    assert skipped["auc"].isna().all() and skipped["note"].str.contains("Not enough").all()
    assert "Not evaluated: 5-day, vs. own sector" in research_run.report
