import json

from sp500_agent import charts
from sp500_agent.research import ARTIFACT_FILES, load_artifacts


OPTIONAL_ARTIFACTS = {"attribution", "factor_returns"}  # need 30+ names per date; the 8-stock sample has fewer


def test_research_run_saves_every_artifact(research_run, research_dir):
    for key, name in ARTIFACT_FILES.items():
        assert key in OPTIONAL_ARTIFACTS or (research_dir / name).exists(), name
    loaded = load_artifacts(research_dir)
    assert len(loaded["model_comparison"]) == 3
    assert set(loaded["backtest_summary"]["strategy"]) == {"long_only", "long_short", "benchmark"}
    assert json.loads((research_dir / "eda.json").read_text())["tickers"] == 8


def test_best_model_is_the_one_saved(research_run):
    assert research_run.bundle["model_name"] == research_run.best_model == research_run.comparison.iloc[0]["model"]
    assert research_run.bundle["metrics"]["auc_mean"] == research_run.comparison.iloc[0]["auc_mean"]


def test_report_covers_data_models_backtest_and_caveats(research_run):
    for heading in ["## 1. Data", "## 2. Model comparison", "## 3. Backtest", "Where the returns come from", "## 4. Experiments", "## 5. Caveats", "Survivorship bias"]:
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
