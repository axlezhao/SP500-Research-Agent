import json

from sp500_agent import charts
from sp500_agent.research import ARTIFACT_FILES, load_artifacts


def test_research_run_saves_every_artifact(research_run, research_dir):
    for name in ARTIFACT_FILES.values():
        assert (research_dir / name).exists(), name
    loaded = load_artifacts(research_dir)
    assert len(loaded["model_comparison"]) == 3
    assert set(loaded["backtest_summary"]["strategy"]) == {"long_only", "long_short", "benchmark"}
    assert json.loads((research_dir / "eda.json").read_text())["tickers"] == 8


def test_best_model_is_the_one_saved(research_run):
    assert research_run.bundle["model_name"] == research_run.best_model == research_run.comparison.iloc[0]["model"]
    assert research_run.bundle["metrics"]["auc_mean"] == research_run.comparison.iloc[0]["auc_mean"]


def test_report_covers_data_models_backtest_and_caveats(research_run):
    for heading in ["## 1. Data", "## 2. Model comparison", "## 3. Backtest", "## 4. Caveats", "Survivorship bias"]:
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
