import csv
import json
from pathlib import Path

from neuro_ant_optimizer.backtest.backtest import main as backtest_main


def _run_backtest(out_dir: Path, extra_args: list[str]) -> None:
    base_args = [
        "--csv",
        "backtest/sample_returns.csv",
        "--lookback",
        "3",
        "--step",
        "2",
        "--objective",
        "sharpe",
        "--seed",
        "123",
        "--out",
        str(out_dir),
        "--skip-plot",
    ]
    backtest_main(base_args + extra_args)


def test_bootstrap_confidence_interval_shrinks(tmp_path: Path) -> None:
    small_dir = tmp_path / "small"
    large_dir = tmp_path / "large"

    _run_backtest(small_dir, ["--bootstrap", "10", "--block", "2"])
    _run_backtest(large_dir, ["--bootstrap", "40", "--block", "2"])

    small_ci = json.loads((small_dir / "metrics_ci.json").read_text())
    large_ci = json.loads((large_dir / "metrics_ci.json").read_text())

    small_width = small_ci["sharpe"]["upper"] - small_ci["sharpe"]["lower"]
    large_width = large_ci["sharpe"]["upper"] - large_ci["sharpe"]["lower"]

    assert large_width <= small_width + 1e-9


def test_cv_results_written(tmp_path: Path) -> None:
    cv_dir = tmp_path / "cv"
    _run_backtest(cv_dir, ["--cv", "k=3"])

    csv_path = cv_dir / "cv_results.csv"
    assert csv_path.exists()

    rows = list(csv.reader(csv_path.open()))
    assert rows
    header = rows[0]
    assert header[:3] == ["fold", "start", "end"]
