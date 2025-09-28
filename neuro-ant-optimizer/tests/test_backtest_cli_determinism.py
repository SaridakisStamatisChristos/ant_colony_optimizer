from __future__ import annotations

from pathlib import Path

from neuro_ant_optimizer.backtest.backtest import main as backtest_main


def _run_cli(out_dir: Path) -> None:
    csv_path = Path("backtest/sample_returns.csv")
    args = [
        "--csv",
        str(csv_path),
        "--lookback",
        "5",
        "--step",
        "2",
        "--cov-model",
        "sample",
        "--objective",
        "sharpe",
        "--seed",
        "123",
        "--rf-bps",
        "50",
        "--out",
        str(out_dir),
        "--save-weights",
        "--skip-plot",
    ]
    backtest_main(args)


def test_backtest_cli_determinism(tmp_path: Path) -> None:
    first_out = tmp_path / "run1"
    second_out = tmp_path / "run2"

    _run_cli(first_out)
    _run_cli(second_out)

    for artifact in ("equity.csv", "rebalance_report.csv", "weights.csv"):
        first_bytes = (first_out / artifact).read_bytes()
        second_bytes = (second_out / artifact).read_bytes()
        assert first_bytes == second_bytes, f"{artifact} mismatch"
