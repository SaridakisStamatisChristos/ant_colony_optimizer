from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuro_ant_optimizer.backtest.backtest import main as backtest_main

pytestmark = pytest.mark.slow


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


def test_backtest_cli_manifest_records_deterministic(tmp_path: Path) -> None:
    out_dir = tmp_path / "dry"
    args = [
        "--csv",
        "backtest/sample_returns.csv",
        "--lookback",
        "5",
        "--step",
        "2",
        "--objective",
        "sharpe",
        "--seed",
        "123",
        "--out",
        str(out_dir),
        "--dry-run",
        "--deterministic",
        "--skip-plot",
    ]
    backtest_main(args)
    manifest = json.loads((out_dir / "run_config.json").read_text())
    assert manifest["deterministic_torch"] is True
