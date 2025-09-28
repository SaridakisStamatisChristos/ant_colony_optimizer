from __future__ import annotations

import json
from pathlib import Path

from neuro_ant_optimizer.backtest.backtest import main as backtest_main


def test_cli_json_logging_schema(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_out"
    log_path = tmp_path / "rebalance.jsonl"

    args = [
        "--csv",
        "backtest/sample_returns.csv",
        "--lookback",
        "4",
        "--step",
        "2",
        "--cov-model",
        "sample",
        "--objective",
        "sharpe",
        "--seed",
        "123",
        "--out",
        str(out_dir),
        "--skip-plot",
        "--log-json",
        str(log_path),
        "--progress",
    ]

    backtest_main(args)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2

    expected_top_keys = {
        "date",
        "seed",
        "objective",
        "cov_model",
        "costs",
        "turnover",
        "feasible",
        "breaches",
        "block",
        "timings",
    }
    expected_cost_keys = {"tx", "slippage"}
    expected_breach_keys = {"active", "group", "factor", "sector"}
    expected_block_keys = {"sharpe", "sortino", "ir", "te"}
    expected_timing_keys = {"cov_ms", "opt_ms"}

    for raw_line in lines[:2]:
        payload = json.loads(raw_line)
        assert set(payload) == expected_top_keys
        assert isinstance(payload["date"], str)
        assert payload["seed"] == 123
        assert payload["objective"] == "sharpe"
        assert payload["cov_model"] == "sample"
        assert isinstance(payload["turnover"], float)
        assert isinstance(payload["feasible"], bool)

        costs = payload["costs"]
        assert set(costs) == expected_cost_keys
        for key in expected_cost_keys:
            assert isinstance(costs[key], float)

        breaches = payload["breaches"]
        assert set(breaches) == expected_breach_keys
        for key in expected_breach_keys:
            assert isinstance(breaches[key], int)

        block = payload["block"]
        assert set(block) == expected_block_keys
        assert isinstance(block["sharpe"], float)
        assert isinstance(block["sortino"], float)
        assert (block["ir"] is None) or isinstance(block["ir"], float)
        assert (block["te"] is None) or isinstance(block["te"], float)

        timings = payload["timings"]
        assert set(timings) == expected_timing_keys
        for key in expected_timing_keys:
            assert isinstance(timings[key], float)
            assert timings[key] >= 0.0
