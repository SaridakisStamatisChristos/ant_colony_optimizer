"""Pytest configuration helpers for the test suite."""
from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Callable, List

import pytest

_HERE = Path(__file__).resolve()
_PKG_ROOT = _HERE.parents[1]
_SRC = _PKG_ROOT / "src"

if _SRC.is_dir():
    p = str(_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)

bt = import_module("neuro_ant_optimizer.backtest.backtest")

_EXPECTED_REBALANCE_HEADER: List[str] = [
    "date",
    "gross_ret",
    "net_tx_ret",
    "net_slip_ret",
    "turnover",
    "turnover_pre_decay",
    "turnover_post_decay",
    "tx_cost",
    "slippage_cost",
    "nt_band_hits",
    "participation_breaches",
    "sector_breaches",
    "sector_penalty",
    "active_breaches",
    "group_breaches",
    "factor_bound_breaches",
    "factor_inf_norm",
    "factor_missing",
    "first_violation",
    "feasible",
    "projection_iterations",
    "block_sharpe",
    "block_sortino",
    "block_info_ratio",
    "block_tracking_error",
    "pre_trade_ok",
    "pre_trade_breach_count",
    "post_trade_breach_count",
    "breach_count",
    "first_breach",
    "pre_trade_reasons",
    "post_trade_reasons",
    "warm_applied",
    "decay",
]


@pytest.fixture
def assert_backtest_artifacts() -> Callable[[Path], List[str]]:
    def _assert(path: Path) -> List[str]:
        manifest_path = path / "run_config.json"
        assert manifest_path.exists(), "run_config.json missing from backtest output"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert (
            manifest.get("schema_version") == bt.SCHEMA_VERSION
        ), "schema version drift detected"

        report_path = path / "rebalance_report.csv"
        assert report_path.exists(), "rebalance_report.csv missing from backtest output"
        header = report_path.read_text(encoding="utf-8").splitlines()[:1]
        assert header, "rebalance_report.csv is empty"
        assert header[0] == ",".join(_EXPECTED_REBALANCE_HEADER)
        return list(_EXPECTED_REBALANCE_HEADER)

    return _assert
