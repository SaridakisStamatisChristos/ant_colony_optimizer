from __future__ import annotations

from importlib import import_module
from pathlib import Path

import math

import numpy as np
import pytest

bt = import_module("neuro_ant_optimizer.backtest.backtest")
from tests.conftest import _EXPECTED_REBALANCE_HEADER


class _Frame:
    def __init__(self, arr: np.ndarray, dates: np.ndarray, cols: list[str]):
        self._arr = arr
        self._idx = list(dates)
        self._cols = cols

    def to_numpy(self, dtype=float):
        return self._arr.astype(dtype)

    @property
    def index(self):  # pragma: no cover - simple accessor
        return self._idx

    @property
    def columns(self):  # pragma: no cover - simple accessor
        return self._cols


class _StubOptimizer:
    def __init__(self, weights_seq: list[np.ndarray]):
        self.weights_seq = weights_seq
        self.calls = 0
        self.cfg = type("Cfg", (), {"use_shrinkage": False, "shrinkage_delta": 0.0})()

    def optimize(self, *_, **__):
        weights = self.weights_seq[self.calls]
        self.calls += 1

        class _Result:
            def __init__(self, w: np.ndarray):
                self.weights = w
                self.feasible = True
                self.projection_iterations = 0

        return _Result(weights)


def test_rebalance_report_and_net_returns(tmp_path: Path, monkeypatch) -> None:
    n_periods = 12
    dates = np.array(
        [np.datetime64("2020-01-01") + np.timedelta64(i, "D") for i in range(n_periods)]
    )
    returns = np.array(
        [
            [0.01, 0.0],
            [0.02, -0.01],
            [0.0, 0.01],
            [0.03, 0.01],
            [0.02, 0.0],
            [0.01, 0.02],
            [0.0, -0.01],
            [0.01, 0.03],
            [0.0, 0.02],
            [0.02, 0.01],
            [0.01, -0.02],
            [0.03, 0.0],
        ],
        dtype=float,
    )
    frame = _Frame(returns, dates, ["A", "B"])

    weights_seq = [
        np.array([0.6, 0.4], dtype=float),
        np.array([0.5, 0.5], dtype=float),
        np.array([0.7, 0.3], dtype=float),
    ]
    stub = _StubOptimizer(weights_seq)
    monkeypatch.setattr(
        bt, "_build_optimizer", lambda n_assets, seed, risk_free_rate=0.0: stub
    )

    results = bt.backtest(
        frame,
        lookback=3,
        step=3,
        seed=0,
        tx_cost_bps=10.0,
        tx_cost_mode="posthoc",
    )

    records = results["rebalance_records"]
    assert len(records) == 3

    tc = 10.0 / 1e4
    for idx, record in enumerate(records):
        start = 3 + 3 * idx
        stop = start + 3
        block = returns[start:stop]
        w = weights_seq[idx]
        gross = block @ w
        expected_turn = float(np.abs(w).sum()) if idx == 0 else float(np.abs(w - weights_seq[idx - 1]).sum())
        expected_gross = float(np.prod(1.0 + gross) - 1.0)
        tx_block = gross - (tc * expected_turn / max(1, gross.size))
        expected_net_tx = float(np.prod(1.0 + tx_block) - 1.0)

        assert record["turnover"] == pytest.approx(expected_turn)
        assert record["turnover_pre_decay"] == pytest.approx(expected_turn)
        assert record["turnover_post_decay"] == pytest.approx(expected_turn)
        assert record["tx_cost"] == pytest.approx(tc * expected_turn)
        assert record["gross_ret"] == pytest.approx(expected_gross)
        assert record["net_tx_ret"] == pytest.approx(expected_net_tx)
        assert record["net_slip_ret"] == pytest.approx(expected_net_tx)
        assert record["nt_band_hits"] == 0
        assert record["participation_breaches"] == 0
        assert record["sector_breaches"] == 0
        assert record["sector_penalty"] == pytest.approx(0.0)
        assert record["active_breaches"] == 0
        assert record["group_breaches"] == 0
        assert record["factor_bound_breaches"] == 0
        assert record["factor_inf_norm"] == pytest.approx(0.0)
        assert record["factor_missing"] is False
        assert record["first_violation"] is None
        assert record["feasible"] is True
        assert record["projection_iterations"] == 0
        block_returns = block @ w
        ann_factor = math.sqrt(252)
        block_mean = float(block_returns.mean())
        block_std = float(block_returns.std())
        expected_block_sharpe = 0.0
        if block_std > 1e-12:
            expected_block_sharpe = float((block_mean * 252) / (block_std * ann_factor))
        downside = block_returns[block_returns < 0]
        expected_block_sortino = 0.0
        if downside.size:
            downside_vol = float(downside.std() * ann_factor)
            if downside_vol > 1e-12:
                expected_block_sortino = float((block_mean * 252) / downside_vol)
        assert record["block_sharpe"] == pytest.approx(expected_block_sharpe)
        assert record["block_sortino"] == pytest.approx(expected_block_sortino)
        assert record["block_info_ratio"] is None
        assert record["block_tracking_error"] is None
        assert record["slippage_cost"] == pytest.approx(0.0)
        assert record["pre_trade_ok"] is True
        assert record["pre_trade_breach_count"] == 0
        assert record["post_trade_breach_count"] == 0
        assert record["breach_count"] == 0
        assert record["first_breach"] is None
        assert record["pre_trade_reasons"] == ""
        assert record["post_trade_reasons"] == ""
        assert record["warm_applied"] is False
        assert record["decay"] == pytest.approx(0.0)

    # Ensure per-period net returns align with report calculations
    np.testing.assert_allclose(
        results["gross_returns"],
        np.concatenate([(returns[3:6] @ weights_seq[0]), (returns[6:9] @ weights_seq[1]), (returns[9:12] @ weights_seq[2])]),
    )

    report_path = tmp_path / "rebalance_report.csv"
    bt._write_rebalance_report(report_path, results)
    text = report_path.read_text().splitlines()
    expected_header = ",".join(_EXPECTED_REBALANCE_HEADER)
    assert text[0] == expected_header


def test_active_bounds_fall_back_for_missing_benchmark(monkeypatch) -> None:
    n_periods = 9
    dates = np.array(
        [np.datetime64("2020-01-01") + np.timedelta64(i, "D") for i in range(n_periods)]
    )
    returns = np.array(
        [[0.01, 0.0, 0.02], [0.02, -0.01, 0.01], [0.0, 0.01, 0.0],
         [0.03, 0.01, 0.02], [0.02, 0.0, 0.03], [0.01, 0.02, -0.01],
         [0.0, -0.01, 0.04], [0.01, 0.03, 0.01], [0.0, 0.02, 0.02]],
        dtype=float,
    )
    frame = _Frame(returns, dates, ["A", "B", "C"])

    weights_seq = [
        np.array([0.1, 0.1, 0.8], dtype=float),
        np.array([0.1, 0.05, 0.85], dtype=float),
    ]
    stub = _StubOptimizer(weights_seq)
    monkeypatch.setattr(
        bt, "_build_optimizer", lambda n_assets, seed, risk_free_rate=0.0: stub
    )

    results = bt.backtest(
        frame,
        lookback=3,
        step=3,
        seed=0,
        benchmark_weights={"A": 0.1, "B": 0.1},
        active_min=-0.1,
        active_max=0.1,
    )

    for record in results["rebalance_records"]:
        assert record["active_breaches"] == 0
        assert record["group_breaches"] == 0
        assert record["first_violation"] is None

    manifest = results["constraint_manifest"]
    assert manifest["benchmark_weights"]["C"] is None

