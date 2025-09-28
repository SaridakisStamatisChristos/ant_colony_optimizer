import math
from importlib import import_module

import numpy as np
import pytest

bt = import_module("neuro_ant_optimizer.backtest.backtest")


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


def _dates(n: int) -> np.ndarray:
    return np.array([np.datetime64("2020-01-01") + np.timedelta64(i, "D") for i in range(n)])


def test_sharpe_sortino_and_block_metrics_use_risk_free(monkeypatch) -> None:
    returns = np.array(
        [[0.05, 0.0], [0.02, 0.0], [0.04, 0.0], [0.01, 0.0], [0.03, 0.0], [0.06, 0.0]],
        dtype=float,
    )
    frame = _Frame(returns, _dates(len(returns)), ["A", "B"])
    stub = _StubOptimizer(
        [np.array([1.0, 0.0], dtype=float), np.array([1.0, 0.0], dtype=float)]
    )
    monkeypatch.setattr(
        bt, "_build_optimizer", lambda n_assets, seed, risk_free_rate=0.0: stub
    )

    annual_rf = 0.10
    trading_days = 2
    result = bt.backtest(
        frame,
        lookback=2,
        step=2,
        seed=0,
        risk_free_rate=annual_rf,
        trading_days=trading_days,
    )

    periodic_rf = (1.0 + annual_rf) ** (1.0 / trading_days) - 1.0
    realized = result["returns"]
    excess = realized - periodic_rf
    ann_factor = math.sqrt(trading_days)
    ann_vol = float(np.std(realized) * ann_factor)
    ann_excess = float(excess.mean() * trading_days)
    expected_sharpe = 0.0 if ann_vol <= 1e-12 else ann_excess / ann_vol
    assert result["sharpe"] == pytest.approx(expected_sharpe)

    negatives = excess[excess < 0]
    expected_sortino = 0.0
    if negatives.size:
        downside = float(negatives.std() * ann_factor)
        if downside > 1e-12:
            expected_sortino = ann_excess / downside
    assert result["sortino"] == pytest.approx(expected_sortino)
    assert result["periodic_risk_free"] == pytest.approx(periodic_rf)
    assert result["trading_days"] == trading_days

    first_block = realized[:2]
    excess_block = first_block - periodic_rf
    block_std = float(np.std(first_block))
    block_mean_excess = float(excess_block.mean())
    expected_block_sharpe = 0.0
    if block_std > 1e-12:
        expected_block_sharpe = (block_mean_excess * trading_days) / (block_std * ann_factor)
    block_negatives = excess_block[excess_block < 0]
    expected_block_sortino = 0.0
    if block_negatives.size:
        block_downside = float(block_negatives.std() * ann_factor)
        if block_downside > 1e-12:
            expected_block_sortino = (block_mean_excess * trading_days) / block_downside

    record = result["rebalance_records"][0]
    assert record["block_sharpe"] == pytest.approx(expected_block_sharpe)
    assert record["block_sortino"] == pytest.approx(expected_block_sortino)


def test_info_ratio_annualization(monkeypatch) -> None:
    returns = np.array(
        [[0.03, 0.0], [0.02, 0.0], [0.05, 0.0], [0.01, 0.0], [0.04, 0.0], [0.00, 0.0]],
        dtype=float,
    )
    benchmark = np.array(
        [[0.02], [0.01], [0.03], [0.02], [0.02], [0.01]],
        dtype=float,
    )
    frame = _Frame(returns, _dates(len(returns)), ["A", "B"])
    bench_frame = _Frame(benchmark, _dates(len(benchmark)), ["B"])
    stub = _StubOptimizer(
        [np.array([1.0, 0.0], dtype=float), np.array([1.0, 0.0], dtype=float)]
    )
    monkeypatch.setattr(
        bt, "_build_optimizer", lambda n_assets, seed, risk_free_rate=0.0: stub
    )

    trading_days = 260
    annual_rf = 0.01
    result = bt.backtest(
        frame,
        lookback=2,
        step=2,
        seed=0,
        benchmark=bench_frame,
        risk_free_rate=annual_rf,
        trading_days=trading_days,
    )

    realized = result["returns"]
    benchmark_realized = result["benchmark_returns"]
    assert benchmark_realized is not None
    active = realized - benchmark_realized
    ann_factor = math.sqrt(trading_days)
    te = float(np.std(active) * ann_factor)
    active_mean = float(active.mean() * trading_days)
    expected_ir = 0.0 if te <= 1e-12 else active_mean / te
    assert result["tracking_error"] == pytest.approx(te)
    assert result["info_ratio"] == pytest.approx(expected_ir)

    first_block_active = active[:2]
    block_te = float(np.std(first_block_active) * ann_factor)
    block_mean = float(first_block_active.mean() * trading_days)
    expected_block_ir = 0.0 if block_te <= 1e-12 else block_mean / block_te
    record = result["rebalance_records"][0]
    assert record["block_tracking_error"] == pytest.approx(block_te)
    assert record["block_info_ratio"] == pytest.approx(expected_block_ir)
