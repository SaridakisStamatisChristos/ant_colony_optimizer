from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

bt = import_module("neuro_ant_optimizer.backtest.backtest")


class _StubOptimizer:
    def __init__(self, weight: np.ndarray):
        self.weight = weight
        self.cfg = type("Cfg", (), {"use_shrinkage": False, "shrinkage_delta": 0.0})()

    def optimize(self, *_, **__):
        class _Result:
            def __init__(self, w: np.ndarray):
                self.weights = w
                self.feasible = True
                self.projection_iterations = 0

        return _Result(self.weight)


def test_decay_blend_math(tmp_path: Path, monkeypatch) -> None:
    returns_path = tmp_path / "returns.csv"
    returns_path.write_text(
        "date,A,B\n"
        "2020-01-01,0.01,0.02\n"
        "2020-01-02,0.03,0.01\n"
        "2020-01-03,0.02,0.00\n"
        "2020-01-04,0.01,0.03\n"
    )

    warm_path = tmp_path / "warm.csv"
    warm_path.write_text(
        "date,A,B\n"
        "2019-12-31,0.0,1.0\n"
    )

    stub = _StubOptimizer(np.array([1.0, 0.0], dtype=float))
    monkeypatch.setattr(bt, "_build_optimizer", lambda n_assets, seed, risk_free_rate=0.0: stub)

    frame = bt._read_csv(returns_path)
    lookback = 2
    step = 2
    results = bt.backtest(
        frame,
        lookback=lookback,
        step=step,
        warm_start=str(warm_path),
        decay=0.25,
    )

    weights = np.asarray(results["weights"][0], dtype=float)
    assert weights == pytest.approx(np.array([0.75, 0.25]))

    record = results["rebalance_records"][0]
    assert record["turnover_pre_decay"] == pytest.approx(2.0)
    assert record["turnover_post_decay"] == pytest.approx(1.5)
    assert record["turnover"] == pytest.approx(1.5)
    assert record["warm_applied"] is True
    assert record["decay"] == pytest.approx(0.25)

    realized = results["returns"][:step]
    returns_block = frame.to_numpy()[lookback : lookback + step]
    expected_returns = returns_block @ weights
    assert realized == pytest.approx(expected_returns)
