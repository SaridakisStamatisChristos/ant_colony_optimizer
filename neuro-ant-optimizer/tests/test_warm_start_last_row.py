from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

bt = import_module("neuro_ant_optimizer.backtest.backtest")


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


def test_warm_start_last_row(tmp_path: Path, monkeypatch) -> None:
    returns_path = tmp_path / "returns.csv"
    returns_path.write_text(
        "date,A,B\n"
        "2020-01-01,0.01,0.00\n"
        "2020-01-02,0.02,0.01\n"
        "2020-01-03,0.00,0.02\n"
        "2020-01-04,0.01,0.03\n"
        "2020-01-05,0.03,0.01\n"
        "2020-01-06,0.02,0.00\n"
    )

    warm_path = tmp_path / "weights.csv"
    warm_path.write_text(
        "date,A,B\n"
        "2020-01-01,0.5,0.5\n"
        "2020-01-02,0.2,0.8\n"
    )

    weights_seq = [np.array([0.6, 0.4], dtype=float)]
    stub = _StubOptimizer(weights_seq)
    monkeypatch.setattr(bt, "_build_optimizer", lambda n_assets, seed, risk_free_rate=0.0: stub)

    results = bt.backtest(
        bt._read_csv(returns_path),
        lookback=3,
        step=3,
        warm_start=str(warm_path),
        warm_align="last_row",
    )

    record = results["rebalance_records"][0]
    expected_turn = abs(0.6 - 0.2) + abs(0.4 - 0.8)
    assert record["turnover_pre_decay"] == pytest.approx(expected_turn)
    assert record["turnover"] == pytest.approx(expected_turn)
    assert record["warm_applied"] is True
    assert results["warm_applied_count"] == 1
    assert "warm_no_overlap" not in results["warnings"]
