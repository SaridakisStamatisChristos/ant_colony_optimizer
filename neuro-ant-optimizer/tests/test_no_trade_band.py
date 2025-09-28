import numpy as np
from importlib import import_module

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


def test_no_trade_band_reduces_turnover(monkeypatch):
    n_periods = 12
    dates = np.array(
        [np.datetime64("2022-01-01") + np.timedelta64(i, "D") for i in range(n_periods)]
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
        np.array([0.50, 0.50], dtype=float),
        np.array([0.52, 0.48], dtype=float),
        np.array([0.51, 0.49], dtype=float),
    ]

    def _build_stub(*_args, **_kwargs):
        return _StubOptimizer([w.copy() for w in weights_seq])

    monkeypatch.setattr(bt, "_build_optimizer", _build_stub)
    baseline = bt.backtest(
        frame,
        lookback=3,
        step=3,
        seed=0,
        tx_cost_mode="none",
        nt_band=0.0,
    )

    monkeypatch.setattr(bt, "_build_optimizer", _build_stub)
    banded = bt.backtest(
        frame,
        lookback=3,
        step=3,
        seed=0,
        tx_cost_mode="none",
        nt_band=0.02,
    )

    assert banded["avg_turnover"] < baseline["avg_turnover"]
    assert any(record["nt_band_hits"] > 0 for record in banded["rebalance_records"])


def test_participation_cap_clips_trades(monkeypatch):
    n_periods = 12
    dates = np.array(
        [np.datetime64("2022-06-01") + np.timedelta64(i, "D") for i in range(n_periods)]
    )
    returns = np.array(
        [
            [0.01, -0.01],
            [0.015, 0.0],
            [0.0, 0.01],
            [0.02, 0.005],
            [0.01, -0.015],
            [0.0, 0.02],
            [0.015, -0.005],
            [0.0, 0.01],
            [0.02, 0.0],
            [0.01, -0.01],
            [0.0, 0.015],
            [0.02, 0.005],
        ],
        dtype=float,
    )
    frame = _Frame(returns, dates, ["A", "B"])

    weights_seq = [
        np.array([0.60, 0.40], dtype=float),
        np.array([0.90, 0.10], dtype=float),
        np.array([0.20, 0.80], dtype=float),
    ]

    def _build_stub(*_args, **_kwargs):
        return _StubOptimizer([w.copy() for w in weights_seq])

    monkeypatch.setattr(bt, "_build_optimizer", _build_stub)
    slippage = bt.parse_slippage("impact:k=20,participation=0.1")
    results = bt.backtest(
        frame,
        lookback=3,
        step=3,
        seed=0,
        tx_cost_mode="none",
        nt_band=0.0,
        slippage=slippage,
    )

    weights = results["weights"]
    assert weights.shape[0] == len(weights_seq)
    np.testing.assert_allclose(weights[0], np.array([0.60, 0.40]), atol=1e-9)
    np.testing.assert_allclose(weights[1], np.array([0.70, 0.30]), atol=1e-9)
    np.testing.assert_allclose(weights[2], np.array([0.60, 0.40]), atol=1e-9)
    assert any(record["participation_breaches"] > 0 for record in results["rebalance_records"])
