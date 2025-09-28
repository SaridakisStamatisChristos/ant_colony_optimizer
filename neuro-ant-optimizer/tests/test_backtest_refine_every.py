from __future__ import annotations

from importlib import import_module

import numpy as np

bt = import_module("neuro_ant_optimizer.backtest.backtest")


class _TrackingOptimizer:
    def __init__(self, weights_seq: list[np.ndarray]):
        self.weights_seq = weights_seq
        self.calls = 0
        self.refine_flags: list[bool] = []
        self.cfg = type("Cfg", (), {"use_shrinkage": False, "shrinkage_delta": 0.0})()

    def optimize(self, *_, refine: bool = True, **__):
        self.refine_flags.append(refine)
        weights = self.weights_seq[self.calls]
        self.calls += 1

        class _Result:
            def __init__(self, w: np.ndarray):
                self.weights = w

        return _Result(weights)


def test_refine_every_skips(monkeypatch) -> None:
    n_periods = 15
    dates = np.array(
        [np.datetime64("2021-01-01") + np.timedelta64(i, "D") for i in range(n_periods)]
    )
    returns = np.tile(np.array([[0.01, 0.0], [0.0, 0.01], [0.02, -0.01], [0.0, 0.02], [0.01, 0.0]], dtype=float), (3, 1))
    returns = returns[:n_periods]

    class _Frame:
        def __init__(self, arr: np.ndarray):
            self._arr = arr
            self._idx = list(dates)
            self._cols = ["A", "B"]

        def to_numpy(self, dtype=float):
            return self._arr.astype(dtype)

        @property
        def index(self):  # pragma: no cover - simple accessor
            return self._idx

        @property
        def columns(self):  # pragma: no cover - simple accessor
            return self._cols

    frame = _Frame(returns)
    weights_seq = [np.array([0.5, 0.5], dtype=float) for _ in range(4)]
    tracker = _TrackingOptimizer(weights_seq)
    monkeypatch.setattr(
        bt, "_build_optimizer", lambda n_assets, seed, risk_free_rate=0.0: tracker
    )

    bt.backtest(frame, lookback=3, step=3, refine_every=2)

    assert tracker.refine_flags == [True, False, True, False]
