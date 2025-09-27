from importlib import import_module

import numpy as np


bt = import_module("neuro_ant_optimizer.backtest.backtest")


class _Frame:
    def __init__(self, arr, dates, cols):
        self._a = arr
        self._d = dates
        self._c = cols

    def to_numpy(self, dtype=float):
        return self._a.astype(dtype)

    @property
    def index(self):
        return self._d

    @property
    def columns(self):
        return self._c


def test_cov_model_determinism():
    rng = np.random.default_rng(0)
    n, m = 160, 6
    X = rng.normal(size=(n, m))
    dates = [np.datetime64("2022-01-01") + np.timedelta64(i, "D") for i in range(n)]
    cols = [f"A{i}" for i in range(m)]
    frame = _Frame(X, dates, cols)

    r1 = bt.backtest(frame, lookback=40, step=10, cov_model="oas", seed=7)
    r2 = bt.backtest(frame, lookback=40, step=10, cov_model="oas", seed=7)

    np.testing.assert_allclose(r1["equity"], r2["equity"], rtol=0, atol=0)
    assert r1["rebalance_records"] == r2["rebalance_records"]
