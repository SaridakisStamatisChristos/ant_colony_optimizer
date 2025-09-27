from importlib import import_module

import numpy as np

from neuro_ant_optimizer.backtest.backtest import (
    _write_equity,
    _write_metrics,
    _write_weights,
)

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


def test_report_csv_determinism_with_benchmark(tmp_path):
    rng = np.random.default_rng(4)
    n, m = 120, 4
    returns = rng.normal(scale=0.01, size=(n, m))
    benchmark = rng.normal(scale=0.008, size=n)
    dates = [np.datetime64("2021-01-01") + np.timedelta64(i, "D") for i in range(n)]
    cols = [f"A{i}" for i in range(m)]
    frame = _Frame(returns, dates, cols)
    bench_frame = _Frame(benchmark.reshape(-1, 1), dates, ["bench"])

    res1 = bt.backtest(
        frame,
        lookback=40,
        step=10,
        cov_model="sample",
        seed=11,
        benchmark=bench_frame,
    )
    res2 = bt.backtest(
        frame,
        lookback=40,
        step=10,
        cov_model="sample",
        seed=11,
        benchmark=bench_frame,
    )

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    out1.mkdir()
    out2.mkdir()

    _write_equity(out1 / "equity.csv", res1)
    _write_equity(out2 / "equity.csv", res2)
    _write_metrics(out1 / "metrics.csv", res1)
    _write_metrics(out2 / "metrics.csv", res2)
    _write_weights(out1 / "weights.csv", res1)
    _write_weights(out2 / "weights.csv", res2)

    for name in ["equity.csv", "metrics.csv", "weights.csv"]:
        data1 = (out1 / name).read_bytes()
        data2 = (out2 / name).read_bytes()
        assert data1 == data2
