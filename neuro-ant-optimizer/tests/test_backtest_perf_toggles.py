from pathlib import Path

import numpy as np

from neuro_ant_optimizer.backtest.backtest import backtest, _read_csv


def _load_sample_returns():
    csv_path = Path("backtest/sample_returns.csv")
    return _read_csv(csv_path)


def test_float32_equity_close() -> None:
    df = _load_sample_returns()

    result64 = backtest(df, lookback=4, step=2, seed=123, dtype=np.float64)
    result32 = backtest(df, lookback=4, step=2, seed=123, dtype=np.float32)

    assert result64["dtype"] == "float64"
    assert result32["dtype"] == "float32"
    assert np.allclose(result64["equity"], result32["equity"], atol=1e-6)


def test_covariance_cache_evictions() -> None:
    df = _load_sample_returns()

    generous = backtest(df, lookback=4, step=2, seed=123, cov_cache_size=16)
    tiny = backtest(df, lookback=4, step=2, seed=123, cov_cache_size=1)

    generous_stats = generous["cov_cache_stats"]
    tiny_stats = tiny["cov_cache_stats"]

    assert generous_stats["size"] == 16
    assert tiny_stats["size"] == 1
    assert generous_stats["misses"] == tiny_stats["misses"]
    assert generous_stats["evictions"] == 0
    assert tiny_stats["evictions"] > 0
