from __future__ import annotations

import numpy as np

from neuro_ant_optimizer.backtest.backtest import backtest


def _make_returns(periods: int = 18, assets: int = 3) -> np.ndarray:
    return np.linspace(0.001, 0.02, num=periods * assets, dtype=np.float64).reshape(
        periods, assets
    )


def test_float32_backtest_matches_float64() -> None:
    df = _make_returns()
    kwargs = dict(lookback=6, step=2, seed=17, trading_days=252)

    result64 = backtest(df, dtype=np.float64, **kwargs)
    result32 = backtest(df, dtype=np.float32, **kwargs)

    assert result64["dtype"] == np.dtype(np.float64).name
    assert result32["dtype"] == np.dtype(np.float32).name

    for key in ("equity", "returns", "gross_returns", "weights"):
        arr64 = np.asarray(result64[key], dtype=np.float64)
        arr32 = np.asarray(result32[key], dtype=np.float64)
        assert np.allclose(arr64, arr32, atol=1e-6), key


def test_covariance_cache_metrics() -> None:
    df = _make_returns(periods=20, assets=2)
    kwargs = dict(lookback=5, step=1, seed=5, trading_days=252)

    # Constant data to ensure repeated covariance windows and cache hits.
    const_df = np.full_like(df, 0.01, dtype=np.float64)
    repeated = backtest(const_df, cov_cache_size=4, **kwargs)
    n_windows = len(repeated["rebalance_dates"])
    assert repeated["cov_cache_size"] == 4
    assert repeated["cov_cache_hits"] == max(n_windows - 1, 0)
    assert repeated["cov_cache_misses"] == min(n_windows, 1)
    assert repeated["cov_cache_evictions"] == 0

    # Disable caching entirely and ensure every window is treated as a miss.
    uncached = backtest(const_df, cov_cache_size=0, **kwargs)
    assert uncached["cov_cache_hits"] == 0
    assert uncached["cov_cache_misses"] == n_windows
    assert uncached["cov_cache_evictions"] == 0

    # Force evictions by using many unique windows with a tiny cache size.
    eviction = backtest(df, cov_cache_size=2, **kwargs)
    expected_evictions = max(len(eviction["rebalance_dates"]) - 2, 0)
    assert eviction["cov_cache_hits"] == 0
    assert eviction["cov_cache_misses"] == len(eviction["rebalance_dates"])
    assert eviction["cov_cache_evictions"] == expected_evictions
