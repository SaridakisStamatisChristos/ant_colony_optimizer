import time
from typing import Any, Dict

import numpy as np
import pytest

try:
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pd = None  # type: ignore

from neuro_ant_optimizer.backtest.backtest import (
    FactorPanel,
    SlippageConfig,
    backtest,
)


pytestmark = pytest.mark.slow


def _run_backtest(workers: int | None) -> Dict[str, Any]:
    if pd is None:
        pytest.skip("pandas is required for this test")
    rng = np.random.default_rng(3)
    returns = rng.normal(0.001, 0.01, size=(36, 3))
    index = pd.date_range("2024-01-01", periods=36, freq="B")
    frame = pd.DataFrame(returns, index=index, columns=["A", "B", "C"])

    return backtest(
        frame,
        lookback=12,
        step=6,
        objective="sharpe",
        seed=11,
        tx_cost_bps=0.0,
        tx_cost_mode="none",
        metric_alpha=0.05,
        cov_model="sample",
        benchmark=None,
        dtype=np.float64,
        cov_cache_size=2,
        workers=workers,
        prefetch=2,
        deterministic=True,
    )


def test_parallel_results_match_single_process() -> None:
    single_start = time.perf_counter()
    single = _run_backtest(workers=None)
    single_elapsed = time.perf_counter() - single_start

    multi_start = time.perf_counter()
    multi = _run_backtest(workers=2)
    multi_elapsed = time.perf_counter() - multi_start

    for key in ("returns", "weights", "rebalance_dates"):
        single_val = single[key]
        multi_val = multi[key]
        if isinstance(single_val, np.ndarray):
            np.testing.assert_allclose(single_val, multi_val)
        else:
            assert single_val == multi_val

    assert multi_elapsed < 5.0
    assert single_elapsed < 5.0


def test_parallel_with_factors_and_slippage() -> None:
    if pd is None:
        pytest.skip("pandas is required for this test")

    rng = np.random.default_rng(5)
    returns = rng.normal(0.0008, 0.012, size=(40, 3))
    index = pd.date_range("2024-02-01", periods=40, freq="B")
    frame = pd.DataFrame(returns, index=index, columns=["A", "B", "C"])

    factors = rng.normal(0.0, 1.0, size=(40, 3, 2))
    panel = FactorPanel(
        dates=list(index),
        assets=["A", "B", "C"],
        loadings=factors,
        factor_names=["value", "size"],
    )

    benchmark = pd.DataFrame(
        rng.normal(0.0005, 0.009, size=(40, 1)), index=index, columns=["SPX"]
    )
    slippage = SlippageConfig(model="proportional", param=5.0)

    single = backtest(
        frame,
        lookback=15,
        step=5,
        objective="sharpe",
        seed=17,
        tx_cost_bps=2.0,
        tx_cost_mode="amortized",
        metric_alpha=0.1,
        cov_model="oas",
        factors=panel,
        factor_align="strict",
        factors_required=True,
        factor_tolerance=1e-6,
        slippage=slippage,
        nt_band=0.01,
        benchmark=benchmark,
        dtype=np.float64,
        cov_cache_size=3,
        workers=None,
        prefetch=2,
        deterministic=True,
        compute_factor_attr=True,
    )

    multi = backtest(
        frame,
        lookback=15,
        step=5,
        objective="sharpe",
        seed=17,
        tx_cost_bps=2.0,
        tx_cost_mode="amortized",
        metric_alpha=0.1,
        cov_model="oas",
        factors=panel,
        factor_align="strict",
        factors_required=True,
        factor_tolerance=1e-6,
        slippage=slippage,
        nt_band=0.01,
        benchmark=benchmark,
        dtype=np.float64,
        cov_cache_size=3,
        workers=2,
        prefetch=3,
        deterministic=True,
        compute_factor_attr=True,
    )

    for key in ("returns", "weights", "rebalance_dates", "factor_attr"):
        single_val = single[key]
        multi_val = multi[key]
        if isinstance(single_val, np.ndarray):
            np.testing.assert_allclose(single_val, multi_val)
        else:
            assert single_val == multi_val
