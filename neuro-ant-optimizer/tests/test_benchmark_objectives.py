import numpy as np
import pytest

from neuro_ant_optimizer.backtest.backtest import backtest, _OBJECTIVE_MAP
from neuro_ant_optimizer.constraints import PortfolioConstraints
from neuro_ant_optimizer.optimizer import (
    BenchmarkStats,
    NeuroAntPortfolioOptimizer,
    OptimizationObjective,
    OptimizerConfig,
)


def _make_benchmark_stats(returns: np.ndarray, benchmark: np.ndarray) -> BenchmarkStats:
    mu = returns.mean(axis=0)
    bench_mean = float(benchmark.mean())
    centered_b = benchmark - bench_mean
    centered_assets = returns - mu
    denom = max(1, centered_b.shape[0] - 1)
    cov_vector = centered_assets.T @ centered_b / denom
    variance = float(np.dot(centered_b, centered_b) / denom)
    return BenchmarkStats(mean=bench_mean, variance=max(variance, 0.0), cov_vector=cov_vector)


def test_tracking_error_and_info_ratio_scoring_prefers_equal_weight() -> None:
    rng = np.random.default_rng(123)
    periods = 160
    benchmark = rng.normal(0.0004, 0.009, size=periods)
    noise = rng.normal(0.0, 0.002, size=(periods, 3))
    bias = np.array([0.0006, -0.0003, 0.0005])
    returns = benchmark[:, None] + noise + bias

    mu = returns.mean(axis=0)
    cov = np.cov(returns, rowvar=False)
    bench_stats = _make_benchmark_stats(returns, benchmark)

    optimizer = NeuroAntPortfolioOptimizer(
        returns.shape[1],
        OptimizerConfig(
            n_ants=4,
            max_iter=2,
            patience=1,
            topk_refine=2,
            topk_train=2,
            use_risk_head=False,
            use_shrinkage=False,
            max_runtime=0.1,
            seed=7,
        ),
    )
    constraints = PortfolioConstraints()
    w_eq = np.ones(returns.shape[1]) / returns.shape[1]
    w_alt = np.array([0.2, 0.6, 0.2])

    manual_te_eq = float(
        np.sqrt(
            max(
                w_eq @ cov @ w_eq
                + bench_stats.variance
                - 2.0 * (w_eq @ bench_stats.cov_vector),
                0.0,
            )
        )
    )
    manual_ir_eq = (
        float((w_eq @ mu - bench_stats.mean) / manual_te_eq)
        if manual_te_eq > 1e-12
        else 0.0
    )

    score_te_eq = optimizer._score(
        w_eq,
        mu,
        cov,
        OptimizationObjective.TRACKING_ERROR_MIN,
        constraints,
        benchmark=bench_stats,
    )
    score_te_alt = optimizer._score(
        w_alt,
        mu,
        cov,
        OptimizationObjective.TRACKING_ERROR_MIN,
        constraints,
        benchmark=bench_stats,
    )

    score_ir_eq = optimizer._score(
        w_eq,
        mu,
        cov,
        OptimizationObjective.INFO_RATIO_MAX,
        constraints,
        benchmark=bench_stats,
    )
    score_ir_alt = optimizer._score(
        w_alt,
        mu,
        cov,
        OptimizationObjective.INFO_RATIO_MAX,
        constraints,
        benchmark=bench_stats,
    )

    assert -score_te_eq == pytest.approx(manual_te_eq, rel=1e-6)
    assert score_ir_eq == pytest.approx(manual_ir_eq, rel=1e-6, abs=1e-12)
    assert score_te_eq > score_te_alt  # lower tracking error preferred
    assert score_ir_eq > score_ir_alt  # higher information ratio preferred


def test_tracking_objectives_require_benchmark() -> None:
    returns = np.random.default_rng(0).normal(0.001, 0.01, size=(64, 3))
    with pytest.raises(ValueError):
        backtest(
            returns,
            lookback=32,
            step=32,
            objective="tracking_error",
            seed=3,
        )

    with pytest.raises(ValueError):
        backtest(
            returns,
            lookback=32,
            step=32,
            objective="info_ratio",
            seed=3,
        )


def test_objective_map_routes_new_objectives() -> None:
    assert _OBJECTIVE_MAP["tracking_error"] is OptimizationObjective.TRACKING_ERROR_MIN
    assert _OBJECTIVE_MAP["info_ratio"] is OptimizationObjective.INFO_RATIO_MAX

