import numpy as np
import pytest

from neuro_ant_optimizer.constraints import PortfolioConstraints
from neuro_ant_optimizer.optimizer import (
    BenchmarkStats,
    NeuroAntPortfolioOptimizer,
    OptimizationObjective,
    OptimizerConfig,
)


def _benchmark_from_weights(mu: np.ndarray, cov: np.ndarray, weights: np.ndarray) -> BenchmarkStats:
    mean = float(mu @ weights)
    cov_vec = cov @ weights
    variance = float(weights @ cov @ weights)
    return BenchmarkStats(mean=mean, variance=variance, cov_vector=cov_vec)


def test_info_ratio_history_is_monotonic() -> None:
    rng = np.random.default_rng(222)
    mu = np.array([0.06, 0.035, 0.04])
    A = rng.normal(0.0, 0.1, size=(3, 3))
    cov = A @ A.T + np.eye(3) * 0.02
    bench_weights = np.array([0.4, 0.35, 0.25])
    bench_stats = _benchmark_from_weights(mu, cov, bench_weights)

    config = OptimizerConfig(
        n_ants=10,
        max_iter=6,
        patience=3,
        topk_refine=4,
        topk_train=4,
        use_risk_head=False,
        use_shrinkage=False,
        seed=99,
    )
    optimizer = NeuroAntPortfolioOptimizer(mu.size, config)
    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=1.0,
        equality_enforce=True,
        leverage_limit=1.0,
    )

    optimizer.optimize(
        mu,
        cov,
        constraints,
        objective=OptimizationObjective.INFO_RATIO_MAX,
        benchmark=bench_stats,
    )

    best_scores = [entry["best"] for entry in optimizer.history]
    assert best_scores, "history should record at least one iteration"
    diffs = np.diff(best_scores)
    assert np.all(diffs >= -1e-8)


def test_te_target_objective_hits_zero_tracking_error() -> None:
    rng = np.random.default_rng(314)
    periods = 128
    benchmark_series = rng.normal(0.0004, 0.009, size=periods)
    noise = rng.normal(0.0, 0.0025, size=(periods, 2))
    returns = np.column_stack(
        [
            benchmark_series,
            benchmark_series + noise[:, 0],
            benchmark_series + 0.5 * noise[:, 1],
        ]
    )

    mu = returns.mean(axis=0)
    cov = np.cov(returns, rowvar=False)
    bench_weights = np.array([1.0, 0.0, 0.0])
    bench_stats = _benchmark_from_weights(mu, cov, bench_weights)

    config = OptimizerConfig(
        n_ants=12,
        max_iter=8,
        patience=4,
        topk_refine=4,
        topk_train=4,
        use_risk_head=False,
        use_shrinkage=False,
        seed=21,
        te_target=0.0,
    )
    optimizer = NeuroAntPortfolioOptimizer(mu.size, config)
    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=1.0,
        equality_enforce=True,
        leverage_limit=1.0,
    )

    result = optimizer.optimize(
        mu,
        cov,
        constraints,
        objective=OptimizationObjective.TRACKING_ERROR_TARGET,
        benchmark=bench_stats,
    )

    te = optimizer._tracking_error(result.weights, mu, cov, bench_stats)
    assert te == pytest.approx(0.0, abs=5e-3)
