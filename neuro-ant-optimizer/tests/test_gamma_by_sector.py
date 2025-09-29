from __future__ import annotations

import numpy as np

from neuro_ant_optimizer.constraints import PortfolioConstraints
from neuro_ant_optimizer.optimizer import (
    BenchmarkStats,
    NeuroAntPortfolioOptimizer,
    OptimizerConfig,
    OptimizationObjective,
)


def test_sector_gamma_penalty_prefers_low_penalty_moves() -> None:
    n_assets = 4
    cfg = OptimizerConfig(
        n_ants=2,
        max_iter=1,
        patience=1,
        topk_refine=1,
        topk_train=1,
        max_runtime=0.01,
        gamma_turnover=0.0,
    )
    optimizer = NeuroAntPortfolioOptimizer(n_assets, cfg)

    gamma_vec = np.array([0.6, 0.6, 0.05, 0.05], dtype=float)
    optimizer.cfg.gamma_turnover_vector = gamma_vec

    prev_weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=1.0,
        equality_enforce=True,
        prev_weights=prev_weights,
        turnover_gamma=gamma_vec,
    )

    mu = np.full(n_assets, 0.05, dtype=float)
    cov = np.eye(n_assets, dtype=float) * 0.01

    weights_high_gamma = prev_weights.copy()
    weights_high_gamma[0] += 0.1
    weights_high_gamma[1] -= 0.1

    weights_low_gamma = prev_weights.copy()
    weights_low_gamma[2] += 0.1
    weights_low_gamma[3] -= 0.1

    benchmark = BenchmarkStats(mean=0.0, variance=0.0, cov_vector=np.zeros(n_assets))

    score_high = optimizer._score(
        weights_high_gamma,
        mu,
        cov,
        OptimizationObjective.MULTI_TERM,
        constraints,
        benchmark=benchmark,
    )
    score_low = optimizer._score(
        weights_low_gamma,
        mu,
        cov,
        OptimizationObjective.MULTI_TERM,
        constraints,
        benchmark=benchmark,
    )

    assert score_low > score_high
