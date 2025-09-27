import numpy as np

from neuro_ant_optimizer.constraints import PortfolioConstraints
from neuro_ant_optimizer.optimizer import (
    NeuroAntPortfolioOptimizer,
    OptimizerConfig,
    OptimizationObjective,
)
from neuro_ant_optimizer.utils import nearest_psd


def test_factor_neutrality_and_sector_caps():
    rng = np.random.default_rng(0)
    n_assets, n_factors = 16, 3
    mu = rng.normal(0.08, 0.05, size=n_assets)
    A = rng.normal(size=(n_assets, n_assets))
    cov = nearest_psd(0.1 * (A @ A.T) / n_assets)

    factor_loadings = rng.normal(size=(n_assets, n_factors))
    factor_targets = np.zeros(n_factors)

    sector_map = [0 if i < n_assets // 2 else 1 for i in range(n_assets)]
    sector_cap = 0.65

    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=0.3,
        equality_enforce=True,
        leverage_limit=1.0,
        sector_map=sector_map,
        max_sector_concentration=sector_cap,
        factor_loadings=factor_loadings,
        factor_targets=factor_targets,
        factor_tolerance=1e-6,
    )

    config = OptimizerConfig(
        n_ants=12,
        max_iter=10,
        topk_refine=4,
        topk_train=4,
        seed=123,
    )
    optimizer = NeuroAntPortfolioOptimizer(n_assets, config)
    result = optimizer.optimize(
        mu,
        cov,
        constraints,
        objective=OptimizationObjective.SHARPE_RATIO,
    )
    weights = result.weights

    left = weights[: n_assets // 2].sum()
    right = weights[n_assets // 2 :].sum()

    assert left <= sector_cap + 1e-6
    assert right <= sector_cap + 1e-6

    neutrality = np.linalg.norm(factor_loadings.T @ weights - factor_targets, ord=np.inf)
    assert neutrality <= constraints.factor_tolerance + 1e-5
