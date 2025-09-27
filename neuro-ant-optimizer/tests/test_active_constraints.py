import numpy as np

from neuro_ant_optimizer.constraints import PortfolioConstraints
from neuro_ant_optimizer.optimizer import NeuroAntPortfolioOptimizer


def test_apply_constraints_respects_active_bounds():
    n = 5
    benchmark = np.ones(n, dtype=float) / n
    opt = NeuroAntPortfolioOptimizer(n_assets=n)
    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=0.7,
        leverage_limit=1.0,
        equality_enforce=True,
        benchmark_weights=benchmark,
        min_active_weight=-0.05,
        max_active_weight=0.05,
    )
    weights = np.linspace(0.0, 1.0, num=n, dtype=float)
    adjusted = opt._apply_constraints(weights, constraints)
    active = adjusted - benchmark
    assert np.all(active <= constraints.max_active_weight + 1e-8)
    assert np.all(active >= constraints.min_active_weight - 1e-8)
    assert abs(adjusted.sum() - 1.0) < 1e-8
    assert opt._feasible(adjusted, constraints)


def test_active_group_bounds_projection():
    n = 4
    benchmark = np.array([0.4, 0.3, 0.2, 0.1], dtype=float)
    opt = NeuroAntPortfolioOptimizer(n_assets=n)
    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=0.6,
        leverage_limit=1.0,
        equality_enforce=True,
        benchmark_weights=benchmark,
        min_active_weight=-0.2,
        max_active_weight=0.2,
        active_group_map=[0, 0, 1, 1],
        active_group_bounds={0: (-0.05, 0.05), 1: (-0.1, 0.1)},
    )
    weights = np.array([0.7, 0.2, 0.05, 0.05], dtype=float)
    adjusted = opt._apply_constraints(weights, constraints)
    groups = np.asarray(constraints.active_group_map)
    active = adjusted - benchmark
    group_zero = active[groups == 0].sum()
    group_one = active[groups == 1].sum()
    assert group_zero <= 0.05 + 1e-8 and group_zero >= -0.05 - 1e-8
    assert group_one <= 0.1 + 1e-8 and group_one >= -0.1 - 1e-8
    assert opt._feasible(adjusted, constraints)


def test_factor_bounds_enforced_without_targets():
    n = 3
    opt = NeuroAntPortfolioOptimizer(n_assets=n)
    loadings = np.array(
        [
            [1.0, 0.0],
            [0.5, 1.0],
            [0.0, 0.5],
        ],
        dtype=float,
    )
    lower = np.array([-0.1, 0.2], dtype=float)
    upper = np.array([0.2, 0.4], dtype=float)
    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=0.8,
        leverage_limit=1.0,
        equality_enforce=True,
        factor_loadings=loadings,
        factor_lower_bounds=lower,
        factor_upper_bounds=upper,
        factor_tolerance=1e-6,
    )
    weights = np.array([0.9, 0.05, 0.05], dtype=float)
    adjusted = opt._apply_constraints(weights, constraints)
    exposures = loadings.T @ adjusted
    assert np.all(exposures <= upper + 1e-6)
    assert np.all(exposures >= lower - 1e-6)
    assert opt._feasible(adjusted, constraints)

