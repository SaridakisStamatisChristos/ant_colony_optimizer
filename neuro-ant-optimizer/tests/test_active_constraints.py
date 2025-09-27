import numpy as np

from neuro_ant_optimizer.constraints import PortfolioConstraints
from neuro_ant_optimizer.optimizer import NeuroAntPortfolioOptimizer
from neuro_ant_optimizer.utils import nearest_psd, shrink_covariance


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


def test_conflicting_factor_bounds_report_infeasible():
    n = 3
    opt = NeuroAntPortfolioOptimizer(n_assets=n)
    mu = np.ones(n, dtype=float) / n
    cov = np.eye(n, dtype=float)
    loadings = np.zeros((n, 1), dtype=float)
    lower = np.array([0.1], dtype=float)
    upper = np.array([0.2], dtype=float)
    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=1.0,
        equality_enforce=True,
        leverage_limit=1.0,
        factor_loadings=loadings,
        factor_lower_bounds=lower,
        factor_upper_bounds=upper,
        factor_tolerance=1e-6,
    )
    result = opt.optimize(mu, cov, constraints)
    assert result.feasible is False
    assert result.projection_iterations >= 1


def test_projection_idempotence_with_active_and_factors():
    n = 6
    opt = NeuroAntPortfolioOptimizer(n_assets=n)
    benchmark = np.array([0.2, 0.15, 0.2, 0.15, 0.2, 0.1], dtype=float)
    prev = benchmark.copy()
    loadings = np.array(
        [
            [1.0, 0.2],
            [0.8, 0.0],
            [0.2, 1.0],
            [0.1, 0.6],
            [0.5, -0.4],
            [0.3, 0.3],
        ],
        dtype=float,
    )
    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=0.6,
        leverage_limit=1.0,
        equality_enforce=True,
        benchmark_weights=benchmark,
        min_active_weight=-0.05,
        max_active_weight=0.05,
        active_group_map=[0, 0, 1, 1, 2, 2],
        active_group_bounds={0: (-0.02, 0.03), 1: (-0.03, 0.04)},
        sector_map=[0, 0, 1, 1, 2, 2],
        max_sector_concentration=0.55,
        factor_loadings=loadings,
        factor_lower_bounds=np.array([0.15, -0.25], dtype=float),
        factor_upper_bounds=np.array([0.65, 0.35], dtype=float),
        factor_tolerance=1e-6,
        prev_weights=prev,
        max_turnover=0.25,
    )

    weights = np.array([0.55, 0.05, 0.05, 0.1, 0.15, 0.1], dtype=float)
    projected = opt._apply_constraints(weights, constraints)
    reprojection = opt._apply_constraints(projected, constraints)

    assert np.allclose(projected, reprojection, atol=1e-8)
    assert opt._feasible(projected, constraints)


def test_near_singular_covariance_feasible_with_active_bounds():
    n = 5
    opt = NeuroAntPortfolioOptimizer(n_assets=n)
    base = np.linspace(0.9, 1.1, n)
    cov_raw = np.outer(base, base)
    cov_raw[0, 1] += 5e-4
    cov_raw[1, 0] -= 5e-4
    mu = np.linspace(0.01, 0.02, n, dtype=float)
    benchmark = np.full(n, 1.0 / n, dtype=float)
    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=0.7,
        equality_enforce=True,
        leverage_limit=1.0,
        benchmark_weights=benchmark,
        min_active_weight=-0.02,
        max_active_weight=0.02,
    )
    cov_processed = nearest_psd(
        shrink_covariance(cov_raw, delta=opt.cfg.shrinkage_delta)
    )
    assert np.min(np.linalg.eigvalsh(cov_processed)) >= -1e-10
    result = opt.optimize(mu, cov_raw, constraints, refine=False)
    assert result.feasible
    assert opt._feasible(result.weights, constraints)


def test_turnover_projection_respects_active_bounds():
    n = 6
    opt = NeuroAntPortfolioOptimizer(n_assets=n)
    benchmark = np.array([0.15, 0.18, 0.17, 0.16, 0.19, 0.15], dtype=float)
    prev = benchmark.copy()
    constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=0.5,
        equality_enforce=True,
        leverage_limit=1.0,
        benchmark_weights=benchmark,
        min_active_weight=-0.03,
        max_active_weight=0.03,
        prev_weights=prev,
        max_turnover=0.12,
    )
    weights = np.array([0.4, 0.05, 0.05, 0.05, 0.3, 0.15], dtype=float)
    adjusted = opt._apply_constraints(weights, constraints)
    active = adjusted - benchmark
    assert np.all(active <= constraints.max_active_weight + 1e-8)
    assert np.all(active >= constraints.min_active_weight - 1e-8)
    turnover = np.abs(adjusted - prev).sum()
    assert turnover <= constraints.max_turnover + 1e-8
    assert opt._feasible(adjusted, constraints)

