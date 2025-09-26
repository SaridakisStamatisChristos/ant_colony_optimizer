import numpy as np
from neuro_ant_optimizer.constraints import PortfolioConstraints
from neuro_ant_optimizer.optimizer import NeuroAntPortfolioOptimizer, OptimizationObjective
from neuro_ant_optimizer.utils import nearest_psd

def test_constraints_feasibility():
    n = 10
    mu = np.random.normal(0.08, 0.02, n)
    A = np.random.normal(0.0, 0.1, (n, n)); cov = nearest_psd(A @ A.T)
    d = np.sqrt(np.clip(np.diag(cov), 1e-12, None)); corr = cov / (np.outer(d, d)+1e-12)
    cov = corr * 0.04; np.fill_diagonal(cov, 0.04)

    cons = PortfolioConstraints(min_weight=0.0, max_weight=0.3, leverage_limit=1.0,
                                equality_enforce=True, sector_map=[i%3 for i in range(n)])
    opt = NeuroAntPortfolioOptimizer(n_assets=n)
    res = opt.optimize(mu, cov, cons, objective=OptimizationObjective.SHARPE_RATIO)
    assert abs(res.weights.sum() - 1.0) < 1e-6
    assert (res.weights >= 0.0).all() and (res.weights <= 0.3 + 1e-9).all()
