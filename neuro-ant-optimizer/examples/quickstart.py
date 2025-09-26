import numpy as np
from neuro_ant_optimizer.optimizer import NeuroAntPortfolioOptimizer, OptimizationObjective
from neuro_ant_optimizer.constraints import PortfolioConstraints
from neuro_ant_optimizer.utils import nearest_psd, set_seed

set_seed(42)
n = 24
mu = np.random.normal(0.08, 0.02, n)
A = np.random.normal(0.0, 0.1, (n, n))
cov = nearest_psd(A @ A.T)
d = np.sqrt(np.clip(np.diag(cov), 1e-12, None)); corr = cov / (np.outer(d, d)+1e-12)
cov = corr * 0.04; np.fill_diagonal(cov, 0.04)

constraints = PortfolioConstraints(
    min_weight=0.0, max_weight=0.2, leverage_limit=1.0,
    max_turnover=0.35, max_sector_concentration=0.35,
    equality_enforce=True, sector_map=[i%6 for i in range(n)]
)

opt = NeuroAntPortfolioOptimizer(n_assets=n)
res = opt.optimize(mu, cov, constraints, objective=OptimizationObjective.SHARPE_RATIO)
print("Sharpe:", round(res.sharpe_ratio, 4), "| Sum w:", round(res.weights.sum(), 6))
print("First 10 weights:", np.round(res.weights[:10], 4))
