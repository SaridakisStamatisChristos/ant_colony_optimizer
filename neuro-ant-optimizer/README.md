# Neuro Ant Optimizer

A hybrid ant-colony + neural policy portfolio optimizer with:
- Deterministic seeding, PSD repair, masked softmax
- Trainable pheromone policy (KL→EMA + entropy), device/dtype-safe
- SLSQP refine with turnover & transaction-cost penalties
- Optional covariance shrinkage and CVaR objective

## Install (minimal)
```bash
python -m pip install numpy scipy torch pytest
```

## Quick start
```python
import numpy as np
from neuro_ant_optimizer.optimizer import (
    NeuroAntPortfolioOptimizer, OptimizerConfig, OptimizationObjective
)
from neuro_ant_optimizer.utils import nearest_psd

n = 20
mu = np.random.default_rng(0).normal(0.08, 0.06, size=n)
A = np.random.default_rng(1).normal(size=(n,n))
cov = nearest_psd(0.2*(A@A.T)/n)

cfg = OptimizerConfig(n_ants=24, max_iter=25, topk_refine=6, topk_train=6)
opt = NeuroAntPortfolioOptimizer(n, cfg)

constraints = type("C", (), dict(
  min_weight=0.0, max_weight=1.0, equality_enforce=True, leverage_limit=1.0,
  sector_map=None, max_sector_concentration=1.0, prev_weights=None, max_turnover=1.0
))()

res = opt.optimize(mu, cov, constraints, objective=OptimizationObjective.SHARPE_RATIO)
print("Sharpe:", res.sharpe_ratio, "Vol:", res.volatility)
```

## Objectives
- `SHARPE_RATIO` (default)
- `MAX_RETURN`
- `MIN_VARIANCE`
- `RISK_PARITY`
- `MIN_CVAR` (normal approx) — configure `OptimizerConfig.cvar_alpha`

## Config highlights
- `use_shrinkage`/`shrinkage_delta`: diagonal shrinkage before PSD
- `risk_weight`: blend risk heuristic in ant decisions
- `topk_refine`/`topk_train`: compute & learning budgets per iteration
- `grad_clip`, `device`, `dtype`
