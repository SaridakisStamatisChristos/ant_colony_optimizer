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

## CLI backtest
Install optional deps then run:
```bash
python -m pip install "neuro-ant-optimizer[backtest]"
neuro-ant-backtest --csv path/to/returns.csv --lookback 252 --step 21 --ewma_span 60 \
  --objective sharpe --out bt_out --save-weights --tx-cost-bps 5 --tx-cost-mode upfront
# tx-cost-mode: upfront | amortized | posthoc | none
# writes metrics.csv (incl. sortino, cvar), equity.csv, equity_net_of_tc.csv (if posthoc), and weights.csv
```
Behavior summary

--tx-cost-mode upfront → costs applied inside the loop on the first day of each block.

--tx-cost-mode amortized → costs applied inside the loop evenly across the block.

--tx-cost-mode posthoc → no costs during loop; after the run, we create equity_net_of_tc.csv with amortized costs.

--tx-cost-mode none → no costs at all.

Outputs `metrics.csv`, `equity.csv`, and (if matplotlib is present) `equity.png`.

## Testing
From the repository root:

```bash
pytest -q
```

The test harness in `tests/conftest.py` automatically adds `neuro-ant-optimizer/src`
to `sys.path`, so no manual `PYTHONPATH` configuration or editable install is required.

## Offline usage (no install)
If your environment blocks package downloads:
```bash
# Run the CLI module directly from the repo checkout
python -m neuro_ant_optimizer.backtest \
  --csv neuro-ant-optimizer/backtest/sample_returns.csv \
  --lookback 5 --step 2 --ewma_span 3 --objective sharpe --out neuro-ant-optimizer/backtest/out_local
# The repo includes a lightweight shim (neuro_ant_optimizer/__init__.py) that
# delegates to src/neuro_ant_optimizer so no PYTHONPATH edits are needed.
```

## Offline wheel build & install
You can build a wheel locally and install it without hitting the internet:
```bash
# Build wheel from source (no network needed for your own package)
python -m pip install --upgrade pip wheel setuptools   # if available locally
python -m pip wheel . -w dist

# Install the wheel offline (no deps)
python -m pip install --no-deps --no-index dist/neuro_ant_optimizer-*.whl
```
> Note: optional extras like `[backtest]` pull external packages. For fully offline use,
> prefer the `python -m neuro_ant_optimizer.backtest` invocation shown above, or pre-stage
> wheels for `pandas`/`matplotlib` on an internal index and install with `--find-links`.
