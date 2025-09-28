# Neuro Ant Optimizer

Welcome to the Neuro Ant Optimizer documentation. The project blends an ant-colony heuristic with learned pheromone and risk models to generate feasible portfolios under rich constraint sets.

## Quickstart (Python API)

Install the core dependencies (Torch is required for the neural policies):

```bash
python -m pip install neuro-ant-optimizer
```

Then optimize a simple synthetic covariance matrix:

```python
import numpy as np

from neuro_ant_optimizer.optimizer import (
    NeuroAntPortfolioOptimizer,
    OptimizerConfig,
    OptimizationObjective,
)
from neuro_ant_optimizer.utils import nearest_psd

n = 20
rng = np.random.default_rng(0)
mu = rng.normal(0.08, 0.06, size=n)
A = rng.normal(size=(n, n))
cov = nearest_psd(0.2 * (A @ A.T) / n)

cfg = OptimizerConfig(n_ants=24, max_iter=25, topk_refine=6, topk_train=6)
opt = NeuroAntPortfolioOptimizer(n, cfg)

constraints = type(
    "C",
    (),
    dict(
        min_weight=0.0,
        max_weight=1.0,
        equality_enforce=True,
        leverage_limit=1.0,
        sector_map=None,
        max_sector_concentration=1.0,
        prev_weights=None,
        max_turnover=1.0,
    ),
)()

result = opt.optimize(mu, cov, constraints, objective=OptimizationObjective.SHARPE_RATIO)
print("Sharpe:", result.sharpe_ratio, "Vol:", result.volatility)
```

The optimizer returns the projected weights, realized statistics, and diagnostic metadata for post-analysis.

## Quickstart (CLI backtest)

After installing the optional backtest extras, run the bundled templates:

```bash
python -m pip install "neuro-ant-optimizer[backtest]"
neuro-ant-backtest --config examples/configs/minimal.yaml
```

Additional ready-to-run scenarios live in both the repository and the packaged module:

- Repository: `examples/configs/`
- Installed package: `import neuro_ant_optimizer.examples as ex; print(list(ex.iter_configs()))`

Each configuration resolves paths relative to the working directory. Update the `out` field inside the YAML to change the artifact location.

## What’s next?

- Read the [configuration reference](config.md) for every CLI option and config-file key.
- Inspect [artifact schemas](artifacts.md) to integrate metrics, rebalance logs, and factor exposures into downstream workflows.
- Follow the [reproducibility playbook](reproducibility.md) to lock down determinism, manifests, and replays.

## Project layout

```text
src/neuro_ant_optimizer/    # Optimizer implementation, backtest, and utils
examples/configs/           # Runnable backtest templates (mirrored in the wheel)
docs/                       # MkDocs documentation sources
```
