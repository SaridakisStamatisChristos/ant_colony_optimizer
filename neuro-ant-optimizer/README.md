# Neuro Ant Optimizer

A hybrid ant-colony + neural policy portfolio optimizer with:

* Deterministic seeding, PSD repair, masked softmax
* Trainable pheromone policy (KL→EMA + entropy), device/dtype-safe
* SLSQP refine with turnover & transaction-cost penalties
* Optional covariance shrinkage and CVaR objective

---

## Install (minimal)

```bash
python -m pip install numpy scipy torch pytest
```

---

## Quick start

```python
import numpy as np
from neuro_ant_optimizer.optimizer import (
    NeuroAntPortfolioOptimizer, OptimizerConfig, OptimizationObjective
)
from neuro_ant_optimizer.utils import nearest_psd

n = 20
mu = np.random.default_rng(0).normal(0.08, 0.06, size=n)
A = np.random.default_rng(1).normal(size=(n, n))
cov = nearest_psd(0.2 * (A @ A.T) / n)

cfg = OptimizerConfig(n_ants=24, max_iter=25, topk_refine=6, topk_train=6)
opt = NeuroAntPortfolioOptimizer(n, cfg)

constraints = type("C", (), dict(
  min_weight=0.0, max_weight=1.0, equality_enforce=True, leverage_limit=1.0,
  sector_map=None, max_sector_concentration=1.0, prev_weights=None, max_turnover=1.0
))()

res = opt.optimize(mu, cov, constraints, objective=OptimizationObjective.SHARPE_RATIO)
print("Sharpe:", res.sharpe_ratio, "Vol:", res.volatility)
```

---

## Objectives

* **SHARPE_RATIO** (default)
* **MAX_RETURN**
* **MIN_VARIANCE**
* **RISK_PARITY**
* **MIN_CVAR** (normal approx) — configure `OptimizerConfig.cvar_alpha`

---

## Config highlights

* `use_shrinkage` / `shrinkage_delta`: diagonal shrinkage before PSD
* `risk_weight`: blend risk heuristic in ant decisions
* `topk_refine` / `topk_train`: compute & learning budgets per iteration
* `grad_clip`, `device`, `dtype`

---

## CLI backtest

Install optional deps then run:

```bash
python -m pip install "neuro-ant-optimizer[backtest]"
neuro-ant-backtest --csv path/to/returns.csv --lookback 252 --step 21 \
  --objective sharpe --cov-model lw --out bt_out \
  --save-weights --tx-cost-bps 5 --tx-cost-mode upfront \
  --factor-align strict --factors path/to/factors.csv \
  --rf-bps 25 --trading-days 260
```

* `--cov-model`: `sample` | `ewma` (use `--ewma_span`) | `lw` | `oas`
* `tx-cost-mode`: `upfront` | `amortized` | `posthoc` | `none`
* `factor-align`: `strict` (require coverage) | `subset` (allow missing windows)
* Use `--factors-required` with `subset` mode to treat missing windows as fatal
* Pass `--skip-plot` to avoid importing matplotlib when running headless
* Writes `metrics.csv` (incl. sortino, cvar), `equity.csv`, `equity_net_of_tc.csv` (if posthoc),
  `factor_diagnostics.json`, `factor_constraints.csv` (when factors are provided) and `weights.csv`

`--rf-bps` sets the annualized risk-free rate (in basis points) used when reporting Sharpe,
Sortino, and information ratios. `--trading-days` controls the number of periods per year
used to annualize returns and tracking error; adjust it if you are working with non-daily
data.

### Behavior summary

* `--tx-cost-mode upfront` → costs applied inside the loop on the first day of each block.
* `--tx-cost-mode amortized` → costs applied inside the loop evenly across the block.
* `--tx-cost-mode posthoc` → no costs during loop; after the run, we create `equity_net_of_tc.csv` with amortized costs.
* `--tx-cost-mode none` → no costs at all.

**Outputs:** `metrics.csv`, `equity.csv`, `factor_diagnostics.json` and (if matplotlib is present) `equity.png`.

---

### Verifying CLI outputs

The CLI writes every artifact into the directory passed via `--out`. After a run, list
the folder to confirm `rebalance_report.csv` and the other CSVs were written:

```bash
ls -l /tmp/te
```

Common files include:

* `metrics.csv` — summary Sharpe (net of the configured risk-free rate), drawdown, turnover, etc.
* `equity.csv` — gross equity curve
* `equity_net_of_tc.csv` — equity net of transaction costs (when `tx_cost_mode=posthoc`)
* `rebalance_report.csv` — per-block turnover, costs, realized returns, and block-level Sharpe/Sortino/IR diagnostics
* `run_config.json` — CLI arguments, resolved constraints, and package metadata

Preview the contents with `head` to make sure data was recorded:

```bash
head -n 5 /tmp/te/rebalance_report.csv
head -n 20 /tmp/te/metrics.csv
```

If the dataset is shorter than the configured `--lookback`, the CLI still emits
`rebalance_report.csv` (header only) so downstream tools do not fail. In that case,
inspect `run_config.json` for `"warnings": ["no_rebalances"]` to confirm the guardrail
was triggered. Provide a longer history or decrease the lookback to produce rebalance
windows.

---

## Factor inputs

Factor panels can be supplied as CSV (wide or multi-index), parquet, or YAML. The loader
normalizes them into a `(T, N, K)` cube and validates:

* Factor names are unique and free of NaNs/inf values.
* Assets overlap with the return universe; missing assets are dropped with a summary in
  `factor_diagnostics.json`.
* Dates align with the rebalance grid. `--factor-align strict` requires every rebalance
  to have factor data; `subset` keeps the overlapping windows and records the gaps so
  factor neutrality can be skipped for those dates.
* Use `--factors-required` to force a hard failure when any window is missing, even in
  subset mode.

**Example CSV (multi-index columns):**

```
date,A,A,B,B
,F0,F1,F0,F1
2020-01-01,0.1,0.0,-0.2,0.4
2020-01-02,0.2,0.1,-0.1,0.3
```

**Equivalent YAML:**

```yaml
2020-01-01:
  A: [0.1, 0.0]
  B: [-0.2, 0.4]
2020-01-02:
  A: [0.2, 0.1]
  B: [-0.1, 0.3]
```

**Common validation errors and fixes:**

* `Factor names must be unique` → rename duplicate column headers.
* `Factor loadings must not contain NaNs or infs` → fill or drop blank cells.
* `Factor panel is missing required rebalance dates` → switch to `--factor-align subset`
  or extend the panel to cover the rebalance date.

`factor_diagnostics.json` reports counts of dropped assets/dates and any missing windows so
automated pipelines can decide whether to proceed.

---

## Active constraints

Active weights are measured relative to the benchmark: `w_active = w - w_bench`. Provide a
benchmark via `benchmark_csv` (CLI) or `benchmark_weights` (API) to unlock the projection logic.

**Flags and keys**

* `active_min` / `active_max` clamp per-asset active weights around the benchmark exposure.
* `active_group_caps` accepts maps or lists describing sector/cluster totals and symmetric caps or
  explicit `[lower, upper]` bounds.
* `factor_bounds` supplies lower/upper limits for factor exposures; `factor_tolerance` controls how
  strict the projection is (tighten to `~1e-8` for hard caps, relax to `1e-4+` for soft bounds).
* `factors_required` forces the run to fail when factor windows are missing, avoiding silent skips
  that would otherwise relax active group projections on those dates.

See the templates in `examples/configs/` for end-to-end wiring:

* `active_box.yaml` — symmetric active bounds around the benchmark weights.
* `active_groups.yaml` — sector-level caps mixing explicit bounds and symmetric group caps.
* `factor_bounds.yaml` — demonstrates soft vs. hard factor bounds via `factor_tolerance`.

**Common failure modes**

* Benchmark weight vectors must align with the return universe; mismatches raise an error before
  training begins.
* Active groups referencing unknown tickers are reported and ignored; duplicated assignments throw
  immediately.
* Factor bounds require the factor loader to emit the named factors; unknown entries are logged and
  skipped. Tighten `factor_tolerance` when hard bounds are desired—the default leaves a small slack
  so SLSQP and the projection stay consistent.

---

## End-to-end config

The CLI accepts YAML/JSON configs. The following snippet wires up returns, benchmark,
and factors with OAS covariance and moderate refinement cadence:

```yaml
csv: data/returns.csv
benchmark_csv: data/benchmark.csv
factors: data/factors.csv
lookback: 252
step: 21
cov_model: oas
refine_every: 2
factor_align: subset
factors_required: false
factor_tolerance: 1e-5
tx_cost_bps: 5
tx_cost_mode: amortized
slippage: proportional:10
out: runs/oas_subset
```

See `examples/configs/` for runnable templates (weekly, monthly, and factor-neutral runs).

---

## Deterministic runs

The optimizer seeds NumPy and PyTorch separately; provide a fixed `--seed` and `--refine-every`
to obtain repeatable runs. The covariance cache now keys off the model, span, lookback, and dtype,
so repeated executions with the same inputs produce identical rebalance records.

`plot_equity.py --overlay` can overlay multiple equity curves from different runs for quick
comparison.

---

## Testing

From the repository root:

```bash
pytest -q
```

The test harness in `tests/conftest.py` automatically adds `neuro-ant-optimizer/src`
to `sys.path`, so no manual `PYTHONPATH` configuration or editable install is required.

---

## Offline usage (no install)

If your environment blocks package downloads:

```bash
# Run the CLI module directly from the repo checkout
python -m neuro_ant_optimizer.backtest \
  --csv neuro-ant-optimizer/backtest/sample_returns.csv \
  --lookback 5 --step 2 --ewma_span 3 --objective sharpe --out neuro-ant-optimizer/backtest/out_local
# The repo includes a lightweight shim (neuro_ant_optimizer/__init__.py) that
# delegates to src/neuro_ant_optimizer so no PYTHONPATH edits are needed.
# This shim is dev-only; wheels/sdists still only include src/neuro_ant_optimizer.
```

---

## Offline wheel build & install

You can build a wheel locally and install it without hitting the internet:

```bash
# Build wheel from source (no network needed for your own package)
python -m pip install --upgrade pip wheel setuptools   # if available locally
python -m pip wheel . -w dist

# Install the wheel offline (no deps)
python -m pip install --no-deps --no-index dist/neuro_ant_optimizer-*.whl
```

> **Note:** optional extras like `[backtest]` pull external packages. For fully offline use,
> prefer the `python -m neuro_ant_optimizer.backtest` invocation shown above, or pre-stage
> wheels for `pandas`/`matplotlib` on an internal index and install with `--find-links`.
