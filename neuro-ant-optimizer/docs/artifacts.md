# Artifact schema

`neuro-ant-backtest` writes a consistent set of CSV (and optionally Parquet) artifacts to the output directory. The sections below document the schema so that downstream tooling can ingest them without reverse-engineering the code.

## `metrics.csv`

A two-column table of run-level aggregates:

| Metric | Description |
| --- | --- |
| `sharpe` | Annualized Sharpe ratio net of the configured risk-free rate. |
| `ann_return` / `ann_vol` | Annualized return and volatility of the gross equity curve. |
| `max_drawdown` | Maximum drawdown depth. |
| `avg_turnover` | Average per-rebalance turnover after decay. |
| `avg_slippage_bps` | Mean slippage in basis points (0 if slippage disabled). |
| `downside_vol` / `sortino` | Annualized downside volatility and Sortino ratio. |
| `realized_cvar` | CVaR of realized returns at the configured `--metric-alpha`. |
| `tracking_error` / `info_ratio` | Annualized tracking error and information ratio (when a benchmark is provided). |
| `te_target` / `lambda_te` / `gamma_turnover` | Objective hyperparameters echoed back for auditability. |
| `cov_cache_*` | Cache size, hits, misses, and evictions for covariance memoization. |
| `baseline_*` | Baseline Sharpe/IR, alpha, and hit-rate versus the selected baseline (only present when `--baseline` is used). |

## `rebalance_report.csv`

One row per rebalance window with feasibility, turnover, and diagnostic metrics.

| Column | Meaning |
| --- | --- |
| `date` | Rebalance date (end of the window). |
| `gross_ret` / `net_tx_ret` / `net_slip_ret` | Block returns before costs, net of transaction costs, and net of slippage. |
| `turnover_pre_decay` / `turnover_post_decay` | Turnover before and after applying the decay blend. `turnover` mirrors the post-decay value. |
| `tx_cost` / `slippage_cost` | Costs incurred during the block. |
| `sector_breaches` / `active_breaches` / `group_breaches` / `factor_bound_breaches` | Count of constraint violations detected after projection. |
| `factor_inf_norm` | Infinity norm of factor exposures relative to the target vector. |
| `factor_missing` | `True` if the factor panel lacked data for this window. |
| `first_violation` | First constraint violation encountered (e.g. `ACTIVE_BOX`, `GROUP_CAP`). |
| `feasible` | Whether the projected solution met all constraints. |
| `projection_iterations` | Number of projection refinement iterations performed. |
| `block_sharpe` / `block_sortino` | Annualized Sharpe and Sortino metrics for the block. |
| `block_info_ratio` / `block_tracking_error` | Block-level IR and tracking error when a benchmark is supplied. |
| `warm_applied` | Indicates whether warm-start weights were used. |
| `decay` | Decay parameter applied on this window. |

## `exposures.csv`

Present when factors are supplied. Each row captures realized factor and sector exposures on a rebalance date.

- `date`: Rebalance date (skips rows where factor data were missing).
- Factor columns: one column per factor in `factor_names`, populated with realized exposures (zeros if beyond the available factor count).
- Sector columns: appended dynamically when sector-level constraints are evaluated; missing sectors report `0.0`.

## Other key outputs

| File | Description |
| --- | --- |
| `weights.csv` | Per-rebalance weights (with a `date` column when the run produced windows). |
| `equity.csv` | Gross equity curve (`date`, `equity`, `ret`). Additional net-of-cost files (`equity_net_of_tc.csv`, `equity_net_of_slippage.csv`) are produced when costs/slippage are configured. |
| `drawdowns.csv` | Peak/trough/recovery timestamps and drawdown depth/length. |
| `contrib.csv` | Per-asset contribution and realized block return for each rebalance. |
| `run_config.json` | Serialized manifest containing CLI args, resolved constraints, deterministic flag, cache stats, warnings, and git/python metadata. |
| `factor_constraints.csv` | Lower/upper bounds for factor targets (written when factor constraints are active). |

### Parquet outputs

Passing `--out-format parquet` writes the same tables as Parquet files alongside the CSVs. The column names and dtypes match the definitions above.
