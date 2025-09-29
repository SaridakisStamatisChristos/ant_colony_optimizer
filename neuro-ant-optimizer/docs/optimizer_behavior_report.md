# Optimizer and Backtest Behaviour Reference

## Core Optimizer Defaults
| Setting | Default | Notes |
| --- | --- | --- |
| `n_ants` | 16 | Colony width for the stochastic search phase (see `OptimizerConfig`). |
| `max_iter` | 20 | Total ant-colony iterations before convergence checks stop the loop. |
| `topk_refine` | 4 | Number of elite portfolios that are passed to SLSQP for deterministic polishing. |
| `refine_every` | 1 | Backtests request refinement on every window by default, so SLSQP receives candidates each rebalance unless overridden. |

## Runtime Budgeting
- `OptimizerConfig.max_runtime` defaults to two seconds and is enforced each optimization loop; exceeding the budget triggers an early exit with a runtime-budget message.

## Refinement Scheduling and SLSQP Usage
- The backtest toggles refinement by computing `should_refine = (len(weights) % refine_every) == 0`, so refinement only fires on windows that align with the `refine_every` cadence.
- Inside `NeuroAntPortfolioOptimizer.optimize`, the expensive SLSQP pass is wrapped in `_refine_topk` and is executed only when `refine=True`, ensuring SLSQP runs exclusively on the scheduled windows.

## Behaviour When No Rebalances Occur
- If no rebalance windows are available, the backtest returns empty arrays, attaches a `"no_rebalances"` warning, and still emits metadata such as constraint manifests and cov-cache stats.
- The rebalance report writer always outputs the full CSV header—including turnover, block-metric, and compliance audit columns (pre/post-trade flags, breach counts, and reason strings)—so a no-rebalance run yields a header-only `rebalance_report.csv` alongside the warning.

## Covariance Model Cache Keying
- Covariance caching keys include the chosen model (or custom spec), sorted parameter items, EWMA span (when relevant), and a hash of the training window, preventing collisions between configurations.

## Benchmark Metrics and Annualisation
- Tracking error is annualised via `sqrt(trading_days)` inside `compute_tracking_error`, aligning TE/IR with the supplied trading-day count.
- Full-period metrics annualise active means and reuse the same TE helper, producing annualised info ratios for completed runs.
- Block-level summaries capture per-window Sharpe, Sortino, info ratio, and tracking error using the same annualisation factors, so CSVs and callbacks reflect consistent scaling.

## Risk-Free Parameterisation
- CLI flag `--rf-bps` feeds into `risk_free_rate=float(parsed.rf_bps)/1e4`, with annualisation handled via `trading_days` to derive a per-period risk-free rate for Sharpe-like metrics.

## Factor Alignment Modes and Diagnostics
- Factor panels can be aligned in `strict` (default) or `subset` mode; strict mode demands exact date coverage, while subset mode drops missing dates/assets and tracks them in diagnostics.
- `FactorDiagnostics` records dropped assets/dates and any rebalance windows that lack factor data, exposing counts plus sorted lists via `to_dict()`. Missing windows discovered during simulation are appended incrementally.
- The CLI persists diagnostics as `factor_diagnostics.json` when present, making alignment issues visible in outputs.

