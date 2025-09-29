# Optimizer and Backtest Behaviour Reference

## Core Optimizer Defaults
| Setting | Default | Notes |
| --- | --- | --- |
| `n_ants` | 16 | Colony width for the stochastic search phase.【F:src/neuro_ant_optimizer/optimizer.py†L28-L76】 |
| `max_iter` | 20 | Total ant-colony iterations before convergence checks stop the loop.【F:src/neuro_ant_optimizer/optimizer.py†L28-L76】 |
| `topk_refine` | 4 | Number of elite portfolios that are passed to SLSQP for deterministic polishing.【F:src/neuro_ant_optimizer/optimizer.py†L28-L76】 |
| `refine_every` | 1 | Backtests request refinement on every window by default, so SLSQP receives candidates each rebalance unless overridden.【F:src/neuro_ant_optimizer/backtest/backtest.py†L3524-L3578】 |

## Runtime Budgeting
- `OptimizerConfig.max_runtime` defaults to two seconds and is enforced each optimization loop; exceeding the budget triggers an early exit with a runtime-budget message.【F:src/neuro_ant_optimizer/optimizer.py†L28-L76】【F:src/neuro_ant_optimizer/optimizer.py†L332-L374】

## Refinement Scheduling and SLSQP Usage
- The backtest toggles refinement by computing `should_refine = (len(weights) % refine_every) == 0`, so refinement only fires on windows that align with the `refine_every` cadence.【F:src/neuro_ant_optimizer/backtest/backtest.py†L4149-L4166】
- Inside `NeuroAntPortfolioOptimizer.optimize`, the expensive SLSQP pass is wrapped in `_refine_topk` and is executed only when `refine=True`, ensuring SLSQP runs exclusively on the scheduled windows.【F:src/neuro_ant_optimizer/optimizer.py†L308-L318】【F:src/neuro_ant_optimizer/optimizer.py†L1092-L1239】

## Behaviour When No Rebalances Occur
- If no rebalance windows are available, the backtest returns empty arrays, attaches a `"no_rebalances"` warning, and still emits metadata such as constraint manifests and cov-cache stats.【F:src/neuro_ant_optimizer/backtest/backtest.py†L3898-L3957】
- The rebalance report writer always outputs the full CSV header—including turnover, block-metric, and compliance audit columns (pre/post-trade flags, breach counts, and reason strings)—so a no-rebalance run yields a header-only `rebalance_report.csv` alongside the warning.【F:src/neuro_ant_optimizer/backtest/backtest.py†L5630-L5670】

## Covariance Model Cache Keying
- Covariance caching keys include the chosen model (or custom spec), sorted parameter items, EWMA span (when relevant), and a hash of the training window, preventing collisions between configurations.【F:src/neuro_ant_optimizer/backtest/backtest.py†L3547-L3548】【F:src/neuro_ant_optimizer/backtest/backtest.py†L3960-L4043】

## Benchmark Metrics and Annualisation
- Tracking error is annualised via `sqrt(trading_days)` inside `compute_tracking_error`, aligning TE/IR with the supplied trading-day count.【F:src/neuro_ant_optimizer/backtest/backtest.py†L1946-L1952】
- Full-period metrics annualise active means and reuse the same TE helper, producing annualised info ratios for completed runs.【F:src/neuro_ant_optimizer/backtest/backtest.py†L4649-L4698】
- Block-level summaries capture per-window Sharpe, Sortino, info ratio, and tracking error using the same annualisation factors, so CSVs and callbacks reflect consistent scaling.【F:src/neuro_ant_optimizer/backtest/backtest.py†L4460-L4528】

## Risk-Free Parameterisation
- CLI flag `--rf-bps` feeds into `risk_free_rate=float(parsed.rf_bps)/1e4`, with annualisation handled via `trading_days` to derive a per-period risk-free rate for Sharpe-like metrics.【F:src/neuro_ant_optimizer/backtest/backtest.py†L6513-L6527】【F:src/neuro_ant_optimizer/backtest/backtest.py†L3549-L3577】

## Factor Alignment Modes and Diagnostics
- Factor panels can be aligned in `strict` (default) or `subset` mode; strict mode demands exact date coverage, while subset mode drops missing dates/assets and tracks them in diagnostics.【F:src/neuro_ant_optimizer/backtest/backtest.py†L3289-L3358】【F:src/neuro_ant_optimizer/backtest/backtest.py†L3359-L3386】
- `FactorDiagnostics` records dropped assets/dates and any rebalance windows that lack factor data, exposing counts plus sorted lists via `to_dict()`. Missing windows discovered during simulation are appended incrementally.【F:src/neuro_ant_optimizer/backtest/backtest.py†L589-L626】【F:src/neuro_ant_optimizer/backtest/backtest.py†L4141-L4143】
- The CLI persists diagnostics as `factor_diagnostics.json` when present, making alignment issues visible in outputs.【F:src/neuro_ant_optimizer/backtest/backtest.py†L6748-L6763】

