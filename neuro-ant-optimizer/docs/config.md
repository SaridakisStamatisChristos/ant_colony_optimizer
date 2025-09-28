# Configuration reference

The CLI and YAML/JSON configs expose the same knobs. The table below lists every flag available on `neuro-ant-backtest`. Boolean flags default to `False` unless noted.

| Flag(s) | Default | Description |
| --- | --- | --- |
| `--config` | `None` | Load run parameters from a YAML/JSON file. Values override CLI defaults. |
| `--csv` | `None` | CSV of asset returns (date index in the first column). |
| `--benchmark-csv` | `None` | Optional CSV of benchmark returns (single column). Must share the returns index. |
| `--baseline` | `None` | Add an equity baseline (`equal` or `cap`). Writes baseline metrics alongside the strategy. |
| `--cap-weights` | `None` | Cap-weight CSV required when `--baseline=cap`. |
| `--active-min` / `--active-max` | `None` | Per-asset active bounds relative to the benchmark weights. |
| `--active-group-caps` | `None` | YAML/JSON file describing sector/group active bounds. |
| `--factor-bounds` | `None` | YAML/JSON file defining factor exposure bounds. |
| `--lookback` | `252` | Rolling window size in periods. |
| `--step` | `21` | Step size between rebalances (in periods). |
| `--ewma_span` | `60` | EWMA span (used only when `--cov-model=ewma`). |
| `--cov-model` | `sample` | Covariance backend (`sample`, `ewma`, `lw`, `oas`, or `custom:module:callable`). |
| `--objective` | `sharpe` | Optimization objective (`sharpe`, `max_return`, `min_variance`, `risk_parity`, `min_cvar`, `tracking_error`, `info_ratio`, `te_target`, `multi_term`, or `custom:module:callable`). |
| `--te-target` | `0.0` | Target tracking-error level for the `te_target` objective. |
| `--lambda-te` | `0.0` | Tracking-error penalty weight for `multi_term`. |
| `--gamma-turnover` | `0.0` | Turnover penalty weight for `multi_term`. |
| `--float32` | `False` | Downcast all NumPy calculations to `float32`. |
| `--deterministic` | `False` | Force `torch.use_deterministic_algorithms(True)` (raises if unavailable). Also recorded in the manifest. |
| `--cache-cov` | `8` | Max number of covariance matrices cached per run (0 disables caching). |
| `--max-workers` | `None` | Maximum threads/processes for async objective evaluation (backend-specific). |
| `--warm-start` | `None` | Path to `weights.csv` from a prior run for warm starts. |
| `--warm-align` | `last_row` | Align warm-start weights to the first rebalance (`by_date`) or use the last row. |
| `--decay` | `0.0` | Blend between prior allocations and the optimizer proposal (0 disables). |
| `--seed` | `7` | Global RNG seed (NumPy + Torch). |
| `--out` | `bt_out` | Output directory for artifacts. |
| `--log-json` | `None` | Append JSON records per rebalance (one line per window). |
| `--progress` | `False` | Stream textual progress updates to stderr. |
| `--out-format` | `csv` | Artifact format (`csv` or `parquet`). Parquet files mirror the CSV outputs. |
| `--rf-bps` | `0.0` | Annualized risk-free rate in basis points used for Sharpe/Sortino reporting. |
| `--trading-days` | `252` | Periods per year for annualization. |
| `--save-weights` | `False` | Write `weights.csv` with per-rebalance allocations. |
| `--skip-plot` | `False` | Skip generating the equity chart (useful for headless environments). |
| `--dry-run` | `False` | Validate configuration and write `run_config.json` without running the optimizer. |
| `--drop-duplicates` | `False` | Drop duplicate dates (keeping the last) from returns/benchmark inputs instead of erroring. |
| `--tx-cost-bps` | `0.0` | Transaction cost in basis points applied per rebalance. |
| `--tx-cost-mode` | `posthoc` | When to apply transaction costs (`none`, `upfront`, `amortized`, `posthoc`). |
| `--metric-alpha` | `0.05` | Tail probability used when reporting realized CVaR. |
| `--factors` | `None` | Path to factor loadings (CSV, Parquet, or YAML). |
| `--factor-tolerance` | `1e-6` | Infinity-norm tolerance for factor neutrality. |
| `--factor-targets` | `None` | CSV/Parquet/YAML vector of factor targets. |
| `--factor-align` | `strict` | Factor alignment policy (`strict` requires full coverage, `subset` allows gaps). |
| `--factors-required` | `False` | Fail if any rebalance window lacks factor data (even in `subset` mode). |
| `--slippage` | `None` | Slippage model spec (for example `proportional:5`). |
| `--refine-every` | `1` | Run SLSQP refinement every *k* rebalances. |

## Configuration files

YAML/JSON configs use the same keys as the CLI flags (replace dashes with underscores). A minimal template looks like:

```yaml
csv: data/returns.csv
out: runs/minimal
lookback: 252
step: 21
cov_model: lw
objective: sharpe
save_weights: true
skip_plot: true
```

You can override any CLI option inside the config file. Command-line flags still win if both are provided. See the bundled templates in `examples/configs/` (also exposed via `neuro_ant_optimizer.examples.iter_configs`) for end-to-end setups covering active constraints, factor bounds, and transaction-cost scenarios.
