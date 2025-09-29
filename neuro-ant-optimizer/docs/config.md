# Configuration reference

`neuro-ant-backtest` accepts the same options via CLI flags or YAML/JSON files (replace
CLI dashes with underscores in config files). The tables below group the most frequently
used parameters by category. Refer to `neuro-ant-backtest --help` for an exhaustive list
including esoteric research toggles.

## Data ingestion

| Flag(s) | Default | Description |
| --- | --- | --- |
| `--csv` | `None` | CSV/Parquet of asset returns. First column must be a timestamp. |
| `--benchmark-csv` | `None` | Optional benchmark series aligned to the returns index. |
| `--io-backend` | `auto` | Dataframe engine (`auto`, `pandas`, `polars`, or the built-in minimal reader). |
| `--data-freq` | `B` | Expected calendar (`B`, `D`, `W`, `M`, etc.) enforced during ingestion. |
| `--data-tz` | `UTC` | Timezone applied before PIT validation and calendar alignment. |
| `--dropna` | `none` | Row-drop policy (`none`, `any`, `all`) before calendar checks. |
| `--columns` | `None` | Optional asset filters. Accepts `SRC` or `SRC:ALIAS` tokens. |
| `--column-map` | `None` | YAML/JSON mapping applied to column names prior to alignment. |
| `--no-pit` | `False` | Disable the future-date guard (enabled by default). |
| `--factors` | `None` | Factor loadings panel (CSV, Parquet, or YAML bundle). |
| `--factor-align` | `strict` | Alignment policy for factor timestamps (`strict`/`subset`). |
| `--factors-required` | `False` | Fail when any window lacks factor data (even in `subset` mode). |
| `--factor-targets` | `None` | Optional factor target vector to enforce neutrality. |
| `--factor-tolerance` | `1e-6` | Infinity-norm tolerance for factor neutrality residuals. |

## Baselines, penalties, and overlays

| Flag(s) | Default | Description |
| --- | --- | --- |
| `--baseline` | `none` | Convex overlay to report alongside the strategy (`minvar`, `maxret`, `riskparity`). |
| `--lambda-tc` | `0.0` | Turnover penalty applied to baseline weights (higher shrinks turnover). |
| `--tx-cost-bps` | `0.0` | Transaction cost per rebalance, in basis points. |
| `--tx-cost-mode` | `posthoc` | When to apply transaction costs (`none`, `upfront`, `amortized`, `posthoc`). |
| `--nt-band` | `0` | No-trade band width around previous weights (accepts raw, `%`, or `bps`). |
| `--slippage` | `None` | Slippage model spec (e.g. `proportional:5`, `impact:k=25,alpha=1.5`). |
| `--active-min` / `--active-max` | `None` | Scalar active bounds relative to the benchmark. |
| `--active-group-caps` | `None` | YAML/JSON structure defining sector or group active caps. |
| `--factor-bounds` | `None` | YAML/JSON file containing factor exposure bounds. |
| `--benchmark-weights` | `None` | Reference weights when evaluating active constraints. |
| `--te-target` | `0.0` | Tracking-error target for the `te_target` objective. |
| `--lambda-te` | `0.0` | Tracking-error penalty weight in multi-term objectives. |
| `--gamma-turnover` | `0.0` | Turnover penalty applied inside the optimizer objective. |

## Optimizer controls

| Flag(s) | Default | Description |
| --- | --- | --- |
| `--lookback` | `252` | Rolling window size in periods. |
| `--step` | `21` | Step size between rebalances. |
| `--cov-model` | `sample` | Covariance backend (`sample`, `ewma`, `oas`, `lw`, `ridge`, etc.). |
| `--ewma-span` | `60` | Span used when `--cov-model=ewma`. |
| `--objective` | `sharpe` | Objective (`sharpe`, `max_return`, `min_variance`, `risk_parity`, `min_cvar`, ...). |
| `--refine-every` | `1` | Apply SLSQP refinement every *k* rebalances. |
| `--seed` | `7` | Global RNG seed (NumPy + Torch). |
| `--deterministic` | `False` | Enable deterministic Torch kernels (raises when unsupported). |
| `--float32` | `False` | Downcast NumPy operations to `float32`. |
| `--warm-start` | `None` | Path to `weights.csv` from a prior run. |
| `--warm-align` | `last_row` | Align warm-start weights by date or use the last row. |
| `--decay` | `0.0` | Blend between previous and proposed weights (0 disables). |
| `--rf-bps` | `0.0` | Annual risk-free rate (basis points) used for reporting. |
| `--trading-days` | `252` | Trading periods per year for annualisation. |
| `--metric-alpha` | `0.05` | Tail probability used for realized CVaR reporting. |

## Performance & outputs

| Flag(s) | Default | Description |
| --- | --- | --- |
| `--workers` | `None` | Number of processes for parallel window evaluation (`None` auto-detects). |
| `--prefetch` | `2` | Number of rebalance windows queued for multiprocessing. |
| `--cache-cov` | `8` | Covariance cache size (set to 0 to disable). |
| `--progress` | `False` | Stream progress updates to stderr. |
| `--out` | `bt_out` | Directory for run artifacts. |
| `--out-format` | `csv` | Artifact format (`csv` or `parquet`). |
| `--save-weights` | `False` | Write `weights.csv` with per-window allocations. |
| `--skip-plot` | `False` | Skip equity plot generation. |
| `--dry-run` | `False` | Validate configuration without running the optimizer. |
| `--log-json` | `None` | Append JSON lines per rebalance. |
| `--runs-csv` | `None` | Append summary stats to a tracking ledger. |
| `--track-artifacts` | `None` | Directory where zipped artifacts are archived per run. |

## Configuration files

Embed any flag inside a YAML/JSON config by replacing dashes with underscores:

```yaml
csv: data/returns_daily.csv
cov_model: ewma
ewma_span: 63
baseline: minvar
lambda_tc: 1.5
workers: 4
prefetch: 4
io_backend: polars
out: runs/daily_ewma
```

The packaged templates in `neuro_ant_optimizer.examples.iter_configs()` cover daily
and weekly workflows, factor-aware runs, and Polars-enabled fast paths.
