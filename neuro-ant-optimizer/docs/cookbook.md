# Cookbooks

## From CSV to equity

This playbook shows how to ingest raw returns, validate them against a business-day
calendar, and run a tracked backtest.

1. **Load returns with PIT safeguards**

   ```python
   from neuro_ant_optimizer.data import load_returns

   returns = load_returns(
       "examples/data/returns_daily.csv",
       freq="B",
       tz="UTC",
       columns=["AAPL", "MSFT", "GOOGL"],
       dropna="any",
   )
   ```

   The loader coerces timestamps onto the requested calendar, fails on duplicate rows,
   and raises `PITViolationError` when the last observation sits in the future.

2. **Run the CLI with multiprocessing**

   ```bash
   neuro-ant-backtest --csv examples/data/returns_daily.csv \
     --lookback 252 --step 21 --cov-model ewma --ewma-span 63 \
     --baseline minvar --lambda-tc 1.5 --workers 4 --prefetch 4 \
     --io-backend auto --out runs/tutorial_daily
   ```

   The command mirrors the `examples/configs/daily_ewma.yaml` template and emits a
   deterministic equity curve along with the convex baseline overlay.

3. **Inspect the outputs**

   Each run produces `equity.csv`, `weights.csv` (when `--save-weights` is enabled),
   and `run_config.json` which captures the loader settings, multiprocessing flags, and
   random seed. Use `python -m neuro_ant_optimizer.backtest.reproduce --run-id ...` to
   replay from the manifest.

## PIT-safe ingestion

The loader normalises timestamps by:

- Aligning them against `freq` (`B`, `W`, `M`, etc.) and the requested timezone.
- Rejecting duplicates after timezone conversion.
- Refusing to read the final row when it points to a future instant in the provided
  timezone (keeping backtests strictly point-in-time).

Toggle PIT checks with the `pit` keyword or `--no-pit` CLI flag when replaying historic
snapshots that intentionally embed future dates (rare but occasionally required for
simulated feeds).

## Fast path with Polars

Install the optional extras and switch the backend:

```bash
python -m pip install "neuro-ant-optimizer[polars,arrow]"
neuro-ant-backtest --config examples/configs/daily_polars_fast.yaml
```

When Polars is available the loader streams CSV/Parquet files directly into Arrow
memory, avoiding Python object overhead. The parallel window executor remains
bit-for-bit deterministic provided you set `--deterministic` (or `deterministic: true`
in configs) and seed the RNG.

For lighter environments without Polars, fall back to pandas by passing
`--io-backend pandas` or rely on the built-in minimal CSV reader.

## Weekly factors with transaction costs

The `examples/configs/weekly_oas_factors.yaml` template demonstrates weekly data with
OAS shrinkage, strict factor coverage, amortised transaction costs, and a risk-parity
baseline with turnover penalties. Use it as a starting point for custom sector bounds
or slippage models when scaling to production universes.
