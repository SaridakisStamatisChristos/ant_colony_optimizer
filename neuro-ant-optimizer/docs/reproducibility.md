# Reproducibility guide

The optimizer is designed to make runs repeatable across machines and architectures. Follow the checklist below to ensure deterministic results and traceability.

## Seeds and deterministic Torch backends

- Every API and CLI entry point calls `set_seed` with the configured `seed` (default `7`).
- Pass `--deterministic` to `neuro-ant-backtest` (or `deterministic=True` to `backtest`) to hard-set `torch.use_deterministic_algorithms(True)`. The flag raises if a deterministic backend is unavailable and records the choice in `run_config.json`.
- The manifest also records the torch version, Python version, git SHA (when available), cache statistics, and warm-start/decay metadata for forensic analysis.

## Input hygiene

- Use `--drop-duplicates` if your returns or benchmark CSVs contain duplicate dates; the flag keeps the last occurrence and guarantees a strictly increasing index. Without it the CLI aborts with a clear error message so that upstream data issues do not silently leak into backtests.
- Factor panels validated in `strict` mode require full coverage. Switch to `--factor-align subset` or omit `--factors-required` to tolerate gaps (diagnostics still report dropped dates).
- The CLI writes `run_config.json` even in `--dry-run` mode, allowing you to verify the manifest without running the optimizer.

## Manifests and replay

Each run emits `run_config.json` with:

- `args`: The normalized CLI/config inputs (including paths resolved at runtime).
- `deterministic_torch`: Whether `--deterministic` was requested.
- `resolved_constraints`: Active bounds, factor limits, and benchmark metadata.
- `cov_cache_*`: Cache hit/miss counters, helpful when profiling performance sensitivity.
- Optional `warnings` when the run skipped rebalances, dropped factors, or detected inconsistent inputs.

Replay a run with `python -m neuro_ant_optimizer.backtest.reproduce --manifest runs/.../run_config.json`. The helper rebuilds the CLI command (including config overrides) so you can verify changes across branches or machines.

## Performance knobs

- Covariance caching (`--cache-cov`) defaults to `8`. Increase it when sliding windows overlap heavily; set it to `0` to force a cold path when benchmarking improvements.
- Slippage and transaction-cost modelling add computation. Disable them when validating core optimization logic to minimize noise.
- Use the new vectorized ant loop (enabled by default) to improve training throughput; benchmark scripts in `bench/` report wall-time improvements when compared to the previous per-ant PyTorch calls.
