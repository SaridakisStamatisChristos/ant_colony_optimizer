# Changelog

## 0.4.1
- Covariance model extensions, benchmark objective tweaks, and risk-free handling refinements.
- Expanded rebalance report schema with stability checks and fixture coverage.
- Factor diagnostics polish plus parquet export path hardening.
- Reproducibility tooling (manifest replayer, safety rails, CLI entry point) and determinism audits.
- CI improvements with optional `[backtest]` leg, slow-test gating, and docs/code sync guards.

## 0.4.0
- MkDocs documentation with quickstart, configuration reference, artifact schema, and reproducibility guide.
- Deterministic CLI flag (`--deterministic`) with manifest tracking and strict Torch enforcement.
- Duplicate-date guardrails with optional `--drop-duplicates` flow and accompanying tests.
- Performance pass: cached ant transition logits, batched risk-network calls, and reusable buffers (≈25% faster on the provided benchmark).
- Packaging polish: example configs shipped in the wheel, classifiers/metadata tightened, README marked as Markdown.

## 0.3.0
- Add diagonal covariance shrinkage + CVaR objective
- Cache pheromone transitions per-iteration; inference under no_grad
- Replace BCE with KL→EMA + entropy PolicyTrainer
- Device/dtype-safe models; deterministic seeding; PSD & masked softmax
- Turnover/tx-cost penalties; gradient/device tests; shrinkage tests
