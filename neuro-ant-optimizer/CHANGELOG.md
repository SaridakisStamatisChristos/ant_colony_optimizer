# Changelog

## 0.3.0
- Add diagonal covariance shrinkage + CVaR objective
- Cache pheromone transitions per-iteration; inference under no_grad
- Replace BCE with KL→EMA + entropy PolicyTrainer
- Device/dtype-safe models; deterministic seeding; PSD & masked softmax
- Turnover/tx-cost penalties; gradient/device tests; shrinkage tests
