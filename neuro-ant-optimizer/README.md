# Neuro–Ant Colony Portfolio Optimizer (Robust Edition)

Neural-guided Ant Colony optimizer for portfolio construction with **robust constraints** (box, leverage,
turnover, sector caps) and optional local **SLSQP refinement**. Includes safe numerics (nearest‑PSD) and
deterministic seeding.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python examples/quickstart.py
```

## Tests
```bash
pytest
```

## Features
- Ant Colony proposals guided by a **pheromone attention** network.
- Optional **risk head** to provide per-asset priors (proxy-trained, bounded).
- **Turnover** and **sector concentration** constraints enforced.
- **Local SLSQP** refinement over top-K candidates.
- **Nearest‑PSD** covariance repair and safe softmax.
