"""CLI for triggering simple scenario runs."""
from __future__ import annotations

import argparse
import json

from .runner import ScenarioConfig, ScenarioRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Neuro Ant scenario runner")
    parser.add_argument("--shock", type=float, default=0.0, help="Uniform return shock")
    args = parser.parse_args(argv)

    runner = ScenarioRunner([0.001, 0.0, -0.0005], transaction_cost=0.0001)
    result = runner.run(ScenarioConfig(return_shock=args.shock))
    payload = {
        "adjusted_returns": result.adjusted_returns.tolist(),
        "portfolio_path": result.portfolio_path.tolist(),
        "breaches": result.breaches,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
