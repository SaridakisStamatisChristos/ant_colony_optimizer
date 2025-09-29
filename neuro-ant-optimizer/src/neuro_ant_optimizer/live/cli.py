"""Console helper for running the simulated broker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..state.positions import PositionsStore
from .broker import SimulatedBroker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Neuro Ant simulated broker")
    parser.add_argument("--state", type=Path, required=True, help="Path to persist broker state")
    parser.add_argument("--account", type=float, default=1_000_000.0, help="Account notional")
    args = parser.parse_args(argv)

    store = PositionsStore(args.state)
    broker = SimulatedBroker(positions_store=store, throttle_window=0.0)
    orders = broker.submit_target_weights({"CASH": 1.0}, account_value=args.account, client_order_id="boot")
    print(json.dumps([order.__dict__ for order in orders], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
