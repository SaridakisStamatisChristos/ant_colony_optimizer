"""Console entry point for the intraday engine."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..state.positions import PositionsStore


def main(argv: list[str] | None = None) -> int:
    """Display the current warm start weights for the intraday engine."""

    parser = argparse.ArgumentParser(description="Neuro Ant intraday engine")
    parser.add_argument("--state", type=Path, help="Path to the warm start state store", default=None)
    args = parser.parse_args(argv)

    store: PositionsStore | None = None
    if args.state is not None:
        store = PositionsStore(args.state)

    weights: dict[str, Any] = {}
    if store is not None:
        weights = store.load()
    print(json.dumps({"warm_start_weights": weights}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
