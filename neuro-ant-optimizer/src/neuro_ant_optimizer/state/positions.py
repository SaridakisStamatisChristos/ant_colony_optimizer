"""Simple JSON-backed position store used by multiple components."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping


class PositionsStore:
    """Persist and retrieve portfolio state."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self) -> Dict[str, float]:
        if not self._path.exists():
            return {}
        with self._path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return {key: float(value) for key, value in raw.items()}

    def save(self, positions: Mapping[str, float]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as handle:
            json.dump({key: float(value) for key, value in positions.items()}, handle, sort_keys=True)
