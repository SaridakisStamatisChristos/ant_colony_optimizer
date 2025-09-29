"""Streaming intraday optimisation engine."""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Dict, MutableMapping, Protocol

from ..state.positions import PositionsStore


class FeedHandler(Protocol):
    """Protocol describing the minimal feed handler contract."""

    def stream(self) -> Iterable[tuple[Any, Mapping[str, float]]]:
        """Yield (timestamp, minibatch) pairs in arrival order."""


@dataclass
class LatencyEvent:
    """Metrics for a processed minibatch."""

    timestamp: Any
    elapsed_ms: float
    dropped: bool = False


class IntradayEngine:
    """Run objective updates for streaming market data."""

    def __init__(
        self,
        feed_handler: FeedHandler,
        objective: Callable[[Mapping[str, float], Mapping[str, float]], MutableMapping[str, float]],
        *,
        state_store: PositionsStore | None = None,
        latency_budget_ms: float = 50.0,
        drop_overrun: bool = False,
    ) -> None:
        self._feed_handler = feed_handler
        self._objective = objective
        self._store = state_store
        self.latency_budget_ms = latency_budget_ms
        self.drop_overrun = drop_overrun
        self.latency_events: list[LatencyEvent] = []
        self._current_weights: Dict[str, float] = {}
        if self._store is not None:
            self._current_weights.update(self._store.load())

    @property
    def current_weights(self) -> Dict[str, float]:
        """Return the latest portfolio weights."""

        return dict(self._current_weights)

    def _iter_feed(self) -> Iterator[tuple[Any, Mapping[str, float]]]:
        stream = self._feed_handler.stream()
        if isinstance(stream, Iterator):
            return stream
        return iter(stream)

    def run(self) -> Dict[str, float]:
        """Consume the entire feed and persist the final weights."""

        for timestamp, minibatch in self._iter_feed():
            warm_start = dict(self._current_weights)
            start = time.perf_counter()
            new_weights = self._objective(minibatch, warm_start)
            elapsed_ms = (time.perf_counter() - start) * 1_000
            dropped = bool(self.drop_overrun and elapsed_ms > self.latency_budget_ms)
            self.latency_events.append(LatencyEvent(timestamp=timestamp, elapsed_ms=elapsed_ms, dropped=dropped))
            if dropped:
                continue
            self._current_weights = dict(new_weights)
            if self._store is not None:
                self._store.save(self._current_weights)
        return dict(self._current_weights)
