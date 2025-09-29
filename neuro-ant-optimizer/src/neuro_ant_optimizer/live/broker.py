"""Simulated broker bridge with idempotent order submission."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Mapping

from ..state.positions import PositionsStore


class ThrottleError(RuntimeError):
    """Raised when submissions violate the throttle window."""


@dataclass(frozen=True)
class OrderSubmission:
    """Representation of a target order request."""

    symbol: str
    target_weight: float
    target_notional: float
    client_order_id: str


class SimulatedBroker:
    """Paper broker that converts target weights into notionals."""

    def __init__(
        self,
        *,
        positions_store: PositionsStore,
        throttle_window: float = 0.5,
    ) -> None:
        self._store = positions_store
        self.throttle_window = throttle_window
        self._last_submit_ts: float = 0.0
        self._order_cache: dict[str, list[OrderSubmission]] = {}

    def _check_throttle(self) -> None:
        now = time.monotonic()
        if now - self._last_submit_ts < self.throttle_window:
            raise ThrottleError("order submissions throttled")
        self._last_submit_ts = now

    def submit_target_weights(
        self,
        target_weights: Mapping[str, float],
        *,
        account_value: float,
        client_order_id: str,
    ) -> list[OrderSubmission]:
        """Submit target weights and persist resulting positions."""

        if client_order_id in self._order_cache:
            return self._order_cache[client_order_id]

        self._check_throttle()
        existing = self._store.load()
        orders: list[OrderSubmission] = []
        updated: Dict[str, float] = dict(existing)
        for symbol, weight in target_weights.items():
            target_notional = weight * account_value
            orders.append(
                OrderSubmission(
                    symbol=symbol,
                    target_weight=weight,
                    target_notional=target_notional,
                    client_order_id=client_order_id,
                )
            )
            updated[symbol] = target_notional
        self._store.save(updated)
        self._order_cache[client_order_id] = orders
        return orders

    def load_positions(self) -> Dict[str, float]:
        """Return the latest stored notionals."""

        return self._store.load()
