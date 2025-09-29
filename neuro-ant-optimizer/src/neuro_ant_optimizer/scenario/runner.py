"""Scenario runner for applying structured shocks to return paths."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping

import numpy as np


@dataclass
class ScenarioConfig:
    """Configuration for a scenario stress test."""

    return_shock: float = 0.0
    vol_spike: float = 0.0
    factor_tilts: Mapping[str, float] = field(default_factory=dict)
    liquidity_haircut: float = 0.0
    transaction_cost_multiplier: float = 1.0
    breach_threshold: float | None = None


@dataclass
class ScenarioResult:
    """Result from running a stress scenario."""

    adjusted_returns: np.ndarray
    portfolio_path: np.ndarray
    breaches: List[str]


class ScenarioRunner:
    """Apply scenario shocks to a stream of returns."""

    def __init__(
        self,
        base_returns: Iterable[float],
        *,
        factor_exposures: Mapping[str, float] | None = None,
        transaction_cost: float = 0.0,
        initial_value: float = 1.0,
    ) -> None:
        self._base_returns = np.asarray(list(base_returns), dtype=float)
        self._factor_exposures = dict(factor_exposures or {})
        self._transaction_cost = float(transaction_cost)
        self._initial_value = float(initial_value)

    def run(self, config: ScenarioConfig) -> ScenarioResult:
        """Run a single scenario and compute the shocked P&L path."""

        returns = self._base_returns.copy()
        if config.return_shock:
            returns = returns + config.return_shock
        if config.vol_spike:
            returns = returns * (1.0 + config.vol_spike)
        if config.factor_tilts:
            factor_contrib = sum(
                tilt * self._factor_exposures.get(factor, 0.0)
                for factor, tilt in config.factor_tilts.items()
            )
            if factor_contrib:
                returns = returns + factor_contrib
        if self._transaction_cost:
            returns = returns - (self._transaction_cost * config.transaction_cost_multiplier)
        path = self._compute_path(returns)
        if config.liquidity_haircut:
            path[-1] *= 1.0 - config.liquidity_haircut
        breaches: List[str] = []
        if config.breach_threshold is not None:
            for idx, value in enumerate(path):
                if value < config.breach_threshold:
                    breaches.append(f"breach@{idx}")
        return ScenarioResult(adjusted_returns=returns, portfolio_path=path, breaches=breaches)

    def _compute_path(self, returns: np.ndarray) -> np.ndarray:
        values = [self._initial_value]
        for period_return in returns:
            values.append(values[-1] * (1.0 + period_return))
        return np.asarray(values[1:], dtype=float)
