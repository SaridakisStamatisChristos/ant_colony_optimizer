"""Validation helpers for portfolio limit enforcement."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from neuro_ant_optimizer.risk.limits import LimitBreach, LimitEvaluator


def build_limit_evaluator(
    spec: Optional[Mapping[str, object]],
    *,
    assets: Sequence[str],
    sector_lookup: Optional[Mapping[str, str]] = None,
) -> Optional[LimitEvaluator]:
    """Instantiate a :class:`LimitEvaluator` from a raw specification."""

    if spec is None:
        return None
    if isinstance(spec, LimitEvaluator):
        return spec
    return LimitEvaluator.from_spec(spec, assets=assets, sector_lookup=sector_lookup)


def _format_reasons(breaches: Sequence[LimitBreach]) -> List[str]:
    return [breach.reason() for breach in breaches]


def pre_trade_check(
    weights: Sequence[float],
    evaluator: Optional[LimitEvaluator],
) -> Tuple[bool, List[str], List[LimitBreach]]:
    """Run pre-trade checks returning a boolean flag and reason tokens."""

    if evaluator is None:
        return True, [], []
    breaches = evaluator.evaluate(weights, phase="PRE")
    return not breaches, _format_reasons(breaches), list(breaches)


def post_trade_check(
    weights: Sequence[float],
    evaluator: Optional[LimitEvaluator],
) -> Tuple[bool, List[str], List[LimitBreach]]:
    """Run post-trade checks on realized weights."""

    if evaluator is None:
        return True, [], []
    breaches = evaluator.evaluate(weights, phase="POST")
    return not breaches, _format_reasons(breaches), list(breaches)


def summarize_breaches(entries: Iterable[LimitBreach]) -> List[Mapping[str, object]]:
    """Aggregate breach counts by phase and limit type."""

    counter: Counter[Tuple[str, str]] = Counter()
    for breach in entries:
        counter[(breach.phase, breach.limit_type)] += 1
    summary: List[Mapping[str, object]] = []
    for (phase, limit_type), count in sorted(counter.items()):
        summary.append({"phase": phase, "type": limit_type, "count": int(count)})
    return summary


def ensure_array(weights: Sequence[float]) -> np.ndarray:
    """Convenience wrapper used in tests to standardize input."""

    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr
