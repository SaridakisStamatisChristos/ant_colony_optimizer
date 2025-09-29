"""Factor attribution helpers used for diagnostics artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np


@dataclass
class FactorContribution:
    date: str
    factor: str
    contribution: float


@dataclass
class FactorAttribution:
    contributions: List[FactorContribution]
    cumulative: List[Mapping[str, float]]

    def contribution_rows(self) -> List[Mapping[str, object]]:
        return [
            {"date": item.date, "factor": item.factor, "contribution": float(item.contribution)}
            for item in self.contributions
        ]

    def cumulative_rows(self) -> List[Mapping[str, object]]:
        return [dict(row) for row in self.cumulative]


def _to_array(value: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def compute_factor_contributions(
    factor_attr_records: Sequence[Mapping[str, object]],
    factor_records: Sequence[Mapping[str, object]],
    factor_names: Sequence[str],
) -> FactorAttribution:
    """Compute per-period factor contributions and cumulative ladders."""

    factor_returns: Dict[str, Dict[str, float]] = {}
    for record in factor_attr_records:
        date = str(record.get("date"))
        factor_returns[date] = {}
        for name in factor_names:
            key = f"{name}_return"
            if key in record and record[key] is not None:
                factor_returns[date][name] = float(record[key])

    contributions: List[FactorContribution] = []
    cumulative_tracker: Dict[str, float] = {str(name): 0.0 for name in factor_names}
    cumulative_rows: List[Mapping[str, float]] = []

    for record in factor_records:
        date_raw = record.get("date")
        date = str(date_raw)
        if record.get("missing"):
            continue
        exposures = _to_array(record.get("exposures"))
        targets = _to_array(record.get("targets"))
        if exposures is None:
            continue
        returns_for_date = factor_returns.get(date)
        if not returns_for_date:
            continue
        if targets is None:
            targets = np.zeros_like(exposures)
        if exposures.shape[0] != len(factor_names):
            continue
        if targets.shape[0] != exposures.shape[0]:
            continue
        active = exposures - targets
        row: Dict[str, float] = {"date": date}
        for idx, name in enumerate(factor_names):
            ret = returns_for_date.get(name)
            if ret is None:
                continue
            contrib = float(active[idx] * ret)
            contributions.append(FactorContribution(date=date, factor=name, contribution=contrib))
            cumulative_tracker[name] += contrib
            row[name] = float(cumulative_tracker[name])
        if len(row) > 1:
            cumulative_rows.append(row)

    return FactorAttribution(contributions=contributions, cumulative=cumulative_rows)
