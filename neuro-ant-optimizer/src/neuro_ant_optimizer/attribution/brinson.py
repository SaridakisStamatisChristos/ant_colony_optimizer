"""Brinson performance attribution utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence

import numpy as np


@dataclass
class BrinsonRecord:
    date: str
    sector: str
    allocation: float
    selection: float
    interaction: float

    @property
    def total(self) -> float:
        return float(self.allocation + self.selection + self.interaction)

    def to_allocation_row(self) -> Mapping[str, object]:
        return {"date": self.date, "sector": self.sector, "contribution": float(self.allocation)}

    def to_selection_row(self) -> Mapping[str, object]:
        return {"date": self.date, "sector": self.sector, "contribution": float(self.selection)}

    def to_total_row(self) -> Mapping[str, object]:
        return {
            "date": self.date,
            "sector": self.sector,
            "allocation": float(self.allocation),
            "selection": float(self.selection),
            "interaction": float(self.interaction),
            "total": float(self.total),
        }


@dataclass
class BrinsonAttribution:
    records: List[BrinsonRecord]
    active_returns: Mapping[str, float]

    def allocation_rows(self) -> List[Mapping[str, object]]:
        return [record.to_allocation_row() for record in self.records]

    def selection_rows(self) -> List[Mapping[str, object]]:
        return [record.to_selection_row() for record in self.records]

    def total_rows(self) -> List[Mapping[str, object]]:
        return [record.to_total_row() for record in self.records]


def _normalize_dates(dates: Sequence[object], length: int) -> List[str]:
    if not dates:
        return [str(idx) for idx in range(length)]
    if len(dates) != length:
        raise ValueError("Date sequence must align with the number of periods")
    return [str(date) for date in dates]


def _normalize_weights(weights: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def compute_brinson_attribution(
    portfolio_weights: Sequence[Sequence[float]],
    benchmark_weights: Sequence[Sequence[float]],
    returns: Sequence[Sequence[float]],
    sectors: Sequence[str],
    dates: Sequence[object],
) -> BrinsonAttribution:
    """Compute Brinson allocation/selection/interaction attribution."""

    portfolio = _normalize_weights(portfolio_weights)
    benchmark = _normalize_weights(benchmark_weights)
    rets = _normalize_weights(returns)
    if portfolio.shape != benchmark.shape or portfolio.shape != rets.shape:
        raise ValueError("Portfolio, benchmark, and returns must share the same shape")
    if portfolio.shape[1] != len(sectors):
        raise ValueError("Sector classification length mismatch with assets")

    dates_norm = _normalize_dates(dates, portfolio.shape[0])
    sectors_norm = [str(sec) for sec in sectors]
    unique_sectors = sorted(set(sectors_norm))
    n_periods, n_assets = portfolio.shape
    records: List[BrinsonRecord] = []
    active_totals: dict[str, float] = {}

    for t in range(n_periods):
        w_p = portfolio[t]
        w_b = benchmark[t]
        r = rets[t]
        bench_total = float(np.dot(w_b, r))
        port_total = float(np.dot(w_p, r))
        active_totals[dates_norm[t]] = float(port_total - bench_total)
        for sector in unique_sectors:
            mask = np.array([sec == sector for sec in sectors_norm], dtype=bool)
            if not np.any(mask):
                continue
            w_p_sector = float(w_p[mask].sum())
            w_b_sector = float(w_b[mask].sum())
            r_b_sector = 0.0
            r_p_sector = 0.0
            if w_b_sector > 1e-12:
                r_b_sector = float(np.dot(w_b[mask], r[mask]) / w_b_sector)
            if w_p_sector > 1e-12:
                r_p_sector = float(np.dot(w_p[mask], r[mask]) / max(w_p_sector, 1e-12))
            allocation = (w_p_sector - w_b_sector) * (r_b_sector - bench_total)
            selection = w_b_sector * (r_p_sector - r_b_sector)
            interaction = (w_p_sector - w_b_sector) * (r_p_sector - r_b_sector)
            records.append(
                BrinsonRecord(
                    date=dates_norm[t],
                    sector=sector,
                    allocation=float(allocation),
                    selection=float(selection),
                    interaction=float(interaction),
                )
            )

    return BrinsonAttribution(records=records, active_returns=active_totals)
