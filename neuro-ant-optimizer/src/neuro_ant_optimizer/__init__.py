"""Neuro Ant Optimizer public API."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:  # runtime package version
    __version__ = version("neuro-ant-optimizer")
except PackageNotFoundError:  # editable/dev env
    __version__ = "0.0.0+local"

from .constraints import PortfolioConstraints
from .optimizer import (
    BenchmarkStats,
    NeuroAntPortfolioOptimizer,
    OptimizationObjective,
    OptimizationResult,
    OptimizerConfig,
)
from .backtest.backtest import ewma_cov

__all__ = [
    "NeuroAntPortfolioOptimizer",
    "OptimizerConfig",
    "OptimizationObjective",
    "OptimizationResult",
    "PortfolioConstraints",
    "BenchmarkStats",
    "__version__",
    "ewma_cov",
]
