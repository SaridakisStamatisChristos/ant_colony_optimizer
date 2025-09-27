"""Neuro Ant Optimizer public API."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .constraints import PortfolioConstraints
from .optimizer import (
    NeuroAntPortfolioOptimizer,
    OptimizationObjective,
    OptimizationResult,
    OptimizerConfig,
)

__all__ = [
    "NeuroAntPortfolioOptimizer",
    "OptimizerConfig",
    "OptimizationObjective",
    "OptimizationResult",
    "PortfolioConstraints",
    "__version__",
]

try:  # runtime package version
    __version__ = version("neuro-ant-optimizer")
except PackageNotFoundError:  # editable/dev env
    __version__ = "0.0.0+local"
