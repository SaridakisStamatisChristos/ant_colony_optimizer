from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

@dataclass
class PortfolioConstraints:
    min_weight: float = 0.0
    max_weight: float = 1.0
    equality_enforce: bool = True
    leverage_limit: float = 1.0
    sector_map: Optional[List[int]] = None
    max_sector_concentration: float = 0.4
    prev_weights: Optional[np.ndarray] = None
    max_turnover: float = 0.3
    factor_loadings: Optional[np.ndarray] = None
    factor_targets: Optional[np.ndarray] = None
    factor_tolerance: float = 1e-6
    benchmark_weights: Optional[np.ndarray] = None
    benchmark_mask: Optional[np.ndarray] = None
    min_active_weight: float = float("-inf")
    max_active_weight: float = float("inf")
    active_group_map: Optional[List[int]] = None
    active_group_bounds: Optional[Dict[int, Tuple[float, float]]] = None
    factor_lower_bounds: Optional[np.ndarray] = None
    factor_upper_bounds: Optional[np.ndarray] = None

    def factors_enabled(self) -> bool:
        if self.factor_loadings is None or self.factor_targets is None:
            return False
        loadings = np.asarray(self.factor_loadings)
        targets = np.asarray(self.factor_targets)
        if loadings.ndim != 2 or targets.ndim != 1:
            return False
        return loadings.shape[1] == targets.shape[0]

    def factor_bounds_enabled(self) -> bool:
        if self.factor_loadings is None:
            return False
        lower = self.factor_lower_bounds
        upper = self.factor_upper_bounds
        if lower is None and upper is None:
            return False
        loadings = np.asarray(self.factor_loadings)
        if loadings.ndim != 2:
            return False
        n_factors = loadings.shape[1]
        if lower is not None:
            lower_arr = np.asarray(lower)
            if lower_arr.ndim != 1 or lower_arr.shape[0] != n_factors:
                return False
        if upper is not None:
            upper_arr = np.asarray(upper)
            if upper_arr.ndim != 1 or upper_arr.shape[0] != n_factors:
                return False
        return True
