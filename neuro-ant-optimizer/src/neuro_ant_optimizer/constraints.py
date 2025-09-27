from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

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

    def factors_enabled(self) -> bool:
        if self.factor_loadings is None or self.factor_targets is None:
            return False
        loadings = np.asarray(self.factor_loadings)
        targets = np.asarray(self.factor_targets)
        if loadings.ndim != 2 or targets.ndim != 1:
            return False
        return loadings.shape[1] == targets.shape[0]
