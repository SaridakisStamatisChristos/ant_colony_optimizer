from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

@dataclass
class PortfolioConstraints:
    min_weight: float = 0.0
    max_weight: float = 1.0
    leverage_limit: float = 1.0
    max_turnover: float = 0.3
    max_sector_concentration: float = 0.4
    equality_enforce: bool = True
    sector_map: Optional[List[int]] = None
    prev_weights: Optional[np.ndarray] = None
