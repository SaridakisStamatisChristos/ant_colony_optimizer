from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, List
from scipy.optimize import minimize
from .constraints import PortfolioConstraints

def refine_slsqp(
    w0: np.ndarray,
    score_fn,
    n_assets: int,
    constraints: PortfolioConstraints,
    maxiter: int = 200,
) -> Tuple[np.ndarray, float, bool]:
    n = n_assets
    c = constraints
    bounds = [(c.min_weight, c.max_weight)] * n

    A = []; b = []; Aeq = []; beq = []

    if c.equality_enforce and abs(c.leverage_limit - 1.0) < 1e-12:
        Aeq.append(np.ones(n)); beq.append(1.0)
    else:
        A.append(np.ones(n)); b.append(c.leverage_limit)

    if c.sector_map is not None:
        sects = np.array(c.sector_map, dtype=int)
        for s in np.unique(sects):
            row = np.zeros(n); row[sects == s] = 1.0
            A.append(row); b.append(c.max_sector_concentration)

    prev = np.asarray(c.prev_weights, dtype=float) if c.prev_weights is not None else None
    T = float(c.max_turnover) if prev is not None else 0.0

    def obj(w):
        base = score_fn(w)
        if prev is not None:
            l1 = np.abs(w - prev).sum()
            if l1 > T:
                base = base - 1000.0 * (T - l1)  # penalty
        return -base

    lin_ineq = [{"type": "ineq", "fun": (lambda w, row=row, rhs=rhs: rhs - float(row @ w))} for row, rhs in zip(A, b)]
    lin_eq = [{"type": "eq", "fun": (lambda w, row=row, rhs=rhs: float(row @ w - rhs))} for row, rhs in zip(Aeq, beq)]
    cons = lin_ineq + lin_eq

    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": maxiter, "ftol": 1e-9, "disp": False})
    return np.asarray(res.x, dtype=float), float(-res.fun), bool(res.success)
