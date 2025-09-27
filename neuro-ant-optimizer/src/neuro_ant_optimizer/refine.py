from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np


def refine_slsqp(
    score_fn: Callable[[np.ndarray], float],
    w0: np.ndarray,
    bounds: Iterable[tuple[float, float]],
    Aeq: Optional[np.ndarray] = None,
    beq: Optional[np.ndarray] = None,
    Aineq: Optional[np.ndarray] = None,
    bineq: Optional[np.ndarray] = None,
    prev: Optional[np.ndarray] = None,
    T: float = 0.0,
    transaction_cost: float = 0.0,
):
    """
    SLSQP refine with turnover allowance (T) and optional linear transaction cost.
    Maximizes score_fn by minimizing its negative with penalties.
    """

    from scipy.optimize import minimize

    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)

    def proj(w: np.ndarray) -> np.ndarray:
        return np.clip(w, lb, ub)

    def obj(w: np.ndarray) -> float:
        base = float(score_fn(w))
        if prev is not None:
            l1 = np.abs(w - prev).sum()
            if l1 > T:
                base -= 1000.0 * (l1 - T)
            base -= float(transaction_cost) * l1
        return -base

    cons = []
    if Aeq is not None and beq is not None:
        cons.append({"type": "eq", "fun": lambda w, A=Aeq, b=beq: A @ w - b})
    if Aineq is not None and bineq is not None:
        cons.append({"type": "ineq", "fun": lambda w, A=Aineq, b=bineq: b - A @ w})

    res = minimize(
        obj,
        proj(w0),
        method="SLSQP",
        bounds=list(bounds),
        constraints=cons,
        options=dict(maxiter=300, ftol=1e-9, disp=False),
    )
    w = proj(res.x if res.success else w0)
    return w, res

