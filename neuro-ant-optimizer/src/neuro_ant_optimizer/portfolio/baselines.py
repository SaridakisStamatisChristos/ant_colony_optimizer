"""Convex baseline portfolio constructors with transaction-cost penalties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

BaselineMode = Literal["minvar", "maxret", "riskparity"]


@dataclass
class BaselineResult:
    label: str
    weights: np.ndarray
    returns: np.ndarray
    turnover: float


def _project_simplex(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    if w.ndim != 1:
        raise ValueError("Weights must be a one-dimensional vector")
    if w.size == 0:
        return w
    sorted_w = np.sort(w)[::-1]
    cssv = np.cumsum(sorted_w) - 1
    ind = np.arange(1, w.size + 1)
    cond = sorted_w - cssv / ind > 0
    if not np.any(cond):
        theta = cssv[-1] / w.size
    else:
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / rho
    projected = np.maximum(w - theta, 0.0)
    total = projected.sum()
    if total <= 0:
        return np.full_like(projected, 1.0 / projected.size)
    return projected / total


def _risk_parity_objective(w: np.ndarray, cov: np.ndarray) -> float:
    cov_w = cov @ w
    port_var = float(np.dot(w, cov_w))
    if port_var <= 0:
        return 0.0
    target = port_var / w.size
    rc = w * cov_w
    return float(np.sum((rc - target) ** 2))


def _make_objective(
    mode: BaselineMode,
    mu: np.ndarray,
    cov: np.ndarray,
    lambda_tc: float,
    prev: np.ndarray,
) -> callable:
    if lambda_tc < 0:
        raise ValueError("lambda_tc must be non-negative")

    def penalty(w: np.ndarray) -> float:
        return lambda_tc * float(np.sum(np.abs(w - prev)))

    if mode == "minvar":
        def objective(w: np.ndarray) -> float:
            return float(w @ cov @ w) + penalty(w)

    elif mode == "maxret":
        def objective(w: np.ndarray) -> float:
            return float(-mu @ w) + penalty(w)

    elif mode == "riskparity":
        def objective(w: np.ndarray) -> float:
            return _risk_parity_objective(w, cov) + penalty(w)

    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported baseline mode '{mode}'")

    return objective


def _solve_baseline(
    mode: BaselineMode,
    returns: np.ndarray,
    *,
    lambda_tc: float,
    prev_weights: Optional[Sequence[float]] = None,
    cov: Optional[np.ndarray] = None,
) -> np.ndarray:
    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Returns array must be two dimensional")
    if matrix.shape[1] == 0:
        raise ValueError("Cannot compute baseline with zero assets")
    mu = matrix.mean(axis=0)
    cov_matrix = np.asarray(cov, dtype=float) if cov is not None else np.cov(matrix, rowvar=False)
    if cov_matrix.shape != (matrix.shape[1], matrix.shape[1]):
        raise ValueError("Covariance matrix shape mismatch")
    prev = np.asarray(prev_weights, dtype=float) if prev_weights is not None else None
    if prev is None or prev.size != matrix.shape[1]:
        prev = np.full(matrix.shape[1], 1.0 / matrix.shape[1], dtype=float)
    prev = prev.astype(float)
    prev = prev / max(prev.sum(), 1e-12)

    objective = _make_objective(mode, mu, cov_matrix, float(lambda_tc), prev)

    bounds = Bounds(0.0, 1.0)
    linear = LinearConstraint(np.ones((1, matrix.shape[1])), lb=1.0, ub=1.0)
    initial = _project_simplex(prev)

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=[linear],
        options={"maxiter": 500, "ftol": 1e-9, "disp": False},
    )
    if not result.success:
        candidate = _project_simplex(result.x if result.x is not None else initial)
    else:
        candidate = _project_simplex(result.x)
    return candidate


def compute_baseline(
    mode: BaselineMode,
    returns: np.ndarray,
    *,
    lambda_tc: float = 0.0,
    prev_weights: Optional[Sequence[float]] = None,
    cov: Optional[np.ndarray] = None,
) -> BaselineResult:
    mode_norm = str(mode).lower()
    if mode_norm not in {"minvar", "maxret", "riskparity"}:
        raise ValueError("mode must be one of minvar, maxret, riskparity")
    weights = _solve_baseline(mode_norm, returns, lambda_tc=float(lambda_tc), prev_weights=prev_weights, cov=cov)
    returns = np.asarray(returns, dtype=float)
    series = returns @ weights
    prev = np.asarray(prev_weights, dtype=float) if prev_weights is not None else np.full(weights.size, 1.0 / weights.size)
    prev = _project_simplex(prev)
    turnover = float(np.sum(np.abs(weights - prev)))
    return BaselineResult(label=mode_norm, weights=weights, returns=series, turnover=turnover)


__all__ = ["BaselineResult", "compute_baseline"]
