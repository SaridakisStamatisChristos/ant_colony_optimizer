from __future__ import annotations

from typing import Optional

import numpy as np

try:  # soft dependency
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Set numpy (and torch if present) seeds; prefer deterministic backends."""

    np.random.seed(seed)
    if torch is None:
        return
    import torch as _t  # local alias to avoid type-checkers complaining

    _t.manual_seed(seed)
    if deterministic_torch:
        try:
            _t.use_deterministic_algorithms(True)
            _t.backends.cudnn.benchmark = False
            _t.backends.cudnn.deterministic = True
        except Exception:
            pass


def nearest_psd(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Fast symmetric eigenvalue clip to nearest PSD matrix."""

    cov = 0.5 * (cov + cov.T)
    w, v = np.linalg.eigh(cov)
    w = np.clip(w, eps, None)
    psd = (v * w) @ v.T
    return 0.5 * (psd + psd.T)


def safe_softmax(
    x: np.ndarray, axis: int = -1, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Numerically-stable softmax with optional boolean mask (True=keep).
    If all entries on an axis are masked or zeroed, returns uniform on unmasked set.
    """

    x = np.asarray(x, dtype=float)
    if mask is not None:
        x = np.where(mask, x, -np.inf)
    x_max = np.nanmax(x, axis=axis, keepdims=True)
    z = np.exp(np.clip(x - x_max, -60, 60))
    if mask is not None:
        z = np.where(mask, z, 0.0)
    denom = z.sum(axis=axis, keepdims=True)
    if np.any(denom == 0):
        u = np.ones_like(z) if mask is None else mask.astype(float)
        s = u.sum(axis=axis, keepdims=True)
        return np.divide(u, np.where(s == 0, 1.0, s), where=True)
    return z / denom

