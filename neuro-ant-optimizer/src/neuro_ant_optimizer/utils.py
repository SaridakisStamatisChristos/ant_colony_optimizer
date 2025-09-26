from __future__ import annotations
import numpy as np
import torch

def nearest_psd(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    eigval, eigvec = np.linalg.eigh(cov)
    eigval_clipped = np.clip(eigval, eps, None)
    cov_psd = (eigvec * eigval_clipped) @ eigvec.T
    return 0.5 * (cov_psd + cov_psd.T)

def safe_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(np.clip(x, -60, 60))
    s = e.sum()
    if s <= 0:
        return np.ones_like(x) / len(x)
    return e / s

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
