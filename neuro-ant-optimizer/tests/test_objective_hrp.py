import numpy as np
from importlib import import_module
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


bt = import_module("neuro_ant_optimizer.backtest.backtest")


def _quasi_diag(linkage_matrix: np.ndarray) -> list[int]:
    link = np.asarray(linkage_matrix, dtype=float)
    if link.size == 0:
        return []
    n = link.shape[0] + 1
    order = [int(link[-1, 0]), int(link[-1, 1])]
    while any(idx >= n for idx in order):
        expanded: list[int] = []
        for idx in order:
            if idx < n:
                expanded.append(int(idx))
            else:
                child = int(idx - n)
                expanded.append(int(link[child, 0]))
                expanded.append(int(link[child, 1]))
        order = expanded
    return [int(i) for i in order]


def _cluster_variance(cov: np.ndarray, cluster: list[int]) -> float:
    if not cluster:
        return 0.0
    sub = cov[np.ix_(cluster, cluster)]
    diag = np.diag(sub)
    diag = np.where(diag <= 0, 1e-12, diag)
    inv_diag = 1.0 / diag
    weights = inv_diag / inv_diag.sum()
    return float(weights @ sub @ weights)


def _reference_hrp(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.array([1.0], dtype=float)
    diag = np.diag(cov)
    denom = np.sqrt(np.outer(diag, diag))
    corr = np.divide(cov, denom, out=np.ones_like(cov), where=denom > 0)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr)))
    condensed = squareform(dist, checks=False)
    if condensed.size == 0:
        return np.ones(n, dtype=float) / n
    link = linkage(condensed, method="single")
    order = _quasi_diag(link)
    if not order:
        return np.ones(n, dtype=float) / n

    def _allocate(cluster: list[int]) -> np.ndarray:
        if len(cluster) == 1:
            return np.array([1.0], dtype=float)
        split = len(cluster) // 2
        left_cluster = cluster[:split]
        right_cluster = cluster[split:]
        left_weights = _allocate(left_cluster)
        right_weights = _allocate(right_cluster)
        var_left = _cluster_variance(cov, [order[i] for i in left_cluster])
        var_right = _cluster_variance(cov, [order[i] for i in right_cluster])
        total = var_left + var_right
        alpha = 0.5 if total <= 0 else 1.0 - var_left / total
        return np.concatenate((alpha * left_weights, (1.0 - alpha) * right_weights))

    weights_sorted = _allocate(list(range(len(order))))
    final = np.zeros(n, dtype=float)
    for pos, asset_idx in enumerate(order):
        final[asset_idx] = weights_sorted[pos]
    return final / final.sum()


def test_hrp_objective_matches_reference():
    rng = np.random.default_rng(42)
    T, N = 40, 5
    returns = rng.normal(scale=0.01, size=(T, N))
    dates = [np.datetime64("2020-01-01") + np.timedelta64(i, "D") for i in range(T)]

    class _Frame:
        def __init__(self, arr, idx):
            self._arr = arr
            self._idx = idx
            self._cols = [f"A{i}" for i in range(arr.shape[1])]

        def to_numpy(self, dtype=float):
            return self._arr.astype(dtype)

        @property
        def index(self):
            return self._idx

        @property
        def columns(self):
            return self._cols

    frame = _Frame(returns, dates)
    result = bt.backtest(
        frame,
        lookback=20,
        step=20,
        cov_model="sample",
        objective="hrp",
        seed=3,
    )
    weights = np.asarray(result["weights"], dtype=float)
    assert weights.shape == (1, N)
    sample_cov = np.cov(returns[:20], rowvar=False)
    sample_cov = 0.5 * (sample_cov + sample_cov.T)
    reference = _reference_hrp(sample_cov)
    np.testing.assert_allclose(weights[0], reference, atol=1e-8)
    np.testing.assert_allclose(weights[0].sum(), 1.0, atol=1e-12)
