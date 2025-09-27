import numpy as np
from importlib import import_module


bt = import_module("neuro_ant_optimizer.backtest.backtest")


def _is_psd(m):
    # Numerical PSD check
    w = np.linalg.eigvalsh(0.5 * (m + m.T))
    return np.all(w > -1e-10)


def test_cov_backends_psd_and_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    S = bt._sample_cov(X)
    E = bt.ewma_cov(X, span=20)
    LW = bt._lw_cov(X)
    OAS = bt._oas_cov(X)
    for M in (S, E, LW, OAS):
        assert M.shape == (8, 8)
        assert _is_psd(M)


def test_backtest_cov_model_routes():
    rng = np.random.default_rng(1)
    T, N = 40, 4
    ret = rng.normal(scale=0.01, size=(T, N))
    dates = [np.datetime64("2020-01-01") + np.timedelta64(i, "D") for i in range(T)]

    class _Frame:
        def __init__(self, a, d):
            self._a = a
            self._d = d
            self._c = [f"A{i}" for i in range(a.shape[1])]

        def to_numpy(self, dtype=float):
            return self._a.astype(dtype)

        @property
        def index(self):
            return self._d

        @property
        def columns(self):
            return self._c

    F = _Frame(ret, dates)
    for model in ("sample", "ewma", "lw", "oas"):
        res = bt.backtest(F, lookback=10, step=5, cov_model=model, ewma_span=5, seed=7)
        assert res["cov_model"] == model
        assert len(res["equity"]) > 0
