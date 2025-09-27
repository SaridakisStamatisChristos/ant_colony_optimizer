from pathlib import Path

import numpy as np

try:
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal envs
    pd = None

from neuro_ant_optimizer.backtest.backtest import backtest


def _load_returns(csv_path: Path):
    if pd is not None:
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)

    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=float, usecols=(1, 2, 3, 4))
    dates = np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=str, usecols=0)

    class _Frame:
        def __init__(self, arr: np.ndarray, idx: np.ndarray):
            self._arr = arr
            self._idx = [np.datetime64(d) for d in idx]

        def to_numpy(self, dtype=float):
            return self._arr.astype(dtype)

        @property
        def index(self):
            return self._idx

    return _Frame(data, dates)


def test_backtest_smoke(tmp_path: Path) -> None:
    csv_path = Path("backtest/sample_returns.csv")
    df = _load_returns(csv_path)
    results = backtest(df, lookback=5, step=2, ewma_span=3, objective="sharpe", seed=1)
    assert "sharpe" in results and "equity" in results
    assert len(results["equity"]) > 0
    assert results["ann_vol"] >= 0.0
