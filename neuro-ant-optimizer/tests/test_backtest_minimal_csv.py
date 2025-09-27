from importlib import import_module
from pathlib import Path

import numpy as np

bt = import_module("neuro_ant_optimizer.backtest.backtest")


def test_read_csv_without_pandas(monkeypatch):
    csv_path = Path("backtest/sample_returns.csv")
    monkeypatch.setattr(bt, "pd", None)

    frame = bt._read_csv(csv_path)
    data = frame.to_numpy()

    assert data.shape[1] == 4
    assert np.issubdtype(data.dtype, np.floating)
    np.testing.assert_allclose(
        data[0],
        np.array([0.001, 0.0, -0.001, 0.002]),
    )
    assert frame.columns == ["A", "B", "C", "D"]
    assert frame.index[0] == np.datetime64("2020-01-01")
