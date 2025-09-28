import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from neuro_ant_optimizer.backtest.backtest import backtest


def _duplicate_frame() -> pd.DataFrame:
    dates = pd.to_datetime([
        "2024-01-02",
        "2024-01-03",
        "2024-01-03",
        "2024-01-04",
    ])
    data = np.array(
        [
            [0.01, 0.02],
            [0.015, -0.01],
            [0.02, 0.005],
            [-0.01, 0.01],
        ]
    )
    return pd.DataFrame(data, index=dates, columns=["A", "B"])


def _unique_frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    data = np.array(
        [
            [0.01, 0.02],
            [0.015, -0.01],
            [0.02, 0.005],
            [-0.01, 0.01],
        ]
    )
    return pd.DataFrame(data, index=dates, columns=["A", "B"])


def test_backtest_duplicate_dates_error():
    frame = _duplicate_frame()
    with pytest.raises(ValueError, match="duplicate dates"):
        backtest(frame, lookback=2, step=1, drop_duplicates=False)


def test_backtest_drop_duplicates_success():
    frame = _duplicate_frame()
    result = backtest(frame, lookback=2, step=1, drop_duplicates=True)
    assert len(result["dates"]) == 3
    assert result["dates"][1] > result["dates"][0]


def test_backtest_deterministic_strict(monkeypatch):
    frame = _unique_frame()
    import neuro_ant_optimizer.utils as utils

    if utils.torch is None:
        pytest.skip("torch not available")

    def boom(flag: bool) -> None:  # pragma: no cover - guarded by strict path
        raise RuntimeError("no deterministic")

    monkeypatch.setattr(utils.torch, "use_deterministic_algorithms", boom)

    with pytest.raises(RuntimeError, match="Deterministic torch backends unavailable"):
        backtest(frame, lookback=2, step=1, deterministic=True)
