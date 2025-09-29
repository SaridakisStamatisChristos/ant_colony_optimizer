import numpy as np
import pytest

from neuro_ant_optimizer.backtest.backtest import FactorPanel, backtest


def test_factor_attr_ic_sign():
    returns = np.array(
        [
            [0.02, 0.010, -0.005, -0.015],
            [0.018, 0.009, -0.004, -0.012],
            [0.019, 0.011, -0.003, -0.010],
            [0.017, 0.008, -0.002, -0.009],
            [0.020, 0.012, -0.001, -0.008],
            [0.018, 0.010, -0.002, -0.009],
            [0.019, 0.011, -0.003, -0.010],
            [0.021, 0.012, -0.001, -0.007],
        ],
        dtype=float,
    )
    dates = list(range(returns.shape[0]))
    assets = ["A", "B", "C", "D"]
    base_loadings = np.array([[1.0], [0.5], [-0.3], [-0.8]], dtype=float)
    loadings = np.repeat(base_loadings[np.newaxis, :, :], len(dates), axis=0)
    panel = FactorPanel(dates, assets, loadings, ["value"])

    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(returns, columns=assets)

    result = backtest(
        frame,
        lookback=3,
        step=1,
        factors=panel,
        factor_targets=np.zeros(1),
        factor_tolerance=1.0,
        compute_factor_attr=True,
        trading_days=252,
        risk_free_rate=0.0,
        deterministic=True,
    )

    attr_rows = result.get("factor_attr") or []
    assert attr_rows, "Expected factor attribution rows to be populated"
    ics = [row["value_ic"] for row in attr_rows if row.get("value_ic") is not None]
    assert ics and all(val > 0 for val in ics)
