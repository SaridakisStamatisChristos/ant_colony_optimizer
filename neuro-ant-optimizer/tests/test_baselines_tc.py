import numpy as np
import pytest

from neuro_ant_optimizer.portfolio.baselines import compute_baseline


@pytest.mark.parametrize("mode", ["minvar", "maxret", "riskparity"])
def test_baseline_turnover_penalty_monotone(mode: str) -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.01, size=(48, 4))
    prev = np.full(4, 0.25)

    baseline_low = compute_baseline(mode, returns, lambda_tc=0.0, prev_weights=prev)
    baseline_high = compute_baseline(mode, returns, lambda_tc=10.0, prev_weights=prev)

    assert baseline_low.weights.shape == baseline_high.weights.shape == (4,)
    assert baseline_low.returns.shape[0] == returns.shape[0]
    assert baseline_high.returns.shape[0] == returns.shape[0]
    assert baseline_high.turnover <= baseline_low.turnover + 1e-6
    assert np.isfinite(baseline_high.turnover)


def test_maxret_prefers_return_direction() -> None:
    returns = np.array(
        [
            [0.02, 0.01, -0.01],
            [0.03, 0.005, -0.02],
            [0.015, 0.004, -0.015],
        ]
    )
    result = compute_baseline("maxret", returns, lambda_tc=0.0)
    # First asset should receive the highest weight when no penalty is applied
    assert result.weights[0] == pytest.approx(max(result.weights))
    assert np.isclose(result.weights.sum(), 1.0)
