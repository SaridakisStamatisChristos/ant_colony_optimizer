from __future__ import annotations

import numpy as np

from neuro_ant_optimizer.scenario.runner import ScenarioConfig, ScenarioRunner


def test_scenario_runner_applies_shocks():
    base_returns = np.array([0.01, -0.005, 0.0])
    runner = ScenarioRunner(
        base_returns,
        factor_exposures={"growth": 0.5},
        transaction_cost=0.001,
        initial_value=1.0,
    )
    config = ScenarioConfig(
        return_shock=0.002,
        vol_spike=0.1,
        factor_tilts={"growth": 0.01},
        liquidity_haircut=0.05,
        transaction_cost_multiplier=2.0,
        breach_threshold=0.96,
    )

    result = runner.run(config)

    expected_returns = (base_returns + 0.002) * 1.1
    expected_returns = expected_returns + 0.5 * 0.01
    expected_returns = expected_returns - 0.001 * 2.0
    assert np.allclose(result.adjusted_returns, expected_returns)

    expected_path = []
    value = 1.0
    for r in expected_returns:
        value *= 1.0 + r
        expected_path.append(value)
    expected_path[-1] *= 0.95
    assert np.allclose(result.portfolio_path, expected_path)
    assert result.breaches == []


def test_scenario_runner_flags_breaches():
    runner = ScenarioRunner([0.0, -0.05, -0.02], initial_value=1.0)
    config = ScenarioConfig(return_shock=0.0, breach_threshold=0.97)
    result = runner.run(config)
    assert result.breaches == ["breach@1", "breach@2"]
