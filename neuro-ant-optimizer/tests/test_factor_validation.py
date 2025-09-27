from __future__ import annotations

import numpy as np
import pytest

from neuro_ant_optimizer.backtest.backtest import (
    FactorPanel,
    backtest,
    validate_factor_panel,
    validate_factor_targets,
)


class _Frame:
    def __init__(self, arr: np.ndarray, dates: list[np.datetime64], cols: list[str]):
        self._arr = arr
        self._dates = dates
        self._cols = cols

    def to_numpy(self, dtype=float):  # pragma: no cover - simple proxy
        return self._arr.astype(dtype)

    @property
    def index(self):  # pragma: no cover - simple accessor
        return self._dates

    @property
    def columns(self):  # pragma: no cover - simple accessor
        return self._cols


def _simple_panel(
    dates: list[np.datetime64],
    assets: list[str],
    loadings: np.ndarray,
) -> FactorPanel:
    factor_names = [f"F{i}" for i in range(loadings.shape[2])]
    return FactorPanel(dates=list(dates), assets=list(assets), loadings=loadings, factor_names=factor_names)


def test_factor_schema_strict_ok() -> None:
    dates = [np.datetime64("2020-01-01") + np.timedelta64(i, "D") for i in range(4)]
    extra_date = np.datetime64("2020-01-10")
    returns = np.random.default_rng(0).normal(scale=0.01, size=(4, 3))
    frame = _Frame(returns, dates, ["A", "B", "C"])

    panel_dates = dates + [extra_date]
    loadings = np.ones((5, 2, 2), dtype=float)
    panel = _simple_panel(panel_dates, ["A", "B"], loadings)

    aligned, diagnostics = validate_factor_panel(panel, frame, align="strict")
    assert aligned.assets == ["A", "B"]
    assert aligned.dates == dates
    assert diagnostics.dropped_assets == ["C"]
    assert diagnostics.dropped_date_count == 1
    assert diagnostics.missing_window_count == 0


def test_factor_schema_strict_fail() -> None:
    dates = [np.datetime64("2020-01-01") + np.timedelta64(i, "D") for i in range(3)]
    returns = np.random.default_rng(1).normal(scale=0.01, size=(3, 2))
    frame = _Frame(returns, dates, ["A", "B"])

    panel_dates = dates[:-1]
    loadings = np.ones((len(panel_dates), 2, 1), dtype=float)
    panel = _simple_panel(panel_dates, ["A", "B"], loadings)

    with pytest.raises(ValueError):
        validate_factor_panel(panel, frame, align="strict")


def test_factor_schema_subset_warn() -> None:
    dates = [np.datetime64("2021-01-01") + np.timedelta64(i, "D") for i in range(4)]
    returns = np.random.default_rng(2).normal(scale=0.01, size=(4, 2))
    frame = _Frame(returns, dates, ["A", "B"])

    panel_dates = dates[:-1]
    loadings = np.ones((len(panel_dates), 2, 2), dtype=float)
    panel = _simple_panel(panel_dates, ["A", "B"], loadings)

    aligned, diagnostics = validate_factor_panel(panel, frame, align="subset")
    assert aligned.dates == panel_dates
    assert diagnostics.missing_window_count == 1
    assert diagnostics.to_dict()["missing_rebalance_dates"][0].startswith("2021-01-04")


def test_factor_targets_validation_mismatch_name_and_len() -> None:
    with pytest.raises(ValueError):
        validate_factor_targets(np.ones(2), ["X", "Y", "Z"])


def test_backtest_factors_required_hard_fail_on_gap() -> None:
    rng = np.random.default_rng(5)
    n_periods = 12
    dates = [np.datetime64("2022-01-01") + np.timedelta64(i, "D") for i in range(n_periods)]
    returns = rng.normal(scale=0.01, size=(n_periods, 3))
    frame = _Frame(returns, dates, ["A", "B", "C"])

    # Factors miss the final rebalance date
    factor_dates = dates[:-1]
    loadings = rng.normal(scale=0.2, size=(len(factor_dates), 3, 2))
    panel = _simple_panel(factor_dates, ["A", "B", "C"], loadings)

    with pytest.raises(ValueError):
        backtest(
            frame,
            lookback=4,
            step=4,
            factors=panel,
            factor_align="subset",
            factors_required=True,
        )


def test_backtest_speed_smoke() -> None:
    rng = np.random.default_rng(7)
    n_periods, n_assets = 80, 5
    returns = rng.normal(scale=0.01, size=(n_periods, n_assets))
    dates = [np.datetime64("2019-01-01") + np.timedelta64(i, "D") for i in range(n_periods)]
    frame = _Frame(returns, dates, [f"A{i}" for i in range(n_assets)])

    result_one = backtest(frame, lookback=20, step=5, cov_model="oas", seed=11)
    result_two = backtest(frame, lookback=20, step=5, cov_model="oas", seed=11)

    np.testing.assert_allclose(result_one["equity"], result_two["equity"], rtol=0, atol=0)
    assert result_one["rebalance_records"] == result_two["rebalance_records"]
    assert result_one["factor_diagnostics"] is None
