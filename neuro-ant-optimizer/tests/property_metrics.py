"""Property-based regression tests for key quantitative invariants."""

from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # type: ignore
from hypothesis import strategies as st  # type: ignore


HYPOTHESIS_EXAMPLES = 50

from neuro_ant_optimizer.backtest.backtest import compute_tracking_error, max_drawdown
from neuro_ant_optimizer.utils import nearest_psd, safe_softmax


@settings(max_examples=HYPOTHESIS_EXAMPLES, deadline=None)
@given(
    st.lists(
        st.lists(
            st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=6,
        ),
        min_size=1,
        max_size=5,
    )
)
def test_safe_softmax_rows_form_simplex(matrix: list[list[float]]) -> None:
    arr = np.asarray(matrix, dtype=float)
    out = safe_softmax(arr, axis=1)
    assert np.all(out >= -1e-9)
    row_sums = out.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


@settings(max_examples=HYPOTHESIS_EXAMPLES, deadline=None)
@given(
    st.integers(min_value=2, max_value=8),
    st.floats(min_value=1e-8, max_value=1e-2, allow_nan=False, allow_infinity=False),
)
def test_nearest_psd_has_non_negative_eigenvalues(dim: int, jitter: float) -> None:
    rng = np.random.default_rng()
    base = rng.normal(size=(dim, dim))
    cov = base @ base.T
    # introduce asymmetry/negativity
    cov = cov - jitter * np.eye(dim)
    psd = nearest_psd(cov)
    eigvals = np.linalg.eigvalsh(psd)
    assert np.all(eigvals >= -1e-9)


@settings(max_examples=HYPOTHESIS_EXAMPLES, deadline=None)
@given(
    st.lists(
        st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=512,
    ),
    st.integers(min_value=1, max_value=504),
)
def test_tracking_error_is_non_negative(active: list[float], trading_days: int) -> None:
    te = compute_tracking_error(np.asarray(active, dtype=float), trading_days)
    assert te >= 0.0


@settings(max_examples=HYPOTHESIS_EXAMPLES, deadline=None)
@given(
    st.lists(
        st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=512,
    )
)
def test_max_drawdown_is_bounded(equity_points: list[float]) -> None:
    equity = np.asarray(equity_points, dtype=float)
    curve = np.cumprod(1.0 + equity / np.maximum(1.0, equity.max()))
    mdd = max_drawdown(curve)
    assert 0.0 <= mdd <= 1.0
