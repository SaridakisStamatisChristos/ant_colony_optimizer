import csv
import math
from pathlib import Path

import numpy as np
import pytest

from neuro_ant_optimizer.backtest.backtest import (
    _compute_rolling_metrics_rows,
    _pain_index,
    _write_metrics,
    _write_rolling_metrics,
    max_drawdown,
)


def test_rolling_metrics_math(tmp_path: Path) -> None:
    returns = np.array([0.01, -0.005, 0.02], dtype=float)
    dates = [0, 1, 2]
    window = 3
    walk_windows = [(0, 3)]
    turnovers = [0.5]
    periodic_rf = 0.0
    trading_days = 252

    rows = _compute_rolling_metrics_rows(
        returns,
        dates,
        periodic_rf,
        trading_days,
        window,
        walk_windows,
        turnovers,
    )
    assert len(rows) == 1
    row = rows[0]

    ann_return_expected = float(returns.mean() * trading_days)
    ann_vol_expected = float(np.std(returns) * math.sqrt(trading_days))
    sharpe_expected = (
        float(ann_return_expected / ann_vol_expected) if ann_vol_expected > 1e-12 else 0.0
    )
    equity = np.cumprod(1.0 + returns)
    calmar_expected = float(ann_return_expected / max_drawdown(equity))
    pain_expected = float(ann_return_expected / _pain_index(equity))
    hit_rate_expected = float(2 / 3)
    turnover_adj_expected = float(sharpe_expected / (1.0 + turnovers[0]))

    assert row["start"] == "0"
    assert row["end"] == "2"
    assert row["hit_rate"] == pytest.approx(hit_rate_expected)
    assert row["ann_return"] == pytest.approx(ann_return_expected)
    assert row["ann_vol"] == pytest.approx(ann_vol_expected)
    assert row["sharpe"] == pytest.approx(sharpe_expected)
    assert row["calmar_ratio"] == pytest.approx(calmar_expected)
    assert row["pain_ratio"] == pytest.approx(pain_expected)
    assert row["turnover_adj_sharpe"] == pytest.approx(turnover_adj_expected)

    out_path = tmp_path / "rolling_metrics.csv"
    results = {
        "returns": returns,
        "dates": dates,
        "periodic_risk_free": periodic_rf,
        "trading_days": trading_days,
        "walk_windows": walk_windows,
        "rebalance_records": [{"turnover": turnovers[0]}],
    }
    _write_rolling_metrics(out_path, results, window)
    assert out_path.exists()
    with out_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) == 1
    recorded = rows[0]
    assert float(recorded["turnover_adj_sharpe"]) == pytest.approx(turnover_adj_expected)
    assert float(recorded["pain_ratio"]) == pytest.approx(pain_expected)


def test_metrics_csv_contains_new_fields(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    results = {
        "turnover_adj_sharpe": 1.5,
        "calmar_ratio": 0.8,
        "pain_ratio": 1.1,
        "hit_rate": 0.6,
    }
    _write_metrics(metrics_path, results)
    with metrics_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    lookup = {row["metric"]: row["value"] for row in rows}
    for key, value in results.items():
        assert key in lookup
        assert float(lookup[key]) == pytest.approx(value)
