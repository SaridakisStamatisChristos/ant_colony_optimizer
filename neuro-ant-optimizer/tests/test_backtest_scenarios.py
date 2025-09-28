import csv
from pathlib import Path

import numpy as np
import pytest

from neuro_ant_optimizer.backtest.backtest import (
    _evaluate_scenarios,
    _load_scenarios_config,
    main,
)
from neuro_ant_optimizer.backtest.backtest import max_drawdown


def test_scenario_engine_computes_metrics_and_weights() -> None:
    dates = [
        np.datetime64("2020-01-01"),
        np.datetime64("2020-01-02"),
        np.datetime64("2020-01-03"),
        np.datetime64("2020-01-04"),
        np.datetime64("2020-01-05"),
    ]
    returns = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.01, 0.02],
            [0.01, -0.02],
            [0.0, 0.01],
        ],
        dtype=float,
    )
    weights = np.full((3, 2), 0.5, dtype=float)
    windows = [(2, 3), (3, 4), (4, 5)]
    scenarios_cfg = {
        "scenarios": [
            {
                "name": "single_shock",
                "shocks": [
                    {"assets": "A", "shift": -0.1, "dates": ["2020-01-05"]},
                ],
                "thresholds": {"max_drawdown": 0.2, "sharpe": 0.0},
            },
            {
                "name": "broad_selloff",
                "shocks": [
                    {"assets": "*", "shift": -0.02},
                ],
            },
        ]
    }

    definitions = _load_scenarios_config(
        scenarios_cfg,
        asset_names=["A", "B"],
        dates=dates,
    )
    benchmark = np.zeros(3, dtype=float)
    report_rows, shocked = _evaluate_scenarios(
        definitions,
        returns=returns,
        windows=windows,
        weights=weights,
        benchmark=benchmark,
        periodic_rf=0.0,
        trading_days=252,
    )

    assert len(report_rows) == 2
    assert set(shocked.keys()) == {"single_shock", "broad_selloff"}

    single = next(row for row in report_rows if row["scenario"] == "single_shock")
    scenario_returns = np.array([0.015, -0.005, -0.045], dtype=float)
    ann_factor = np.sqrt(252)
    expected_ann_vol = float(np.std(scenario_returns) * ann_factor)
    expected_ann_return = float(np.mean(scenario_returns) * 252)
    excess_mean = expected_ann_return
    expected_sharpe = float(excess_mean / expected_ann_vol)
    equity = np.cumprod(1.0 + scenario_returns)
    expected_mdd = max_drawdown(equity)
    expected_te = expected_ann_vol

    assert single["ann_vol"] == pytest.approx(expected_ann_vol)
    assert single["ann_return"] == pytest.approx(expected_ann_return)
    assert single["sharpe"] == pytest.approx(expected_sharpe)
    assert single["max_drawdown"] == pytest.approx(expected_mdd)
    assert single["tracking_error"] == pytest.approx(expected_te)
    assert single["info_ratio"] == pytest.approx(expected_sharpe)
    assert single["breaches"] == 1

    shocked_weights = shocked["single_shock"]
    expected_weights = np.array([0.5 * 0.9, 0.5 * 1.01])
    expected_weights = expected_weights / expected_weights.sum()
    np.testing.assert_allclose(shocked_weights, expected_weights)


def test_backtest_cli_writes_scenario_outputs(tmp_path: Path) -> None:
    returns_path = tmp_path / "returns.csv"
    returns_path.write_text(
        "date,A,B\n"
        "2020-01-01,0.0,0.0\n"
        "2020-01-02,0.01,0.02\n"
        "2020-01-03,0.01,-0.01\n"
        "2020-01-04,-0.02,0.00\n"
        "2020-01-05,0.00,0.01\n",
        encoding="utf-8",
    )
    scenario_path = tmp_path / "scenarios.json"
    scenario_path.write_text(
        (
            "{\n"
            "  \"scenarios\": [{\n"
            "    \"name\": \"shock\",\n"
            "    \"shocks\": [{\n"
            "      \"assets\": \"A\",\n"
            "      \"shift\": -0.1,\n"
            "      \"dates\": [\"2020-01-05\"]\n"
            "    }]\n"
            "  }]\n"
            "}\n"
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    main(
        [
            "--csv",
            str(returns_path),
            "--out",
            str(out_dir),
            "--lookback",
            "3",
            "--step",
            "1",
            "--scenarios",
            str(scenario_path),
            "--skip-plot",
        ]
    )

    report_path = out_dir / "scenarios_report.csv"
    weights_path = out_dir / "weights_after_shock.csv"
    assert report_path.exists()
    assert weights_path.exists()

    with report_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0][:4] == ["scenario", "sharpe", "ann_return", "ann_vol"]
    assert any(row[0] == "shock" for row in rows[1:])

    with weights_path.open(newline="", encoding="utf-8") as fh:
        weight_rows = list(csv.reader(fh))
    assert weight_rows[0][0] == "asset"
    assert weight_rows[0][1] == "shock"
