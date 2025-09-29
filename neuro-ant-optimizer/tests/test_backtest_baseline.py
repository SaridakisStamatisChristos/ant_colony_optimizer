from __future__ import annotations

from pathlib import Path

import numpy as np

from neuro_ant_optimizer.backtest.backtest import main


def _write_returns_csv(path: Path, returns: np.ndarray, assets: list[str], dates: list[np.datetime64]) -> None:
    header = "date," + ",".join(assets)
    rows = [header]
    for idx, date in enumerate(dates):
        row_vals = ",".join(f"{returns[idx, j]:.6f}" for j in range(returns.shape[1]))
        rows.append(f"{str(date)},{row_vals}")
    path.write_text("\n".join(rows), encoding="utf-8")


def test_backtest_cli_baseline_maxret(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n_periods, n_assets = 32, 4
    returns = rng.normal(scale=0.01, size=(n_periods, n_assets))
    dates = [np.datetime64("2021-01-01") + np.timedelta64(i, "D") for i in range(n_periods)]
    assets = [f"A{i}" for i in range(n_assets)]

    returns_path = tmp_path / "returns_maxret.csv"
    _write_returns_csv(returns_path, returns, assets, dates)

    out_dir = tmp_path / "baseline_equal"
    main(
        [
            "--csv",
            str(returns_path),
            "--lookback",
            "12",
            "--step",
            "5",
            "--baseline",
            "maxret",
            "--lambda-tc",
            "2.5",
            "--skip-plot",
            "--out",
            str(out_dir),
        ]
    )

    metrics_path = out_dir / "metrics.csv"
    baseline_equity = out_dir / "equity_baseline_maxret.csv"
    equity_plot = out_dir / "equity.png"

    assert metrics_path.exists()
    assert baseline_equity.exists()
    assert not equity_plot.exists()

    metrics_text = metrics_path.read_text(encoding="utf-8")
    assert "baseline_sharpe" in metrics_text
    assert "hit_rate_vs_baseline" in metrics_text


def test_backtest_cli_baseline_riskparity(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n_periods, n_assets = 28, 3
    returns = rng.normal(scale=0.008, size=(n_periods, n_assets))
    dates = [np.datetime64("2022-01-01") + np.timedelta64(i, "D") for i in range(n_periods)]
    assets = [f"A{i}" for i in range(n_assets)]

    returns_path = tmp_path / "returns_rp.csv"
    _write_returns_csv(returns_path, returns, assets, dates)

    out_dir = tmp_path / "baseline_rp"
    main(
        [
            "--csv",
            str(returns_path),
            "--lookback",
            "10",
            "--step",
            "4",
            "--baseline",
            "riskparity",
            "--lambda-tc",
            "1.0",
            "--skip-plot",
            "--out",
            str(out_dir),
        ]
    )

    metrics_path = out_dir / "metrics.csv"
    baseline_equity = out_dir / "equity_baseline_riskparity.csv"

    assert metrics_path.exists()
    assert baseline_equity.exists()

    metrics_text = metrics_path.read_text(encoding="utf-8")
    assert "alpha_vs_baseline" in metrics_text
    assert "baseline_info_ratio" in metrics_text
