from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import numpy.typing as npt
import pytest
from importlib import import_module

bt = import_module("neuro_ant_optimizer.backtest.backtest")
from neuro_ant_optimizer.backtest.backtest import backtest, load_factor_panel, main


class _Frame:
    def __init__(self, arr: np.ndarray, dates: np.ndarray, cols: list[str]):
        self._arr = arr
        self._idx = list(dates)
        self._cols = cols

    def to_numpy(self, dtype=float):
        return self._arr.astype(dtype)

    @property
    def index(self):
        return self._idx

    @property
    def columns(self):
        return self._cols


def _write_factor_csv(path: Path, dates: np.ndarray, loadings: np.ndarray, assets: list[str], factors: list[str]) -> None:
    # loadings shape: (T, N, K)
    header = ["date"]
    factor_header = ["factor"]
    for asset in assets:
        for factor in factors:
            header.append(asset)
            factor_header.append(factor)
    rows = [",".join(header), ",".join(factor_header)]
    T, N, K = loadings.shape
    for t in range(T):
        parts = [str(dates[t])]
        for i in range(N):
            for j in range(K):
                parts.append(f"{loadings[t, i, j]:.6f}")
        rows.append(",".join(parts))
    path.write_text("\n".join(rows))


def test_backtest_factor_neutrality(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    n_assets, n_factors = 6, 2
    n_periods = 36
    returns = rng.normal(scale=0.01, size=(n_periods, n_assets))
    dates = np.array([np.datetime64("2020-01-01") + np.timedelta64(i, "D") for i in range(n_periods)])
    assets = [f"A{i}" for i in range(n_assets)]
    df = _Frame(returns, dates, assets)

    base_loadings = rng.normal(scale=0.3, size=(n_assets, n_factors))
    factor_panel = np.broadcast_to(base_loadings, (n_periods, n_assets, n_factors)).copy()
    factor_path = tmp_path / "factors.csv"
    _write_factor_csv(factor_path, dates, factor_panel, assets, [f"F{j}" for j in range(n_factors)])
    panel = load_factor_panel(factor_path)

    results = backtest(
        df,
        lookback=12,
        step=6,
        ewma_span=3,
        objective="sharpe",
        seed=7,
        factors=panel,
        factor_targets=np.zeros(n_factors),
        factor_tolerance=1e-5,
    )

    assert results["factor_records"], "expected factor neutrality records"
    assert results["factor_diagnostics"]["missing_window_count"] == 0
    tol = results["factor_tolerance"] + 1e-4
    for record in results["factor_records"]:
        exposures = record["exposures"]
        targets = record["targets"]
        if targets is None:
            targets = np.zeros_like(exposures)
        diff = exposures - targets
        assert np.linalg.norm(diff, ord=np.inf) <= tol
        assert record["missing"] is False

    exposures_path = tmp_path / "exposures.csv"
    bt._write_exposures(exposures_path, results)
    exposures_data: npt.NDArray[np.void] = np.genfromtxt(
        exposures_path, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    rows = exposures_data if exposures_data.ndim else [exposures_data]
    assert len(rows) == len(results["factor_records"])
    for row, record in zip(rows, results["factor_records"]):
        for idx, name in enumerate(results["factor_names"]):
            assert pytest.approx(record["exposures"][idx]) == row[name]


def test_backtest_cli_with_factors_and_slippage(tmp_path: Path) -> None:
    n_assets, n_periods, n_factors = 3, 24, 2
    rng = np.random.default_rng(1)
    returns = rng.normal(scale=0.01, size=(n_periods, n_assets))
    dates = [np.datetime64("2021-01-01") + np.timedelta64(i, "D") for i in range(n_periods)]
    assets = ["X", "Y", "Z"]

    csv_lines = ["date," + ",".join(assets)]
    for t, date in enumerate(dates):
        vals = ",".join(f"{returns[t, j]:.6f}" for j in range(n_assets))
        csv_lines.append(f"{str(date)},{vals}")
    returns_path = tmp_path / "returns.csv"
    returns_path.write_text("\n".join(csv_lines))

    factor_loadings = rng.normal(scale=0.2, size=(n_periods, n_assets, n_factors))
    factor_path = tmp_path / "factor_loadings.csv"
    _write_factor_csv(factor_path, np.array(dates), factor_loadings, assets, ["F0", "F1"])

    out_dir = tmp_path / "bt_out"
    main(
        [
            "--csv",
            str(returns_path),
            "--lookback",
            "8",
            "--step",
            "4",
            "--factors",
            str(factor_path),
            "--slippage",
            "impact:k=25,alpha=1.5",
            "--nt-band",
            "5bps",
            "--out",
            str(out_dir),
        ]
    )

    metrics_path = out_dir / "metrics.csv"
    factor_csv = out_dir / "factor_constraints.csv"
    slippage_equity = out_dir / "equity_net_of_slippage.csv"
    diag_path = out_dir / "factor_diagnostics.json"
    assert metrics_path.exists()
    assert factor_csv.exists()
    assert slippage_equity.exists()
    assert diag_path.exists()
    diag = json.loads(diag_path.read_text())
    assert diag["align_mode"] == "strict"
    metrics_text = metrics_path.read_text()
    assert "avg_slippage_bps" in metrics_text
