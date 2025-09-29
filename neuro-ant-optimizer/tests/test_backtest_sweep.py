from __future__ import annotations

import csv
from pathlib import Path

import json
from importlib import import_module

import pytest

bt = import_module("neuro_ant_optimizer.backtest.backtest")

pytestmark = pytest.mark.slow


def _write_returns(path: Path) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "A", "B"])
        writer.writerow(["2020-01-01", 0.01, 0.0])
        writer.writerow(["2020-01-02", 0.0, 0.01])
        writer.writerow(["2020-01-03", 0.02, -0.01])
        writer.writerow(["2020-01-04", -0.01, 0.015])
        writer.writerow(["2020-01-05", 0.015, 0.005])


def test_sweep_config_runs(tmp_path: Path) -> None:
    csv_path = tmp_path / "returns.csv"
    _write_returns(csv_path)
    config_path = tmp_path / "sweep.json"
    config_path.write_text(
        json.dumps(
            {
                "base": {
                    "csv": str(csv_path),
                    "lookback": 3,
                    "step": 2,
                    "skip_plot": True,
                },
                "grid": {"seed": [1, 3]},
            }
        )
    )

    out_dir = tmp_path / "runs"
    bt.main(["sweep", "--config", str(config_path), "--out", str(out_dir)])

    results = sorted(out_dir.glob("run_*"))
    assert len(results) == 2
    summary = out_dir / "sweep_results.csv"
    assert summary.exists()
    rows = list(csv.DictReader(summary.open()))
    assert len(rows) == 2
    seeds = {row["param_seed"] for row in rows}
    assert seeds == {"1", "3"}


def test_sweep_grid_cli(tmp_path: Path) -> None:
    csv_path = tmp_path / "returns.csv"
    _write_returns(csv_path)
    out_dir = tmp_path / "runs"
    bt.main(
        [
            "sweep",
            "--csv",
            str(csv_path),
            "--lookback",
            "3",
            "--step",
            "2",
            "--grid",
            "seed=4,5",
            "--out",
            str(out_dir),
            "--skip-plot",
        ]
    )
    summary = out_dir / "sweep_results.csv"
    rows = list(csv.DictReader(summary.open()))
    assert {row["param_seed"] for row in rows} == {"4", "5"}
