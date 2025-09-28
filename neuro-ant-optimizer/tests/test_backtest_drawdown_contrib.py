from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import math

import numpy as np

bt = import_module("neuro_ant_optimizer.backtest.backtest")


class _StubOptimizer:
    def __init__(self, weights_seq: list[np.ndarray]):
        self.weights_seq = weights_seq
        self.calls = 0
        self.cfg = SimpleNamespace(use_shrinkage=False, shrinkage_delta=0.0)
        self.objectives: list[object] = []

    def optimize(self, *_, objective=None, **__):  # type: ignore[override]
        if objective is not None:
            self.objectives.append(objective)
        weights = self.weights_seq[self.calls]
        self.calls += 1
        return SimpleNamespace(
            weights=weights,
            feasible=True,
            projection_iterations=0,
        )


def test_drawdowns_and_contributions(monkeypatch) -> None:
    returns = np.array(
        [
            [0.02, 0.01],
            [-0.01, 0.005],
            [0.015, -0.02],
            [0.01, 0.012],
            [-0.03, -0.015],
            [0.02, 0.01],
        ]
    )
    weights_seq = [
        np.array([0.7, 0.3], dtype=float),
        np.array([0.4, 0.6], dtype=float),
    ]
    stub = _StubOptimizer(weights_seq)
    monkeypatch.setattr(bt, "_build_optimizer", lambda *args, **kwargs: stub)

    results = bt.backtest(returns, lookback=3, step=2, seed=1)

    drawdowns = results["drawdowns"]
    assert drawdowns, "drawdown events should be recorded"
    expected_drawdowns = bt.compute_drawdown_events(results["equity"], results["dates"])
    assert drawdowns == expected_drawdowns

    contrib_rows = results["contributions"]
    assert contrib_rows
    totals: dict[object, float] = {}
    sums: dict[object, float] = {}
    for row in contrib_rows:
        key = row["date"]
        totals[key] = row["block_return"]
        sums[key] = sums.get(key, 0.0) + float(row["contribution"])
    for key, total in totals.items():
        assert math.isclose(sums[key], float(total), rel_tol=1e-9, abs_tol=1e-9)


def test_cli_emits_drawdown_and_contrib(tmp_path: Path) -> None:
    csv_path = Path("backtest/sample_returns.csv")
    out_dir = tmp_path / "cli"
    bt.main(
        [
            "--csv",
            str(csv_path),
            "--lookback",
            "5",
            "--step",
            "3",
            "--out",
            str(out_dir),
            "--skip-plot",
        ]
    )
    assert (out_dir / "drawdowns.csv").exists()
    assert (out_dir / "contrib.csv").exists()
