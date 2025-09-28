from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

bt = import_module("neuro_ant_optimizer.backtest.backtest")


class _StubOptimizer:
    def __init__(self):
        self.cfg = SimpleNamespace(use_shrinkage=False, shrinkage_delta=0.0)
        self.seen_objectives: list[object] = []

    def optimize(self, mu, cov, constraints, objective=None, **_):  # type: ignore[override]
        if objective is not None:
            self.seen_objectives.append(objective)
        n = mu.shape[0]
        return SimpleNamespace(
            weights=np.full(n, 1.0 / n),
            feasible=True,
            projection_iterations=0,
        )


def test_register_objective(monkeypatch) -> None:
    returns = np.array(
        [
            [0.01, 0.02, -0.01],
            [0.0, 0.01, 0.005],
            [0.02, -0.01, 0.0],
            [0.015, 0.01, -0.005],
        ]
    )
    stub = _StubOptimizer()
    monkeypatch.setattr(bt, "_build_optimizer", lambda *args, **kwargs: stub)

    def my_return(weights, mu, cov, constraints, benchmark, workspace=None):
        return float(weights @ mu)

    bt.register_objective("my_return", my_return)
    bt.backtest(returns, lookback=2, step=2, objective="my_return")
    assert stub.seen_objectives and callable(stub.seen_objectives[0])


def test_register_cov_model(monkeypatch) -> None:
    returns = np.array(
        [
            [0.01, 0.02],
            [0.0, 0.01],
            [0.015, -0.01],
            [-0.01, 0.005],
        ]
    )

    def identity_cov(data: np.ndarray, **_: object) -> np.ndarray:
        n = data.shape[1]
        return np.eye(n)

    bt.register_cov_model("identity", identity_cov)
    results = bt.backtest(returns, lookback=2, step=2, cov_model="identity")
    assert "equity" in results


def test_custom_objective_import_error(tmp_path: Path) -> None:
    csv_path = tmp_path / "returns.csv"
    csv_path.write_text("date,A\n2020-01-01,0.01\n2020-01-02,0.02\n")
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError) as excinfo:
        bt.main(
            [
                "--csv",
                str(csv_path),
                "--lookback",
                "1",
                "--step",
                "1",
                "--objective",
                "custom:not.real:fn",
                "--out",
                str(out_dir),
            ]
        )
    assert "Failed to import" in str(excinfo.value)


def test_custom_cov_model_import(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "my_cov.py"
    module_path.write_text(
        "import numpy as np\n"
        "def build_cov(returns, **kwargs):\n"
        "    n = returns.shape[1]\n"
        "    return np.eye(n)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    csv_path = tmp_path / "returns.csv"
    csv_path.write_text(
        "date,A,B\n"
        "2020-01-01,0.01,0.02\n"
        "2020-01-02,0.0,0.01\n"
        "2020-01-03,0.02,-0.01\n"
    )
    out_dir = tmp_path / "out"
    bt.main(
        [
            "--csv",
            str(csv_path),
            "--lookback",
            "2",
            "--step",
            "1",
            "--cov-model",
            "custom:my_cov:build_cov",
            "--out",
            str(out_dir),
            "--skip-plot",
        ]
    )
    assert (out_dir / "metrics.csv").exists()
