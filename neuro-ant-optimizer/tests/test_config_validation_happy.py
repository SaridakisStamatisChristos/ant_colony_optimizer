import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

bt = import_module("neuro_ant_optimizer.backtest.backtest")


class _StubOptimizer:
    def __init__(self, weight: np.ndarray):
        self.weight = weight
        self.cfg = type("Cfg", (), {"use_shrinkage": False, "shrinkage_delta": 0.0})()

    def optimize(self, *_, **__):
        class _Result:
            def __init__(self, w: np.ndarray):
                self.weights = w
                self.feasible = True
                self.projection_iterations = 0

        return _Result(self.weight)


def test_config_validation_happy(tmp_path: Path, monkeypatch) -> None:
    returns_path = tmp_path / "returns.csv"
    returns_path.write_text(
        "date,A,B\n"
        "2020-01-01,0.01,0.02\n"
        "2020-01-02,0.00,0.01\n"
        "2020-01-03,0.01,0.00\n"
        "2020-01-04,0.02,0.01\n"
        "2020-01-05,0.01,0.03\n"
        "2020-01-06,0.00,0.02\n"
    )

    out_dir = tmp_path / "bt_out"
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"csv: {returns_path}",
                f"out: {out_dir}",
                "lookback: 4",
                "step: 2",
                "decay: 0.1",
                "objective: sharpe",
            ]
        )
    )

    stub = _StubOptimizer(np.array([0.6, 0.4], dtype=float))
    monkeypatch.setattr(bt, "_build_optimizer", lambda n_assets, seed, risk_free_rate=0.0: stub)

    bt.main(["--config", str(config_path)])

    manifest_path = out_dir / "run_config.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["validated"] is True
    assert manifest["decay"] == pytest.approx(0.1)
    assert manifest["warm_align"] == "last_row"
    assert manifest["warm_start"] is None
    assert manifest["warm_applied_count"] == 0
    assert manifest["args"]["decay"] == pytest.approx(0.1)
