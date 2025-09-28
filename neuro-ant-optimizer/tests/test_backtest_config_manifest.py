from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import numpy as np

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


def test_config_overrides_and_manifest(tmp_path: Path, monkeypatch) -> None:
    returns_path = tmp_path / "returns.csv"
    returns_path.write_text(
        "date,A,B\n"
        "2020-01-01,0.01,0.00\n"
        "2020-01-02,0.02,-0.01\n"
        "2020-01-03,0.00,0.01\n"
        "2020-01-04,0.01,0.02\n"
        "2020-01-05,0.02,0.01\n"
        "2020-01-06,0.00,0.03\n"
    )

    out_dir = tmp_path / "bt_out"
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"csv: {returns_path}",
                "lookback: 4",
                "step: 3",
                "seed: 11",
                f"out: {out_dir}",
                "objective: sharpe",
                "tx_cost_bps: 0",
                "refine_every: 2",
            ]
        )
    )

    stub = _StubOptimizer(np.array([0.6, 0.4], dtype=float))
    monkeypatch.setattr(
        bt, "_build_optimizer", lambda n_assets, seed, risk_free_rate=0.0: stub
    )

    bt.main(["--config", str(config_path), "--lookback", "5"])

    manifest_path = out_dir / "run_config.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["args"]["lookback"] == 5
    assert manifest["args"]["csv"] == str(returns_path)
    assert manifest["args"]["refine_every"] == 2
    assert manifest["config_path"] == str(config_path)
    assert "package_version" in manifest
    assert "python_version" in manifest
    assert "resolved_constraints" in manifest
