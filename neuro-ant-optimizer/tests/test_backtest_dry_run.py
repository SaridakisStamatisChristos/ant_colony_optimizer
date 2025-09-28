import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

bt = import_module("neuro_ant_optimizer.backtest.backtest")


class _StubOptimizer:
    def __init__(self, n_assets: int):
        weight = (
            np.full(n_assets, 1.0 / n_assets, dtype=float) if n_assets else np.array([], dtype=float)
        )
        self.weight = weight
        self.cfg = type(
            "Cfg",
            (),
            {
                "use_shrinkage": False,
                "shrinkage_delta": 0.0,
                "te_target": 0.0,
                "lambda_te": 0.0,
                "gamma_turnover": 0.0,
            },
        )()

    def optimize(self, *_, **__):
        class _Result:
            def __init__(self, w: np.ndarray):
                self.weights = w
                self.feasible = True
                self.projection_iterations = 0

        return _Result(self.weight)


def _patch_optimizer(monkeypatch):
    def _factory(n_assets: int, seed: int, risk_free_rate: float = 0.0):  # noqa: ARG001
        return _StubOptimizer(n_assets)

    monkeypatch.setattr(bt, "_build_optimizer", _factory)


def _write_returns(path: Path, rows: list[tuple[str, str, str]]) -> None:
    contents = ["date,A,B"] + ["%s,%s,%s" % row for row in rows]
    path.write_text("\n".join(contents))


def test_dry_run_no_rebalances_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_optimizer(monkeypatch)
    returns_path = tmp_path / "returns.csv"
    _write_returns(
        returns_path,
        [
            ("2020-01-01", "0.01", "0.00"),
            ("2020-01-02", "-0.02", "0.01"),
        ],
    )

    out_dir = tmp_path / "dry_no_rebalances"
    bt.main(
        [
            "--csv",
            str(returns_path),
            "--lookback",
            "5",
            "--dry-run",
            "--out",
            str(out_dir),
        ]
    )

    files = sorted(p.name for p in out_dir.iterdir())
    assert files == ["run_config.json"]
    manifest = json.loads((out_dir / "run_config.json").read_text())
    assert manifest["warnings"] == ["no_rebalances"]


def test_dry_run_strict_missing_window_respects_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_optimizer(monkeypatch)
    returns_path = tmp_path / "returns.csv"
    _write_returns(
        returns_path,
        [
            ("2020-01-01", "0.01", "0.00"),
            ("2020-01-02", "0.02", "-0.01"),
            ("2020-01-03", "0.00", "0.01"),
            ("2020-01-04", "0.01", "0.02"),
            ("2020-01-05", "0.02", "0.01"),
            ("2020-01-06", "0.00", "0.03"),
        ],
    )

    factor_path = tmp_path / "factors.csv"
    factor_rows = [
        "date,A,B",
        "2020-01-01,0.1,0.0",
        "2020-01-02,0.1,0.0",
        "2020-01-03,0.1,0.0",
        "2020-01-04,0.1,0.0",
    ]
    factor_path.write_text("\n".join(factor_rows))

    strict_out = tmp_path / "strict_fail"
    with pytest.raises(ValueError):
        bt.main(
            [
                "--csv",
                str(returns_path),
                "--factors",
                str(factor_path),
                "--factor-align",
                "strict",
                "--step",
                "1",
                "--lookback",
                "3",
                "--dry-run",
                "--out",
                str(strict_out),
            ]
        )

    subset_fail = tmp_path / "subset_fail"
    with pytest.raises(ValueError):
        bt.main(
            [
                "--csv",
                str(returns_path),
                "--factors",
                str(factor_path),
                "--factor-align",
                "subset",
                "--factors-required",
                "--step",
                "1",
                "--lookback",
                "3",
                "--dry-run",
                "--out",
                str(subset_fail),
            ]
        )

    strict_ok = tmp_path / "strict_ok"
    bt.main(
        [
            "--csv",
            str(returns_path),
            "--factors",
            str(factor_path),
            "--factor-align",
            "subset",
            "--step",
            "1",
            "--lookback",
            "3",
            "--dry-run",
            "--out",
            str(strict_ok),
        ]
    )

    manifest = json.loads((strict_ok / "run_config.json").read_text())
    assert manifest["factor_diagnostics"]["missing_window_count"] == 2


def test_dry_run_subset_records_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_optimizer(monkeypatch)
    returns_path = tmp_path / "returns.csv"
    _write_returns(
        returns_path,
        [
            ("2020-01-01", "0.01", "0.00"),
            ("2020-01-02", "0.02", "-0.01"),
            ("2020-01-03", "0.00", "0.01"),
            ("2020-01-04", "0.01", "0.02"),
            ("2020-01-05", "0.02", "0.01"),
            ("2020-01-06", "0.00", "0.03"),
        ],
    )

    factor_path = tmp_path / "subset_factors.csv"
    factor_rows = [
        "date,A,B",
        "2020-01-01,0.1,0.0",
        "2020-01-02,0.1,0.0",
        "2020-01-03,0.1,0.0",
        "2020-01-04,0.1,0.0",
    ]
    factor_path.write_text("\n".join(factor_rows))

    out_dir = tmp_path / "subset_dry"
    bt.main(
        [
            "--csv",
            str(returns_path),
            "--factors",
            str(factor_path),
            "--factor-align",
            "subset",
            "--step",
            "1",
            "--lookback",
            "3",
            "--dry-run",
            "--out",
            str(out_dir),
        ]
    )

    files = sorted(p.name for p in out_dir.iterdir())
    assert files == ["run_config.json"]
    manifest = json.loads((out_dir / "run_config.json").read_text())
    diagnostics = manifest["factor_diagnostics"]
    assert diagnostics["align_mode"] == "subset"
    assert diagnostics["missing_window_count"] == 2
    missing_dates = {entry.split("T")[0] for entry in diagnostics["missing_rebalance_dates"]}
    assert missing_dates == {"2020-01-05", "2020-01-06"}
