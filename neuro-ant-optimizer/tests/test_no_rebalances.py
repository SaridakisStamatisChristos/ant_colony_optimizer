import argparse
import json
from importlib import import_module
from pathlib import Path

import numpy as np

bt = import_module("neuro_ant_optimizer.backtest.backtest")


class _Frame:
    def __init__(self, arr: np.ndarray, cols: list[str]):
        self._arr = arr
        self._cols = cols

    def to_numpy(self, dtype=float):
        return self._arr.astype(dtype)

    @property
    def index(self):  # pragma: no cover - simple accessor
        return list(range(self._arr.shape[0]))

    @property
    def columns(self):  # pragma: no cover - simple accessor
        return self._cols


def test_backtest_emits_warning_and_header_only_report(tmp_path: Path) -> None:
    arr = np.zeros((10, 3), dtype=float)
    frame = _Frame(arr, ["A", "B", "C"])

    results = bt.backtest(frame, lookback=252, step=21)

    assert results["warnings"] == ["no_rebalances"]
    assert results["rebalance_records"] == []
    assert results["returns"].size == 0

    report_path = tmp_path / "rebalance_report.csv"
    bt._write_rebalance_report(report_path, results)
    lines = [line for line in report_path.read_text().splitlines() if line]
    assert len(lines) == 1
    assert lines[0].startswith("date,")

    manifest_args = argparse.Namespace(
        csv="returns.csv",
        lookback=252,
        out="bt_out",
        out_format="csv",
        save_weights=False,
    )
    bt._write_run_manifest(
        tmp_path,
        manifest_args,
        config_path=None,
        results=results,
        extras=None,
    )
    manifest = json.loads((tmp_path / "run_config.json").read_text())
    assert manifest["warnings"] == ["no_rebalances"]
    assert manifest["schema_version"] == bt.SCHEMA_VERSION
