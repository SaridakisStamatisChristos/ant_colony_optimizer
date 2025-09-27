from importlib import import_module
from pathlib import Path

import numpy as np


backtest = import_module("neuro_ant_optimizer.backtest.backtest")


def test_write_weights_without_pandas(tmp_path, monkeypatch):
    monkeypatch.setattr(backtest, "pd", None)
    results = {
        "weights": np.array(
            [[0.25, 0.25, 0.25, 0.25], [0.4, 0.3, 0.2, 0.1]],
            dtype=float,
        ),
        "rebalance_dates": ["2020-01-06", "2020-01-08"],
        "asset_names": ["A", "B", "C", "D"],
    }
    out_path = Path(tmp_path) / "weights.csv"
    backtest._write_weights(out_path, results)

    contents = out_path.read_text().strip().splitlines()
    header = contents[0].lstrip("# ").split(",")
    assert header == ["date", "A", "B", "C", "D"]
    assert contents[1].startswith("2020-01-06,")
    assert contents[2].startswith("2020-01-08,")
