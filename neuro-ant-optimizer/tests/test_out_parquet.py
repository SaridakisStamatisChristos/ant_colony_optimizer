import argparse
import csv
import json
from importlib import import_module
from pathlib import Path

bt = import_module("neuro_ant_optimizer.backtest.backtest")


class _DummyFrame:
    def __init__(self, rows: list[list[str]]):
        self._rows = rows

    def to_parquet(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"rows": len(self._rows)}))


class _DummyPandas:
    def read_csv(self, path: Path):  # pragma: no cover - simple helper
        with Path(path).open(newline="") as fh:
            reader = csv.reader(fh)
            rows = list(reader)[1:]
        return _DummyFrame(rows)


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def test_parquet_outputs_created_when_requested(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(bt, "pd", _DummyPandas())

    equity_csv = tmp_path / "equity.csv"
    _write_csv(
        equity_csv,
        ["date", "equity", "ret"],
        [["2020-01-01", "1.0", "0.0"], ["2020-01-02", "1.1", "0.1"]],
    )
    rebalance_csv = tmp_path / "rebalance_report.csv"
    _write_csv(
        rebalance_csv,
        ["date", "gross_ret", "net_tx_ret", "net_slip_ret", "turnover", "tx_cost", "slippage_cost", "sector_breaches", "active_breaches", "group_breaches", "factor_bound_breaches", "factor_inf_norm", "factor_missing", "first_violation", "feasible", "projection_iterations", "block_sharpe", "block_sortino", "block_info_ratio", "block_tracking_error"],
        [["2020-01-02", "0.01", "0.01", "0.01", "0.1", "0.0", "0.0", "0", "0", "0", "0", "0", "False", "", "True", "0", "0.0", "0.0", "", ""]],
    )
    metrics_csv = tmp_path / "metrics.csv"
    _write_csv(metrics_csv, ["metric", "value"], [["sharpe", "0.0"]])
    weights_csv = tmp_path / "weights.csv"
    _write_csv(weights_csv, ["date", "w0", "w1"], [["2020-01-02", "0.5", "0.5"]])

    args = argparse.Namespace(out_format="parquet", save_weights=True)
    bt._maybe_write_parquet(tmp_path, args)

    equity_meta = json.loads((tmp_path / "equity.parquet").read_text())
    rebalance_meta = json.loads((tmp_path / "rebalance_report.parquet").read_text())
    metrics_meta = json.loads((tmp_path / "metrics.parquet").read_text())
    weights_meta = json.loads((tmp_path / "weights.parquet").read_text())

    def _count_rows(csv_path: Path) -> int:
        with csv_path.open(newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            return sum(1 for _ in reader)

    assert equity_meta["rows"] == _count_rows(equity_csv)
    assert rebalance_meta["rows"] == _count_rows(rebalance_csv)
    assert metrics_meta["rows"] == _count_rows(metrics_csv)
    assert weights_meta["rows"] == _count_rows(weights_csv)
