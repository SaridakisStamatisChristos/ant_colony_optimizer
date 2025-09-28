from importlib import import_module
from pathlib import Path

import pytest

bt = import_module("neuro_ant_optimizer.backtest.backtest")


def _write_returns(tmp_path: Path) -> Path:
    returns_path = tmp_path / "returns.csv"
    returns_path.write_text(
        "date,A,B\n"
        "2020-01-01,0.01,0.02\n"
        "2020-01-02,0.00,0.01\n"
        "2020-01-03,0.01,0.00\n"
        "2020-01-04,0.02,0.01\n"
    )
    return returns_path


def test_invalid_cov_model(tmp_path: Path) -> None:
    returns_path = _write_returns(tmp_path)
    config_path = tmp_path / "bad_cov.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"csv: {returns_path}",
                "lookback: 3",
                "step: 2",
                "cov_model: imaginary",
            ]
        )
    )

    with pytest.raises(SystemExit) as excinfo:
        bt.main(["--config", str(config_path)])

    message = str(excinfo.value)
    assert "Invalid config" in message
    assert "cov_model" in message


def test_negative_lookback(tmp_path: Path) -> None:
    returns_path = _write_returns(tmp_path)
    config_path = tmp_path / "bad_lookback.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"csv: {returns_path}",
                "lookback: -1",
                "step: 2",
            ]
        )
    )

    with pytest.raises(SystemExit) as excinfo:
        bt.main(["--config", str(config_path)])

    message = str(excinfo.value)
    assert "lookback" in message
    assert "greater than or equal to 1" in message


def test_bad_ewma_span(tmp_path: Path) -> None:
    returns_path = _write_returns(tmp_path)
    config_path = tmp_path / "bad_ewma.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"csv: {returns_path}",
                "lookback: 3",
                "step: 2",
                "cov_model: ewma",
                "ewma_span: 1",
            ]
        )
    )

    with pytest.raises(SystemExit) as excinfo:
        bt.main(["--config", str(config_path)])

    message = str(excinfo.value)
    assert "ewma_span" in message
    assert "greater than or equal to 2" in message
