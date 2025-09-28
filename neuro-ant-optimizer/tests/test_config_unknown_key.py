from importlib import import_module
from pathlib import Path

import pytest

bt = import_module("neuro_ant_optimizer.backtest.backtest")


def test_unknown_key(tmp_path: Path) -> None:
    returns_path = tmp_path / "returns.csv"
    returns_path.write_text(
        "date,A\n"
        "2020-01-01,0.01\n"
        "2020-01-02,0.02\n"
        "2020-01-03,0.00\n"
    )

    config_path = tmp_path / "extra.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"csv: {returns_path}",
                "lookback: 2",
                "foo: bar",
            ]
        )
    )

    with pytest.raises(SystemExit) as excinfo:
        bt.main(["--config", str(config_path)])

    message = str(excinfo.value)
    assert "Invalid config" in message
    assert "foo" in message
    assert "extra fields not permitted" in message
