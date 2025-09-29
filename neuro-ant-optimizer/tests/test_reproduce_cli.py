from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

bt = import_module("neuro_ant_optimizer.backtest.backtest")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_BOOTSTRAP = (
    "import runpy,sys;"
    "sys.argv=['neuro-ant-reproduce']+sys.argv[1:];"
    "sys.exit(runpy.run_module('neuro_ant_optimizer.backtest.reproduce', run_name='__main__'))"
)


def _cli_args(*args: str) -> list[str]:
    return [sys.executable, "-c", CLI_BOOTSTRAP, *args]


@pytest.fixture(scope="session")
def cli_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing}" if existing else src_path
    return env


def test_reproduce_console_script(
    tmp_path: Path, cli_env: dict[str, str], assert_backtest_artifacts
) -> None:
    returns_path = tmp_path / "returns.csv"
    returns_path.write_text(
        "date,A,B\n"
        "2020-01-01,0.01,0.00\n"
        "2020-01-02,0.02,-0.01\n"
        "2020-01-03,0.00,0.01\n"
        "2020-01-04,0.01,0.02\n"
        "2020-01-05,0.02,0.01\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "orig"
    bt.main(
        [
            "--csv",
            str(returns_path),
            "--lookback",
            "3",
            "--step",
            "2",
            "--out",
            str(out_dir),
            "--skip-plot",
        ]
    )

    manifest_path = out_dir / "run_config.json"
    assert manifest_path.exists()
    manifest_blob = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_blob["args"]["max_iter"] = 2
    manifest_blob["args"]["n_ants"] = 4
    manifest_path.write_text(json.dumps(manifest_blob), encoding="utf-8")

    replay_dir = tmp_path / "replay"
    subprocess.check_call(
        _cli_args("--manifest", str(manifest_path), "--out", str(replay_dir)),
        cwd=PROJECT_ROOT,
        env=cli_env,
    )

    assert (replay_dir / "equity.csv").exists()
    assert_backtest_artifacts(replay_dir)


def test_reproduce_missing_inputs(tmp_path: Path, cli_env: dict[str, str]) -> None:
    manifest = {
        "schema_version": bt.SCHEMA_VERSION,
        "args": {
            "csv": "backtest/missing.csv",
            "lookback": 5,
            "step": 2,
        },
    }
    manifest_path = tmp_path / "run_config.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    out_dir = tmp_path / "out"
    proc = subprocess.run(
        _cli_args("--manifest", str(manifest_path), "--out", str(out_dir)),
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=cli_env,
    )
    assert proc.returncode != 0
    output = proc.stdout + proc.stderr
    assert "Manifest references missing inputs" in output
    assert "missing.csv" in output
