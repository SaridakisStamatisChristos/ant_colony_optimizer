"""End-to-end quickstart runner for reproducible demos."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

BASE_URL = "https://raw.githubusercontent.com/quantresearch/neuro-ant-optimizer/main/examples/data"
DATA_FILES = {
    "returns.csv": "Daily asset return sample",
    "benchmark.csv": "Benchmark returns sample",
    "factors.csv": "Factor loadings sample",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _download_file(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as response, dest.open("wb") as fh:  # type: ignore[arg-type]
            shutil.copyfileobj(response, fh)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False


def _ensure_data(files: Iterable[str], download_dir: Path) -> None:
    fallback_dir = _project_root() / "examples" / "data"
    for name in files:
        target = download_dir / name
        if target.exists():
            continue
        remote = f"{BASE_URL}/{name}"
        if _download_file(remote, target):
            print(f"Downloaded {name} from {remote}")
            continue
        fallback = fallback_dir / name
        if not fallback.exists():
            raise FileNotFoundError(f"Missing fallback data file: {fallback}")
        shutil.copy2(fallback, target)
        print(f"Copied {name} from local fallback")


def _write_config(path: Path, data_dir: Path, out_dir: Path) -> None:
    payload = {
        "csv": str(data_dir / "returns.csv"),
        "benchmark_csv": str(data_dir / "benchmark.csv"),
        "factors": str(data_dir / "factors.csv"),
        "lookback": 126,
        "step": 21,
        "objective": "max_return",
        "cov_model": "sample",
        "refine_every": 1,
        "seed": 7,
        "skip_plot": True,
        "out": str(out_dir),
        "save_weights": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    root = _project_root()
    artifacts_dir = root / "quickstart_artifacts"
    archives_dir = root / "quickstart_archives"
    data_dir = root / "quickstart_data"
    runs_tracker = root / "runs.csv"
    config_path = root / "quickstart_config.json"

    _ensure_data(DATA_FILES.keys(), data_dir)
    _write_config(config_path, data_dir, artifacts_dir)

    run_id = f"quickstart-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    cmd = [
        sys.executable,
        "-m",
        "neuro_ant_optimizer.backtest.backtest",
        "--config",
        str(config_path),
        "--runs-csv",
        str(runs_tracker),
        "--track-artifacts",
        str(archives_dir),
        "--run-id",
        run_id,
    ]
    print("Running quickstart backtest...", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"Artifacts written to {artifacts_dir}")
    print(f"Run tracker appended at {runs_tracker}")
if __name__ == "__main__":
    main()
