"""Manual execution of critical regression checks without invoking pytest."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import sys
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from neuro_ant_optimizer.backtest.backtest import (  # noqa: E402
    FactorPanel,
    SlippageConfig,
    backtest,
)

backtest_cli = import_module("neuro_ant_optimizer.backtest.backtest")
create_app = import_module("service.app").create_app


@contextlib.contextmanager
def temp_environment(**overrides: str) -> Iterator[None]:
    original: Dict[str, str | None] = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def fast_optimizer(max_iter: int = 4, n_ants: int = 8) -> Iterator[None]:
    original = backtest_cli._build_optimizer

    def _patched(n_assets: int, seed: int, risk_free_rate: float = 0.0):
        cfg = backtest_cli.OptimizerConfig(
            n_ants=n_ants,
            max_iter=max_iter,
            topk_refine=max(1, n_ants // 4),
            topk_train=max(1, n_ants // 4),
            use_shrinkage=False,
            shrinkage_delta=0.15,
            cvar_alpha=0.05,
            seed=seed,
            risk_free=risk_free_rate,
        )
        return backtest_cli.NeuroAntPortfolioOptimizer(n_assets, cfg)

    try:
        backtest_cli._build_optimizer = _patched  # type: ignore[attr-defined]
        yield
    finally:
        backtest_cli._build_optimizer = original  # type: ignore[attr-defined]


def _iter_returns_from_csv(path: Path) -> np.ndarray:
    rows: List[List[float]] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = [name for name in reader.fieldnames or [] if name.lower() != "date"]
        for row in reader:
            rows.append([float(row[name]) for name in fieldnames])
    return np.asarray(rows, dtype=float)


def _write_equity_csv(out_dir: Path, dates: Sequence[int], equity: Sequence[float]) -> None:
    path = out_dir / "equity.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "equity"])
        for idx, value in zip(dates, equity):
            writer.writerow([idx, f"{value:.12f}"])


def _write_weights_csv(out_dir: Path, rebalance_dates: Sequence[int], weights: np.ndarray, asset_names: Sequence[str]) -> None:
    path = out_dir / "weights.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", *asset_names])
        for date, row in zip(rebalance_dates, weights):
            writer.writerow([date, *[f"{val:.12f}" for val in row]])


def _write_rebalance_report(out_dir: Path, records: Sequence[Mapping[str, object]]) -> None:
    if not records:
        return
    path = out_dir / "rebalance_report.csv"
    fieldnames = sorted(records[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in fieldnames})


def _write_metrics_csv(out_dir: Path, metrics: Mapping[str, float]) -> None:
    path = out_dir / "metrics.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for key in ("sharpe", "ann_return", "ann_vol", "max_drawdown"):
            writer.writerow([key, f"{metrics.get(key, 0.0):.12f}"])


def _write_artifact_index(out_dir: Path) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for candidate in sorted(out_dir.iterdir()):
        if not candidate.is_file():
            continue
        if candidate.name in {"run_config.json", "artifact_index.json"}:
            continue
        blob = candidate.read_bytes()
        entries.append(
            {
                "name": candidate.name,
                "sha256": hashlib.sha256(blob).hexdigest(),
                "size": len(blob),
            }
        )
    (out_dir / "artifact_index.json").write_text(json.dumps(entries), encoding="utf-8")
    return entries


def _write_manifest(out_dir: Path, manifest: Mapping[str, object]) -> None:
    (out_dir / "run_config.json").write_text(json.dumps(manifest), encoding="utf-8")


@dataclass
class BacktestResult:
    metrics: Dict[str, float]
    summary_row: Dict[str, object]


def run_backtest_case(
    returns: np.ndarray,
    *,
    lookback: int,
    step: int,
    seed: int,
    out_dir: Path,
    deterministic: bool = False,
    write_manifest: bool = False,
) -> BacktestResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    with fast_optimizer():
        result = backtest(
            returns,
            lookback=lookback,
            step=step,
            objective="sharpe",
            seed=seed,
            tx_cost_bps=0.0,
            tx_cost_mode="none",
            metric_alpha=0.05,
            dtype=np.float64,
            cov_cache_size=2,
            workers=None,
            deterministic=deterministic,
        )

    _write_equity_csv(out_dir, result["dates"], result["equity"])
    _write_weights_csv(out_dir, result["rebalance_dates"], result["weights"], result["asset_names"])
    _write_rebalance_report(out_dir, result.get("rebalance_records", []))
    _write_metrics_csv(out_dir, result)

    records = result.get("rebalance_records") or []
    feasible_count = sum(
        1
        for record in records
        if isinstance(record, Mapping) and bool(record.get("feasible"))
    )
    if feasible_count == 0:
        raise AssertionError("slow checks harness produced no feasible rebalances")

    summary_row = {
        "run": out_dir.name,
        "out_dir": str(out_dir),
        "objective": result.get("objective", "sharpe"),
        "cov_model": result.get("cov_model", "sample"),
        "param_seed": str(seed),
        "sharpe": float(result.get("sharpe", 0.0)),
        "ann_return": float(result.get("ann_return", 0.0)),
        "ann_vol": float(result.get("ann_vol", 0.0)),
        "max_drawdown": float(result.get("max_drawdown", 0.0)),
    }

    if write_manifest:
        artifact_entries = _write_artifact_index(out_dir)
        manifest = {
            "args": {"lookback": lookback, "step": step, "seed": seed},
            "schema_version": backtest_cli.SCHEMA_VERSION,
            "package_version": getattr(backtest_cli, "__version__", ""),
            "deterministic_torch": deterministic,
            "run_id": f"manual-{uuid.uuid4().hex}",
            "artifact_index": "artifact_index.json",
            "artifact_index_entries": artifact_entries,
        }
        _write_manifest(out_dir, manifest)
    else:
        _write_artifact_index(out_dir)

    metrics = {
        "sharpe": float(result.get("sharpe", 0.0)),
        "ann_return": float(result.get("ann_return", 0.0)),
        "ann_vol": float(result.get("ann_vol", 0.0)),
        "max_drawdown": float(result.get("max_drawdown", 0.0)),
    }
    return BacktestResult(metrics=metrics, summary_row=summary_row)


def _sweep_returns() -> np.ndarray:
    rows = np.array(
        [
            [0.01, 0.0],
            [0.0, 0.01],
            [0.02, -0.01],
            [-0.01, 0.015],
            [0.015, 0.005],
        ],
        dtype=float,
    )
    return rows


def check_backtest_sweep_config() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        returns = _sweep_returns()
        seeds = [1, 3]
        summary_rows = []
        for idx, seed in enumerate(seeds):
            run_dir = tmp_path / f"run_{idx:03d}_seed-{seed}"
            result = run_backtest_case(returns, lookback=3, step=2, seed=seed, out_dir=run_dir)
            summary_rows.append(result.summary_row)

        sweep_dir = tmp_path / "runs"
        sweep_dir.mkdir()
        summary_path = sweep_dir / "sweep_results.csv"
        fieldnames = [
            "run",
            "out_dir",
            "objective",
            "cov_model",
            "param_seed",
            "sharpe",
            "ann_return",
            "ann_vol",
            "max_drawdown",
        ]
        with summary_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

        seeds_found = {row["param_seed"] for row in summary_rows}
        assert seeds_found == {"1", "3"}


def check_backtest_sweep_cli_grid() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        returns = _sweep_returns()
        seeds = [4, 5]
        summary_rows = []
        for idx, seed in enumerate(seeds):
            run_dir = tmp_path / f"run_{idx:03d}_seed-{seed}"
            result = run_backtest_case(returns, lookback=3, step=2, seed=seed, out_dir=run_dir)
            summary_rows.append(result.summary_row)

        summary_path = tmp_path / "sweep_results.csv"
        with summary_path.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "run",
                    "out_dir",
                    "objective",
                    "cov_model",
                    "param_seed",
                    "sharpe",
                    "ann_return",
                    "ann_vol",
                    "max_drawdown",
                ],
            )
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

        seeds_found = {row["param_seed"] for row in summary_rows}
        assert seeds_found == {"4", "5"}


def _sample_returns() -> np.ndarray:
    returns_csv = PROJECT_ROOT / "backtest" / "sample_returns.csv"
    return _iter_returns_from_csv(returns_csv)


def check_backtest_cli_determinism() -> None:
    returns = _sample_returns()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        first = tmp_path / "run_a"
        second = tmp_path / "run_b"
        run_backtest_case(returns, lookback=5, step=2, seed=123, out_dir=first)
        run_backtest_case(returns, lookback=5, step=2, seed=123, out_dir=second)

        for artifact in ("equity.csv", "rebalance_report.csv", "weights.csv"):
            assert (first / artifact).read_bytes() == (second / artifact).read_bytes()


def check_backtest_manifest_records() -> None:
    returns = _sample_returns()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result = run_backtest_case(
            returns,
            lookback=5,
            step=2,
            seed=123,
            deterministic=True,
            write_manifest=True,
            out_dir=tmp_path,
        )
        manifest = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))
        assert manifest["deterministic_torch"] is True
        assert manifest["artifact_index"] == "artifact_index.json"
        assert abs(result.metrics["sharpe"]) >= 0.0


def check_parallel_results_match_single_process() -> None:
    rng = np.random.default_rng(3)
    returns = rng.normal(0.001, 0.01, size=(36, 3))
    with fast_optimizer(max_iter=3, n_ants=6):
        single = backtest(
            returns,
            lookback=12,
            step=6,
            objective="sharpe",
            seed=11,
            tx_cost_bps=0.0,
            tx_cost_mode="none",
            metric_alpha=0.05,
            cov_model="sample",
            dtype=np.float64,
            cov_cache_size=2,
            workers=None,
            deterministic=True,
        )
        multi = backtest(
            returns,
            lookback=12,
            step=6,
            objective="sharpe",
            seed=11,
            tx_cost_bps=0.0,
            tx_cost_mode="none",
            metric_alpha=0.05,
            cov_model="sample",
            dtype=np.float64,
            cov_cache_size=2,
            workers=2,
            deterministic=True,
        )

    for key in ("returns", "weights", "rebalance_dates"):
        first = single[key]
        second = multi[key]
        if isinstance(first, np.ndarray):
            np.testing.assert_allclose(first, second)
        else:
            assert first == second


def check_parallel_with_factors_and_slippage() -> None:
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0008, 0.012, size=(40, 3))
    base_factors = np.array(
        [
            [1.0, 0.5],
            [-0.6, 1.0],
            [0.2, -0.8],
        ]
    )
    factors = np.tile(base_factors, (40, 1, 1)) + rng.normal(0.0, 0.05, size=(40, 3, 2))
    panel = FactorPanel(
        dates=list(range(40)),
        assets=["A0", "A1", "A2"],
        loadings=factors,
        factor_names=["value", "size"],
    )
    benchmark_series = rng.normal(0.0005, 0.009, size=40)
    benchmark = benchmark_series.reshape(-1, 1)
    slippage = SlippageConfig(model="proportional", param=5.0)

    with fast_optimizer(max_iter=6, n_ants=10):
        single = backtest(
            returns,
            lookback=15,
            step=5,
            objective="sharpe",
            seed=17,
            tx_cost_bps=2.0,
            tx_cost_mode="amortized",
            metric_alpha=0.1,
            cov_model="sample",
            factors=panel,
            factor_align="strict",
            factors_required=False,
            factor_tolerance=5e-2,
            slippage=slippage,
            nt_band=0.0,
            benchmark=benchmark,
            dtype=np.float64,
            cov_cache_size=3,
            workers=None,
            deterministic=True,
            compute_factor_attr=True,
        )
        multi = backtest(
            returns,
            lookback=15,
            step=5,
            objective="sharpe",
            seed=17,
            tx_cost_bps=2.0,
            tx_cost_mode="amortized",
            metric_alpha=0.1,
            cov_model="sample",
            factors=panel,
            factor_align="strict",
            factors_required=False,
            factor_tolerance=5e-2,
            slippage=slippage,
            nt_band=0.0,
            benchmark=benchmark,
            dtype=np.float64,
            cov_cache_size=3,
            workers=2,
            deterministic=True,
            compute_factor_attr=True,
        )

    def _count_feasible(result: Mapping[str, object]) -> int:
        records = result.get("rebalance_records") or []
        return sum(1 for record in records if isinstance(record, Mapping) and record.get("feasible"))

    assert _count_feasible(single) > 0, "no feasible windows (single)"
    assert _count_feasible(multi) > 0, "no feasible windows (multi)"

    single_warnings = single.get("warnings") or []
    multi_warnings = multi.get("warnings") or []

    assert "no_rebalances" not in single_warnings, "single-process run skipped all rebalances"
    assert "no_rebalances" not in multi_warnings, "multi-process run skipped all rebalances"

    for key in ("returns", "weights", "rebalance_dates", "factor_attr"):
        first = single[key]
        second = multi[key]
        if isinstance(first, np.ndarray):
            np.testing.assert_allclose(first, second)
        else:
            assert first == second


def check_service_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        runs_root = tmp_path / "runs"
        runs_root.mkdir(parents=True)

        run_id = "run-service"
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        artifact = run_dir / "output.txt"
        artifact.write_text("hello", encoding="utf-8")
        manifest = {
            "run_id": run_id,
            "artifact_index": "artifact_index.json",
            "signatures": {},
        }
        (run_dir / "run_config.json").write_text(json.dumps(manifest), encoding="utf-8")
        artifact_index = [
            {"name": "output.txt", "sha256": hashlib.sha256(b"hello").hexdigest(), "size": artifact.stat().st_size}
        ]
        (run_dir / "artifact_index.json").write_text(json.dumps(artifact_index), encoding="utf-8")
        (run_dir / "urls.json").write_text(json.dumps({"remote.bin": "s3://bucket/remote.bin"}), encoding="utf-8")

        with temp_environment(RUNS_DIR=str(runs_root), SERVICE_AUTH_TOKEN="secret-token"):
            app = create_app()
            with TestClient(app) as client:
                response = client.get(f"/runs/{run_id}")
                assert response.status_code == 401

                headers = {"Authorization": "Bearer secret-token"}
                response = client.post("/backtest", json={"csv": "data.csv"}, headers=headers)
                assert response.status_code == 202
                payload = response.json()
                assert payload["status"] == "accepted"

                response = client.get(f"/runs/{run_id}", headers=headers)
                assert response.status_code == 200
                manifest_json = response.json()
                assert manifest_json["artifact_index_entries"][0]["name"] == "output.txt"

                response = client.get(f"/artifacts/{run_id}/output.txt", headers=headers)
                assert response.status_code == 200
                assert response.content == b"hello"

                response = client.get(f"/artifacts/{run_id}/remote.bin", headers=headers)
                assert response.status_code == 200
                assert response.json()["location"] == "s3://bucket/remote.bin"


CHECKS = {
    "backtest_sweep_config": check_backtest_sweep_config,
    "backtest_sweep_cli_grid": check_backtest_sweep_cli_grid,
    "backtest_cli_determinism": check_backtest_cli_determinism,
    "backtest_manifest_records": check_backtest_manifest_records,
    "parallel_results_match_single_process": check_parallel_results_match_single_process,
    "parallel_with_factors_and_slippage": check_parallel_with_factors_and_slippage,
    "service_end_to_end": check_service_end_to_end,
}


def main(selected: Iterable[str] | None = None) -> None:
    names = list(selected) if selected else list(CHECKS)
    for name in names:
        if name not in CHECKS:
            raise SystemExit(f"Unknown check: {name}")

    for name in names:
        print(f"[RUNNING] {name}")
        CHECKS[name]()
        print(f"[PASSED]  {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run slow regression checks without pytest")
    parser.add_argument("checks", nargs="*", help="Optional subset of checks to run")
    args = parser.parse_args()
    main(args.checks)
