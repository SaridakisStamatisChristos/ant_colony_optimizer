"""Walk-forward backtesting utilities built around the neuro-ant optimizer."""

from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Union,
)
from types import SimpleNamespace

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from neuro_ant_optimizer import __version__
from neuro_ant_optimizer.constraints import PortfolioConstraints
from neuro_ant_optimizer.optimizer import (
    BenchmarkStats,
    NeuroAntPortfolioOptimizer,
    OptimizationObjective,
    OptimizerConfig,
)
from neuro_ant_optimizer.utils import nearest_psd, set_seed, shrink_covariance

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - minimal environments
    pd = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import polars as pl  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - minimal environments
    pl = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - minimal environments
    yaml = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from numba import njit  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - minimal environments
    njit = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

    _PYDANTIC_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - minimal environments
    BaseModel = object  # type: ignore[assignment]
    ConfigDict = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]

    def model_validator(*_args: Any, **_kwargs: Any):  # type: ignore[override]
        def decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
            return func

        return decorator

    class ValidationError(Exception):  # type: ignore[override]
        def __init__(self, errors: Sequence[Mapping[str, Any]]):
            super().__init__("Validation failed")
            self._errors = list(errors)

        def errors(self) -> Sequence[Mapping[str, Any]]:  # type: ignore[override]
            return list(self._errors)

    _PYDANTIC_AVAILABLE = False


SCHEMA_VERSION = "1.0.0"


ProgressCallback = Callable[[int, int], None]
RebalanceLogCallback = Callable[[Dict[str, Any]], None]

ObjectiveFunction = Callable[
    [np.ndarray, np.ndarray, np.ndarray, PortfolioConstraints, Optional[BenchmarkStats]],
    float,
]
ObjectiveLike = Union[OptimizationObjective, ObjectiveFunction]
CovarianceFunction = Callable[..., np.ndarray]

_CUSTOM_OBJECTIVE_REGISTRY: Dict[str, ObjectiveFunction] = {}
_COVARIANCE_REGISTRY: Dict[str, CovarianceFunction] = {}


@dataclass(frozen=True)
class CovarianceModelSpec:
    fn: CovarianceFunction
    params: Mapping[str, Any]
    base: str
    label: str
    is_custom: bool


def _coerce_parameter_value(value: str) -> Any:
    text = str(value).strip()
    lower = text.lower()
    if lower == "none":
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return int(text)
    except (TypeError, ValueError):
        pass
    try:
        return float(text)
    except (TypeError, ValueError):
        return text


def _parse_cov_model_spec(raw: str) -> Tuple[str, Dict[str, Any]]:
    base, *tokens = raw.split(":")
    params: Dict[str, Any] = {}
    for token in tokens:
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                "Covariance model parameters must be provided as key=value pairs"
            )
        key, value = token.split("=", 1)
        key = key.strip().lower()
        if not key:
            raise ValueError("Covariance model parameter keys must be non-empty")
        params[key] = _coerce_parameter_value(value)
    return base, params


@dataclass(frozen=True)
class ScenarioShock:
    asset_indices: Tuple[int, ...]
    date_indices: Optional[Tuple[int, ...]]
    shift: float = 0.0
    scale: float = 0.0


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    shocks: Tuple[ScenarioShock, ...]
    thresholds: Mapping[str, float] = field(default_factory=dict)


def register_objective(name: str, fn: ObjectiveFunction) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Objective name must be a non-empty string")
    if not callable(fn):
        raise TypeError("Objective function must be callable")
    _CUSTOM_OBJECTIVE_REGISTRY[key] = fn


def register_cov_model(name: str, fn: CovarianceFunction) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Covariance model name must be a non-empty string")
    if not callable(fn):
        raise TypeError("Covariance model must be callable")
    _COVARIANCE_REGISTRY[key] = fn


def get_registered_objectives() -> List[str]:
    return sorted(_CUSTOM_OBJECTIVE_REGISTRY.keys())


def get_available_cov_models() -> List[str]:
    return sorted(_COVARIANCE_REGISTRY.keys())


def _mapping_to_cli(mapping: Mapping[str, Any]) -> List[str]:
    args: List[str] = []
    for key, value in mapping.items():
        opt = f"--{str(key).replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(opt)
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                args.extend([opt, str(item)])
            continue
        args.extend([opt, str(value)])
    return args


def _parse_assignment_list(items: Sequence[str]) -> Dict[str, List[str]]:
    assignments: Dict[str, List[str]] = {}
    for entry in items:
        if "=" not in entry:
            raise ValueError(f"Sweep option '{entry}' must use KEY=VAL1,VAL2 syntax")
        key, raw_values = entry.split("=", 1)
        key = key.strip().replace("-", "_")
        values = [val.strip() for val in raw_values.split(",") if val.strip()]
        if not values:
            raise ValueError(f"Sweep option '{entry}' must provide at least one value")
        assignments[key] = values
    return assignments


def _normalize_sweep_values(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _strip_out_arg(args: Sequence[str]) -> Tuple[List[str], Optional[str]]:
    cleaned: List[str] = []
    out_value: Optional[str] = None
    skip_next = False
    for idx, token in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if token == "--out":
            if idx + 1 < len(args):
                out_value = args[idx + 1]
                skip_next = True
            continue
        cleaned.append(token)
    return cleaned, out_value


def _args_to_mapping(args: Sequence[str]) -> Dict[str, Any]:
    mapping: Dict[str, Any] = {}
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            next_idx = idx + 1
            if next_idx < len(args) and not args[next_idx].startswith("--"):
                mapping[key] = args[next_idx]
                idx += 2
            else:
                mapping[key] = True
                idx += 1
        else:
            idx += 1
    return mapping


def _format_run_name(index: int, params: Mapping[str, Any]) -> str:
    if not params:
        return f"run_{index:03d}"
    pieces = []
    for key in sorted(params.keys()):
        value = str(params[key])
        safe = (
            value.replace("/", "_")
            .replace("\\", "_")
            .replace(":", "-")
            .replace(" ", "")
        )
        pieces.append(f"{key}-{safe}")
    return f"run_{index:03d}_" + "_".join(pieces)


def _maybe_float(value: Any) -> Any:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _collect_sweep_summary(
    run_dir: Path, params: Mapping[str, Any], base_mapping: Mapping[str, Any]
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        with metrics_path.open(newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    metrics[row[0]] = row[1]
    breaches = {
        "sector_breaches": 0,
        "active_breaches": 0,
        "group_breaches": 0,
        "factor_bound_breaches": 0,
    }
    report_path = run_dir / "rebalance_report.csv"
    if report_path.exists():
        with report_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for key in breaches:
                    try:
                        breaches[key] += int(float(row.get(key, 0) or 0))
                    except (TypeError, ValueError):
                        continue
    combined = dict(base_mapping)
    combined.update(params)
    summary: Dict[str, Any] = {
        "run": run_dir.name,
        "out_dir": str(run_dir),
        "objective": combined.get("objective"),
        "cov_model": combined.get("cov_model"),
    }
    for field in ("sharpe", "ann_return", "ann_vol", "max_drawdown"):
        summary[field] = _maybe_float(metrics.get(field))
    summary.update(breaches)
    for key, value in params.items():
        summary[f"param_{key}"] = value
    return summary


def _main_sweep(argv: Sequence[str]) -> None:
    sweep_parser = argparse.ArgumentParser(
        prog="neuro-ant-backtest sweep",
        description="Run multiple backtests across hyper-parameter sweeps",
    )
    sweep_parser.add_argument("--config", type=str, default=None, help="Sweep config YAML/JSON")
    sweep_parser.add_argument(
        "--out",
        type=str,
        default="sweep_runs",
        help="Directory to store sweep outputs",
    )
    sweep_parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="KEY=V1,V2",
        help="Grid search specification (repeatable)",
    )
    sweep_parser.add_argument(
        "--random",
        action="append",
        default=[],
        metavar="KEY=V1,V2",
        help="Random sweep specification (repeatable)",
    )
    sweep_parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Random samples per grid point (defaults to 1 when random sweeps are used)",
    )
    sweep_parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed for random sweeps",
    )

    argv_list = list(argv)
    sweep_args, base_args = sweep_parser.parse_known_args(argv_list)

    base_cli_args: List[str] = list(base_args)
    grid_map: Dict[str, List[str]] = {}
    random_map: Dict[str, List[str]] = {}
    samples = sweep_args.samples
    cfg_out: Optional[str] = None

    if sweep_args.config:
        cfg = _load_sweep_config(Path(sweep_args.config))
        base_section = cfg.get("base")
        if isinstance(base_section, Mapping):
            base_cli_args = _mapping_to_cli(base_section) + base_cli_args
        elif isinstance(base_section, Sequence) and not isinstance(base_section, (str, bytes)):
            base_cli_args = [str(item) for item in base_section] + base_cli_args
        grid_section = cfg.get("grid")
        if isinstance(grid_section, Mapping):
            for key, value in grid_section.items():
                values = _normalize_sweep_values(value)
                if not values:
                    raise ValueError(f"Grid parameter '{key}' must contain values")
                grid_map[str(key).replace("-", "_")] = values
        random_section = cfg.get("random")
        if isinstance(random_section, Mapping):
            for key, value in random_section.items():
                values = _normalize_sweep_values(value)
                if not values:
                    raise ValueError(f"Random parameter '{key}' must contain values")
                random_map[str(key).replace("-", "_")] = values
        if "samples" in cfg and samples == 0:
            try:
                samples = int(cfg["samples"])
            except (TypeError, ValueError):
                samples = sweep_args.samples
        out_value = cfg.get("out")
        if isinstance(out_value, str):
            cfg_out = out_value

    if sweep_args.grid:
        grid_map.update(_parse_assignment_list(sweep_args.grid))
    if sweep_args.random:
        random_map.update(_parse_assignment_list(sweep_args.random))

    base_cli_args, inline_out = _strip_out_arg(base_cli_args)
    cli_out_override: Optional[str] = None
    for idx, token in enumerate(argv_list):
        if token == "--out" and idx + 1 < len(argv_list):
            cli_out_override = argv_list[idx + 1]
            break

    base_out = sweep_args.out
    default_out = sweep_parser.get_default("out")
    if cli_out_override is not None:
        base_out = cli_out_override
    else:
        if cfg_out is not None and base_out == default_out:
            base_out = cfg_out
        if inline_out is not None and base_out == default_out:
            base_out = inline_out

    base_mapping = _args_to_mapping(base_cli_args)

    if grid_map:
        grid_keys = sorted(grid_map.keys())
        grid_values = [grid_map[key] for key in grid_keys]
        grid_specs = [dict(zip(grid_keys, combo)) for combo in itertools.product(*grid_values)]
    else:
        grid_specs = [{}]

    final_specs: List[Dict[str, Any]] = []
    if random_map:
        rng = np.random.default_rng(sweep_args.random_seed)
        random_count = samples if samples > 0 else 1
        for base_spec in grid_specs:
            for _ in range(random_count):
                sampled = {key: rng.choice(values) for key, values in random_map.items()}
                merged = dict(base_spec)
                merged.update(sampled)
                final_specs.append(merged)
    else:
        final_specs = [dict(spec) for spec in grid_specs]

    if not final_specs:
        final_specs = [{}]

    out_dir = Path(base_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    for idx, params in enumerate(final_specs):
        run_name = _format_run_name(idx, params)
        run_dir = out_dir / run_name
        run_args = list(base_cli_args)
        run_args.extend(_mapping_to_cli(params))
        run_args.extend(["--out", str(run_dir)])
        cmd = [sys.executable, "-m", "neuro_ant_optimizer.backtest.backtest", *run_args]
        print(f"Running {run_name} -> {run_dir}")
        subprocess.run(cmd, check=True)
        summary_rows.append(_collect_sweep_summary(run_dir, params, base_mapping))

    param_columns = sorted({key for row in summary_rows for key in row.keys() if key.startswith("param_")})
    metric_columns = ["sharpe", "ann_return", "ann_vol", "max_drawdown"]
    breach_columns = [
        "sector_breaches",
        "sector_penalty",
        "active_breaches",
        "group_breaches",
        "factor_bound_breaches",
    ]
    header = ["run", "out_dir", "objective", "cov_model"] + param_columns + metric_columns + breach_columns
    summary_path = out_dir / "sweep_results.csv"
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field) for field in header})
    print(f"Wrote {summary_path}")


@dataclass
class FactorPanel:
    dates: List[Any]
    assets: List[str]
    loadings: np.ndarray  # shape (T, N, K)
    factor_names: List[str]

    def align_assets(self, asset_order: Sequence[str]) -> "FactorPanel":
        asset_map = {asset: idx for idx, asset in enumerate(self.assets)}
        selected_assets: List[str] = [asset for asset in asset_order if asset in asset_map]
        indices = [asset_map[asset] for asset in selected_assets]
        if not selected_assets:
            raise ValueError("No overlapping assets between returns and factor panel")
        aligned = self.loadings[:, indices, :]
        return FactorPanel(self.dates, selected_assets, aligned, list(self.factor_names))

    def index_map(self) -> Dict[Any, int]:
        if not hasattr(self, "_index_cache"):
            self._index_cache = {date: idx for idx, date in enumerate(self.dates)}
        return self._index_cache  # type: ignore[attr-defined]


def _sort_key(value: Any) -> Tuple[int, Any]:
    if isinstance(value, (np.datetime64,)):
        return (0, value)
    if pd is not None and isinstance(value, pd.Timestamp):  # pragma: no branch - optional
        return (0, value.to_datetime64())
    return (1, str(value))


def _stringify(value: Any) -> str:
    if isinstance(value, (np.datetime64,)):
        return str(np.datetime_as_string(value, unit="s"))
    if pd is not None and isinstance(value, pd.Timestamp):  # pragma: no branch - optional
        return value.isoformat()
    return str(value)


def _hash_returns_window(window: np.ndarray) -> str:
    arr = np.ascontiguousarray(window, dtype=np.float64)
    digest = hashlib.blake2b(arr.view(np.uint8), digest_size=16)
    return digest.hexdigest()


@dataclass
class FactorDiagnostics:
    align_mode: str
    total_assets: int
    total_dates: int
    dropped_assets: List[str]
    dropped_dates: List[Any]
    missing_rebalance_dates: List[Any] = field(default_factory=list)

    _missing_set: Set[Any] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self.align_mode = str(self.align_mode)
        self.dropped_assets = sorted(dict.fromkeys(self.dropped_assets))
        self.dropped_dates = list(self.dropped_dates)
        self._missing_set.update(self.missing_rebalance_dates)
        self.missing_rebalance_dates = list(self._missing_set)

    @property
    def dropped_asset_count(self) -> int:
        return len(self.dropped_assets)

    @property
    def dropped_date_count(self) -> int:
        return len(self.dropped_dates)

    @property
    def missing_window_count(self) -> int:
        return len(self._missing_set)

    def record_missing(self, date: Any) -> None:
        if date not in self._missing_set:
            self._missing_set.add(date)
            self.missing_rebalance_dates.append(date)

    def to_dict(self) -> Dict[str, Any]:
        dropped_dates = sorted(self.dropped_dates, key=_sort_key)
        missing_dates = sorted(self._missing_set, key=_sort_key)
        return {
            "align_mode": self.align_mode,
            "total_assets": self.total_assets,
            "total_dates": self.total_dates,
            "dropped_asset_count": self.dropped_asset_count,
            "dropped_assets": sorted(self.dropped_assets),
            "dropped_date_count": self.dropped_date_count,
            "dropped_dates": [_stringify(val) for val in dropped_dates],
            "missing_window_count": self.missing_window_count,
            "missing_rebalance_dates": [_stringify(val) for val in missing_dates],
        }


_STANDARD_METRIC_KEYS: Tuple[str, ...] = (
    "sharpe",
    "ann_return",
    "ann_vol",
    "max_drawdown",
    "avg_turnover",
    "avg_sector_penalty",
    "avg_slippage_bps",
    "downside_vol",
    "sortino",
    "realized_cvar",
    "tracking_error",
    "info_ratio",
    "turnover_adj_sharpe",
    "calmar_ratio",
    "pain_ratio",
    "hit_rate",
    "te_target",
    "lambda_te",
    "gamma_turnover",
    "cov_cache_size",
    "cov_cache_hits",
    "cov_cache_misses",
    "cov_cache_evictions",
)

_BASELINE_METRIC_KEYS: Tuple[str, ...] = (
    "baseline_sharpe",
    "baseline_info_ratio",
    "alpha_vs_baseline",
    "hit_rate_vs_baseline",
)

_CI_METRIC_KEYS: Tuple[str, ...] = _STANDARD_METRIC_KEYS + _BASELINE_METRIC_KEYS

RUN_TRACKER_METRICS: Tuple[str, ...] = (
    "sharpe",
    "ann_return",
    "max_drawdown",
    "tracking_error",
    "info_ratio",
)


@dataclass
class SlippageConfig:
    model: str
    param: float
    params: Dict[str, float] = field(default_factory=dict)


class _ProgressPrinter:
    """Render incremental progress updates for CLI runs."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._is_tty = bool(getattr(stream, "isatty", lambda: False)())
        self._last: Tuple[int, int] = (-1, -1)
        self._finished = False

    def __call__(self, current: int, total: int) -> None:
        if current < 0:
            current = 0
        if total < 0:
            total = 0
        if not self._is_tty and (current, total) == self._last:
            return
        if total > 0:
            pct = (current / total) * 100.0
        else:
            pct = 0.0
        if self._is_tty:
            msg = f"\rProgress: {current}/{total} ({pct:5.1f}%)"
            self._stream.write(msg)
            if total == 0 or current >= total:
                if not self._finished:
                    self._stream.write("\n")
                    self._finished = True
        else:
            msg = f"Progress: {current}/{total}"
            self._stream.write(msg + "\n")
            if total == 0 or current >= total:
                self._finished = True
        self._stream.flush()
        self._last = (current, total)

    def close(self) -> None:
        if self._is_tty and not self._finished and self._last != (-1, -1):
            self._stream.write("\n")
            self._stream.flush()
        self._finished = True


class _JsonlWriter:
    """Write JSON objects line-by-line to a file handle."""

    def __init__(self, handle: TextIO) -> None:
        self._handle = handle

    def write(self, payload: Mapping[str, Any]) -> None:
        json.dump(payload, self._handle, sort_keys=True)
        self._handle.write("\n")
        self._handle.flush()

    def close(self) -> None:
        try:
            self._handle.flush()
        finally:
            self._handle.close()


def _coerce_scalar(text: str) -> Any:
    lowered = text.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", ""}:
        return None
    try:
        if text.startswith("[") or text.startswith("{"):
            return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _parse_simple_mapping(text: str) -> Dict[str, Any]:
    mapping: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError("Unable to parse config line: missing ':' delimiter")
        key, value = line.split(":", 1)
        mapping[key.strip()] = _coerce_scalar(value.strip())
    return mapping


def _load_run_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    data: Any
    if suffix in {".yaml", ".yml"}:
        if yaml is not None:
            data = yaml.safe_load(text)
        else:
            data = _parse_simple_mapping(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        data = _parse_simple_mapping(text)
    if not isinstance(data, dict):
        raise ValueError("Run config must evaluate to a mapping")
    normalized: Dict[str, Any] = {}
    for key, value in data.items():
        norm_key = str(key).replace("-", "_")
        normalized[norm_key] = value
    return normalized


def _load_sweep_config(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    data: Any
    if suffix in {".yaml", ".yml"} and yaml is not None:
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError("Sweep config must be YAML or JSON")
    if not isinstance(data, Mapping):
        raise ValueError("Sweep config must evaluate to a mapping")
    return data


def _parse_fractional_value(text: str) -> float:
    lowered = text.strip().lower()
    if lowered.endswith("bps"):
        lowered = lowered[:-3]
        return float(lowered) / 1e4
    if lowered.endswith("bp"):
        lowered = lowered[:-2]
        return float(lowered) / 1e4
    if lowered.endswith("%"):
        lowered = lowered[:-1]
        return float(lowered) / 100.0
    return float(lowered)


def _parse_bps_value(text: str) -> float:
    lowered = text.strip().lower()
    if lowered.endswith("bps"):
        lowered = lowered[:-3]
    elif lowered.endswith("bp"):
        lowered = lowered[:-2]
    elif lowered.endswith("%"):
        return float(lowered[:-1]) * 100.0
    return float(lowered)


if _PYDANTIC_AVAILABLE:

    class RunConfig(BaseModel):
        model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

        csv: str
        benchmark_csv: Optional[str] = None
        baseline: Optional[Literal["equal", "cap"]] = None
        cap_weights: Optional[str] = None
        out: str = "bt_out"
        lookback: int = Field(ge=1, default=252)
        step: int = Field(ge=1, default=21)
        objective: str = "sharpe"
        cov_model: str = "sample"
        ewma_span: Optional[int] = Field(default=None, ge=2)
        seed: int = 7
        tx_cost_bps: float = Field(ge=0.0, default=0.0)
        tx_cost_mode: Literal["none", "upfront", "amortized", "posthoc"] = "posthoc"
        rf_bps: float = 0.0
        trading_days: int = Field(ge=1, default=252)
        factors: Optional[str] = None
        factor_align: Literal["strict", "subset"] = "strict"
        factors_required: bool = False
        factor_targets: Optional[str] = None
        factor_tolerance: float = Field(ge=0.0, default=1e-6)
        benchmark_weights: Optional[Any] = None
        active_min: Optional[float] = None
        active_max: Optional[float] = None
        active_group_caps: Optional[Any] = None
        factor_bounds: Optional[Any] = None
        te_target: float = Field(ge=0.0, default=0.0)
        lambda_te: float = Field(ge=0.0, default=0.0)
        gamma_turnover: float = Field(ge=0.0, default=0.0)
        gamma_by_sector: Optional[Any] = None
        refine_every: int = Field(ge=1, default=1)
        out_format: Literal["csv", "parquet"] = "csv"
        save_weights: bool = False
        skip_plot: bool = False
        dry_run: bool = False
        deterministic: bool = False
        drop_duplicates: bool = False
        float32: bool = False
        cache_cov: int = Field(ge=0, default=8)
        max_workers: Optional[int] = Field(default=None, ge=1)
        log_json: Optional[str] = None
        progress: bool = False
        slippage: Optional[str] = None
        nt_band: float = Field(ge=0.0, default=0.0)
        metric_alpha: float = Field(ge=0.0, le=1.0, default=0.05)
        warm_start: Optional[str] = None
        warm_align: Literal["by_date", "last_row"] = "last_row"
        decay: float = Field(ge=0.0, le=1.0, default=0.0)
        scenarios: Optional[Any] = None
        bootstrap: int = Field(ge=0, default=0)
        bootstrap_method: Literal["stationary", "circular"] = "stationary"
        block: int = Field(ge=1, default=21)
        cv: Optional[str] = None

        @model_validator(mode="after")
        def check_covariance_deps(self) -> "RunConfig":
            try:
                _resolve_objective(self.objective)
            except ValueError as exc:  # pragma: no cover - validation surface
                raise ValueError(str(exc)) from exc

            try:
                _resolve_cov_model(self.cov_model)
            except ValueError as exc:  # pragma: no cover - validation surface
                raise ValueError(str(exc)) from exc

            if str(self.cov_model).lower() != "ewma":
                object.__setattr__(self, "ewma_span", None)
            return self

else:

    class RunConfig:
        __slots__ = ("_data",)

        _defaults: Dict[str, Any] = {
            "csv": None,
            "benchmark_csv": None,
            "baseline": None,
            "cap_weights": None,
            "out": "bt_out",
            "lookback": 252,
            "step": 21,
            "objective": "sharpe",
            "cov_model": "sample",
            "ewma_span": None,
            "seed": 7,
            "tx_cost_bps": 0.0,
            "tx_cost_mode": "posthoc",
            "rf_bps": 0.0,
            "trading_days": 252,
            "factors": None,
            "factor_align": "strict",
            "factors_required": False,
            "factor_targets": None,
            "factor_tolerance": 1e-6,
            "benchmark_weights": None,
            "active_min": None,
            "active_max": None,
            "active_group_caps": None,
            "factor_bounds": None,
            "te_target": 0.0,
            "lambda_te": 0.0,
            "gamma_turnover": 0.0,
            "gamma_by_sector": None,
            "refine_every": 1,
            "out_format": "csv",
            "save_weights": False,
            "skip_plot": False,
            "dry_run": False,
            "deterministic": False,
            "drop_duplicates": False,
            "float32": False,
            "cache_cov": 8,
            "max_workers": None,
            "log_json": None,
            "progress": False,
            "slippage": None,
            "nt_band": 0.0,
            "metric_alpha": 0.05,
            "warm_start": None,
            "warm_align": "last_row",
            "decay": 0.0,
            "scenarios": None,
            "bootstrap": 0,
            "bootstrap_method": "stationary",
            "block": 21,
            "cv": None,
        }

        _tx_cost_modes = {"none", "upfront", "amortized", "posthoc"}
        _warm_align = {"by_date", "last_row"}
        _factor_align = {"strict", "subset"}
        _baseline_modes = {"equal", "cap"}
        _out_formats = {"csv", "parquet"}
        _bootstrap_methods = {"stationary", "circular"}

        def __init__(self, data: Dict[str, Any]) -> None:
            self._data = data

        @classmethod
        def _bool(cls, value: Any, name: str, errors: List[Dict[str, Any]]) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, np.integer)):
                return bool(value)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "yes", "1", "on"}:
                    return True
                if lowered in {"false", "no", "0", "off"}:
                    return False
            errors.append({"loc": (name,), "msg": "Input should be a valid boolean"})
            return False

        @classmethod
        def _int(
            cls, value: Any, name: str, errors: List[Dict[str, Any]], *, ge: Optional[int] = None
        ) -> Optional[int]:
            try:
                ivalue = int(value)
            except (TypeError, ValueError):
                errors.append({"loc": (name,), "msg": "Input should be a valid integer"})
                return None
            if ge is not None and ivalue < ge:
                errors.append({"loc": (name,), "msg": f"Input should be greater than or equal to {ge}"})
                return None
            return ivalue

        @classmethod
        def _float(
            cls,
            value: Any,
            name: str,
            errors: List[Dict[str, Any]],
            *,
            ge: Optional[float] = None,
            le: Optional[float] = None,
        ) -> Optional[float]:
            try:
                fvalue = float(value)
            except (TypeError, ValueError):
                errors.append({"loc": (name,), "msg": "Input should be a valid number"})
                return None
            if ge is not None and fvalue < ge:
                errors.append({"loc": (name,), "msg": f"Input should be greater than or equal to {ge}"})
                return None
            if le is not None and fvalue > le:
                errors.append({"loc": (name,), "msg": f"Input should be less than or equal to {le}"})
                return None
            return fvalue

        @classmethod
        def model_validate(cls, raw: Mapping[str, Any]) -> "RunConfig":
            if not isinstance(raw, Mapping):
                raise ValidationError([{ "loc": ("<root>",), "msg": "Input should be a mapping" }])

            errors: List[Dict[str, Any]] = []
            data: Dict[str, Any] = dict(cls._defaults)

            for key in raw.keys():
                if key not in data:
                    errors.append({"loc": (key,), "msg": "extra fields not permitted"})

            csv_value = raw.get("csv")
            if not csv_value:
                errors.append({"loc": ("csv",), "msg": "Field required"})
            else:
                data["csv"] = str(csv_value).strip()

            optional_str_fields = [
                "benchmark_csv",
                "cap_weights",
                "out",
                "factors",
                "factor_targets",
                "log_json",
                "slippage",
                "warm_start",
                "cv",
            ]
            for field in optional_str_fields:
                if field in raw:
                    value = raw[field]
                    data[field] = None if value is None else str(value).strip()

            if "gamma_by_sector" in raw:
                data["gamma_by_sector"] = raw["gamma_by_sector"]

            if "baseline" in raw:
                value = raw["baseline"]
                if value is None:
                    data["baseline"] = None
                else:
                    choice = str(value).strip()
                    if choice not in cls._baseline_modes:
                        errors.append({"loc": ("baseline",), "msg": "Input should be 'equal' or 'cap'"})
                    else:
                        data["baseline"] = choice

            for field in ("active_min", "active_max"):
                if field in raw and raw[field] is not None:
                    result = cls._float(raw[field], field, errors)
                    if result is not None:
                        data[field] = result

            for field in ("lookback", "step"):
                if field in raw:
                    result = cls._int(raw[field], field, errors, ge=1)
                    if result is not None:
                        data[field] = result

            if "bootstrap" in raw:
                result = cls._int(raw["bootstrap"], "bootstrap", errors, ge=0)
                if result is not None:
                    data["bootstrap"] = result

            if "block" in raw:
                result = cls._int(raw["block"], "block", errors, ge=1)
                if result is not None:
                    data["block"] = result

            if "objective" in raw:
                obj = str(raw["objective"]).strip()
                try:
                    _resolve_objective(obj)
                    data["objective"] = obj
                except ValueError as exc:
                    errors.append({"loc": ("objective",), "msg": str(exc)})

            if "cov_model" in raw:
                cov = str(raw["cov_model"]).strip()
                try:
                    _resolve_cov_model(cov)
                    data["cov_model"] = cov
                except ValueError as exc:
                    errors.append({"loc": ("cov_model",), "msg": str(exc)})

            if "bootstrap_method" in raw:
                method = str(raw["bootstrap_method"]).strip().lower()
                if method not in cls._bootstrap_methods:
                    errors.append({"loc": ("bootstrap_method",), "msg": "Input should be 'stationary' or 'circular'"})
                else:
                    data["bootstrap_method"] = method

            if "ewma_span" in raw and raw["ewma_span"] is not None:
                span = cls._int(raw["ewma_span"], "ewma_span", errors, ge=2)
                if span is not None:
                    data["ewma_span"] = span

            if "seed" in raw:
                result = cls._int(raw["seed"], "seed", errors)
                if result is not None:
                    data["seed"] = result

            float_fields = {
                "tx_cost_bps": (0.0, None),
                "rf_bps": (None, None),
                "te_target": (0.0, None),
                "lambda_te": (0.0, None),
                "gamma_turnover": (0.0, None),
                "metric_alpha": (0.0, 1.0),
                "factor_tolerance": (0.0, None),
                "nt_band": (0.0, None),
                "decay": (0.0, 1.0),
            }
            for field, bounds in float_fields.items():
                if field in raw:
                    result = cls._float(raw[field], field, errors, ge=bounds[0], le=bounds[1])
                    if result is not None:
                        data[field] = result

            if "trading_days" in raw:
                result = cls._int(raw["trading_days"], "trading_days", errors, ge=1)
                if result is not None:
                    data["trading_days"] = result

            if "cache_cov" in raw:
                result = cls._int(raw["cache_cov"], "cache_cov", errors, ge=0)
                if result is not None:
                    data["cache_cov"] = result

            if "max_workers" in raw and raw["max_workers"] is not None:
                result = cls._int(raw["max_workers"], "max_workers", errors, ge=1)
                if result is not None:
                    data["max_workers"] = result

            bool_fields = [
                "factors_required",
                "save_weights",
                "skip_plot",
                "dry_run",
                "float32",
                "progress",
            ]
            for field in bool_fields:
                if field in raw:
                    data[field] = cls._bool(raw[field], field, errors)

            if "tx_cost_mode" in raw:
                mode = str(raw["tx_cost_mode"]).strip()
                if mode not in cls._tx_cost_modes:
                    errors.append({"loc": ("tx_cost_mode",), "msg": "Input should be one of none, upfront, amortized, posthoc"})
                else:
                    data["tx_cost_mode"] = mode

            if "factor_align" in raw:
                align = str(raw["factor_align"]).strip()
                if align not in cls._factor_align:
                    errors.append({"loc": ("factor_align",), "msg": "Input should be 'strict' or 'subset'"})
                else:
                    data["factor_align"] = align

            if "warm_align" in raw:
                align = str(raw["warm_align"]).strip()
                if align not in cls._warm_align:
                    errors.append({"loc": ("warm_align",), "msg": "Input should be 'by_date' or 'last_row'"})
                else:
                    data["warm_align"] = align

            if "out_format" in raw:
                fmt = str(raw["out_format"]).strip()
                if fmt not in cls._out_formats:
                    errors.append({"loc": ("out_format",), "msg": "Input should be 'csv' or 'parquet'"})
                else:
                    data["out_format"] = fmt

            if "refine_every" in raw:
                result = cls._int(raw["refine_every"], "refine_every", errors, ge=1)
                if result is not None:
                    data["refine_every"] = result

            passthrough_fields = [
                "benchmark_weights",
                "active_group_caps",
                "factor_bounds",
            ]
            for field in passthrough_fields:
                if field in raw:
                    data[field] = raw[field]

            if errors:
                raise ValidationError(errors)

            if str(data["cov_model"]).lower() != "ewma":
                data["ewma_span"] = None

            return cls(data)

        def model_dump(self) -> Dict[str, Any]:
            return dict(self._data)


def _read_weights_csv(path: Path) -> Tuple[List[Any], np.ndarray, List[str]]:
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if raw.size == 0:
        return [], np.zeros((0, 0), dtype=float), []
    names = list(raw.dtype.names or [])
    dates: List[Any] = []
    if "date" in names:
        names.remove("date")
        dates = np.atleast_1d(raw["date"]).tolist()
    matrix_rows: List[np.ndarray] = []
    for name in names:
        matrix_rows.append(np.atleast_1d(raw[name]).astype(float))
    if not matrix_rows:
        return dates, np.zeros((0, 0), dtype=float), []
    matrix = np.vstack(matrix_rows).T
    return dates, matrix, names


def _align_warm_to_assets(
    warm_names: Sequence[str], warm_w: np.ndarray, dest_assets: Sequence[str]
) -> np.ndarray:
    index = {name: idx for idx, name in enumerate(warm_names)}
    aligned = np.zeros(len(dest_assets), dtype=float)
    for j, asset in enumerate(dest_assets):
        idx = index.get(asset)
        if idx is not None and warm_w.size:
            aligned[j] = float(warm_w[idx])
    total = float(aligned.sum())
    if total > 1e-12:
        aligned /= total
    return aligned

def _load_data_structure(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is not None:
            return yaml.safe_load(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError(
                "PyYAML is required to parse structured YAML constraint files"
            ) from exc
    if suffix == ".json":
        return json.loads(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _parse_simple_mapping(text)


def _maybe_load_structure(spec: Any) -> Any:
    if spec is None:
        return None
    if isinstance(spec, (str, Path)):
        path = Path(spec)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file '{path}' not found")
        return _load_data_structure(path)
    return spec


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "null"}:
            return None
        return float(value)
    raise TypeError(f"Unable to coerce value '{value}' to float")


def _normalize_active_bounds(
    min_active: Optional[float], max_active: Optional[float]
) -> Tuple[Optional[float], Optional[float]]:
    if min_active is not None:
        if min_active < -1.0 - 1e-12:
            raise ValueError("active_min must be greater than or equal to -1")
        if min_active > 0.0 + 1e-12:
            raise ValueError("active_min must be less than or equal to 0")
    if max_active is not None:
        if max_active > 1.0 + 1e-12:
            raise ValueError("active_max must be less than or equal to 1")
        if max_active < 0.0 - 1e-12:
            raise ValueError("active_max must be greater than or equal to 0")
    if min_active is not None and max_active is not None and min_active > max_active + 1e-12:
        raise ValueError("active_min must not exceed active_max")
    return min_active, max_active


def _manifest_bound(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def _prepare_benchmark_weights(
    spec: Any, asset_names: Sequence[str]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Optional[float]]]:
    data = _maybe_load_structure(spec)
    resolved: Dict[str, Optional[float]] = {name: None for name in asset_names}
    if data is None:
        return None, None, resolved

    weights = np.zeros(len(asset_names), dtype=float)
    mask = np.zeros(len(asset_names), dtype=bool)
    index = {name: idx for idx, name in enumerate(asset_names)}
    unknown: Set[str] = set()

    def _assign(name: Any, value: Any) -> None:
        if name is None:
            raise ValueError("Benchmark weight entry missing asset name")
        asset = str(name)
        if value is None:
            return
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return
        val = float(value)
        idx = index.get(asset)
        if idx is None:
            unknown.add(asset)
            return
        weights[idx] = val
        mask[idx] = True
        resolved[asset] = val

    if isinstance(data, Mapping):
        for key, value in data.items():
            _assign(key, value)
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        if len(data) == len(asset_names) and all(
            not isinstance(item, Mapping) for item in data
        ):
            for idx, value in enumerate(data):
                if value is None:
                    continue
                weights[idx] = float(value)
                mask[idx] = True
                resolved[asset_names[idx]] = float(value)
        else:
            for entry in data:
                if isinstance(entry, Mapping):
                    name = entry.get("asset") or entry.get("name") or entry.get("ticker")
                    weight_val = entry.get("weight")
                    if weight_val is None:
                        weight_val = entry.get("value")
                    _assign(name, weight_val)
                elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
                    _assign(entry[0], entry[1])
                else:
                    raise ValueError("Unable to parse benchmark weight entry")
    else:
        raise ValueError("Benchmark weights must be a mapping or list")

    if unknown:
        print(
            "Warning: benchmark weights provided for unknown assets: "
            + ", ".join(sorted(unknown))
        )

    if not mask.any():
        return None, None, resolved

    return weights, mask, resolved


def _coerce_group_entry(
    raw_name: Optional[str], value: Any, index: int
) -> Tuple[str, List[str], Optional[float], Optional[float]]:
    name = raw_name
    members: Optional[List[str]] = None
    lower: Optional[float] = None
    upper: Optional[float] = None

    if isinstance(value, Mapping):
        maybe_name = value.get("name") or value.get("label") or value.get("group_name")
        if maybe_name is not None:
            name = str(maybe_name)
        maybe_group = value.get("group")
        if members is None and isinstance(maybe_group, (list, tuple, set)):
            members = [str(item) for item in maybe_group]
        if members is None:
            alt_members = value.get("members") or value.get("assets") or value.get("tickers")
            if isinstance(alt_members, (list, tuple, set)):
                members = [str(item) for item in alt_members]
        if name is None and isinstance(maybe_group, str):
            name = maybe_group
        lower_val = value.get("lower")
        if lower_val is None:
            lower_val = value.get("min")
        upper_val = value.get("upper")
        if upper_val is None:
            upper_val = value.get("max")
        cap_val = value.get("cap")
        if lower_val is None and upper_val is None and cap_val is not None:
            cap = float(cap_val)
            lower = -abs(cap)
            upper = abs(cap)
        else:
            if lower_val is not None:
                lower = float(lower_val)
            if upper_val is not None:
                upper = float(upper_val)
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        if name is None:
            name = str(value[0])
        members_candidate = value[1]
        if isinstance(members_candidate, (list, tuple, set)):
            members = [str(item) for item in members_candidate]
        cap = value[2] if len(value) > 2 else None
        if cap is not None:
            cap_val = float(cap)
            lower = -abs(cap_val)
            upper = abs(cap_val)
    elif isinstance(value, (float, int, np.floating, np.integer)):
        cap = float(value)
        lower = -abs(cap)
        upper = abs(cap)
    else:
        raise ValueError("Unable to parse active group entry")

    if members is None:
        raise ValueError("Active group entry must provide a list of members")
    if name is None:
        name = f"group_{index}"
    if lower is None and upper is None:
        raise ValueError("Active group entry must provide a cap or explicit bounds")
    if lower is None:
        lower = float("-inf")
    if upper is None:
        upper = float("inf")
    if lower > upper + 1e-12:
        raise ValueError("Active group lower bound exceeds upper bound")
    return str(name), members, lower, upper


def _prepare_active_groups(
    spec: Any, asset_names: Sequence[str]
) -> Tuple[
    Optional[List[int]],
    Optional[Dict[int, Tuple[float, float]]],
    List[Dict[str, Any]],
]:
    data = _maybe_load_structure(spec)
    if data is None:
        return None, None, []

    entries: List[Tuple[str, List[str], float, float]] = []
    if isinstance(data, Mapping):
        for idx, (key, value) in enumerate(data.items()):
            entries.append(_coerce_group_entry(str(key), value, idx))
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        for idx, value in enumerate(data):
            raw_name = None
            if isinstance(value, Mapping):
                maybe_name = value.get("name") or value.get("label")
                raw_name = str(maybe_name) if maybe_name is not None else None
            entries.append(_coerce_group_entry(raw_name, value, idx))
    else:
        raise ValueError("Active group caps must be a mapping or list")

    if not entries:
        return None, None, []

    asset_index = {name: idx for idx, name in enumerate(asset_names)}
    group_map = np.full(len(asset_names), -1, dtype=int)
    bounds: Dict[int, Tuple[float, float]] = {}
    manifest: List[Dict[str, Any]] = []
    assigned: Dict[str, str] = {}
    unknown: Set[str] = set()
    group_id = 0

    for name, members, lower, upper in entries:
        known_members: List[str] = []
        mask = np.zeros(len(asset_names), dtype=bool)
        for member in members:
            idx = asset_index.get(member)
            if idx is None:
                unknown.add(member)
                continue
            mask[idx] = True
            known_members.append(member)
        if not known_members:
            continue
        for member in known_members:
            if member in assigned:
                raise ValueError(
                    f"Asset '{member}' assigned to multiple active groups: {assigned[member]} and {name}"
                )
            assigned[member] = name
            group_map[asset_index[member]] = group_id
        bounds[group_id] = (float(lower), float(upper))
        manifest.append(
            {
                "name": name,
                "members": known_members,
                "lower": _manifest_bound(lower),
                "upper": _manifest_bound(upper),
            }
        )
        group_id += 1

    if unknown:
        print(
            "Warning: active group definitions include unknown assets: "
            + ", ".join(sorted(unknown))
        )

    if group_id == 0:
        return None, None, []

    return group_map.tolist(), bounds, manifest


def _prepare_factor_bounds(
    spec: Any, factor_names: Sequence[str]
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Dict[str, Dict[str, Optional[float]]],
]:
    if not factor_names:
        return None, None, {}
    data = _maybe_load_structure(spec)
    if data is None:
        return None, None, {}

    lower = np.full(len(factor_names), -np.inf, dtype=float)
    upper = np.full(len(factor_names), np.inf, dtype=float)
    manifest: Dict[str, Dict[str, Optional[float]]] = {}
    name_to_idx = {name: idx for idx, name in enumerate(factor_names)}
    unknown: Set[str] = set()

    def _assign(name: Any, bounds_val: Any) -> None:
        if name is None:
            raise ValueError("Factor bound entry missing factor name")
        factor = str(name)
        idx = name_to_idx.get(factor)
        if idx is None:
            unknown.add(factor)
            return
        lo: Optional[float] = None
        hi: Optional[float] = None
        if isinstance(bounds_val, Mapping):
            if "bounds" in bounds_val and isinstance(bounds_val["bounds"], Sequence):
                seq = bounds_val["bounds"]
                if len(seq) >= 1:
                    lo = _coerce_optional_float(seq[0])
                if len(seq) >= 2:
                    hi = _coerce_optional_float(seq[1])
            else:
                lo = _coerce_optional_float(
                    bounds_val.get("lower", bounds_val.get("min"))
                )
                hi = _coerce_optional_float(bounds_val.get("upper", bounds_val.get("max")))
        elif isinstance(bounds_val, Sequence) and not isinstance(bounds_val, (str, bytes)):
            if len(bounds_val) == 0:
                return
            lo = _coerce_optional_float(bounds_val[0])
            hi = _coerce_optional_float(bounds_val[1]) if len(bounds_val) > 1 else None
        else:
            raise ValueError("Factor bounds must be specified as a mapping or sequence")

        lo_val = -np.inf if lo is None else float(lo)
        hi_val = np.inf if hi is None else float(hi)
        if lo_val > hi_val + 1e-12:
            raise ValueError("Factor lower bound exceeds upper bound")
        lower[idx] = lo_val
        upper[idx] = hi_val
        manifest[factor] = {"lower": _manifest_bound(lo_val), "upper": _manifest_bound(hi_val)}

    if isinstance(data, Mapping):
        for key, value in data.items():
            _assign(key, value)
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        for entry in data:
            if isinstance(entry, Mapping):
                name = entry.get("factor") or entry.get("name")
                if name is None and "group" in entry:
                    name = entry["group"]
                _assign(name, entry)
            else:
                raise ValueError("Factor bounds list entries must be mappings")
    else:
        raise ValueError("Factor bounds must be a mapping or list")

    if unknown:
        print(
            "Warning: factor bounds specified for unknown factors: "
            + ", ".join(sorted(unknown))
        )

    if not manifest:
        return None, None, {}

    return lower, upper, manifest


def _infer_sector_map(
    frame: Any, asset_names: Sequence[str]
) -> Tuple[Optional[List[int]], Dict[int, str]]:
    if pd is None or not hasattr(frame, "columns"):
        return None, {}
    columns = getattr(frame, "columns")
    if isinstance(columns, pd.MultiIndex) and columns.nlevels >= 2:
        sectors = [str(val) for val in columns.get_level_values(0)]
        sector_index: Dict[str, int] = {}
        sector_map: List[int] = []
        sector_name_map: Dict[int, str] = {}
        for idx, label in enumerate(sectors):
            key = label or f"sector_{idx}"
            sec_id = sector_index.setdefault(key, len(sector_index))
            sector_map.append(sec_id)
            sector_name_map[sec_id] = key
        if len(sector_map) == len(asset_names):
            return sector_map, sector_name_map
    return None, {}


def _prepare_turnover_penalties(
    spec: Any,
    asset_names: Sequence[str],
    inferred_sector_map: Optional[List[int]],
    inferred_sector_names: Dict[int, str],
) -> Tuple[
    Optional[np.ndarray],
    Optional[List[int]],
    Dict[int, str],
    List[Dict[str, Any]],
]:
    data = _maybe_load_structure(spec)
    if data is None:
        return None, inferred_sector_map, dict(inferred_sector_names), []

    entries: List[Tuple[str, Any]] = []
    if isinstance(data, Mapping):
        entries = [(str(key), value) for key, value in data.items()]
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        for value in data:
            if not isinstance(value, Mapping):
                raise ValueError("gamma_by_sector list entries must be mappings")
            name = value.get("sector") or value.get("name") or value.get("label") or value.get("asset")
            if name is None:
                raise ValueError("gamma_by_sector entries must include a sector or asset name")
            entries.append((str(name), value))
    else:
        raise ValueError("gamma_by_sector specification must be a mapping or list of mappings")

    n_assets = len(asset_names)
    gamma_vector = np.zeros(n_assets, dtype=float)
    assigned = np.zeros(n_assets, dtype=bool)
    asset_index = {name: idx for idx, name in enumerate(asset_names)}
    sector_map_arr = (
        np.asarray(inferred_sector_map, dtype=int) if inferred_sector_map is not None else None
    )
    sector_name_map = dict(inferred_sector_names)
    sector_index = {name: sid for sid, name in sector_name_map.items()}
    manifest: List[Dict[str, Any]] = []
    default_gamma: Optional[float] = None

    def _ensure_sector_array() -> np.ndarray:
        nonlocal sector_map_arr
        if sector_map_arr is None:
            sector_map_arr = np.full(n_assets, -1, dtype=int)
        return sector_map_arr

    for raw_name, raw_value in entries:
        key_lower = raw_name.lower()
        if key_lower == "default":
            if isinstance(raw_value, Mapping):
                gamma_raw = (
                    raw_value.get("gamma")
                    or raw_value.get("penalty")
                    or raw_value.get("weight")
                )
            else:
                gamma_raw = raw_value
            gamma_val = _coerce_optional_float(gamma_raw)
            if gamma_val is None:
                raise ValueError("gamma_by_sector default entry must include a numeric value")
            default_gamma = float(gamma_val)
            continue

        if isinstance(raw_value, Mapping):
            members = raw_value.get("members") or raw_value.get("assets")
            gamma_raw = raw_value.get("gamma") or raw_value.get("penalty") or raw_value.get("weight")
            if gamma_raw is None:
                raise ValueError(f"gamma_by_sector entry '{raw_name}' missing gamma value")
            gamma_val_opt = _coerce_optional_float(gamma_raw)
            if gamma_val_opt is None:
                raise ValueError(f"gamma_by_sector entry '{raw_name}' gamma must be numeric")
            gamma_val = float(gamma_val_opt)
            if members is not None:
                members_seq = list(members)
                sector_array = _ensure_sector_array()
                sec_id = sector_index.setdefault(raw_name, len(sector_index))
                sector_name_map[sec_id] = raw_name
                assigned_members: List[str] = []
                for member in members_seq:
                    asset = str(member)
                    idx = asset_index.get(asset)
                    if idx is None:
                        raise ValueError(f"gamma_by_sector references unknown asset '{asset}'")
                    if sector_array[idx] not in {-1, sec_id}:
                        raise ValueError(
                            f"Asset '{asset}' assigned to multiple sectors in gamma_by_sector"
                        )
                    sector_array[idx] = sec_id
                    gamma_vector[idx] = gamma_val
                    assigned[idx] = True
                    assigned_members.append(asset)
                manifest.append(
                    {
                        "sector": raw_name,
                        "gamma": gamma_val,
                        "members": assigned_members,
                    }
                )
                continue
            gamma_target = raw_name
            sector_array = _ensure_sector_array()
            sec_id = sector_index.get(gamma_target)
            if sec_id is None:
                raise ValueError(
                    f"gamma_by_sector entry '{raw_name}' missing members and no inferred sector mapping"
                )
            member_idx = np.nonzero(sector_array == sec_id)[0]
            if member_idx.size == 0:
                raise ValueError(
                    f"gamma_by_sector entry '{raw_name}' has no assets in inferred sector mapping"
                )
            members_list = [asset_names[i] for i in member_idx]
            for idx in member_idx:
                gamma_vector[idx] = gamma_val
                assigned[idx] = True
            manifest.append(
                {
                    "sector": raw_name,
                    "gamma": gamma_val,
                    "members": members_list,
                }
            )
            continue

        if isinstance(raw_value, (float, int, np.floating, np.integer)):
            gamma_val = float(raw_value)
            idx = asset_index.get(raw_name)
            if idx is not None:
                gamma_vector[idx] = gamma_val
                assigned[idx] = True
                manifest.append({"asset": raw_name, "gamma": gamma_val})
                continue
            sector_array = _ensure_sector_array()
            sec_id = sector_index.get(raw_name)
            if sec_id is None:
                raise ValueError(
                    f"gamma_by_sector entry '{raw_name}' does not match an asset or inferred sector"
                )
            member_idx = np.nonzero(sector_array == sec_id)[0]
            if member_idx.size == 0:
                raise ValueError(
                    f"gamma_by_sector entry '{raw_name}' has no members to assign"
                )
            members_list = [asset_names[i] for i in member_idx]
            for idx in member_idx:
                gamma_vector[idx] = gamma_val
                assigned[idx] = True
            manifest.append(
                {
                    "sector": raw_name,
                    "gamma": gamma_val,
                    "members": members_list,
                }
            )
            continue

        raise ValueError(
            "gamma_by_sector entries must be numeric values or mappings with 'members'/'gamma' keys"
        )

    if default_gamma is not None:
        for idx in range(n_assets):
            if not assigned[idx]:
                gamma_vector[idx] = default_gamma
                assigned[idx] = True

    gamma_out: Optional[np.ndarray]
    if np.any(assigned) or default_gamma is not None:
        gamma_out = gamma_vector
    else:
        gamma_out = None

    sector_map_list = sector_map_arr.tolist() if sector_map_arr is not None else inferred_sector_map
    return gamma_out, sector_map_list, sector_name_map, manifest


def _serialize_args(args: argparse.Namespace) -> Dict[str, Any]:
    blob: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            blob[key] = str(value)
        elif isinstance(value, (list, tuple)):
            blob[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            blob[key] = value
    return blob


def _resolve_git_sha() -> Optional[str]:
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        return sha or None
    except Exception:
        return None


def _generate_run_id(seed: Optional[str] = None) -> str:
    if seed:
        return str(seed)
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"run-{timestamp}-{suffix}"


def compute_tracking_error(active_returns: np.ndarray, trading_days: int) -> float:
    arr = np.asarray(active_returns, dtype=float)
    if arr.size == 0:
        return 0.0
    trading = max(1, int(trading_days))
    te = float(np.std(arr) * math.sqrt(trading))
    return float(te if te >= 0.0 else 0.0)


def _archive_run_artifacts(out_dir: Path, artifact_dir: Path, run_id: str) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    archive_base = artifact_dir / run_id
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=out_dir)
    return Path(archive_path)


def _append_run_tracker(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "run_id",
        "timestamp",
        "git_sha",
        "out_dir",
        "manifest",
        "config",
        "objective",
        "artifact",
        "args_json",
        *RUN_TRACKER_METRICS,
    ]
    row = {key: record.get(key) for key in header}
    exists = path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_run_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    config_path: Optional[Path],
    results: Optional[Dict[str, Any]] = None,
    extras: Optional[Dict[str, Any]] = None,
    *,
    validated: bool = False,
) -> None:
    manifest: Dict[str, Any] = {
        "args": _serialize_args(args),
        "schema_version": SCHEMA_VERSION,
        "package_version": __version__,
        "python_version": sys.version,
        "validated": bool(validated),
    }
    if getattr(args, "run_id", None):
        manifest["run_id"] = str(args.run_id)
    manifest["deterministic_torch"] = bool(getattr(args, "deterministic", False))
    if config_path is not None:
        manifest["config_path"] = str(config_path)
    if extras:
        try:
            manifest["resolved_constraints"] = json.loads(json.dumps(extras))
        except TypeError:
            manifest["resolved_constraints"] = extras
    try:  # optional torch dependency
        import torch

        manifest["torch_version"] = torch.__version__
    except ModuleNotFoundError:  # pragma: no cover - environments without torch
        manifest["torch_version"] = None

    git_sha = _resolve_git_sha()
    if git_sha:
        manifest["git_sha"] = git_sha

    if results is not None:
        warnings = results.get("warnings")
        if warnings:
            manifest["warnings"] = list(warnings)
        diagnostics = results.get("factor_diagnostics")
        if diagnostics is not None:
            manifest["factor_diagnostics"] = diagnostics
        manifest["decay"] = results.get("decay")
        manifest["warm_start"] = results.get("warm_start")
        manifest["warm_align"] = results.get("warm_align")
        manifest["warm_applied_count"] = results.get("warm_applied_count", 0)

    (out_dir / "run_config.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _coerce_date(value: Any) -> Any:
    if isinstance(value, (np.datetime64,)):
        return value
    if pd is not None and isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return value
    try:
        return np.datetime64(text)
    except Exception:  # pragma: no cover - fallback for unusual inputs
        return text


def _build_factor_panel(
    dates: Sequence[Any],
    data: np.ndarray,
    asset_headers: Sequence[str],
    factor_headers: Optional[Sequence[str]],
) -> FactorPanel:
    if data.ndim != 2:
        raise ValueError("Factor data must be a 2D array")
    if len(asset_headers) != data.shape[1]:
        raise ValueError("Header column count does not match factor data columns")

    ordered: "OrderedDict[str, List[Tuple[int, Optional[str]]]]" = OrderedDict()
    for col_idx, raw_asset in enumerate(asset_headers):
        asset = str(raw_asset).strip()
        if not asset:
            raise ValueError("Factor file contains empty asset column name")
        factor_name: Optional[str] = None
        if factor_headers is not None:
            factor_name = str(factor_headers[col_idx]).strip() or None
        ordered.setdefault(asset, []).append((col_idx, factor_name))

    asset_list = list(ordered.keys())
    first_asset = asset_list[0]
    base_columns = ordered[first_asset]
    n_factors = len(base_columns)
    if n_factors == 0:
        raise ValueError("Factor panel contains no factor columns")

    def _factor_names() -> List[str]:
        names: List[str] = []
        for pos, (_, maybe_name) in enumerate(base_columns):
            if maybe_name is None:
                names.append(f"factor_{pos}")
            else:
                names.append(maybe_name)
        return names

    factor_names = _factor_names()
    for asset, cols in ordered.items():
        if len(cols) != n_factors:
            raise ValueError("Factor columns per asset must be consistent")
        if factor_headers is not None:
            for idx, (_, maybe_name) in enumerate(cols):
                name = (maybe_name or factor_names[idx])
                factor_names[idx] = name

    loadings: List[np.ndarray] = []
    for row in data:
        row_exposures: List[List[float]] = []
        for asset in asset_list:
            cols = ordered[asset]
            values: List[float] = []
            for col_idx, _ in cols:
                cell = row[col_idx]
                if isinstance(cell, str):
                    cell = cell.strip()
                if cell == "" or cell is None:
                    values.append(float("nan"))
                else:
                    values.append(float(cell))
            row_exposures.append(values)
        loadings.append(np.asarray(row_exposures, dtype=float))

    panel = np.stack(loadings, axis=0)
    coerced_dates = [_coerce_date(date) for date in dates]
    return FactorPanel(coerced_dates, asset_list, panel, factor_names)


def _load_factor_csv(csv_path: Path) -> FactorPanel:
    if pd is not None:
        try:
            frame = pd.read_csv(csv_path, header=[0, 1], index_col=0)
            assets = list(frame.columns.get_level_values(0))
            factors = list(frame.columns.get_level_values(1))
            return _build_factor_panel(frame.index.tolist(), frame.to_numpy(), assets, factors)
        except ValueError:
            frame = pd.read_csv(csv_path, index_col=0)
            assets = list(frame.columns)
            return _build_factor_panel(frame.index.tolist(), frame.to_numpy(), assets, None)

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        rows = [row for row in reader if row]

    if not rows:
        raise ValueError("Factor CSV is empty")

    header_row = rows[0]
    if len(header_row) < 2:
        raise ValueError("Factor CSV must contain at least one asset column")
    asset_headers = [cell.strip() for cell in header_row[1:]]

    factor_headers: Optional[List[str]] = None
    start_idx = 1
    if len(rows) > 1:
        sample = rows[1][1:]
        def _is_float(cell: str) -> bool:
            try:
                float(cell)
                return True
            except ValueError:
                return False

        if not all(_is_float(cell) or cell.strip() == "" for cell in sample):
            factor_headers = [cell.strip() for cell in rows[1][1:]]
            start_idx = 2

    data_rows = rows[start_idx:]
    dates: List[Any] = []
    values: List[List[float]] = []
    for row in data_rows:
        if not row:
            continue
        dates.append(row[0])
        row_vals: List[float] = []
        for cell in row[1:]:
            cell = cell.strip()
            if not cell:
                row_vals.append(float("nan"))
            else:
                row_vals.append(float(cell))
        values.append(row_vals)

    data = np.asarray(values, dtype=float) if values else np.empty((0, len(asset_headers)), dtype=float)
    return _build_factor_panel(dates, data, asset_headers, factor_headers)


def _load_factor_parquet(path: Path) -> FactorPanel:
    if pd is None:
        raise RuntimeError("Reading parquet requires pandas to be installed")
    frame = pd.read_parquet(path)
    if isinstance(frame.columns, pd.MultiIndex):
        assets = list(frame.columns.get_level_values(0))
        factors = list(frame.columns.get_level_values(1))
    else:
        assets = list(frame.columns)
        factors = None
    return _build_factor_panel(frame.index.tolist(), frame.to_numpy(), assets, factors)


def _load_factor_yaml(path: Path) -> FactorPanel:
    if yaml is None:
        raise RuntimeError("Loading YAML factor files requires pyyaml to be installed")
    with path.open("r", encoding="utf-8") as fh:
        parsed = yaml.safe_load(fh)
    if not isinstance(parsed, dict):
        raise ValueError("YAML factor file must be a mapping of dates to assets")

    dates: List[Any] = []
    asset_order: List[str] = []
    loadings: List[np.ndarray] = []
    factor_names: Optional[List[str]] = None
    for date_key, payload in parsed.items():
        if not isinstance(payload, dict):
            raise ValueError("Each date entry must map to asset exposures")
        if not asset_order:
            asset_order = list(payload.keys())
        exposures: List[List[float]] = []
        for asset in asset_order:
            raw = payload.get(asset, [])
            if not isinstance(raw, (list, tuple)):
                raise ValueError("Factor exposures must be sequences")
            values = [float(x) for x in raw]
            if factor_names is None:
                factor_names = [f"factor_{i}" for i in range(len(values))]
            exposures.append(values)
        dates.append(date_key)
        loadings.append(np.asarray(exposures, dtype=float))

    if factor_names is None:
        raise ValueError("YAML factor file did not contain any exposures")
    panel = np.stack(loadings, axis=0)
    coerced_dates = [_coerce_date(d) for d in dates]
    return FactorPanel(coerced_dates, asset_order, panel, factor_names)


def load_factor_panel(path: Optional[Path]) -> Optional[FactorPanel]:
    if path is None:
        return None
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_factor_csv(path)
    if suffix in {".yaml", ".yml"}:
        return _load_factor_yaml(path)
    if suffix in {".parquet", ".pq"}:
        return _load_factor_parquet(path)
    raise ValueError(f"Unsupported factor file extension '{path.suffix}'")


def _align_targets(
    values: Any, factor_names: Sequence[str], expected_len: int
) -> np.ndarray:
    if isinstance(values, dict):
        return np.asarray([float(values.get(name, 0.0)) for name in factor_names], dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.size < expected_len:
        raise ValueError("Factor targets length does not match factor loadings")
    return arr[:expected_len]


def load_factor_targets(
    path: Optional[Path], factor_names: Sequence[str]
) -> Optional[np.ndarray]:
    if path is None:
        return None
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("Loading YAML factor targets requires pyyaml to be installed")
        with path.open("r", encoding="utf-8") as fh:
            parsed = yaml.safe_load(fh)
        if isinstance(parsed, dict):
            return _align_targets(parsed, factor_names, len(factor_names))
        if isinstance(parsed, (list, tuple)):
            return _align_targets(list(parsed), factor_names, len(factor_names))
        raise ValueError("Unsupported YAML structure for factor targets")
    if suffix == ".csv":
        if pd is not None:
            frame = pd.read_csv(path)
            if frame.empty:
                raise ValueError("Factor targets CSV is empty")
            if frame.shape[1] == 1:
                return _align_targets(frame.iloc[:, 0].to_numpy(), factor_names, len(factor_names))
            if frame.shape[1] >= 2:
                mapping = {
                    str(row[0]).strip(): float(row[1])
                    for row in frame.to_numpy()
                    if len(row) >= 2
                }
                return _align_targets(mapping, factor_names, len(factor_names))
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = list(csv.reader(fh))
        if not reader:
            raise ValueError("Factor targets CSV is empty")
        if len(reader[0]) >= len(factor_names):
            return _align_targets(reader[0][: len(factor_names)], factor_names, len(factor_names))
        mapping: Dict[str, float] = {}
        for row in reader:
            if len(row) >= 2:
                key = row[0].strip()
                if key:
                    mapping[key] = float(row[1])
        if not mapping:
            raise ValueError("Unable to parse factor target CSV contents")
        return _align_targets(mapping, factor_names, len(factor_names))
    if suffix in {".parquet", ".pq"}:
        if pd is None:
            raise RuntimeError("Reading parquet requires pandas to be installed")
        frame = pd.read_parquet(path)
        if frame.empty:
            raise ValueError("Factor targets parquet is empty")
        if frame.shape[1] == 1:
            return _align_targets(frame.iloc[:, 0].to_numpy(), factor_names, len(factor_names))
        mapping = {str(idx): float(val) for idx, val in zip(frame.iloc[:, 0], frame.iloc[:, 1])}
        return _align_targets(mapping, factor_names, len(factor_names))
    raise ValueError(f"Unsupported factor target extension '{path.suffix}'")


def parse_slippage(spec: Optional[str]) -> Optional[SlippageConfig]:
    if spec is None or not spec.strip():
        return None
    text = spec.strip().lower()
    if ":" in text:
        model, param_text = text.split(":", 1)
        model = model.strip()
        param_text = param_text.strip()
    else:
        model = text
        param_text = ""
    defaults = {
        "proportional": 5.0,
        "square": 1.0,
        "vol_scaled": 1.0,
        "impact": 25.0,
    }
    if model not in defaults:
        raise ValueError(f"Unsupported slippage model '{model}'")
    params: Dict[str, float] = {}
    if model == "impact":
        if param_text:
            for chunk in param_text.split(","):
                if not chunk.strip():
                    continue
                if "=" not in chunk:
                    raise ValueError(
                        "Impact slippage parameters must be provided as key=value pairs"
                    )
                key, value = chunk.split("=", 1)
                key = key.strip().lower()
                value = value.strip()
                if key == "k":
                    params["k"] = _parse_bps_value(value)
                elif key == "alpha":
                    params["alpha"] = float(value)
                elif key == "spread":
                    params["spread"] = _parse_fractional_value(value)
                elif key == "participation":
                    params["participation"] = _parse_fractional_value(value)
                else:
                    raise ValueError(f"Unknown impact slippage parameter '{key}'")
        params.setdefault("k", defaults[model])
        params.setdefault("alpha", 1.5)
        params.setdefault("spread", 0.0)
        param_value = params["k"]
    else:
        if not param_text:
            param_value = defaults[model]
        else:
            try:
                param_value = float(param_text)
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid slippage parameter '{param_text}'") from exc
    return SlippageConfig(model=model, param=param_value, params=params)


def _cross_sectional_volatility(block: np.ndarray) -> float:
    if block.size == 0:
        return 0.0
    if block.ndim == 1:
        return float(np.nanstd(block))
    row_std = np.nanstd(block, axis=1)
    if row_std.size == 0:
        return 0.0
    return float(np.nanmean(row_std))


def _ewma_factors(history: np.ndarray) -> np.ndarray:
    if history.shape[0] == 1:
        return history[0]
    span = history.shape[0]
    lam = max(1.0 - 2.0 / (1.0 + span), 0.0)
    current = history[0]
    for row in history[1:]:
        current = lam * current + (1.0 - lam) * row
    return current


def _compute_factor_snapshot(
    panel: FactorPanel,
    dates: Sequence[Any],
    start_idx: int,
    lookback: int,
    index_map: Optional[Dict[Any, int]] = None,
) -> Optional[np.ndarray]:
    if lookback <= 0:
        return None
    mapping = index_map or panel.index_map()
    rebalance_date = dates[start_idx]
    history_indices: List[int] = []
    for offset in range(max(0, start_idx - lookback), start_idx + 1):
        mapped = mapping.get(dates[offset])
        if mapped is not None:
            history_indices.append(mapped)
    if not history_indices:
        return None
    history = panel.loadings[history_indices]
    current_idx = mapping.get(rebalance_date)
    snapshot: Optional[np.ndarray]
    if current_idx is not None:
        snapshot = np.array(panel.loadings[current_idx], dtype=float)
        if np.isnan(snapshot).any():
            snapshot = _ewma_factors(history)
    else:
        snapshot = _ewma_factors(history)
    if snapshot is None:
        return None
    if np.isnan(snapshot).any():
        means = np.nanmean(history, axis=0)
        snapshot = snapshot.copy()
        mask = np.isnan(snapshot)
        snapshot[mask] = means[mask]
    if np.isnan(snapshot).any():
        return None
    return snapshot


def _rankdata(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = x.size
    if n < 3:
        return 0.0, 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx_mean = rx.mean()
    ry_mean = ry.mean()
    cov = float(np.dot(rx - rx_mean, ry - ry_mean))
    denom = float(np.linalg.norm(rx - rx_mean) * np.linalg.norm(ry - ry_mean))
    if denom <= 1e-12:
        return 0.0, 0.0
    corr = float(np.clip(cov / denom, -1.0, 1.0))
    if abs(corr) >= 1.0:
        return corr, 0.0
    t_stat = corr * math.sqrt((n - 2) / max(1e-12, 1.0 - corr**2))
    return corr, float(t_stat)


def _ols_factor_returns(
    exposures: np.ndarray, returns: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(exposures, dtype=float)
    y = np.asarray(returns, dtype=float).reshape(-1)
    if x.ndim != 2 or y.ndim != 1:
        raise ValueError("Invalid exposure or return dimensions for attribution")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Exposure and return length mismatch")
    n_obs, n_fac = x.shape
    if n_obs <= n_fac:
        return np.zeros(n_fac, dtype=float), np.zeros(n_fac, dtype=float)
    design = np.column_stack([np.ones(n_obs, dtype=float), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    residuals = y - fitted
    dof = max(n_obs - design.shape[1], 1)
    sigma2 = float(np.dot(residuals, residuals) / dof)
    xtx = design.T @ design
    xtx_inv = np.linalg.pinv(xtx)
    variances = np.diag(xtx_inv) * sigma2
    std_err = np.sqrt(np.maximum(variances[1:], 0.0))
    factor_returns = beta[1:]
    t_stats = np.divide(
        factor_returns,
        std_err,
        out=np.zeros_like(factor_returns),
        where=std_err > 1e-12,
    )
    return factor_returns.astype(float), t_stats.astype(float)


def _compute_factor_attr_row(
    date: Any,
    exposures: np.ndarray,
    asset_returns: np.ndarray,
    factor_names: Sequence[str],
) -> Optional[Dict[str, Any]]:
    exp_arr = np.asarray(exposures, dtype=float)
    ret_arr = np.asarray(asset_returns, dtype=float)
    if exp_arr.ndim != 2 or ret_arr.ndim != 1:
        return None
    if exp_arr.shape[0] != ret_arr.shape[0]:
        return None
    mask = np.isfinite(ret_arr)
    if exp_arr.size:
        mask = mask & np.all(np.isfinite(exp_arr), axis=1)
    if not np.any(mask):
        return None
    filtered_exposures = exp_arr[mask]
    filtered_returns = ret_arr[mask]
    if filtered_exposures.shape[0] <= filtered_exposures.shape[1]:
        return None
    factor_returns, t_stats = _ols_factor_returns(filtered_exposures, filtered_returns)
    if factor_returns.size != len(factor_names):
        return None
    row: Dict[str, Any] = {"date": _stringify(date)}
    for idx, name in enumerate(factor_names):
        corr, ic_t = _spearman_ic(filtered_exposures[:, idx], filtered_returns)
        row[f"{name}_return"] = float(factor_returns[idx])
        row[f"{name}_return_t"] = float(t_stats[idx])
        row[f"{name}_ic"] = float(corr)
        row[f"{name}_ic_t"] = float(ic_t)
    return row


def _compute_slippage_cost(
    cfg: Optional[SlippageConfig],
    delta: Optional[np.ndarray],
    asset_block: np.ndarray,
    turn: float,
) -> float:
    if cfg is None:
        return 0.0
    if cfg.model == "impact":
        if delta is None:
            return 0.0
        abs_delta = np.abs(np.asarray(delta, dtype=float))
        if not abs_delta.size:
            return 0.0
        k_bps = cfg.params.get("k", cfg.param)
        alpha = cfg.params.get("alpha", 1.5)
        spread = cfg.params.get("spread", 0.0)
        impact_component = (k_bps / 1e4) * float(np.sum(abs_delta ** alpha))
        spread_component = float(spread) * float(abs_delta.sum())
        return impact_component + spread_component
    if turn <= 0:
        return 0.0
    if cfg.model == "proportional":
        return (cfg.param / 1e4) * turn
    if cfg.model == "square":
        return cfg.param * (turn ** 2)
    if cfg.model == "vol_scaled":
        vol = _cross_sectional_volatility(asset_block)
        return cfg.param * turn * vol
    return 0.0


def _apply_no_trade_band(
    target: np.ndarray,
    previous: Optional[np.ndarray],
    band: float,
) -> Tuple[np.ndarray, int]:
    if previous is None or band <= 0.0:
        return target, 0
    prev = np.asarray(previous, dtype=float)
    desired = np.asarray(target, dtype=float)
    if prev.shape != desired.shape:
        raise ValueError("Weight vectors must share the same shape for no-trade band")
    delta = desired - prev
    mask = np.abs(delta) <= band
    hits = int(np.count_nonzero(mask))
    if hits == 0:
        return desired, 0
    adjusted = desired.copy()
    adjusted[mask] = prev[mask]
    if hits == adjusted.size:
        return prev.copy(), hits
    target_sum = float(desired[~mask].sum())
    remaining = float(1.0 - prev[mask].sum())
    if abs(target_sum) > 1e-12:
        scale = remaining / target_sum
        adjusted[~mask] = desired[~mask] * scale
    else:
        adjusted = prev.copy()
    return adjusted, hits


def _apply_participation_cap(
    previous: Optional[np.ndarray],
    target: np.ndarray,
    cap: Optional[float],
) -> Tuple[np.ndarray, int]:
    if previous is None or cap is None or cap <= 0.0:
        return target, 0
    prev = np.asarray(previous, dtype=float)
    desired = np.asarray(target, dtype=float)
    if prev.shape != desired.shape:
        raise ValueError("Weight vectors must share the same shape for participation cap")
    delta = desired - prev
    abs_delta = np.abs(delta)
    breaches = abs_delta > cap
    if not np.any(breaches):
        return desired, 0
    clipped = np.clip(delta, -cap, cap)
    adjusted = prev + clipped
    total = float(adjusted.sum())
    if abs(total) > 1e-12:
        adjusted = adjusted / total
    return adjusted, int(np.count_nonzero(breaches))


def ewma_cov(returns: np.ndarray, span: int = 60) -> np.ndarray:
    """Compute an exponentially weighted covariance matrix."""

    if span <= 1:
        raise ValueError("span must be greater than one for EWMA covariance")

    lam = max(1.0 - 2.0 / (1.0 + span), 0.0)
    demeaned = returns - returns.mean(axis=0, keepdims=True)
    cov = np.zeros((demeaned.shape[1], demeaned.shape[1]), dtype=float)
    for row in demeaned:
        cov = lam * cov + (1.0 - lam) * np.outer(row, row)
    cov = 0.5 * (cov + cov.T)
    return cov


def _sample_cov(returns: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance (rowvar=False)."""

    if returns.size == 0:
        return np.zeros((0, 0), dtype=float)
    cov = np.cov(returns, rowvar=False)
    cov = 0.5 * (cov + cov.T)
    return cov


def _lw_cov(returns: np.ndarray) -> np.ndarray:
    """Ledoit–Wolf shrinkage toward identity (σ^2 I)."""

    X = np.asarray(returns, dtype=float)
    T, N = X.shape
    if T <= 1:
        return np.eye(N, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / (T - 1)
    mu = np.trace(S) / N
    F = mu * np.eye(N, dtype=float)
    X2 = Xc ** 2
    phi_mat = (X2.T @ X2) / (T - 1) - S ** 2
    pi_hat = np.sum(phi_mat)
    gamma_hat = np.linalg.norm(S - F, ord="fro") ** 2
    rho_hat = pi_hat
    kappa = max(0.0, min(1.0, rho_hat / max(gamma_hat, 1e-18)))
    Sigma = (1 - kappa) * S + kappa * F
    return 0.5 * (Sigma + Sigma.T)


def _oas_cov(returns: np.ndarray) -> np.ndarray:
    """Oracle Approximating Shrinkage toward scaled identity."""

    X = np.asarray(returns, dtype=float)
    T, N = X.shape
    if T <= 1:
        return np.eye(N, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / (T - 1)
    mu = np.trace(S) / N
    tr_S2 = np.sum(S * S)
    num = (1 - 2 / N) * tr_S2 + (np.trace(S) ** 2)
    den = (T + 1 - 2 / N) * (tr_S2 - (np.trace(S) ** 2) / N)
    alpha = 0.0 if den <= 0 else min(1.0, max(0.0, num / (den + 1e-18)))
    F = mu * np.eye(N, dtype=float)
    Sigma = (1 - alpha) * S + alpha * F
    return 0.5 * (Sigma + Sigma.T)


def _ridge_cov(returns: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    X = np.asarray(returns, dtype=float)
    if X.size == 0:
        return np.zeros((0, 0), dtype=float)
    cov = _sample_cov(X)
    if cov.size == 0:
        return cov
    lam = float(alpha)
    if lam < 0.0:
        lam = 0.0
    n = cov.shape[0]
    ridge = cov + lam * np.eye(n, dtype=float)
    return 0.5 * (ridge + ridge.T)


def _soft_threshold(value: float, lam: float) -> float:
    if value > lam:
        return value - lam
    if value < -lam:
        return value + lam
    return 0.0


def _lasso_coordinate_descent(
    gram: np.ndarray,
    target: np.ndarray,
    alpha: float,
    *,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> np.ndarray:
    size = gram.shape[0]
    if size == 0:
        return np.zeros(0, dtype=float)
    diag = np.diag(gram).copy()
    diag = np.where(diag <= 0, 1e-12, diag)
    beta = np.zeros(size, dtype=float)
    lam = float(max(alpha, 0.0))
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(size):
            residual = target[j] - np.dot(gram[j], beta) + gram[j, j] * beta[j]
            beta[j] = _soft_threshold(residual, lam) / max(gram[j, j], 1e-12)
        if np.max(np.abs(beta - beta_old)) <= tol:
            break
    return beta


def _graphical_lasso(emp_cov: np.ndarray, alpha: float, *, max_iter: int, tol: float) -> np.ndarray:
    S = np.asarray(emp_cov, dtype=float)
    n_features = S.shape[0]
    if n_features == 0:
        return np.zeros((0, 0), dtype=float)
    covariance = S.copy()
    precision = np.linalg.pinv(covariance + alpha * np.eye(n_features, dtype=float))
    indices = np.arange(n_features)
    lam = float(max(alpha, 0.0))
    for _ in range(max(1, int(max_iter))):
        precision_prev = precision.copy()
        for idx in range(n_features):
            mask = indices != idx
            cov_11 = covariance[np.ix_(mask, mask)]
            if cov_11.size == 0:
                covariance[idx, idx] = max(float(S[idx, idx]), 1e-12)
                precision[idx, idx] = 1.0 / max(covariance[idx, idx], 1e-12)
                continue
            s12 = S[mask, idx]
            beta = _lasso_coordinate_descent(
                cov_11,
                s12,
                lam,
                tol=max(tol, 1e-8),
                max_iter=max_iter,
            )
            cov12 = cov_11 @ beta
            covariance[idx, mask] = cov12
            covariance[mask, idx] = cov12
            w_ii = max(float(S[idx, idx]), 1e-12)
            theta_ii_denom = max(w_ii - float(np.dot(beta, cov12)), 1e-12)
            theta_ii = 1.0 / theta_ii_denom
            precision[idx, idx] = theta_ii
            precision[mask, idx] = precision[idx, mask] = -theta_ii * beta
        if np.max(np.abs(precision - precision_prev)) <= tol:
            break
    covariance = 0.5 * (covariance + covariance.T)
    covariance[np.diag_indices_from(covariance)] = np.maximum(
        covariance.diagonal(), 1e-10
    )
    return nearest_psd(covariance)


def _glasso_cov(
    returns: np.ndarray,
    *,
    alpha: float = 0.01,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> np.ndarray:
    X = np.asarray(returns, dtype=float)
    if X.size == 0:
        return np.zeros((0, 0), dtype=float)
    emp_cov = _sample_cov(X)
    lam = float(alpha)
    if lam <= 0.0:
        return nearest_psd(emp_cov)
    return _graphical_lasso(emp_cov, lam, max_iter=max_iter, tol=tol)


def _bayesian_cov(
    returns: np.ndarray,
    *,
    nu: Optional[float] = None,
    prior_scale: Optional[float] = None,
) -> np.ndarray:
    X = np.asarray(returns, dtype=float)
    T, N = X.shape
    if N == 0:
        return np.zeros((0, 0), dtype=float)
    if T <= 1:
        scale = float(prior_scale) if prior_scale is not None else 1.0
        return scale * np.eye(N, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    sample = (Xc.T @ Xc) / (T - 1)
    nu_value = float(nu) if nu is not None else float(N + 2)
    nu_value = max(nu_value, 0.0)
    if prior_scale is None:
        prior = np.diag(np.diag(sample))
    else:
        prior = float(prior_scale) * np.eye(N, dtype=float)
    total = float(T + nu_value)
    if total <= 0:
        return nearest_psd(sample)
    shrunk = (T / total) * sample + (nu_value / total) * prior
    return 0.5 * (shrunk + shrunk.T)


register_cov_model("sample", lambda returns, **_: _sample_cov(np.asarray(returns, dtype=float)))
register_cov_model("lw", lambda returns, **_: _lw_cov(np.asarray(returns, dtype=float)))
register_cov_model("oas", lambda returns, **_: _oas_cov(np.asarray(returns, dtype=float)))
register_cov_model(
    "ridge",
    lambda returns, **kw: _ridge_cov(
        np.asarray(returns, dtype=float), alpha=float(kw.get("alpha", 0.05))
    ),
)
register_cov_model(
    "glasso",
    lambda returns, **kw: _glasso_cov(
        np.asarray(returns, dtype=float),
        alpha=float(kw.get("alpha", 0.01)),
        max_iter=int(kw.get("max_iter", 100)),
        tol=float(kw.get("tol", 1e-4)),
    ),
)
register_cov_model(
    "bayesian",
    lambda returns, **kw: _bayesian_cov(
        np.asarray(returns, dtype=float),
        nu=kw.get("nu"),
        prior_scale=kw.get("prior_scale"),
    ),
)


def _ewma_adapter(returns: np.ndarray, *, span: Optional[int] = None, **_: Any) -> np.ndarray:
    span_value = 60 if span is None else int(span)
    return ewma_cov(np.asarray(returns, dtype=float), span=span_value)


register_cov_model("ewma", _ewma_adapter)


def _hrp_quasi_diag(linkage_matrix: np.ndarray) -> List[int]:
    link = np.asarray(linkage_matrix, dtype=float)
    if link.size == 0:
        return []
    n = link.shape[0] + 1
    order = [int(link[-1, 0]), int(link[-1, 1])]
    while any(idx >= n for idx in order):
        new_order: List[int] = []
        for idx in order:
            if idx < n:
                new_order.append(int(idx))
            else:
                child = int(idx - n)
                new_order.append(int(link[child, 0]))
                new_order.append(int(link[child, 1]))
        order = new_order
    return [int(i) for i in order]


def _hrp_cluster_variance(cov: np.ndarray, indices: Sequence[int]) -> float:
    if not indices:
        return 0.0
    sub = cov[np.ix_(indices, indices)]
    diag = np.diag(sub)
    diag = np.where(diag <= 0, 1e-12, diag)
    inv_diag = 1.0 / diag
    weights = inv_diag / inv_diag.sum()
    variance = float(weights @ sub @ weights)
    return max(variance, 0.0)


def _hierarchical_risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.array([1.0], dtype=float)
    diag = np.diag(cov)
    denom = np.sqrt(np.outer(diag, diag))
    corr = np.divide(cov, denom, out=np.ones_like(cov), where=denom > 0)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr)))
    condensed = squareform(dist, checks=False)
    if condensed.size == 0:
        return np.ones(n, dtype=float) / n
    link = linkage(condensed, method="single")
    order = _hrp_quasi_diag(link)
    if not order:
        return np.ones(n, dtype=float) / n
    weights = np.ones(len(order), dtype=float)
    stack: List[List[int]] = [list(range(len(order)))]
    while stack:
        cluster = stack.pop()
        if len(cluster) <= 1:
            continue
        split = len(cluster) // 2
        left = cluster[:split]
        right = cluster[split:]
        left_idx = [order[i] for i in left]
        right_idx = [order[i] for i in right]
        var_left = _hrp_cluster_variance(cov, left_idx)
        var_right = _hrp_cluster_variance(cov, right_idx)
        total = var_left + var_right
        alloc_left = 0.5 if total <= 0 else 1.0 - var_left / total
        for idx in left:
            weights[idx] *= alloc_left
        for idx in right:
            weights[idx] *= 1.0 - alloc_left
        stack.append(right)
        stack.append(left)
    final = np.zeros(n, dtype=float)
    for position, asset_idx in enumerate(order):
        final[asset_idx] = weights[position]
    total_weight = final.sum()
    if total_weight <= 0:
        return np.ones(n, dtype=float) / n
    return final / total_weight

if njit is not None:  # pragma: no cover - exercised in environments with numba
    @njit(cache=True)
    def _abs_assign_nb(out: np.ndarray, src: np.ndarray) -> None:
        for i in range(src.shape[0]):
            out[i] = abs(src[i])

    @njit(cache=True)
    def _diff_abs_assign_nb(out: np.ndarray, prev: np.ndarray, curr: np.ndarray) -> None:
        for i in range(curr.shape[0]):
            out[i] = abs(curr[i] - prev[i])
else:  # pragma: no cover - fallback
    def _abs_assign_nb(out: np.ndarray, src: np.ndarray) -> None:
        out[:] = np.abs(src)

    def _diff_abs_assign_nb(out: np.ndarray, prev: np.ndarray, curr: np.ndarray) -> None:
        np.subtract(curr, prev, out=out)
        np.abs(out, out=out)


def _turnover_penalty_components(
    previous: Optional[np.ndarray],
    current: np.ndarray,
    gamma: Optional[np.ndarray] = None,
    *,
    out: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    current_arr = np.asarray(current, dtype=float)
    buffer = out
    if buffer is None or buffer.shape != current_arr.shape:
        buffer = np.empty_like(current_arr, dtype=float)
    else:
        buffer = np.asarray(buffer, dtype=float)
    if previous is None:
        _abs_assign_nb(buffer, current_arr)
    else:
        prev_arr = np.asarray(previous, dtype=float)
        if prev_arr.shape != current_arr.shape:
            raise ValueError("previous weights dimension mismatch")
        _diff_abs_assign_nb(buffer, prev_arr, current_arr)
    turnover_val = float(buffer.sum())
    penalty_val = 0.0
    if gamma is not None:
        gamma_arr = np.asarray(gamma, dtype=float)
        if gamma_arr.shape != buffer.shape:
            raise ValueError("turnover penalty vector dimension mismatch")
        penalty_val = float(buffer @ gamma_arr)
    return turnover_val, penalty_val, buffer


def turnover(
    previous: Optional[np.ndarray],
    current: np.ndarray,
    *,
    out: Optional[np.ndarray] = None,
) -> float:
    """Compute the L1 turnover between two weight vectors."""

    turnover_val, _, _ = _turnover_penalty_components(previous, current, out=out)
    return float(turnover_val)


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Return the maximum drawdown for the supplied equity curve."""

    equity = np.asarray(equity_curve, dtype=float)
    if equity.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(equity)
    drawdown = 1.0 - np.divide(
        equity,
        running_peak,
        out=np.ones_like(equity),
        where=running_peak > 0,
    )
    return float(np.max(drawdown))


def _pain_index(equity_curve: np.ndarray) -> float:
    equity = np.asarray(equity_curve, dtype=float)
    if equity.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(equity)
    drawdown = 1.0 - np.divide(
        equity,
        running_peak,
        out=np.ones_like(equity),
        where=running_peak > 0,
    )
    return float(drawdown.mean())


def compute_drawdown_events(equity: np.ndarray, dates: Sequence[Any]) -> List[Dict[str, Any]]:
    curve = np.asarray(equity, dtype=float)
    if curve.size == 0:
        return []
    if len(dates) != curve.size:
        raise ValueError("Dates length must match equity length")
    running_peak = np.maximum.accumulate(curve)
    drawdown = 1.0 - np.divide(
        curve,
        running_peak,
        out=np.ones_like(curve),
        where=running_peak > 0,
    )
    events: List[Dict[str, Any]] = []
    in_drawdown = False
    peak_idx = 0
    trough_idx = 0
    trough_depth = 0.0
    peak_value = running_peak[0]
    for idx, depth in enumerate(drawdown):
        if depth > 1e-12:
            if not in_drawdown:
                in_drawdown = True
                peak_value = running_peak[idx]
                peak_idx = idx
                for back in range(idx, -1, -1):
                    if abs(curve[back] - peak_value) <= max(1e-12, 1e-9 * abs(peak_value)):
                        peak_idx = back
                        break
                trough_idx = idx
                trough_depth = float(depth)
            elif depth > trough_depth + 1e-12:
                trough_depth = float(depth)
                trough_idx = idx
        elif in_drawdown:
            recovery_idx = idx
            peak = dates[peak_idx]
            trough = dates[trough_idx]
            recovery = dates[recovery_idx]
            peak_val = running_peak[peak_idx]
            trough_val = curve[trough_idx]
            depth_val = 0.0 if peak_val <= 0 else float(1.0 - (trough_val / peak_val))
            length = recovery_idx - peak_idx
            events.append(
                {
                    "peak": peak,
                    "trough": trough,
                    "recovery": recovery,
                    "depth": depth_val,
                    "length": int(length),
                }
            )
            in_drawdown = False
            trough_depth = 0.0
    if in_drawdown:
        peak = dates[peak_idx]
        trough = dates[trough_idx]
        peak_val = running_peak[peak_idx]
        trough_val = curve[trough_idx]
        depth_val = 0.0 if peak_val <= 0 else float(1.0 - (trough_val / peak_val))
        length = len(curve) - peak_idx
        events.append(
            {
                "peak": peak,
                "trough": trough,
                "recovery": None,
                "depth": depth_val,
                "length": int(length),
            }
        )
    return events


def _build_optimizer(
    n_assets: int, seed: int, risk_free_rate: float = 0.0
) -> NeuroAntPortfolioOptimizer:
    cfg = OptimizerConfig(
        n_ants=24,
        max_iter=25,
        topk_refine=6,
        topk_train=6,
        use_shrinkage=False,
        shrinkage_delta=0.15,
        cvar_alpha=0.05,
        seed=seed,
        risk_free=risk_free_rate,
    )
    return NeuroAntPortfolioOptimizer(n_assets, cfg)


def _build_constraints(n_assets: int) -> PortfolioConstraints:
    return PortfolioConstraints(
        min_weight=0.0,
        max_weight=1.0,
        equality_enforce=True,
        leverage_limit=1.0,
        sector_map=None,
        max_sector_concentration=1.0,
        prev_weights=None,
        max_turnover=1.0,
    )


def _frame_to_numpy(frame: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    np_dtype = np.dtype(dtype) if dtype is not None else np.float64
    if hasattr(frame, "to_numpy"):
        return frame.to_numpy(dtype=np_dtype)  # type: ignore[no-any-return]
    return np.asarray(frame, dtype=np_dtype)


def _frame_index(frame: Any, length: int) -> List[Any]:
    if hasattr(frame, "index"):
        return list(frame.index)  # type: ignore[attr-defined]
    return list(range(length))


def _normalize_index(
    dates: Sequence[Any], *, drop_duplicates: bool, label: str
) -> Tuple[List[Any], np.ndarray]:
    values = list(dates)
    keep_mask = np.ones(len(values), dtype=bool)
    if not values:
        return [], keep_mask

    counts = Counter(values)
    duplicates = [val for val, count in counts.items() if count > 1]
    if duplicates:
        if not drop_duplicates:
            offenders = ", ".join(_stringify(val) for val in duplicates[:5])
            suffix = "" if len(duplicates) <= 5 else ", ..."
            raise ValueError(
                f"{label} contains duplicate dates ({offenders}{suffix}). "
                "Pass --drop-duplicates to keep the last occurrence."
            )
        last_seen: Dict[Any, int] = {}
        for idx, val in enumerate(values):
            prev = last_seen.get(val)
            if prev is not None:
                keep_mask[prev] = False
            last_seen[val] = idx
        values = [val for idx, val in enumerate(values) if keep_mask[idx]]

    arr = np.asarray(values)
    if arr.size > 1:
        try:
            non_increasing = np.asarray(arr[1:] <= arr[:-1], dtype=bool)
        except TypeError:
            non_increasing = np.array(
                [values[i + 1] <= values[i] for i in range(len(values) - 1)],
                dtype=bool,
            )
        if np.any(non_increasing):
            bad_idx = int(np.where(non_increasing)[0][0])
            before = _stringify(values[bad_idx])
            after = _stringify(values[bad_idx + 1])
            raise ValueError(
                f"{label} must be strictly increasing (saw {after} <= {before})."
            )

    return values, keep_mask


def _sanitize_frame(
    frame: Any,
    *,
    drop_duplicates: bool,
    label: str,
) -> Tuple[Any, np.ndarray, List[Any]]:
    arr = _frame_to_numpy(frame)
    arr = np.atleast_2d(arr)
    dates = _frame_index(frame, arr.shape[0])
    normalized, keep_mask = _normalize_index(
        dates, drop_duplicates=drop_duplicates, label=label
    )
    if keep_mask.size and not np.all(keep_mask):
        if hasattr(frame, "iloc"):
            frame = frame.iloc[np.nonzero(keep_mask)[0]]
        elif hasattr(frame, "filter_rows"):
            frame = frame.filter_rows(keep_mask)
        arr = arr[keep_mask]
    return frame, arr, normalized


def validate_factor_panel(
    panel: FactorPanel,
    returns_frame: Any,
    *,
    align: str = "strict",
) -> Tuple[FactorPanel, FactorDiagnostics]:
    mode = str(align).lower()
    if mode not in {"strict", "subset"}:
        raise ValueError("align must be either 'strict' or 'subset'")

    loadings = np.asarray(panel.loadings, dtype=float)
    if loadings.ndim != 3:
        raise ValueError("Factor loadings must be a 3D array (T, N, K)")
    if len(panel.dates) != loadings.shape[0]:
        raise ValueError("Factor date axis mismatch")
    if len(panel.assets) != loadings.shape[1]:
        raise ValueError("Factor asset axis mismatch")
    if len(panel.factor_names) != loadings.shape[2]:
        raise ValueError("Factor name axis mismatch")
    if len(panel.factor_names) == 0:
        raise ValueError("Factor panel must contain at least one factor")
    if len(set(panel.factor_names)) != len(panel.factor_names):
        raise ValueError("Factor names must be unique")
    if loadings.size and not np.isfinite(loadings).all():
        raise ValueError("Factor loadings must not contain NaNs or infs")

    returns_arr = _frame_to_numpy(returns_frame)
    returns_arr = np.atleast_2d(returns_arr)
    if returns_arr.ndim != 2:
        raise ValueError("Returns frame must coerce to a 2D array")

    if hasattr(returns_frame, "columns") and getattr(returns_frame, "columns") is not None:
        raw_assets = [str(col) for col in returns_frame.columns]
    else:
        raw_assets = [f"A{i}" for i in range(returns_arr.shape[1])]

    asset_map = {asset: idx for idx, asset in enumerate(panel.assets)}
    keep_asset_indices: List[int] = []
    aligned_assets: List[str] = []
    dropped_assets: List[str] = []
    for asset in raw_assets:
        idx = asset_map.get(asset)
        if idx is None:
            dropped_assets.append(asset)
            continue
        keep_asset_indices.append(idx)
        aligned_assets.append(asset)

    if not keep_asset_indices:
        raise ValueError("No overlapping assets between returns and factor panel")

    aligned_loadings = loadings[:, keep_asset_indices, :]

    return_dates = _frame_index(returns_frame, returns_arr.shape[0])
    return_date_set = set(return_dates)
    keep_date_indices = [idx for idx, date in enumerate(panel.dates) if date in return_date_set]
    if not keep_date_indices:
        raise ValueError("Factor panel does not overlap with returns dates")

    aligned_dates = [panel.dates[idx] for idx in keep_date_indices]
    aligned_loadings = aligned_loadings[keep_date_indices, :, :]

    dropped_dates = [panel.dates[idx] for idx in range(len(panel.dates)) if idx not in keep_date_indices]
    panel_date_set = set(panel.dates)
    missing_rebalance_dates = [date for date in return_dates if date not in panel_date_set]
    if mode == "strict" and missing_rebalance_dates:
        raise ValueError("Factor panel is missing required rebalance dates")

    diagnostics = FactorDiagnostics(
        align_mode=mode,
        total_assets=len(raw_assets),
        total_dates=len(return_dates),
        dropped_assets=dropped_assets,
        dropped_dates=dropped_dates,
        missing_rebalance_dates=missing_rebalance_dates if mode == "subset" else [],
    )

    aligned_panel = FactorPanel(aligned_dates, aligned_assets, aligned_loadings, list(panel.factor_names))
    return aligned_panel, diagnostics


def validate_factor_targets(targets: np.ndarray, factor_names: Sequence[str]) -> np.ndarray:
    arr = np.asarray(targets, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    arr = arr.reshape(-1)
    if arr.size != len(factor_names):
        raise ValueError("Factor target length does not match factor names")
    if not np.isfinite(arr).all():
        raise ValueError("Factor targets must not contain NaNs or infs")
    return arr


_OBJECTIVE_MAP: Dict[str, OptimizationObjective] = {
    "sharpe": OptimizationObjective.SHARPE_RATIO,
    "max_return": OptimizationObjective.MAX_RETURN,
    "min_variance": OptimizationObjective.MIN_VARIANCE,
    "risk_parity": OptimizationObjective.RISK_PARITY,
    "hrp": OptimizationObjective.HRP,
    "min_cvar": OptimizationObjective.MIN_CVAR,
    "tracking_error": OptimizationObjective.TRACKING_ERROR_MIN,
    "min_tracking_error": OptimizationObjective.TRACKING_ERROR_MIN,
    "info_ratio": OptimizationObjective.INFO_RATIO_MAX,
    "te_target": OptimizationObjective.TRACKING_ERROR_TARGET,
    "multi_term": OptimizationObjective.MULTI_TERM,
}

def _import_callable(target: str, kind: str) -> Callable[..., Any]:
    if ":" not in target:
        raise ValueError(
            f"Custom {kind} must be provided as '<module>:<callable>'"
        )
    module_name, attr_name = target.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ValueError(f"Failed to import {kind} '{target}': {exc}") from exc
    try:
        fn = getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(
            f"Module '{module_name}' does not define {kind} '{attr_name}'"
        ) from exc
    if not callable(fn):
        raise ValueError(f"Resolved {kind} '{target}' is not callable")
    return fn


def _resolve_objective(name: str) -> ObjectiveLike:
    raw = str(name).strip()
    if not raw:
        raise ValueError("Objective must be a non-empty string")
    lower = raw.lower()
    if lower in _OBJECTIVE_MAP:
        return _OBJECTIVE_MAP[lower]
    if lower in _CUSTOM_OBJECTIVE_REGISTRY:
        return _CUSTOM_OBJECTIVE_REGISTRY[lower]
    if raw.startswith("custom:"):
        parts = raw.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "Custom objective must be specified as 'custom:<module>:<callable>'"
            )
        target = f"{parts[1]}:{parts[2]}"
        return _import_callable(target, "custom objective")
    raise ValueError(
        f"Unknown objective '{name}' (available: {sorted(_OBJECTIVE_MAP)} + {get_registered_objectives()})"
    )


def _resolve_cov_model(name: str) -> CovarianceModelSpec:
    raw = str(name).strip()
    if not raw:
        raise ValueError("cov_model must be a non-empty string")
    if raw.startswith("custom:"):
        parts = raw.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "Custom cov_model must be specified as 'custom:<module>:<callable>'"
            )
        target = f"{parts[1]}:{parts[2]}"
        fn = _import_callable(target, "covariance model")
        return CovarianceModelSpec(fn=fn, params={}, base=raw, label=raw, is_custom=True)
    base, params = _parse_cov_model_spec(raw)
    lower = base.strip().lower()
    if lower in _COVARIANCE_REGISTRY:
        fn = _COVARIANCE_REGISTRY[lower]
        label = raw if params else lower
        return CovarianceModelSpec(
            fn=fn,
            params=params,
            base=lower,
            label=label,
            is_custom=False,
        )
    raise ValueError(
        f"Unknown cov_model '{name}' (available: {get_available_cov_models()})"
    )


def backtest(
    df: Any,
    lookback: int = 252,
    step: int = 21,
    ewma_span: Optional[int] = None,
    objective: str = "sharpe",
    seed: int = 7,
    tx_cost_bps: float = 0.0,
    tx_cost_mode: str = "none",
    metric_alpha: float = 0.05,
    factors: Optional[FactorPanel] = None,
    factor_targets: Optional[np.ndarray] = None,
    factor_tolerance: float = 1e-6,
    factor_align: str = "strict",
    factors_required: bool = False,
    slippage: Optional[SlippageConfig] = None,
    nt_band: float = 0.0,
    refine_every: int = 1,
    cov_model: str = "sample",
    benchmark: Optional[Any] = None,
    benchmark_weights: Optional[Any] = None,
    active_min: Optional[Any] = None,
    active_max: Optional[Any] = None,
    active_group_caps: Optional[Any] = None,
    factor_bounds: Optional[Any] = None,
    te_target: float = 0.0,
    lambda_te: float = 0.0,
    gamma_turnover: float = 0.0,
    gamma_by_sector: Optional[Any] = None,
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
    progress_callback: Optional[ProgressCallback] = None,
    rebalance_callback: Optional[RebalanceLogCallback] = None,
    dtype: np.dtype = np.float64,
    cov_cache_size: int = 8,
    max_workers: Optional[int] = None,
    warm_start: Optional[Any] = None,
    warm_align: str = "last_row",
    decay: float = 0.0,
    drop_duplicates: bool = False,
    deterministic: bool = False,
    compute_factor_attr: bool = False,
) -> Dict[str, Any]:
    """Run a rolling-window backtest on a return dataframe."""

    if lookback <= 0 or step <= 0:
        raise ValueError("lookback and step must be positive integers")
    if refine_every <= 0:
        raise ValueError("refine_every must be positive")
    objective_spec = _resolve_objective(objective)
    cov_model_spec = str(cov_model)
    cov_spec = _resolve_cov_model(cov_model_spec)
    cov_callable = cov_spec.fn
    cov_params = dict(cov_spec.params)
    cov_is_custom = cov_spec.is_custom
    cov_model_lower = cov_spec.base
    cov_model_label = cov_spec.label
    if cov_model_lower != "ewma":
        ewma_span = None
    else:
        span_override = cov_params.pop("span", None)
        if span_override is not None:
            try:
                ewma_span = int(span_override)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError("ewma span parameter must be an integer") from exc
        elif ewma_span is None:
            ewma_span = 60
        try:
            ewma_span = int(ewma_span)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("ewma_span must be an integer") from exc
        if ewma_span < 2:
            raise ValueError("ewma_span must be >= 2")
    cov_cache_base = cov_model_spec if cov_is_custom else cov_model_lower
    cov_param_items = tuple(sorted(cov_params.items()))

    min_active_val = _coerce_optional_float(active_min)
    max_active_val = _coerce_optional_float(active_max)
    min_active_val, max_active_val = _normalize_active_bounds(
        min_active_val, max_active_val
    )
    constraint_manifest: Dict[str, Any] = {
        "active_bounds": {
            "min": _manifest_bound(min_active_val),
            "max": _manifest_bound(max_active_val),
        }
    }
    warm_warnings: List[str] = []

    if trading_days <= 0:
        raise ValueError("trading_days must be positive")

    annual_rf = float(risk_free_rate)
    if annual_rf <= -1.0:
        raise ValueError("risk_free_rate must be greater than -100%")
    periodic_rf = float((1.0 + annual_rf) ** (1.0 / trading_days) - 1.0)

    warm_align_mode = str(warm_align or "last_row").lower()
    if warm_align_mode not in {"by_date", "last_row"}:
        raise ValueError("warm_align must be either 'by_date' or 'last_row'")
    decay_value = float(decay)
    if decay_value < 0.0 or decay_value > 1.0:
        raise ValueError("decay must be between 0 and 1")
    nt_band_value = float(nt_band)
    if nt_band_value < 0.0:
        raise ValueError("nt_band must be non-negative")

    float_dtype = np.dtype(dtype)
    if float_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        float_dtype = np.dtype(np.float64)
    max_workers_value = None if max_workers is None else int(max_workers)
    if max_workers_value is not None and max_workers_value <= 0:
        raise ValueError("max_workers must be positive when provided")

    df, returns_arr, dates = _sanitize_frame(
        df, drop_duplicates=drop_duplicates, label="returns"
    )
    returns = returns_arr.astype(float_dtype)
    if returns.size == 0:
        raise ValueError("input dataframe must contain returns")

    set_seed(seed, deterministic_torch=True, strict=deterministic)
    n_periods, n_assets = returns.shape
    benchmark_series: Optional[np.ndarray] = None
    if benchmark is not None:
        benchmark, bench_values_raw, bench_dates = _sanitize_frame(
            benchmark, drop_duplicates=drop_duplicates, label="benchmark"
        )
        bench_values = np.asarray(bench_values_raw, dtype=float_dtype)
        if bench_values.ndim > 2:
            raise ValueError("Benchmark series must be one or two dimensional")
        if bench_values.ndim == 2:
            if bench_values.shape[1] == 0:
                raise ValueError("Benchmark series contains no columns")
            if bench_values.shape[1] > 1:
                raise ValueError("Benchmark series must contain exactly one column")
            bench_values = bench_values[:, 0]
        benchmark_series = np.asarray(bench_values, dtype=float_dtype).reshape(-1)
        if benchmark_series.size != n_periods:
            raise ValueError("Benchmark length must match returns length")
        if list(bench_dates) != list(dates):
            raise ValueError(
                "Benchmark index must match returns index after duplicate handling"
            )

    objective_needs_benchmark = (
        isinstance(objective_spec, OptimizationObjective)
        and objective_spec
        in {
            OptimizationObjective.TRACKING_ERROR_MIN,
            OptimizationObjective.INFO_RATIO_MAX,
            OptimizationObjective.TRACKING_ERROR_TARGET,
            OptimizationObjective.MULTI_TERM,
        }
    )
    if objective_needs_benchmark and benchmark_series is None:
        raise ValueError(
            "Benchmark series required for tracking_error/info_ratio/te_target/multi_term objectives"
        )

    asset_names = _extract_asset_names(df, n_assets)
    inferred_sector_map, inferred_sector_names = _infer_sector_map(df, asset_names)
    sector_lookup = (
        {name: inferred_sector_map[idx] for idx, name in enumerate(asset_names)}
        if inferred_sector_map is not None
        else None
    )

    factor_panel: Optional[FactorPanel] = None
    factor_target_vec: Optional[np.ndarray] = None
    factor_names: List[str] = []
    factor_diagnostics: Optional[FactorDiagnostics] = None
    factor_missing_dates: Set[Any] = set()
    if factors is not None:
        factor_panel, factor_diagnostics = validate_factor_panel(
            factors, df, align=factor_align
        )
        factor_missing_dates = set(factor_diagnostics.missing_rebalance_dates)
        if factor_panel.loadings.dtype != float_dtype:
            factor_panel = FactorPanel(
                factor_panel.dates,
                factor_panel.assets,
                factor_panel.loadings.astype(float_dtype, copy=False),
                list(factor_panel.factor_names),
            )
        if factor_diagnostics.dropped_assets:
            print(
                "Warning: dropping assets without factor data: "
                + ", ".join(sorted(factor_diagnostics.dropped_assets))
            )
        asset_lookup = {name: idx for idx, name in enumerate(asset_names)}
        reorder_indices = [asset_lookup[name] for name in factor_panel.assets if name in asset_lookup]
        if len(reorder_indices) != len(factor_panel.assets):
            raise ValueError("Factor assets failed to align with returns columns")
        returns = returns[:, reorder_indices]
        asset_names = list(factor_panel.assets)
        if sector_lookup is not None:
            remapped = [sector_lookup.get(name, -1) for name in asset_names]
            if any(val != -1 for val in remapped):
                inferred_sector_map = remapped
            else:
                inferred_sector_map = None
        n_periods, n_assets = returns.shape
        if factor_panel.loadings.shape[1] != n_assets:
            raise ValueError("Factor panel asset dimension mismatch after alignment")
        factor_names = list(factor_panel.factor_names)
        if factor_targets is not None:
            factor_target_vec = validate_factor_targets(factor_targets, factor_names)
            factor_target_vec = factor_target_vec.astype(float_dtype, copy=False)
        else:
            factor_target_vec = np.zeros(len(factor_names), dtype=float_dtype)
        if factors_required and factor_missing_dates:
            sorted_missing = sorted(
                factor_diagnostics.missing_rebalance_dates, key=_sort_key
            )
            missing_str = ", ".join(_stringify(val) for val in sorted_missing)
            raise ValueError(f"Missing factor data for rebalance dates: {missing_str}")

    if returns.shape[1] == 0:
        raise ValueError("No assets remain after aligning factors with returns")

    turnover_gamma_vec, effective_sector_map, sector_name_map, gamma_manifest = (
        _prepare_turnover_penalties(
            gamma_by_sector,
            asset_names,
            inferred_sector_map,
            inferred_sector_names,
        )
    )
    if gamma_manifest:
        constraint_manifest["sector_penalties"] = gamma_manifest
    elif gamma_by_sector is not None:
        constraint_manifest["sector_penalties"] = []

    warm_path_obj: Optional[Path] = None
    warm_dates: List[Any] = []
    warm_matrix: np.ndarray = np.zeros((0, 0), dtype=float)
    warm_names: List[str] = []
    if warm_start is not None:
        warm_path_obj = Path(str(warm_start))
        if not warm_path_obj.exists():
            raise FileNotFoundError(f"Warm-start weights file '{warm_path_obj}' not found")
        warm_dates, warm_matrix, warm_names = _read_weights_csv(warm_path_obj)
        if warm_matrix.ndim == 1:
            warm_matrix = warm_matrix.reshape(1, -1)

    bench_vector, bench_mask, bench_manifest = _prepare_benchmark_weights(
        benchmark_weights, asset_names
    )
    if bench_vector is not None:
        bench_vector = bench_vector.astype(float_dtype, copy=False)
    if bench_mask is not None:
        bench_mask = bench_mask.astype(bool, copy=False)
    constraint_manifest["benchmark_weights"] = bench_manifest
    group_map, group_bounds, group_manifest = _prepare_active_groups(
        active_group_caps, asset_names
    )
    constraint_manifest["active_group_caps"] = group_manifest
    factor_lower_arr, factor_upper_arr, factor_bounds_manifest = _prepare_factor_bounds(
        factor_bounds, factor_names
    )
    constraint_manifest["factor_bounds"] = factor_bounds_manifest

    optimizer = _build_optimizer(returns.shape[1], seed, risk_free_rate=periodic_rf)
    if max_workers_value is not None:
        setattr(optimizer.cfg, "max_workers", max_workers_value)
    if te_target < 0.0:
        raise ValueError("te_target must be non-negative")
    if lambda_te < 0.0:
        raise ValueError("lambda_te must be non-negative")
    if gamma_turnover < 0.0:
        raise ValueError("gamma_turnover must be non-negative")
    optimizer.cfg.te_target = float(te_target)
    optimizer.cfg.lambda_te = float(lambda_te)
    optimizer.cfg.gamma_turnover = float(gamma_turnover)
    if turnover_gamma_vec is not None:
        optimizer.cfg.gamma_turnover_vector = turnover_gamma_vec.astype(float, copy=False)
    else:
        optimizer.cfg.gamma_turnover_vector = None
    constraints = _build_constraints(returns.shape[1])
    constraints.factor_tolerance = factor_tolerance
    constraints.min_active_weight = float("-inf") if min_active_val is None else float(min_active_val)
    constraints.max_active_weight = float("inf") if max_active_val is None else float(max_active_val)
    if bench_vector is not None:
        constraints.benchmark_weights = bench_vector
        constraints.benchmark_mask = bench_mask
    else:
        constraints.benchmark_weights = None
        constraints.benchmark_mask = None
    if effective_sector_map is not None:
        constraints.sector_map = list(effective_sector_map)
    if sector_name_map:
        constraints.sector_name_map = dict(sector_name_map)
    else:
        constraints.sector_name_map = None
    if turnover_gamma_vec is not None:
        constraints.turnover_gamma = turnover_gamma_vec.astype(float_dtype, copy=False)
    else:
        constraints.turnover_gamma = None
    turnover_gamma_array = (
        np.asarray(constraints.turnover_gamma, dtype=float)
        if constraints.turnover_gamma is not None
        else None
    )
    if group_map is not None and group_bounds is not None:
        constraints.active_group_map = group_map
        constraints.active_group_bounds = group_bounds
    else:
        constraints.active_group_map = None
        constraints.active_group_bounds = None
    if factor_lower_arr is not None:
        factor_lower_arr = factor_lower_arr.astype(float_dtype, copy=False)
        constraints.factor_lower_bounds = factor_lower_arr
    if factor_upper_arr is not None:
        factor_upper_arr = factor_upper_arr.astype(float_dtype, copy=False)
        constraints.factor_upper_bounds = factor_upper_arr

    weights: List[np.ndarray] = []
    rebalance_dates: List[Any] = []
    realized_dates: List[Any] = []
    feasible_flags: List[bool] = []
    projection_iters: List[int] = []
    prev_weights: Optional[np.ndarray] = None
    prev_weights_is_warm = False
    warm_applied_count = 0
    tc = float(tx_cost_bps) / 1e4
    missing_factor_logged: Set[Any] = set()
    factor_records: List[Dict[str, Any]] = []
    factor_attr_records: List[Dict[str, Any]] = []
    rebalance_records: List[Dict[str, Any]] = []
    walk_windows: List[Tuple[int, int]] = []

    rebalance_points = list(range(lookback, n_periods, step))
    n_windows = len(rebalance_points)
    total_test_periods = max(n_periods - lookback, 0)
    gross_returns_arr = np.empty(total_test_periods, dtype=float_dtype)
    realized_returns_arr = np.empty(total_test_periods, dtype=float_dtype)
    net_tx_returns_arr = np.empty(total_test_periods, dtype=float_dtype)
    net_slip_returns_arr = np.empty(total_test_periods, dtype=float_dtype)
    benchmark_realized_arr = (
        np.empty(total_test_periods, dtype=float_dtype) if benchmark_series is not None else None
    )
    slippage_costs = np.zeros(n_windows, dtype=float_dtype)
    turnovers_arr = np.zeros(n_windows, dtype=float_dtype)
    sector_penalties_arr = np.zeros(n_windows, dtype=float_dtype)
    cursor = 0

    factor_index_map: Optional[Dict[Any, int]] = None
    if factor_panel is not None:
        factor_index_map = factor_panel.index_map()

    turnover_pre_buffer = np.empty(n_assets, dtype=float)
    turnover_buffer = np.empty(n_assets, dtype=float)

    warm_vector: Optional[np.ndarray] = None
    if warm_matrix.size and len(asset_names):
        row_idx = warm_matrix.shape[0] - 1
        matched = False
        if warm_align_mode == "by_date" and warm_dates:
            if rebalance_points:
                target_date = dates[rebalance_points[0]]
                target_str = _stringify(target_date)
                for idx, raw_date in enumerate(warm_dates):
                    if _stringify(raw_date) == target_str:
                        row_idx = idx
                        matched = True
                        break
            if not matched and warm_matrix.shape[0] > 0 and rebalance_points:
                if "warm_date_fallback" not in warm_warnings:
                    warm_warnings.append("warm_date_fallback")
        candidate = np.asarray(warm_matrix[row_idx], dtype=float)
        aligned = _align_warm_to_assets(warm_names, candidate, asset_names)
        if float(np.abs(aligned).sum()) <= 1e-12:
            if "warm_no_overlap" not in warm_warnings:
                warm_warnings.append("warm_no_overlap")
        else:
            warm_vector = aligned.astype(float_dtype, copy=False)
    if warm_vector is not None:
        prev_weights = warm_vector
        prev_weights_is_warm = True

    max_cov_cache = max(0, int(cov_cache_size))

    if not rebalance_points:
        if progress_callback is not None:
            progress_callback(0, 0)
        empty = np.array([], dtype=float_dtype)
        benchmark_returns = empty if benchmark_series is not None else None
        warn_list = ["no_rebalances"]
        for item in warm_warnings:
            if item not in warn_list:
                warn_list.append(item)
        return {
            "dates": [],
            "returns": empty,
            "gross_returns": empty,
            "net_tx_returns": empty,
            "net_slip_returns": empty,
            "equity": empty,
            "weights": np.empty((0, n_assets), dtype=float_dtype),
            "rebalance_dates": [],
            "asset_names": asset_names,
            "cov_model": cov_model_label,
            "benchmark_returns": benchmark_returns,
            "sharpe": 0.0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
            "max_drawdown": 0.0,
            "avg_turnover": 0.0,
            "downside_vol": 0.0,
            "sortino": 0.0,
            "realized_cvar": 0.0,
            "tracking_error": None,
            "info_ratio": None,
            "te_target": float(optimizer.cfg.te_target),
            "lambda_te": float(optimizer.cfg.lambda_te),
            "gamma_turnover": float(optimizer.cfg.gamma_turnover),
            "risk_free_rate": annual_rf,
            "periodic_risk_free": periodic_rf,
            "trading_days": int(trading_days),
            "factor_records": [],
            "factor_names": factor_names,
            "factor_tolerance": factor_tolerance,
            "avg_slippage_bps": 0.0,
            "slippage_costs": empty,
            "slippage_net_returns": None,
            "rebalance_records": [],
            "rebalance_feasible": [],
            "projection_iterations": [],
            "factor_diagnostics": factor_diagnostics.to_dict() if factor_diagnostics else None,
            "constraint_manifest": constraint_manifest,
            "warnings": warn_list,
            "cov_cache_size": int(max_cov_cache),
            "cov_cache_hits": 0,
            "cov_cache_misses": 0,
            "cov_cache_evictions": 0,
            "max_workers": max_workers_value,
            "dtype": float_dtype.name,
            "decay": decay_value,
            "warm_start": str(warm_path_obj) if warm_path_obj is not None else None,
            "warm_align": warm_align_mode,
            "warm_applied_count": 0,
        }

    # include model + params in the cache key to avoid collisions across models/spans
    cov_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
    cov_cache_hits = 0
    cov_cache_misses = 0
    cov_cache_evictions = 0
    block_contributions: List[Dict[str, Any]] = []

    if progress_callback is not None:
        progress_callback(0, n_windows)

    for window_idx, start in enumerate(rebalance_points):
        end = min(start + step, n_periods)
        train = returns[start - lookback : start]
        test = returns[start:end]
        walk_windows.append((int(start), int(end)))
        mu = train.mean(axis=0)
        bench_stats: Optional[BenchmarkStats] = None
        if benchmark_series is not None:
            bench_train = benchmark_series[start - lookback : start]
            if bench_train.shape[0] != train.shape[0]:
                raise ValueError("Benchmark lookback does not match returns lookback")
            bench_mean = float(bench_train.mean())
            centered_b = bench_train - bench_mean
            centered_assets = train - mu
            denom = max(1, centered_b.shape[0] - 1)
            cov_vector = centered_assets.T @ centered_b / denom
            variance = float(np.dot(centered_b, centered_b) / denom)
            bench_stats = BenchmarkStats(
                mean=bench_mean,
                variance=max(variance, 0.0),
                cov_vector=cov_vector,
            )
        span = ewma_span if cov_model_lower == "ewma" and ewma_span is not None else None
        window_hash = _hash_returns_window(train)
        cov_key: Optional[Tuple[Any, ...]] = (
            cov_cache_base,
            cov_param_items,
            span,
            window_hash,
        )
        cov_elapsed_ms = 0.0
        cov: Optional[np.ndarray] = None
        cached_cov = (
            cov_cache.get(cov_key) if (cov_key is not None and max_cov_cache > 0) else None
        )
        if cached_cov is not None:
            cov_start = time.perf_counter()
            cov = cached_cov.copy()
            cov_elapsed_ms = (time.perf_counter() - cov_start) * 1000.0
            cov_cache_hits += 1
        if cov is None:
            cov_start = time.perf_counter()
            cov_kwargs: Dict[str, Any] = dict(cov_params)
            if span is not None:
                cov_kwargs["span"] = span
            cov_raw = np.asarray(cov_callable(train, **cov_kwargs), dtype=float)
            if optimizer.cfg.use_shrinkage:
                cov_raw = shrink_covariance(cov_raw, delta=optimizer.cfg.shrinkage_delta)
            cov = nearest_psd(cov_raw)
            cov_elapsed_ms = (time.perf_counter() - cov_start) * 1000.0
            if cov_key is not None and max_cov_cache > 0:
                cov_cache[cov_key] = cov.copy()
                while len(cov_cache) > max_cov_cache:
                    cov_cache.popitem(last=False)
                    cov_cache_evictions += 1
            cov_cache_misses += 1

        rebalance_date = dates[start]
        factors_missing = False
        current_factor_snapshot: Optional[np.ndarray] = None
        if factor_panel is not None:
            if rebalance_date in factor_missing_dates:
                factors_missing = True
            else:
                snapshot = _compute_factor_snapshot(
                    factor_panel, dates, start, lookback, index_map=factor_index_map
                )
                if snapshot is None:
                    factors_missing = True
                else:
                    snapshot = snapshot.astype(float_dtype, copy=False)
                    constraints.factor_loadings = snapshot
                    constraints.factor_targets = factor_target_vec
                    current_factor_snapshot = snapshot
        if factor_panel is None or factors_missing:
            constraints.factor_loadings = None
            constraints.factor_targets = None
            if factor_panel is not None:
                if rebalance_date not in missing_factor_logged:
                    print(
                        f"Skipping factor neutrality on {rebalance_date} (missing factor data)"
                    )
                    missing_factor_logged.add(rebalance_date)
                if factor_diagnostics is not None:
                    factor_diagnostics.record_missing(rebalance_date)
                factor_missing_dates.add(rebalance_date)
                if factors_required:
                    raise ValueError(
                        f"Missing factor data for rebalance date {rebalance_date}"
                    )

        constraints.prev_weights = prev_weights
        should_refine = (len(weights) % refine_every) == 0
        opt_start = time.perf_counter()
        if objective_spec == OptimizationObjective.HRP:
            hrp_weights = _hierarchical_risk_parity_weights(cov)
            result = SimpleNamespace(
                weights=hrp_weights,
                feasible=True,
                projection_iterations=0,
            )
        else:
            result = optimizer.optimize(
                mu,
                cov,
                constraints,
                objective=objective_spec,
                refine=should_refine,
                benchmark=bench_stats,
            )
        opt_elapsed_ms = (time.perf_counter() - opt_start) * 1000.0
        feasible_flags.append(bool(getattr(result, "feasible", True)))
        projection_iters.append(int(getattr(result, "projection_iterations", 0)))
        w_opt = np.asarray(result.weights, dtype=float_dtype)
        warm_applied_flag = prev_weights_is_warm
        turn_pre_val, _, _ = _turnover_penalty_components(
            prev_weights, w_opt, out=turnover_pre_buffer
        )
        turn_pre = float(turn_pre_val)
        w_target = w_opt
        nt_band_hits = 0
        if nt_band_value > 0.0:
            w_target, nt_band_hits = _apply_no_trade_band(w_target, prev_weights, nt_band_value)
            w_target = np.asarray(w_target, dtype=float_dtype)
        w = w_target
        if prev_weights is not None and decay_value > 0.0:
            w = (1.0 - decay_value) * w_target + decay_value * prev_weights
        participation_cap = None
        if slippage is not None:
            participation_cap = slippage.params.get("participation")
        w, participation_breaches = _apply_participation_cap(prev_weights, w, participation_cap)
        w = np.asarray(w, dtype=float_dtype)
        turn_val, penalty_val, _ = _turnover_penalty_components(
            prev_weights,
            w,
            gamma=turnover_gamma_array,
            out=turnover_buffer,
        )
        turn = float(turn_val)
        sector_penalty_val = float(penalty_val)
        weights.append(w)
        rebalance_dates.append(rebalance_date)

        gross_block_returns = np.asarray(test @ w, dtype=float_dtype).reshape(-1)
        block_len = gross_block_returns.size
        idx_slice = slice(cursor, cursor + block_len)
        gross_returns_arr[idx_slice] = gross_block_returns
        length = max(1, block_len)
        contrib_vals = np.zeros_like(w, dtype=float)
        if test.size:
            contrib_vals = np.asarray(test, dtype=float) * np.asarray(w, dtype=float)
            contrib_vals = contrib_vals.sum(axis=0)
        block_total = float(gross_block_returns.sum())
        for asset_name, contrib_val in zip(asset_names, contrib_vals):
            block_contributions.append(
                {
                    "date": rebalance_date,
                    "asset": asset_name,
                    "contribution": float(contrib_val),
                    "block_return": block_total,
                }
            )

        if compute_factor_attr and current_factor_snapshot is not None:
            block_asset_returns = np.prod(1.0 + np.asarray(test, dtype=float), axis=0) - 1.0
            attr_row = _compute_factor_attr_row(
                rebalance_date,
                current_factor_snapshot,
                block_asset_returns,
                factor_names,
            )
            if attr_row is not None:
                factor_attr_records.append(attr_row)

        turnovers_arr[window_idx] = float(turn)
        sector_penalties_arr[window_idx] = float(sector_penalty_val)
        turnover_violation = False
        if np.isfinite(constraints.max_turnover):
            if float(turn) > float(constraints.max_turnover) + 1e-9:
                turnover_violation = True
        tx_cost_value = tc * turn if (tc > 0.0 and tx_cost_mode != "none") else 0.0
        tx_block_returns = gross_block_returns.copy()
        block_returns_metrics = gross_block_returns.copy()
        if tx_cost_value > 0.0:
            if tx_cost_mode == "upfront":
                if block_returns_metrics.size > 0:
                    block_returns_metrics = block_returns_metrics.copy()
                    block_returns_metrics[0] -= tx_cost_value
                tx_block_returns = block_returns_metrics.copy()
            elif tx_cost_mode == "amortized":
                adjust = tx_cost_value / length
                block_returns_metrics = block_returns_metrics - adjust
                tx_block_returns = block_returns_metrics.copy()
            elif tx_cost_mode == "posthoc":
                tx_block_returns = tx_block_returns - (tx_cost_value / length)

        trade_delta = w if prev_weights is None else w - prev_weights
        slip_cost = _compute_slippage_cost(slippage, trade_delta, test, turn)
        slippage_costs[window_idx] = float(slip_cost)
        slip_block_returns = tx_block_returns.copy()
        if slippage is not None and slip_cost > 0.0:
            if tx_cost_mode in ("upfront", "amortized"):
                if tx_cost_mode == "upfront" and block_returns_metrics.size > 0:
                    block_returns_metrics = block_returns_metrics.copy()
                    block_returns_metrics[0] -= slip_cost
                else:
                    block_returns_metrics = block_returns_metrics - (slip_cost / length)
            if tx_cost_mode == "upfront" and slip_block_returns.size > 0:
                slip_block_returns = slip_block_returns.copy()
                slip_block_returns[0] -= slip_cost
            else:
                slip_block_returns = slip_block_returns - (slip_cost / length)

        net_tx_returns_arr[idx_slice] = tx_block_returns
        net_slip_returns_arr[idx_slice] = slip_block_returns

        block_returns = block_returns_metrics
        realized_returns_arr[idx_slice] = block_returns
        realized_dates.extend(dates[start:end])
        if benchmark_series is not None and benchmark_realized_arr is not None:
            benchmark_realized_arr[idx_slice] = benchmark_series[start:end]
        prev_weights = w
        if warm_applied_flag:
            warm_applied_count += 1
        prev_weights_is_warm = False

        sector_breaches = 0
        sector_exposures: Dict[str, float] = {}
        if constraints.sector_map is not None:
            sectors = np.asarray(constraints.sector_map)
            if sectors.shape[0] != w.shape[0]:
                raise ValueError("Sector map dimension mismatch with weights")
            if sectors.size:
                sectors_int = sectors.astype(int)
                offset = int(sectors_int.min())
                if offset < 0:
                    shifted = sectors_int - offset
                else:
                    offset = 0
                    shifted = sectors_int
                counts = np.bincount(shifted, weights=w, minlength=int(shifted.max()) + 1)
                unique_shifted = np.unique(shifted)
                exposures_values = counts[unique_shifted]
                labels = unique_shifted + offset
                sector_name_lookup = getattr(constraints, "sector_name_map", None) or {}
                sector_exposures = {}
                for raw_label, value in zip(labels, exposures_values):
                    label_int = int(raw_label)
                    if label_int < 0:
                        continue
                    display = sector_name_lookup.get(label_int)
                    key = f"sector_{display}" if display else f"sector_{label_int}"
                    sector_exposures[key] = float(value)
                cap = float(constraints.max_sector_concentration)
                sector_breaches = int(np.count_nonzero(exposures_values > cap + 1e-9))
        active_breaches = 0
        group_breaches = 0
        factor_bound_breaches = 0
        asset_active_violation = False
        group_violation = False
        factor_violation = False
        bench_weights = constraints.benchmark_weights
        bench_mask_arr = None
        bench_arr = np.zeros_like(w)
        active = w.copy()
        tol_active = 1e-9
        if bench_weights is not None:
            bench_arr = np.asarray(bench_weights, dtype=float_dtype).ravel()
            if bench_arr.shape != w.shape:
                raise ValueError("benchmark_weights dimension mismatch with weights")
            if constraints.benchmark_mask is not None:
                bench_mask_arr = np.asarray(constraints.benchmark_mask, dtype=bool).ravel()
                if bench_mask_arr.shape != bench_arr.shape:
                    raise ValueError("benchmark_mask dimension mismatch with weights")
            else:
                bench_mask_arr = np.ones_like(bench_arr, dtype=bool)
            active = w - bench_arr
            if np.isfinite(constraints.min_active_weight):
                violations = np.logical_and(
                    bench_mask_arr,
                    active < float(constraints.min_active_weight) - tol_active,
                )
                if np.any(violations):
                    active_breaches += int(np.count_nonzero(violations))
                    asset_active_violation = True
            if np.isfinite(constraints.max_active_weight):
                violations = np.logical_and(
                    bench_mask_arr,
                    active > float(constraints.max_active_weight) + tol_active,
                )
                if np.any(violations):
                    active_breaches += int(np.count_nonzero(violations))
                    asset_active_violation = True
        if (
            constraints.active_group_map is not None
            and constraints.active_group_bounds
        ):
            groups = np.asarray(constraints.active_group_map, dtype=int)
            if groups.shape[0] != w.shape[0]:
                raise ValueError("active_group_map dimension mismatch with weights")
            if bench_mask_arr is None:
                bench_mask_arr = np.ones_like(w, dtype=bool)
            for gid, bound in constraints.active_group_bounds.items():
                mask = np.logical_and(groups == gid, bench_mask_arr)
                if not np.any(mask):
                    continue
                active_sum = float(active[mask].sum())
                lower_b, upper_b = bound
                violated = False
                if np.isfinite(upper_b) and active_sum > float(upper_b) + tol_active:
                    group_breaches += 1
                    violated = True
                if np.isfinite(lower_b) and active_sum < float(lower_b) - tol_active:
                    group_breaches += 1
                    violated = True
                if violated:
                    group_violation = True
        factor_inf_norm = 0.0
        if factor_panel is not None:
            if factors_missing:
                factor_records.append(
                    {
                        "date": rebalance_date,
                        "exposures": None,
                        "targets": None,
                        "tolerance": factor_tolerance,
                        "sector_exposures": sector_exposures if sector_exposures else None,
                        "missing": True,
                    }
                )
            elif current_factor_snapshot is not None and factor_target_vec is not None:
                exposures = current_factor_snapshot.T @ w
                targets = factor_target_vec
                factor_inf_norm = float(np.linalg.norm(exposures - targets, ord=np.inf))
                tol_bounds = max(1e-9, float(constraints.factor_tolerance))
                if factor_lower_arr is not None:
                    lower_mask = np.isfinite(factor_lower_arr)
                    below = exposures < factor_lower_arr - tol_bounds
                    if np.any(np.logical_and(lower_mask, below)):
                        count = int(np.count_nonzero(np.logical_and(lower_mask, below)))
                        factor_bound_breaches += count
                        factor_violation = True
                if factor_upper_arr is not None:
                    upper_mask = np.isfinite(factor_upper_arr)
                    above = exposures > factor_upper_arr + tol_bounds
                    if np.any(np.logical_and(upper_mask, above)):
                        count = int(np.count_nonzero(np.logical_and(upper_mask, above)))
                        factor_bound_breaches += count
                        factor_violation = True
                factor_records.append(
                    {
                        "date": rebalance_date,
                        "exposures": exposures,
                        "targets": targets,
                        "tolerance": factor_tolerance,
                        "sector_exposures": sector_exposures if sector_exposures else None,
                        "missing": False,
                    }
                )

        first_violation: Optional[str] = None
        if asset_active_violation:
            first_violation = "ACTIVE_BOX"
        elif group_violation:
            first_violation = "GROUP_CAP"
        elif factor_violation:
            first_violation = "FACTOR_BOUND"
        elif sector_breaches > 0:
            first_violation = "SECTOR_CAP"
        elif turnover_violation:
            first_violation = "TURNOVER"

        block_info_ratio: Optional[float] = None
        block_tracking_error: Optional[float] = None
        if benchmark_series is not None:
            active_block = block_returns - benchmark_series[start:end]
            if active_block.size:
                block_tracking_error = compute_tracking_error(active_block, trading_days)
                active_mean = float(active_block.mean() * trading_days)
                if block_tracking_error <= 1e-12:
                    if abs(active_mean) <= 1e-12:
                        block_info_ratio = 0.0
                    else:
                        block_info_ratio = float(math.copysign(1e6, active_mean))
                else:
                    block_info_ratio = float(active_mean / block_tracking_error)

        excess_block = block_returns - periodic_rf
        block_std = float(np.std(excess_block)) if excess_block.size else 0.0
        ann_factor = math.sqrt(trading_days)
        block_mean_excess = float(excess_block.mean()) if excess_block.size else 0.0
        block_sharpe = 0.0
        if block_std > 1e-12:
            block_sharpe = float((block_mean_excess * trading_days) / (block_std * ann_factor))
        block_sortino = 0.0
        downside = excess_block[excess_block < 0]
        if downside.size:
            downside_vol = float(downside.std() * ann_factor)
            if downside_vol > 1e-12:
                block_sortino = float((block_mean_excess * trading_days) / downside_vol)

        turn_pre_val = float(turn_pre)
        turn_post_val = float(turn)

        rebalance_records.append(
            {
                "date": rebalance_date,
                "gross_ret": float(np.prod(1.0 + gross_block_returns) - 1.0),
                "net_tx_ret": float(np.prod(1.0 + tx_block_returns) - 1.0),
                "net_slip_ret": float(np.prod(1.0 + slip_block_returns) - 1.0),
                "turnover": turn_post_val,
                "turnover_pre_decay": turn_pre_val,
                "turnover_post_decay": turn_post_val,
                "tx_cost": float(tx_cost_value),
                "slippage_cost": float(slip_cost),
                "nt_band_hits": int(nt_band_hits),
                "participation_breaches": int(participation_breaches),
                "sector_breaches": int(sector_breaches),
                "sector_penalty": float(sector_penalty_val),
                "active_breaches": int(active_breaches),
                "group_breaches": int(group_breaches),
                "factor_bound_breaches": int(factor_bound_breaches),
                "factor_inf_norm": float(factor_inf_norm),
                "factor_missing": bool(factors_missing),
                "first_violation": first_violation,
                "feasible": bool(feasible_flags[-1]),
                "projection_iterations": int(projection_iters[-1]),
                "block_sharpe": block_sharpe,
                "block_sortino": block_sortino,
                "block_info_ratio": block_info_ratio,
                "block_tracking_error": block_tracking_error,
                "warm_applied": bool(warm_applied_flag),
                "decay": float(decay_value),
            }
        )
        if rebalance_callback is not None:
            log_record: Dict[str, Any] = {
                "date": _stringify(rebalance_date),
                "seed": int(seed),
                "objective": str(objective),
                "cov_model": cov_model_label,
                "costs": {
                    "tx": float(tx_cost_value),
                    "slippage": float(slip_cost),
                },
                "turnover": float(turn),
                "turnover_pre_decay": float(turn_pre),
                "turnover_post_decay": float(turn),
                "nt_band_hits": int(nt_band_hits),
                "participation_breaches": int(participation_breaches),
                "feasible": bool(feasible_flags[-1]),
                "breaches": {
                    "active": int(active_breaches),
                    "group": int(group_breaches),
                    "factor": int(factor_bound_breaches),
                    "sector": int(sector_breaches),
                },
                "sector_penalty": float(sector_penalty_val),
                "block": {
                    "sharpe": float(block_sharpe),
                    "sortino": float(block_sortino),
                    "ir": None if block_info_ratio is None else float(block_info_ratio),
                    "te": None
                    if block_tracking_error is None
                    else float(block_tracking_error),
                },
                "timings": {
                    "cov_ms": float(cov_elapsed_ms),
                    "opt_ms": float(opt_elapsed_ms),
                },
                "warm_applied": bool(warm_applied_flag),
                "decay": float(decay_value),
            }
            rebalance_callback(log_record)
        if progress_callback is not None:
            progress_callback(window_idx + 1, n_windows)
        cursor += block_len

    gross_returns_arr = gross_returns_arr[:cursor]
    net_tx_returns_arr = net_tx_returns_arr[:cursor]
    net_slip_returns_arr = net_slip_returns_arr[:cursor]
    realized_returns_arr = realized_returns_arr[:cursor]
    if benchmark_realized_arr is not None:
        benchmark_returns_arr = benchmark_realized_arr[:cursor]
    else:
        benchmark_returns_arr = np.array([])
    equity = np.cumprod(1.0 + realized_returns_arr, dtype=float_dtype)
    drawdown_events = compute_drawdown_events(equity, realized_dates)
    slippage_costs_arr = slippage_costs[: len(weights)] if n_windows else np.array([])
    avg_slippage_bps = (
        float(slippage_costs_arr.mean() * 1e4) if slippage_costs_arr.size else 0.0
    )
    turnovers_used = turnovers_arr[: len(weights)] if n_windows else np.array([])
    sector_penalties_used = (
        sector_penalties_arr[: len(weights)] if n_windows else np.array([])
    )
    slippage_net_returns: Optional[np.ndarray] = None
    if slippage is not None:
        slippage_net_returns = net_slip_returns_arr.copy()

    ann_factor = math.sqrt(trading_days)
    ann_vol = (
        float(np.std(realized_returns_arr) * ann_factor)
        if realized_returns_arr.size
        else 0.0
    )
    ann_return = (
        float(np.mean(realized_returns_arr) * trading_days)
        if realized_returns_arr.size
        else 0.0
    )
    excess_returns = realized_returns_arr - periodic_rf
    excess_mean = (
        float(excess_returns.mean() * trading_days)
        if excess_returns.size
        else 0.0
    )
    sharpe = 0.0
    if ann_vol > 1e-12:
        sharpe = float(excess_mean / ann_vol)
    negatives = excess_returns[excess_returns < 0]
    downside_vol = (
        float(negatives.std() * ann_factor)
        if negatives.size
        else 0.0
    )
    sortino = float(excess_mean / downside_vol) if downside_vol > 1e-12 else 0.0
    alpha = float(np.clip(metric_alpha, 1e-4, 0.5))
    if realized_returns_arr.size:
        tail_len = max(1, int(math.floor(alpha * realized_returns_arr.size)))
        tail = np.sort(realized_returns_arr)[:tail_len]
        realized_cvar = float(-tail.mean()) if tail.size else 0.0
    else:
        realized_cvar = 0.0
    mdd = max_drawdown(equity)
    avg_turn = float(turnovers_used.mean()) if turnovers_used.size else 0.0
    avg_sector_penalty = (
        float(sector_penalties_used.mean()) if sector_penalties_used.size else 0.0
    )
    pain_idx = _pain_index(equity)
    calmar_ratio = float(ann_return / mdd) if mdd > 1e-12 else 0.0
    pain_ratio = float(ann_return / pain_idx) if pain_idx > 1e-12 else 0.0
    turnover_adj_sharpe = (
        float(sharpe / (1.0 + avg_turn)) if (1.0 + avg_turn) > 1e-12 else 0.0
    )
    if realized_returns_arr.size:
        hit_rate = float(np.count_nonzero(realized_returns_arr > 0) / realized_returns_arr.size)
    else:
        hit_rate = 0.0

    tracking_error = None
    info_ratio = None
    if benchmark_returns_arr.size == realized_returns_arr.size and realized_returns_arr.size:
        active = realized_returns_arr - benchmark_returns_arr
        te = compute_tracking_error(active, trading_days)
        tracking_error = te
        active_mean = float(active.mean() * trading_days)
        if te <= 1e-12:
            info_ratio = (
                0.0
                if abs(active_mean) <= 1e-12
                else float(math.copysign(1e6, active_mean))
            )
        else:
            info_ratio = float(active_mean / te)

    warnings_list = list(dict.fromkeys(warm_warnings))

    return {
        "dates": realized_dates,
        "returns": realized_returns_arr,
        "gross_returns": gross_returns_arr,
        "net_tx_returns": net_tx_returns_arr,
        "net_slip_returns": net_slip_returns_arr,
        "equity": equity,
        "weights": np.asarray(weights, dtype=float_dtype),
        "rebalance_dates": rebalance_dates,
        "asset_names": asset_names,
        "walk_windows": [(int(start), int(end)) for start, end in walk_windows],
        "cov_model": cov_model_label,
        "benchmark_returns": benchmark_returns_arr if benchmark_returns_arr.size else None,
        "sharpe": sharpe,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_drawdown": mdd,
        "avg_turnover": avg_turn,
        "avg_sector_penalty": avg_sector_penalty,
        "downside_vol": downside_vol,
        "sortino": sortino,
        "realized_cvar": realized_cvar,
        "tracking_error": tracking_error,
        "info_ratio": info_ratio,
        "turnover_adj_sharpe": turnover_adj_sharpe,
        "sector_penalties": sector_penalties_used,
        "calmar_ratio": calmar_ratio,
        "pain_ratio": pain_ratio,
        "hit_rate": hit_rate,
        "te_target": float(optimizer.cfg.te_target),
        "lambda_te": float(optimizer.cfg.lambda_te),
        "gamma_turnover": float(optimizer.cfg.gamma_turnover),
        "gamma_turnover_vector": turnover_gamma_vec,
        "risk_free_rate": annual_rf,
        "periodic_risk_free": periodic_rf,
        "trading_days": int(trading_days),
        "factor_records": factor_records,
        "factor_names": factor_names,
        "factor_tolerance": factor_tolerance,
        "avg_slippage_bps": avg_slippage_bps,
        "slippage_costs": slippage_costs_arr,
        "slippage_net_returns": slippage_net_returns,
        "rebalance_records": rebalance_records,
        "rebalance_feasible": feasible_flags,
        "projection_iterations": projection_iters,
        "drawdowns": drawdown_events,
        "contributions": block_contributions,
        "factor_attr": factor_attr_records,
        "factor_diagnostics": factor_diagnostics.to_dict() if factor_diagnostics else None,
        "constraint_manifest": constraint_manifest,
        "sector_name_map": sector_name_map,
        "warnings": warnings_list,
        "cov_cache_size": int(max_cov_cache),
        "cov_cache_hits": int(cov_cache_hits),
        "cov_cache_misses": int(cov_cache_misses),
        "cov_cache_evictions": int(cov_cache_evictions),
        "max_workers": max_workers_value,
        "dtype": float_dtype.name,
        "decay": decay_value,
        "warm_start": str(warm_path_obj) if warm_path_obj is not None else None,
        "warm_align": warm_align_mode,
        "warm_applied_count": int(warm_applied_count),
    }


def _write_metrics(metrics_path: Path, results: Dict[str, Any]) -> None:
    with metrics_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for key in _STANDARD_METRIC_KEYS:
            writer.writerow([key, results.get(key)])

        for key in _BASELINE_METRIC_KEYS:
            if key in results:
                writer.writerow([key, results.get(key)])


def _window_turnover(
    start: int, end: int, windows: Sequence[Tuple[int, int]], turnovers: Sequence[float]
) -> float:
    if not windows or not turnovers:
        return 0.0
    values: List[float] = []
    limit = min(len(windows), len(turnovers))
    for idx in range(limit):
        win_start, win_end = windows[idx]
        if win_end <= start or win_start >= end:
            continue
        values.append(float(turnovers[idx]))
    if not values:
        return 0.0
    return float(np.mean(values))


def _compute_rolling_metrics_rows(
    returns: np.ndarray,
    dates: Sequence[Any],
    periodic_rf: float,
    trading_days: int,
    window: int,
    walk_windows: Sequence[Tuple[int, int]],
    turnovers: Sequence[float],
) -> List[Dict[str, Any]]:
    returns_arr = np.asarray(returns, dtype=float)
    if window <= 1 or returns_arr.size < window:
        return []
    ann_factor = math.sqrt(trading_days)
    rows: List[Dict[str, Any]] = []
    for end_idx in range(window, returns_arr.size + 1):
        start_idx = end_idx - window
        window_returns = returns_arr[start_idx:end_idx]
        equity = np.cumprod(1.0 + window_returns, dtype=float)
        ann_return = float(window_returns.mean() * trading_days)
        ann_vol = float(np.std(window_returns) * ann_factor)
        excess = window_returns - periodic_rf
        excess_mean = float(excess.mean() * trading_days)
        sharpe = float(excess_mean / ann_vol) if ann_vol > 1e-12 else 0.0
        negatives = excess[excess < 0]
        downside_vol = float(negatives.std() * ann_factor) if negatives.size else 0.0
        sortino = float(excess_mean / downside_vol) if downside_vol > 1e-12 else 0.0
        mdd = max_drawdown(equity)
        pain_idx = _pain_index(equity)
        calmar_ratio = float(ann_return / mdd) if mdd > 1e-12 else 0.0
        pain_ratio = float(ann_return / pain_idx) if pain_idx > 1e-12 else 0.0
        hit_rate = (
            float(np.count_nonzero(window_returns > 0) / window_returns.size)
            if window_returns.size
            else 0.0
        )
        avg_turnover = _window_turnover(start_idx, end_idx, walk_windows, turnovers)
        turnover_adj_sharpe = (
            float(sharpe / (1.0 + avg_turnover))
            if (1.0 + avg_turnover) > 1e-12
            else 0.0
        )
        row = {
            "start": _stringify(dates[start_idx]) if dates else start_idx,
            "end": _stringify(dates[end_idx - 1]) if dates else end_idx - 1,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "turnover_adj_sharpe": turnover_adj_sharpe,
            "calmar_ratio": calmar_ratio,
            "pain_ratio": pain_ratio,
            "hit_rate": hit_rate,
            "max_drawdown": mdd,
        }
        rows.append(row)
    return rows


def _write_rolling_metrics(
    path: Path, results: Dict[str, Any], window: int
) -> None:
    returns = results.get("returns")
    dates = results.get("dates")
    if not isinstance(returns, np.ndarray) or returns.size == 0:
        return
    if not isinstance(window, int) or window <= 1:
        return
    periodic_rf = float(results.get("periodic_risk_free", 0.0))
    trading_days = int(results.get("trading_days", 252))
    walk_windows: Sequence[Tuple[int, int]] = results.get("walk_windows", [])  # type: ignore[assignment]
    rebalance_records: Sequence[Mapping[str, Any]] = results.get("rebalance_records", [])  # type: ignore[assignment]
    turnovers = [float(record.get("turnover", 0.0)) for record in rebalance_records]
    rows = _compute_rolling_metrics_rows(
        returns,
        dates or [],
        periodic_rf,
        trading_days,
        window,
        walk_windows,
        turnovers,
    )
    if not rows:
        return
    header = [
        "start",
        "end",
        "ann_return",
        "ann_vol",
        "sharpe",
        "sortino",
        "turnover_adj_sharpe",
        "calmar_ratio",
        "pain_ratio",
        "hit_rate",
        "max_drawdown",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _slice_frame(frame: Any, indices: Sequence[int], *, reset_index: bool) -> Any:
    if frame is None:
        return None
    index_list = [int(idx) for idx in indices]
    if hasattr(frame, "iloc"):
        subset = frame.iloc[index_list].copy()
        if reset_index and pd is not None and hasattr(subset, "index"):
            subset.index = pd.RangeIndex(len(subset))
        return subset
    arr = _frame_to_numpy(frame)
    arr = np.asarray(arr)
    return arr[index_list]


def _slice_factor_panel(
    panel: Optional[FactorPanel],
    indices: Sequence[int],
    *,
    reset_index: bool,
) -> Optional[FactorPanel]:
    if panel is None:
        return None
    idx_list = [int(idx) for idx in indices]
    if not idx_list:
        empty = np.zeros((0, panel.loadings.shape[1], panel.loadings.shape[2]), dtype=panel.loadings.dtype)
        dates: List[Any] = []
        return FactorPanel(dates, list(panel.assets), empty, list(panel.factor_names))
    selected = panel.loadings[idx_list, :, :]
    if reset_index:
        dates = list(range(len(idx_list)))
    else:
        dates = [panel.dates[i] for i in idx_list]
    return FactorPanel(dates, list(panel.assets), selected, list(panel.factor_names))


def _stationary_bootstrap_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=int)
    block = max(1, int(block_length))
    p = 1.0 / float(block)
    indices: List[int] = []
    while len(indices) < n:
        start = int(rng.integers(0, n))
        block_len = int(rng.geometric(p))
        for offset in range(block_len):
            indices.append((start + offset) % n)
            if len(indices) >= n:
                break
    return np.asarray(indices[:n], dtype=int)


def _circular_bootstrap_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=int)
    block = max(1, int(block_length))
    indices: List[int] = []
    while len(indices) < n:
        start = int(rng.integers(0, n))
        for offset in range(block):
            indices.append((start + offset) % n)
            if len(indices) >= n:
                break
    return np.asarray(indices[:n], dtype=int)


def _bootstrap_indices(
    n: int, block_length: int, method: str, rng: np.random.Generator
) -> np.ndarray:
    mode = method.lower()
    if mode == "stationary":
        return _stationary_bootstrap_indices(n, block_length, rng)
    if mode == "circular":
        return _circular_bootstrap_indices(n, block_length, rng)
    raise ValueError(f"Unknown bootstrap method '{method}'")


def _compute_confidence_intervals(
    samples: Mapping[str, Sequence[float]], *, alpha: float = 0.05
) -> Dict[str, Dict[str, float]]:
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    z_score = 1.959963984540054  # two-sided 95% normal quantile
    results: Dict[str, Dict[str, float]] = {}
    for key, values in samples.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        mean = float(arr.mean())
        if arr.size > 1:
            std = float(arr.std(ddof=1))
        else:
            std = 0.0
        margin = float(z_score * std / math.sqrt(arr.size)) if arr.size else 0.0
        results[key] = {
            "mean": mean,
            "lower": mean - margin,
            "upper": mean + margin,
            "n": int(arr.size),
        }
    return results


def _parse_cv_spec(spec: Optional[str]) -> int:
    if spec is None:
        return 0
    text = str(spec).strip()
    if not text:
        return 0
    if "=" in text:
        key, value = text.split("=", 1)
        if key.strip().lower() not in {"k", "folds", "n"}:
            raise ValueError("--cv must be provided as an integer or k=<folds>")
        text = value
    try:
        folds = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("--cv must be an integer (e.g. k=5)") from exc
    if folds <= 1:
        raise ValueError("--cv requires at least two folds")
    return folds


def _run_bootstrap_evaluations(
    *,
    df: Any,
    benchmark: Optional[Any],
    factor_panel: Optional[FactorPanel],
    backtest_kwargs: Mapping[str, Any],
    bootstrap: int,
    block_length: int,
    method: str,
    seed: int,
    alpha: float,
) -> Dict[str, Dict[str, float]]:
    if bootstrap <= 0:
        return {}
    arr = _frame_to_numpy(df)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        n_periods = arr.shape[0]
    else:
        n_periods = arr.shape[0]
    if n_periods == 0:
        return {}
    rng = np.random.default_rng(int(seed))
    samples: MutableMapping[str, List[float]] = defaultdict(list)
    eval_kwargs = dict(backtest_kwargs)
    for _ in range(int(bootstrap)):
        draw = _bootstrap_indices(n_periods, block_length, method, rng)
        boot_df = _slice_frame(df, draw, reset_index=True)
        boot_benchmark = (
            _slice_frame(benchmark, draw, reset_index=True) if benchmark is not None else None
        )
        boot_panel = _slice_factor_panel(factor_panel, draw, reset_index=True)
        res = backtest(
            boot_df,
            benchmark=boot_benchmark,
            factors=boot_panel,
            progress_callback=None,
            rebalance_callback=None,
            **eval_kwargs,
        )
        for key in _CI_METRIC_KEYS:
            if key not in res:
                continue
            value = res.get(key)
            if isinstance(value, (float, int, np.floating, np.integer)) and np.isfinite(value):
                samples[key].append(float(value))
    return _compute_confidence_intervals(samples, alpha=alpha)


def _run_cross_validation(
    *,
    df: Any,
    benchmark: Optional[Any],
    factor_panel: Optional[FactorPanel],
    backtest_kwargs: Mapping[str, Any],
    folds: int,
    lookback: int,
    dates: Sequence[Any],
) -> List[Dict[str, Any]]:
    if folds <= 1:
        return []
    n_periods = len(dates)
    if n_periods == 0:
        return []
    splits = np.array_split(np.arange(n_periods), folds)
    records: List[Dict[str, Any]] = []
    eval_kwargs = dict(backtest_kwargs)
    for fold_idx, fold in enumerate(splits, start=1):
        fold_indices = np.asarray(fold, dtype=int)
        if fold_indices.size == 0 or fold_indices.size <= lookback:
            continue
        subset_df = _slice_frame(df, fold_indices, reset_index=False)
        subset_benchmark = (
            _slice_frame(benchmark, fold_indices, reset_index=False) if benchmark is not None else None
        )
        subset_panel = _slice_factor_panel(factor_panel, fold_indices, reset_index=False)
        res = backtest(
            subset_df,
            benchmark=subset_benchmark,
            factors=subset_panel,
            progress_callback=None,
            rebalance_callback=None,
            **eval_kwargs,
        )
        record: Dict[str, Any] = {
            "fold": fold_idx,
            "start": _stringify(dates[int(fold_indices[0])]),
            "end": _stringify(dates[int(fold_indices[-1])]),
        }
        for key in _CI_METRIC_KEYS:
            if key not in res:
                continue
            value = res.get(key)
            if isinstance(value, (float, int, np.floating, np.integer)):
                record[key] = float(value)
            elif value is None:
                record[key] = None
        records.append(record)
    return records


def _write_equity(equity_path: Path, results: Dict[str, Any]) -> None:
    if pd is not None:
        frame = pd.DataFrame(
            {
                "date": results["dates"],
                "equity": results["equity"],
                "ret": results["returns"],
            }
        )
        frame.to_csv(equity_path, index=False)
        return

    data = np.column_stack(
        [
            np.asarray(results["dates"], dtype=str),
            np.asarray(results["equity"], dtype=float),
            np.asarray(results["returns"], dtype=float),
        ]
    )
    header = "date,equity,ret"
    np.savetxt(equity_path, data, fmt="%s", delimiter=",", header=header, comments="")


def _write_weights(weights_path: Path, results: Dict[str, Any]) -> None:
    W = np.asarray(results["weights"], dtype=float)
    if W.ndim == 1 and W.size:
        W = W.reshape(1, -1)
    n_assets = W.shape[1] if W.ndim > 1 else 0
    dates = results.get("rebalance_dates", [])
    cols = results.get("asset_names")
    if pd is not None:
        header_cols = cols if cols else [f"w{i}" for i in range(n_assets)]
        df = pd.DataFrame(W, columns=header_cols)
        if dates:
            df.insert(0, "date", dates)
        df.to_csv(weights_path, index=False)
        return

    header_cols = [f"w{i}" for i in range(n_assets)] if n_assets else []
    if cols:
        header_cols = list(cols)
        if n_assets and len(header_cols) != n_assets:
            header_cols = [f"w{i}" for i in range(n_assets)]
    if dates:
        header = ",".join(["date", *header_cols]) if header_cols else "date"
        if W.size:
            data = np.column_stack([np.asarray(dates, dtype=str), W])
        else:
            data = np.asarray(dates, dtype=str)[:, None]
    else:
        header = ",".join(header_cols)
        data = W
    np.savetxt(weights_path, data, delimiter=",", header=header, comments="", fmt="%s")


_SCENARIO_THRESHOLD_RULES: Dict[str, str] = {
    "sharpe": "min",
    "info_ratio": "min",
    "ann_return": "min",
    "max_drawdown": "max",
    "tracking_error": "max",
}


def _load_scenarios_config(
    spec: Any,
    *,
    asset_names: Sequence[str],
    dates: Sequence[Any],
) -> List[ScenarioDefinition]:
    if spec is None:
        return []
    raw = spec
    if isinstance(raw, Mapping) and "scenarios" in raw:
        scenarios_raw = raw["scenarios"]
    else:
        scenarios_raw = raw
    if not isinstance(scenarios_raw, Sequence):
        raise ValueError("Scenario configuration must be a sequence or mapping with 'scenarios'")
    asset_map = {str(name): idx for idx, name in enumerate(asset_names)}
    date_map = {_stringify(date): idx for idx, date in enumerate(dates)}
    definitions: List[ScenarioDefinition] = []
    for entry in scenarios_raw:
        if not isinstance(entry, Mapping):
            raise ValueError("Scenario entries must be mappings")
        name_raw = entry.get("name") or entry.get("id")
        if name_raw is None:
            raise ValueError("Scenario entries must include a name")
        name = str(name_raw)
        shocks_raw = entry.get("shocks") or []
        if not isinstance(shocks_raw, Sequence):
            raise ValueError(f"Scenario '{name}' shocks must be a sequence")
        shocks: List[ScenarioShock] = []
        for shock in shocks_raw:
            if not isinstance(shock, Mapping):
                raise ValueError(f"Scenario '{name}' shock specifications must be mappings")
            assets_spec = shock.get("assets", "*")
            asset_indices: List[int] = []
            if assets_spec in {"*", "all", None}:
                asset_indices = list(range(len(asset_names)))
            else:
                if isinstance(assets_spec, (str, Path)):
                    assets_iter: Sequence[Any] = [assets_spec]
                else:
                    if not isinstance(assets_spec, Sequence):
                        raise ValueError(
                            f"Scenario '{name}' assets specification must be a string or sequence"
                        )
                    assets_iter = assets_spec
                for asset_name in assets_iter:
                    asset_key = str(asset_name)
                    if asset_key in {"*", "all"}:
                        asset_indices = list(range(len(asset_names)))
                        break
                    idx = asset_map.get(asset_key)
                    if idx is None:
                        raise ValueError(f"Scenario '{name}' references unknown asset '{asset_key}'")
                    asset_indices.append(idx)
            asset_indices = sorted(dict.fromkeys(int(idx) for idx in asset_indices))
            if not asset_indices:
                raise ValueError(f"Scenario '{name}' shock does not target any known assets")
            raw_dates = shock.get("dates")
            if raw_dates is None and "date" in shock:
                raw_dates = shock.get("date")
            date_indices: Optional[Tuple[int, ...]] = None
            if raw_dates is not None:
                if isinstance(raw_dates, Sequence) and not isinstance(raw_dates, (str, bytes)):
                    dates_iter: Sequence[Any] = raw_dates
                else:
                    dates_iter = [raw_dates]
                indices: List[int] = []
                for raw_date in dates_iter:
                    if isinstance(raw_date, (int, np.integer)):
                        idx_val = int(raw_date)
                    else:
                        key = str(raw_date).strip()
                        if key.lower() == "last":
                            idx_val = len(dates) - 1
                        else:
                            mapped = date_map.get(key)
                            if mapped is None:
                                normalized_key: Optional[str] = None
                                try:
                                    normalized_key = _stringify(np.datetime64(key))
                                except Exception:  # pragma: no cover - best effort coercion
                                    normalized_key = None
                                if normalized_key is not None:
                                    mapped = date_map.get(normalized_key)
                            if mapped is None and pd is not None:  # pragma: no branch - optional
                                try:
                                    normalized_pd = _stringify(pd.Timestamp(key))
                                except Exception:  # pragma: no cover - best effort coercion
                                    normalized_pd = None
                                if normalized_pd is not None:
                                    mapped = date_map.get(normalized_pd)
                            if mapped is None:
                                raise ValueError(
                                    f"Scenario '{name}' references unknown date '{raw_date}'"
                                )
                            idx_val = int(mapped)
                    indices.append(idx_val)
                date_indices = tuple(sorted(dict.fromkeys(indices)))
            shift_val = shock.get("shift")
            scale_val = shock.get("scale")
            shift = float(shift_val) if shift_val is not None else 0.0
            scale = float(scale_val) if scale_val is not None else 0.0
            if abs(shift) <= 0.0 and abs(scale) <= 0.0:
                raise ValueError(f"Scenario '{name}' shock must specify a non-zero shift or scale")
            shocks.append(
                ScenarioShock(
                    asset_indices=tuple(asset_indices),
                    date_indices=date_indices,
                    shift=shift,
                    scale=scale,
                )
            )
        thresholds_raw = entry.get("thresholds") or entry.get("limits") or {}
        if thresholds_raw and not isinstance(thresholds_raw, Mapping):
            raise ValueError(f"Scenario '{name}' thresholds must be a mapping")
        thresholds: Dict[str, float] = {}
        if isinstance(thresholds_raw, Mapping):
            for key, value in thresholds_raw.items():
                try:
                    thresholds[str(key)] = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Scenario '{name}' threshold for '{key}' must be numeric"
                    ) from exc
        definitions.append(
            ScenarioDefinition(name=name, shocks=tuple(shocks), thresholds=thresholds)
        )
    return definitions


def _apply_scenario_shocks(
    base_returns: np.ndarray,
    scenario: ScenarioDefinition,
    default_rows: np.ndarray,
) -> np.ndarray:
    shocked = np.asarray(base_returns, dtype=float).copy()
    if not scenario.shocks:
        return shocked
    for shock in scenario.shocks:
        cols = np.asarray(shock.asset_indices, dtype=int)
        if cols.size == 0:
            continue
        if shock.date_indices is None:
            rows = np.asarray(default_rows, dtype=int)
        else:
            rows = np.asarray(shock.date_indices, dtype=int)
        if rows.size == 0:
            continue
        grid = np.ix_(rows, cols)
        if abs(shock.scale) > 0.0:
            shocked[grid] = shocked[grid] * (1.0 + shock.scale)
        if abs(shock.shift) > 0.0:
            shocked[grid] = shocked[grid] + shock.shift
    return shocked


def _replay_walk_returns(
    returns: np.ndarray,
    windows: Sequence[Tuple[int, int]],
    weights: np.ndarray,
) -> np.ndarray:
    weights_arr = np.asarray(weights, dtype=float)
    if weights_arr.ndim == 1:
        weights_arr = weights_arr.reshape(1, -1)
    if weights_arr.shape[0] != len(windows):
        raise ValueError("Weight history length does not match walk windows")
    blocks: List[np.ndarray] = []
    for (start, end), w in zip(windows, weights_arr):
        block = np.asarray(returns[start:end], dtype=float)
        if block.size == 0:
            continue
        contrib = block @ np.asarray(w, dtype=float)
        blocks.append(np.asarray(contrib, dtype=float).reshape(-1))
    if not blocks:
        return np.zeros(0, dtype=float)
    return np.concatenate(blocks)


def _compute_scenario_metrics(
    realized_returns: np.ndarray,
    benchmark_returns: Optional[np.ndarray],
    periodic_rf: float,
    trading_days: int,
) -> Dict[str, Any]:
    realized_arr = np.asarray(realized_returns, dtype=float)
    ann_factor = math.sqrt(max(1, int(trading_days)))
    ann_vol = (
        float(np.std(realized_arr) * ann_factor) if realized_arr.size else 0.0
    )
    ann_return = (
        float(np.mean(realized_arr) * trading_days) if realized_arr.size else 0.0
    )
    excess = realized_arr - float(periodic_rf)
    excess_mean = (
        float(excess.mean() * trading_days) if excess.size else 0.0
    )
    sharpe = float(excess_mean / ann_vol) if ann_vol > 1e-12 else 0.0
    equity = np.cumprod(1.0 + realized_arr, dtype=float) if realized_arr.size else np.ones(0)
    mdd = max_drawdown(equity) if equity.size else 0.0
    tracking_error: Optional[float] = None
    info_ratio: Optional[float] = None
    if benchmark_returns is not None:
        bench = np.asarray(benchmark_returns, dtype=float)
        if bench.size == realized_arr.size and realized_arr.size:
            active = realized_arr - bench
            te = compute_tracking_error(active, trading_days)
            tracking_error = te
            active_mean = float(active.mean() * trading_days)
            if te <= 1e-12:
                if abs(active_mean) <= 1e-12:
                    info_ratio = 0.0
                else:
                    info_ratio = float(math.copysign(1e6, active_mean))
            else:
                info_ratio = float(active_mean / te)
    return {
        "sharpe": sharpe,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_drawdown": mdd,
        "tracking_error": tracking_error,
        "info_ratio": info_ratio,
    }


def _count_breaches(metrics: Mapping[str, Any], thresholds: Mapping[str, float]) -> int:
    if not thresholds:
        return 0
    breaches = 0
    for key, limit in thresholds.items():
        if key not in metrics:
            continue
        value = metrics.get(key)
        if value is None:
            continue
        rule = _SCENARIO_THRESHOLD_RULES.get(key, "max")
        try:
            limit_val = float(limit)
        except (TypeError, ValueError):
            continue
        val = float(value)
        if rule == "min":
            if val < limit_val - 1e-12:
                breaches += 1
        else:
            if val > limit_val + 1e-12:
                breaches += 1
    return breaches


def _evaluate_scenarios(
    scenarios: Sequence[ScenarioDefinition],
    *,
    returns: np.ndarray,
    windows: Sequence[Tuple[int, int]],
    weights: np.ndarray,
    benchmark: Optional[np.ndarray],
    periodic_rf: float,
    trading_days: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    if not scenarios:
        return [], {}
    returns_arr = np.asarray(returns, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    if weights_arr.ndim == 1 and weights_arr.size:
        weights_arr = weights_arr.reshape(1, -1)
    if weights_arr.size == 0 or not windows:
        return [], {}
    if weights_arr.shape[0] != len(windows):
        raise ValueError("Scenario replay requires weight history for each walk window")
    row_mask = np.zeros(returns_arr.shape[0], dtype=bool)
    for start, end in windows:
        row_mask[int(start) : int(end)] = True
    default_rows = np.nonzero(row_mask)[0]
    if default_rows.size == 0:
        return [], {}
    bench_arr = None
    if benchmark is not None:
        bench_arr = np.asarray(benchmark, dtype=float)
    reports: List[Dict[str, Any]] = []
    shocked_weights: Dict[str, np.ndarray] = {}
    last_index = int(default_rows[-1])
    for scenario in scenarios:
        shocked_returns = _apply_scenario_shocks(returns_arr, scenario, default_rows)
        walk_returns = _replay_walk_returns(shocked_returns, windows, weights_arr)
        bench_series = None
        if bench_arr is not None and bench_arr.size:
            if bench_arr.size != walk_returns.size:
                raise ValueError("Benchmark series length mismatch for scenario replay")
            bench_series = bench_arr
        metrics = _compute_scenario_metrics(walk_returns, bench_series, periodic_rf, trading_days)
        breaches = _count_breaches(metrics, scenario.thresholds)
        reports.append(
            {
                "scenario": scenario.name,
                "sharpe": metrics["sharpe"],
                "ann_return": metrics["ann_return"],
                "ann_vol": metrics["ann_vol"],
                "max_drawdown": metrics["max_drawdown"],
                "tracking_error": metrics["tracking_error"],
                "info_ratio": metrics["info_ratio"],
                "breaches": breaches,
            }
        )
        last_returns = shocked_returns[last_index]
        base_weights = np.asarray(weights_arr[-1], dtype=float)
        updated = base_weights * (1.0 + np.asarray(last_returns, dtype=float))
        total = float(updated.sum())
        if total > 1e-12:
            updated = updated / total
        else:
            updated = np.zeros_like(updated)
        shocked_weights[scenario.name] = updated
    return reports, shocked_weights


def _write_scenarios_report(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        return
    columns = [
        "scenario",
        "sharpe",
        "ann_return",
        "ann_vol",
        "max_drawdown",
        "tracking_error",
        "info_ratio",
        "breaches",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for record in records:
            row = []
            for col in columns:
                value = record.get(col)
                row.append("" if value is None else value)
            writer.writerow(row)


def _write_shocked_weights(
    path: Path,
    asset_names: Sequence[str],
    shocked: Mapping[str, np.ndarray],
) -> None:
    if not shocked:
        return
    scenario_names = list(shocked.keys())
    assets = list(asset_names) if asset_names else [f"A{i}" for i in range(len(next(iter(shocked.values()))))]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["asset", *scenario_names])
        for idx, asset in enumerate(assets):
            row: List[Any] = [asset]
            for name in scenario_names:
                weights = np.asarray(shocked[name], dtype=float)
                value = weights[idx] if idx < weights.size else ""
                row.append(value)
            writer.writerow(row)


def _write_factor_constraints(path: Path, results: Dict[str, Any]) -> None:
    records: Sequence[Dict[str, Any]] = results.get("factor_records", [])  # type: ignore[assignment]
    factor_names: Sequence[str] = results.get("factor_names", [])  # type: ignore[assignment]
    if not records or not factor_names:
        return
    header = ["date"]
    for name in factor_names:
        header.extend([
            f"{name}_exposure",
            f"{name}_target",
            f"{name}_diff",
        ])
    header.extend(["tolerance", "missing"])

    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for record in records:
            date = record.get("date")
            exposures_raw = record.get("exposures")
            exposures_arr = np.zeros(len(factor_names), dtype=float)
            if exposures_raw is not None:
                exp_vals = np.asarray(exposures_raw, dtype=float)
                limit = min(exp_vals.size, exposures_arr.size)
                if limit:
                    exposures_arr[:limit] = exp_vals[:limit]
            targets_raw = record.get("targets")
            targets_arr = np.zeros(len(factor_names), dtype=float)
            if targets_raw is not None:
                tgt_vals = np.asarray(targets_raw, dtype=float)
                limit = min(tgt_vals.size, targets_arr.size)
                if limit:
                    targets_arr[:limit] = tgt_vals[:limit]
            tolerance = float(record.get("tolerance", 0.0))
            missing_flag = 1 if record.get("missing") else 0
            row: List[Any] = [date]
            for idx in range(len(factor_names)):
                exp_val = float(exposures_arr[idx])
                tgt_val = float(targets_arr[idx])
                diff_val = exp_val - tgt_val
                row.extend([exp_val, tgt_val, diff_val])
            row.extend([tolerance, missing_flag])
            writer.writerow(row)


def _write_factor_attr(path: Path, results: Dict[str, Any]) -> None:
    rows: Sequence[Dict[str, Any]] = results.get("factor_attr", [])  # type: ignore[assignment]
    factor_names: Sequence[str] = results.get("factor_names", [])  # type: ignore[assignment]
    if not rows or not factor_names:
        return
    header = ["date"]
    for name in factor_names:
        header.extend(
            [
                f"{name}_return",
                f"{name}_return_t",
                f"{name}_ic",
                f"{name}_ic_t",
            ]
        )
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in header})


def _write_rebalance_report(path: Path, results: Dict[str, Any]) -> None:
    records: Sequence[Dict[str, Any]] = results.get("rebalance_records", [])  # type: ignore[assignment]
    header = [
        "date",
        "gross_ret",
        "net_tx_ret",
        "net_slip_ret",
        "turnover",
        "turnover_pre_decay",
        "turnover_post_decay",
        "tx_cost",
        "slippage_cost",
        "nt_band_hits",
        "participation_breaches",
        "sector_breaches",
        "sector_penalty",
        "active_breaches",
        "group_breaches",
        "factor_bound_breaches",
        "factor_inf_norm",
        "factor_missing",
        "first_violation",
        "feasible",
        "projection_iterations",
        "block_sharpe",
        "block_sortino",
        "block_info_ratio",
        "block_tracking_error",
        "warm_applied",
        "decay",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for record in records or []:
            row = {key: record.get(key) for key in header}
            writer.writerow(row)


def _write_drawdowns(path: Path, results: Dict[str, Any]) -> None:
    events: Sequence[Dict[str, Any]] = results.get("drawdowns", [])  # type: ignore[assignment]
    header = ["peak", "trough", "recovery", "depth", "length"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for event in events:
            row = {
                "peak": _stringify(event.get("peak")) if event.get("peak") is not None else None,
                "trough": _stringify(event.get("trough")) if event.get("trough") is not None else None,
                "recovery": _stringify(event.get("recovery")) if event.get("recovery") is not None else None,
                "depth": event.get("depth"),
                "length": event.get("length"),
            }
            writer.writerow(row)


def _write_contributions(path: Path, results: Dict[str, Any]) -> None:
    records: Sequence[Dict[str, Any]] = results.get("contributions", [])  # type: ignore[assignment]
    header = ["date", "asset", "contribution", "block_return"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "date": _stringify(record.get("date")) if record.get("date") is not None else None,
                    "asset": record.get("asset"),
                    "contribution": record.get("contribution"),
                    "block_return": record.get("block_return"),
                }
            )


def _maybe_write_parquet(out_dir: Path, args: argparse.Namespace) -> None:
    if getattr(args, "out_format", "csv") != "parquet":
        return
    if pd is None:
        return
    try:
        equity_csv = out_dir / "equity.csv"
        if equity_csv.exists():
            pd.read_csv(equity_csv).to_parquet(out_dir / "equity.parquet")
        rebalance_csv = out_dir / "rebalance_report.csv"
        if rebalance_csv.exists():
            pd.read_csv(rebalance_csv).to_parquet(out_dir / "rebalance_report.parquet")
        metrics_csv = out_dir / "metrics.csv"
        if metrics_csv.exists():
            pd.read_csv(metrics_csv).to_parquet(out_dir / "metrics.parquet")
        weights_csv = out_dir / "weights.csv"
        if getattr(args, "save_weights", False) and weights_csv.exists():
            pd.read_csv(weights_csv).to_parquet(out_dir / "weights.parquet")
    except Exception:
        pass


def _write_exposures(path: Path, results: Dict[str, Any]) -> None:
    records: Sequence[Dict[str, Any]] = results.get("factor_records", [])  # type: ignore[assignment]
    factor_names: Sequence[str] = results.get("factor_names", [])  # type: ignore[assignment]
    if not records or not factor_names:
        return
    sector_columns: List[str] = []
    for record in records:
        sectors = record.get("sector_exposures")
        if isinstance(sectors, dict):
            for name in sectors.keys():
                if name not in sector_columns:
                    sector_columns.append(name)
    sector_columns.sort()
    header = ["date", *factor_names, *sector_columns]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for record in records:
            if record.get("missing"):
                continue
            exposures_raw = record.get("exposures")
            values = np.zeros(len(factor_names), dtype=float)
            if exposures_raw is not None:
                exp_vals = np.asarray(exposures_raw, dtype=float)
                limit = min(exp_vals.size, values.size)
                if limit:
                    values[:limit] = exp_vals[:limit]
            row: List[Any] = [record.get("date")]
            row.extend(values.tolist())
            sectors = record.get("sector_exposures") or {}
            for name in sector_columns:
                row.append(float(sectors.get(name, 0.0)))
            writer.writerow(row)


def _read_csv(csv_path: Path):
    class _Frame:
        def __init__(
            self,
            arr: np.ndarray,
            idx: Sequence[str],
            cols: Optional[Sequence[str]] = None,
        ) -> None:
            self._arr = arr
            converted: List[Any] = []
            for val in idx:
                try:
                    converted.append(np.datetime64(val))
                except Exception:
                    converted.append(val)
            self._idx = converted
            self._cols = list(cols) if cols is not None else []

        def to_numpy(self, dtype=float):  # pragma: no cover - simple proxy
            return self._arr.astype(dtype)

        @property
        def index(self):  # pragma: no cover - simple accessor
            return self._idx

        @property
        def columns(self):  # pragma: no cover - simple accessor
            return self._cols

        def filter_rows(self, mask: np.ndarray) -> "_Frame":
            mask = np.asarray(mask, dtype=bool)
            indices = [self._idx[i] for i in np.nonzero(mask)[0]]
            return _Frame(self._arr[mask], indices, self._cols)

    size_bytes = 0
    try:
        size_bytes = csv_path.stat().st_size
    except OSError:  # pragma: no cover - filesystem edge case
        size_bytes = 0

    use_polars = pl is not None and (pd is None or size_bytes > 8_000_000)
    if use_polars:
        try:
            frame = pl.read_csv(csv_path, try_parse_dates=True)
            if pd is not None:
                pdf = frame.to_pandas()
                if not pdf.empty and pdf.columns.size:
                    first_col = pdf.columns[0]
                    pdf = pdf.set_index(first_col)
                    if pd is not None:
                        pdf.index = pd.to_datetime(pdf.index, errors="ignore")
                return pdf
            if frame.width == 0:
                return _Frame(np.empty((0, 0), dtype=float), [], [])
            cols = frame.columns
            if not cols:
                return _Frame(np.empty((0, 0), dtype=float), [], [])
            values_np = frame.select(cols[1:]).to_numpy()
            if values_np.size:
                values_np = values_np.astype(float)
            dates = frame.select(cols[0]).to_series().to_list()
            return _Frame(values_np, dates, cols[1:])
        except Exception:  # pragma: no cover - fallback on parse issues
            if pd is None:
                raise
            # fall back to pandas parsing for complex headers
    if pd is not None:
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)

    header_cols: Optional[List[str]] = None
    # Peek the header row to capture asset names (and strip BOM/whitespace)
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header_row = next(reader)
        except StopIteration:
            header_row = []

    if header_row:
        if header_row[0]:
            # strip UTF-8 BOM and whitespace from the first header cell (date column)
            header_row[0] = header_row[0].lstrip("\ufeff").strip()
        # collect asset column names (strip whitespace)
        extracted = [col.strip() for col in header_row[1:]]
        header_cols = extracted if any(extracted) else None

    # Load the data block (all rows except header)
    raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=str)
    if raw.size == 0:
        values = np.empty((0, 0), dtype=float)
        dates: Sequence[str] = []
    else:
        raw = np.atleast_2d(raw)
        dates = raw[:, 0]
        values = raw[:, 1:].astype(float)

    # If header count doesn't match parsed columns, synthesize names
    if header_cols and values.size and values.shape[1] != len(header_cols):
        header_cols = [f"w{i}" for i in range(values.shape[1])]

    return _Frame(values, dates, header_cols)


def _extract_asset_names(frame: Any, n_assets: int) -> List[str]:
    if hasattr(frame, "columns") and getattr(frame, "columns") is not None:
        cols_obj = frame.columns
        if pd is not None and isinstance(cols_obj, pd.MultiIndex):
            level = cols_obj.get_level_values(-1)
            cols = [str(col) for col in level]
        else:
            cols = [str(col) for col in cols_obj]
        if len(cols) == n_assets:
            return cols
    return [f"A{i}" for i in range(n_assets)]


def _load_cap_weight_vector(path: Path, asset_names: Sequence[str]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Cap weights file not found: {path}")

    weights_map: Dict[str, float] = {}
    if pd is not None:
        frame = pd.read_csv(path)
        lower_cols = {str(col).lower() for col in frame.columns}
        if {"asset", "weight"}.issubset(lower_cols):
            # Normalize column names without mutating original dataframe ordering
            name_col = next(col for col in frame.columns if str(col).lower() == "asset")
            weight_col = next(col for col in frame.columns if str(col).lower() == "weight")
            for _, row in frame.iterrows():
                asset = str(row[name_col])
                weight = float(row[weight_col])
                if not math.isfinite(weight):
                    raise ValueError("Cap weight entries must be finite")
                weights_map[asset] = weight
        elif not frame.empty:
            first_row = frame.iloc[0]
            for col in frame.columns:
                weight = float(first_row[col])
                if not math.isfinite(weight):
                    raise ValueError("Cap weight entries must be finite")
                weights_map[str(col)] = weight
    if not weights_map:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            rows = [row for row in reader if any(cell.strip() for cell in row)]
        if not rows:
            raise ValueError("Cap weights file is empty")
        header = [cell.strip() for cell in rows[0]]
        if header and header[0].lower() in {"asset", "name"} and len(header) >= 2:
            for row in rows[1:]:
                if len(row) < 2:
                    continue
                asset = row[0].strip()
                if not asset:
                    continue
                weight = float(row[1])
                if not math.isfinite(weight):
                    raise ValueError("Cap weight entries must be finite")
                weights_map[asset] = weight
        elif len(rows) >= 2:
            for idx, col in enumerate(header):
                if idx >= len(rows[1]):
                    break
                weight = float(rows[1][idx])
                if not math.isfinite(weight):
                    raise ValueError("Cap weight entries must be finite")
                weights_map[col or f"A{idx}"] = weight
        else:
            raise ValueError("Cap weights file must contain at least one row of weights")

    if not weights_map:
        raise ValueError("Unable to parse cap weights file")

    weights = np.zeros(len(asset_names), dtype=float)
    found = 0
    for idx, name in enumerate(asset_names):
        weight = weights_map.get(name)
        if weight is None:
            raise ValueError(f"Missing cap weight for asset '{name}'")
        weights[idx] = weight
        found += 1

    if found == 0:
        raise ValueError("No overlapping assets found in cap weights file")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Cap weights must sum to a positive value")
    weights /= total
    return weights


def _compute_baseline_returns(
    baseline: str,
    returns_arr: np.ndarray,
    asset_names: Sequence[str],
    *,
    cap_weights_path: Optional[Path] = None,
) -> Tuple[str, np.ndarray, np.ndarray]:
    if returns_arr.ndim != 2:
        raise ValueError("Returns array must be two dimensional for baseline computation")
    n_assets = returns_arr.shape[1]
    if n_assets == 0:
        raise ValueError("Cannot compute baseline with zero assets")

    mode = str(baseline).lower()
    if mode == "equal":
        weights = np.full(n_assets, 1.0 / n_assets, dtype=float)
        label = "equal"
    elif mode == "cap":
        if cap_weights_path is None:
            raise ValueError("--cap-weights must be provided when --baseline=cap")
        weights = _load_cap_weight_vector(cap_weights_path, asset_names)
        label = "cap"
    else:
        raise ValueError(f"Unknown baseline mode '{baseline}'")

    returns = returns_arr @ weights
    return label, returns.astype(float, copy=False), weights


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the backtest CLI."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML/JSON file containing run parameters",
    )
    parser.add_argument("--csv", type=str, default=None, help="CSV of daily returns with date index")
    parser.add_argument(
        "--benchmark-csv",
        type=str,
        default=None,
        help="Optional CSV of benchmark returns (single column)",
    )
    parser.add_argument(
        "--baseline",
        choices=["equal", "cap"],
        default=None,
        help="Include baseline overlay and metrics (equal or cap)",
    )
    parser.add_argument(
        "--cap-weights",
        type=str,
        default=None,
        help="CSV file containing cap weights (required when --baseline=cap)",
    )
    parser.add_argument(
        "--active-min",
        type=float,
        default=None,
        help="Minimum per-asset active weight relative to the benchmark",
    )
    parser.add_argument(
        "--active-max",
        type=float,
        default=None,
        help="Maximum per-asset active weight relative to the benchmark",
    )
    parser.add_argument(
        "--active-group-caps",
        type=str,
        default=None,
        help="YAML/JSON file describing active group caps",
    )
    parser.add_argument(
        "--factor-bounds",
        type=str,
        default=None,
        help="YAML/JSON file describing factor exposure bounds",
    )
    parser.add_argument(
        "--attr",
        action="store_true",
        help="Compute factor attribution diagnostics (factor returns and IC)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="YAML/JSON file describing stress scenarios to replay",
    )
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--step", type=int, default=21)
    parser.add_argument(
        "--ewma_span",
        type=int,
        default=60,
        help="EWMA span (only used when --cov-model=ewma)",
    )
    parser.add_argument(
        "--cov-model",
        type=str,
        default="sample",
        help=(
            "Covariance backend (sample|ewma|lw|oas|ridge|glasso|bayesian"
            " or custom:<module>:<callable>; optional :param=value suffixes allowed)"
        ),
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="sharpe",
        help="Optimization objective name (e.g. sharpe, hrp) or custom:<module>:<callable>",
    )
    parser.add_argument(
        "--te-target",
        type=float,
        default=0.0,
        help="Target tracking error level for te_target objective",
    )
    parser.add_argument(
        "--lambda-te",
        type=float,
        default=0.0,
        help="Penalty weight applied to tracking error in multi_term objective",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Use float32 for all numpy computations",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enforce torch deterministic algorithms (errors if unavailable)",
    )
    parser.add_argument(
        "--cache-cov",
        type=int,
        default=8,
        help="Maximum number of covariance matrices to cache",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers to use (if supported)",
    )
    parser.add_argument(
        "--gamma-turnover",
        type=float,
        default=0.0,
        help="Penalty weight applied to turnover in multi_term objective",
    )
    parser.add_argument(
        "--gamma-by-sector",
        type=str,
        default=None,
        help="YAML/JSON mapping of sector-level turnover penalties",
    )
    parser.add_argument(
        "--warm-start",
        type=str,
        default=None,
        help="weights.csv from prior run for initial prev_weights",
    )
    parser.add_argument(
        "--warm-align",
        choices=["by_date", "last_row"],
        default="last_row",
        help="match warm-start weights to first rebalance date or use last row",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.0,
        help="Blend current optimal weights with previous allocations (0..1)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="bt_out")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier override when tracking runs",
    )
    parser.add_argument(
        "--runs-csv",
        type=str,
        default=None,
        help="Append run metadata to the specified CSV tracker",
    )
    parser.add_argument(
        "--track-artifacts",
        type=str,
        default=None,
        help="Directory to store zipped run artifacts when tracking",
    )
    parser.add_argument(
        "--log-json",
        type=str,
        default=None,
        help="Write per-rebalance JSON lines to the given file",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Display progress updates during the backtest run",
    )
    parser.add_argument(
        "--out-format",
        choices=["csv", "parquet"],
        default="csv",
        help="Emit parquet artifacts alongside CSV outputs",
    )
    parser.add_argument(
        "--rf-bps",
        type=float,
        default=0.0,
        help="Annualized risk-free rate in basis points (default: 0)",
    )
    parser.add_argument(
        "--trading-days",
        type=int,
        default=252,
        help="Trading periods per year used for annualization",
    )
    parser.add_argument(
        "--save-weights",
        action="store_true",
        help="Write weights.csv with per-step allocations",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip generating the equity plot",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and configuration without writing backtest artifacts",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Drop duplicate dates from returns/benchmark instead of raising",
    )
    parser.add_argument(
        "--tx-cost-bps",
        type=float,
        default=0.0,
        help="Per-rebalance transaction cost in basis points",
    )
    parser.add_argument(
        "--tx-cost-mode",
        choices=["none", "upfront", "amortized", "posthoc"],
        default="posthoc",
        help="When to apply transaction costs",
    )
    parser.add_argument(
        "--metric-alpha",
        type=float,
        default=0.05,
        help="Tail probability (alpha) for realized CVaR metric",
    )
    parser.add_argument(
        "--factors",
        type=str,
        default=None,
        help="Path to factor loadings panel (csv/parquet/yaml)",
    )
    parser.add_argument(
        "--factor-tolerance",
        type=float,
        default=1e-6,
        help="Infinity-norm tolerance for factor neutrality",
    )
    parser.add_argument(
        "--factor-targets",
        type=str,
        default=None,
        help="Optional factor target vector (csv/parquet/yaml)",
    )
    parser.add_argument(
        "--factor-align",
        choices=["strict", "subset"],
        default="strict",
        help="How to align factor panel dates with returns",
    )
    parser.add_argument(
        "--factors-required",
        action="store_true",
        help="Fail if any rebalance window lacks factor data",
    )
    parser.add_argument(
        "--slippage",
        type=str,
        default=None,
        help="Slippage model specification (e.g. proportional:5)",
    )
    parser.add_argument(
        "--nt-band",
        type=str,
        default="0",
        help="No-trade band threshold (e.g. 5bps or 0.001)",
    )
    parser.add_argument(
        "--refine-every",
        type=int,
        default=1,
        help="Run SLSQP refinement every k rebalances",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Number of block bootstrap resamples to evaluate (0 disables)",
    )
    parser.add_argument(
        "--bootstrap-method",
        choices=["stationary", "circular"],
        default="stationary",
        help="Block bootstrap method to use for confidence intervals",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=0,
        help="Window length (in periods) for rolling performance metrics (0 disables)",
    )
    parser.add_argument(
        "--block",
        type=int,
        default=21,
        help="Average block length used for bootstrap resampling",
    )
    parser.add_argument(
        "--cv",
        type=str,
        default=None,
        help="Walk-forward cross-validation specification (e.g. k=5)",
    )

    return parser


def main(args: Optional[Iterable[str]] = None) -> None:
    argv = list(args) if args is not None else sys.argv[1:]
    if argv and argv[0] == "sweep":
        _main_sweep(argv[1:])
        return

    parser = build_parser()

    preliminary, _ = parser.parse_known_args(args=argv)
    config_path: Optional[Path] = None
    cfg_validated = False
    cfg_blob: Dict[str, Any] = {}
    if preliminary.config:
        config_path = Path(preliminary.config)
        raw_config = _load_run_config(config_path)
        try:
            model = RunConfig.model_validate(raw_config)
            cfg_blob = model.model_dump()
            cfg_validated = True
        except ValidationError as exc:
            lines = []
            for err in exc.errors():
                loc = ".".join(map(str, err.get("loc", []))) or "<root>"
                lines.append(f"{loc}: {err.get('msg')}")
            message = "Invalid config:\n  " + "\n  ".join(lines)
            raise SystemExit(message)
        cfg_blob.pop("config", None)
        parser.set_defaults(**cfg_blob)

    parsed = parser.parse_args(args=argv)
    parsed.run_id = _generate_run_id(getattr(parsed, "run_id", None))
    if not parsed.csv:
        raise ValueError("--csv must be provided via CLI or config")

    cv_folds = _parse_cv_spec(parsed.cv) if parsed.cv else 0
    bootstrap_count = int(parsed.bootstrap)
    if bootstrap_count < 0:
        raise ValueError("--bootstrap must be non-negative")
    block_length = int(parsed.block)
    if block_length <= 0:
        raise ValueError("--block must be a positive integer")
    bootstrap_method = str(parsed.bootstrap_method or "stationary")

    float_dtype = np.float32 if parsed.float32 else np.float64
    df = _read_csv(Path(parsed.csv))
    df, returns_preview, date_index = _sanitize_frame(
        df, drop_duplicates=parsed.drop_duplicates, label="returns"
    )
    benchmark_df = _read_csv(Path(parsed.benchmark_csv)) if parsed.benchmark_csv else None
    factor_panel = load_factor_panel(Path(parsed.factors)) if parsed.factors else None
    slippage_cfg = parse_slippage(parsed.slippage)
    nt_band_spec = getattr(parsed, "nt_band", 0.0)
    if isinstance(nt_band_spec, str):
        nt_band_value = _parse_fractional_value(nt_band_spec)
    else:
        nt_band_value = float(nt_band_spec)
    if nt_band_value < 0.0:
        raise ValueError("--nt-band must be non-negative")
    factor_target_vec = None
    if parsed.factor_targets:
        if factor_panel is None:
            raise ValueError("Factor targets provided without factor loadings")
        factor_target_vec = load_factor_targets(Path(parsed.factor_targets), factor_panel.factor_names)

    factor_panel_prepared = factor_panel
    if factor_panel_prepared is not None:
        factor_panel_prepared, _ = validate_factor_panel(
            factor_panel_prepared, df, align=parsed.factor_align
        )
    if parsed.attr and factor_panel_prepared is None:
        raise ValueError("--attr requires factor loadings (--factors)")

    benchmark_weights_spec = getattr(parsed, "benchmark_weights", None)
    active_group_spec = parsed.active_group_caps or None
    factor_bounds_spec = parsed.factor_bounds or None

    tx_cost_bps = float(parsed.tx_cost_bps) if parsed.tx_cost_mode != "none" else 0.0
    returns_arr_for_baseline = np.atleast_2d(returns_preview.astype(float_dtype))
    asset_names_for_baseline = _extract_asset_names(df, returns_arr_for_baseline.shape[1])
    all_dates = list(date_index)

    if parsed.baseline == "cap" and not parsed.cap_weights:
        raise ValueError("--cap-weights must be supplied when --baseline=cap")

    baseline_series_full: Optional[np.ndarray] = None
    baseline_label: Optional[str] = None
    baseline_weights: Optional[np.ndarray] = None
    if parsed.baseline:
        baseline_label, baseline_series_full, baseline_weights = _compute_baseline_returns(
            parsed.baseline,
            returns_arr_for_baseline,
            asset_names_for_baseline,
            cap_weights_path=Path(parsed.cap_weights) if parsed.cap_weights else None,
        )

    shared_backtest_kwargs: Dict[str, Any] = dict(
        lookback=parsed.lookback,
        step=parsed.step,
        ewma_span=parsed.ewma_span,
        objective=parsed.objective,
        seed=parsed.seed,
        tx_cost_bps=tx_cost_bps,
        tx_cost_mode=parsed.tx_cost_mode,
        metric_alpha=parsed.metric_alpha,
        factor_targets=factor_target_vec,
        factor_tolerance=parsed.factor_tolerance,
        factor_align=parsed.factor_align,
        factors_required=parsed.factors_required,
        slippage=slippage_cfg,
        nt_band=nt_band_value,
        refine_every=parsed.refine_every,
        cov_model=parsed.cov_model,
        benchmark_weights=benchmark_weights_spec,
        active_min=parsed.active_min,
        active_max=parsed.active_max,
        active_group_caps=active_group_spec,
        factor_bounds=factor_bounds_spec,
        te_target=parsed.te_target,
        lambda_te=parsed.lambda_te,
        gamma_turnover=parsed.gamma_turnover,
        gamma_by_sector=parsed.gamma_by_sector,
        risk_free_rate=float(parsed.rf_bps) / 1e4,
        trading_days=parsed.trading_days,
        dtype=float_dtype,
        cov_cache_size=parsed.cache_cov,
        max_workers=parsed.max_workers,
        decay=parsed.decay,
        drop_duplicates=parsed.drop_duplicates,
        deterministic=parsed.deterministic,
        compute_factor_attr=parsed.attr,
    )
    base_backtest_kwargs = dict(shared_backtest_kwargs)
    base_backtest_kwargs.update(
        warm_start=parsed.warm_start,
        warm_align=parsed.warm_align,
    )
    eval_backtest_kwargs = dict(shared_backtest_kwargs)
    eval_backtest_kwargs.update(
        warm_start=None,
        warm_align=parsed.warm_align,
    )

    jsonl_writer: Optional[_JsonlWriter] = None
    progress_printer: Optional[_ProgressPrinter] = None
    try:
        if parsed.log_json:
            log_path = Path(parsed.log_json)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_writer = _JsonlWriter(log_path.open("w", encoding="utf-8"))
        if parsed.progress:
            progress_printer = _ProgressPrinter(sys.stderr)

        results = backtest(
            df,
            benchmark=benchmark_df,
            factors=factor_panel_prepared,
            progress_callback=progress_printer,
            rebalance_callback=jsonl_writer.write if jsonl_writer else None,
            **base_backtest_kwargs,
        )
        results["run_id"] = parsed.run_id
    finally:
        if progress_printer is not None:
            progress_printer.close()
        if jsonl_writer is not None:
            jsonl_writer.close()

    if baseline_series_full is not None:
        date_to_ret = {date: float(ret) for date, ret in zip(all_dates, baseline_series_full)}
        baseline_returns: List[float] = []
        for date in results["dates"]:
            if date not in date_to_ret:
                raise ValueError("Baseline returns missing date alignment")
            baseline_returns.append(date_to_ret[date])
        baseline_arr = np.asarray(baseline_returns, dtype=float_dtype)
        baseline_equity = np.cumprod(1.0 + baseline_arr, dtype=float_dtype)
        results["baseline_returns"] = baseline_arr
        results["baseline_equity"] = baseline_equity
        results["baseline_label"] = baseline_label
        if baseline_weights is not None:
            results["baseline_weights"] = np.asarray(baseline_weights, dtype=float_dtype)

        trading_days = max(1, int(results.get("trading_days", 252)))
        ann_factor = math.sqrt(trading_days)
        periodic_rf = float(results.get("periodic_risk_free", 0.0))
        excess_baseline = baseline_arr - periodic_rf
        baseline_vol = float(np.std(baseline_arr) * ann_factor) if baseline_arr.size else 0.0
        baseline_excess_mean = (
            float(excess_baseline.mean() * trading_days) if baseline_arr.size else 0.0
        )
        baseline_sharpe = (
            float(baseline_excess_mean / baseline_vol) if baseline_vol > 1e-12 else 0.0
        )

        benchmark_returns = results.get("benchmark_returns")
        baseline_info_ratio: Optional[float] = None
        if isinstance(benchmark_returns, np.ndarray) and benchmark_returns.size == baseline_arr.size:
            active_baseline = baseline_arr - benchmark_returns
            te = compute_tracking_error(active_baseline, trading_days)
            active_mean = float(active_baseline.mean() * trading_days)
            if te <= 1e-12:
                if abs(active_mean) <= 1e-12:
                    baseline_info_ratio = 0.0
                else:
                    baseline_info_ratio = float(math.copysign(1e6, active_mean))
            else:
                baseline_info_ratio = float(active_mean / te)

        realized_arr = np.asarray(results["returns"], dtype=float_dtype)
        active_vs_baseline = realized_arr - baseline_arr
        alpha_vs_baseline = (
            float(active_vs_baseline.mean() * trading_days) if active_vs_baseline.size else 0.0
        )
        if active_vs_baseline.size:
            hit_rate_vs_baseline = float(np.mean(realized_arr > baseline_arr))
        else:
            hit_rate_vs_baseline = 0.0

        results["baseline_sharpe"] = baseline_sharpe
        results["baseline_info_ratio"] = baseline_info_ratio
        results["alpha_vs_baseline"] = alpha_vs_baseline
        results["hit_rate_vs_baseline"] = hit_rate_vs_baseline

    out_dir = Path(parsed.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    constraint_manifest = results.get("constraint_manifest")

    if parsed.dry_run:
        _write_run_manifest(
            out_dir,
            parsed,
            config_path,
            results=results,
            extras=constraint_manifest,
            validated=cfg_validated,
        )
        return

    bootstrap_cis: Dict[str, Dict[str, float]] = {}
    cv_records: List[Dict[str, Any]] = []
    if bootstrap_count:
        bootstrap_cis = _run_bootstrap_evaluations(
            df=df,
            benchmark=benchmark_df,
            factor_panel=factor_panel_prepared,
            backtest_kwargs=eval_backtest_kwargs,
            bootstrap=bootstrap_count,
            block_length=block_length,
            method=bootstrap_method,
            seed=parsed.seed,
            alpha=0.05,
        )
    if cv_folds:
        cv_records = _run_cross_validation(
            df=df,
            benchmark=benchmark_df,
            factor_panel=factor_panel_prepared,
            backtest_kwargs=eval_backtest_kwargs,
            folds=cv_folds,
            lookback=parsed.lookback,
            dates=all_dates,
        )

    _write_metrics(out_dir / "metrics.csv", results)
    _write_rebalance_report(out_dir / "rebalance_report.csv", results)
    _write_drawdowns(out_dir / "drawdowns.csv", results)
    _write_contributions(out_dir / "contrib.csv", results)
    _write_equity(out_dir / "equity.csv", results)
    if bootstrap_count:
        ci_path = out_dir / "metrics_ci.json"
        with ci_path.open("w", encoding="utf-8") as fh:
            json.dump(bootstrap_cis, fh, indent=2, sort_keys=True)
    if cv_folds:
        cv_path = out_dir / "cv_results.csv"
        metric_columns = [
            key for key in _CI_METRIC_KEYS if any(key in record for record in cv_records)
        ]
        with cv_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["fold", "start", "end", *metric_columns])
            for record in cv_records:
                row = [record.get("fold"), record.get("start"), record.get("end")]
                row.extend(record.get(col) for col in metric_columns)
                writer.writerow(row)
    if baseline_series_full is not None and baseline_label is not None:
        baseline_equity_path = out_dir / f"equity_baseline_{baseline_label}.csv"
        _write_equity(
            baseline_equity_path,
            {
                "dates": results["dates"],
                "equity": results["baseline_equity"],
                "returns": results["baseline_returns"],
            },
        )
    net_tx_returns = results.get("net_tx_returns")
    if isinstance(net_tx_returns, np.ndarray) and net_tx_returns.size:
        net_results = dict(results)
        net_results["returns"] = net_tx_returns
        net_results["equity"] = np.cumprod(1.0 + net_tx_returns, dtype=float_dtype)
        _write_equity(out_dir / "equity_net_of_tc.csv", net_results)
    slip_returns = results.get("slippage_net_returns")
    if isinstance(slip_returns, np.ndarray) and slip_returns.size:
        slip_results = dict(results)
        slip_results["returns"] = slip_returns
        slip_results["equity"] = np.cumprod(1.0 + slip_returns, dtype=float_dtype)
        _write_equity(out_dir / "equity_net_of_slippage.csv", slip_results)
    if parsed.factors:
        _write_factor_constraints(out_dir / "factor_constraints.csv", results)
        _write_exposures(out_dir / "exposures.csv", results)
        if parsed.attr:
            _write_factor_attr(out_dir / "factor_attr.csv", results)
    elif parsed.attr:
        _write_factor_attr(out_dir / "factor_attr.csv", results)
    if parsed.save_weights:
        _write_weights(out_dir / "weights.csv", results)
    if getattr(parsed, "rolling_window", 0):
        _write_rolling_metrics(
            out_dir / "rolling_metrics.csv", results, int(parsed.rolling_window)
        )

    scenario_spec = getattr(parsed, "scenarios", None)
    if scenario_spec:
        scenario_data = _maybe_load_structure(scenario_spec)
        scenario_asset_names = results.get("asset_names") or asset_names_for_baseline
        if scenario_asset_names is None:
            scenario_asset_names = []
        scenario_defs = _load_scenarios_config(
            scenario_data,
            asset_names=list(scenario_asset_names),
            dates=date_index,
        )
        if scenario_defs:
            windows = results.get("walk_windows") or []
            weights_history = results.get("weights")
            benchmark_series = results.get("benchmark_returns")
            scenario_rows, shocked_weights = _evaluate_scenarios(
                scenario_defs,
                returns=returns_preview,
                windows=windows,
                weights=weights_history,
                benchmark=benchmark_series,
                periodic_rf=float(results.get("periodic_risk_free", 0.0)),
                trading_days=int(results.get("trading_days", 252)),
            )
            if scenario_rows:
                _write_scenarios_report(out_dir / "scenarios_report.csv", scenario_rows)
            if shocked_weights:
                _write_shocked_weights(
                    out_dir / "weights_after_shock.csv",
                    list(scenario_asset_names),
                    shocked_weights,
                )

    diagnostics = results.get("factor_diagnostics")
    if diagnostics:
        (out_dir / "factor_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
        )

    _maybe_write_parquet(out_dir, parsed)

    _write_run_manifest(
        out_dir,
        parsed,
        config_path,
        results=results,
        extras=constraint_manifest,
        validated=cfg_validated,
    )

    tracker_path = getattr(parsed, "runs_csv", None)
    if tracker_path:
        tracker_file = Path(tracker_path)
        artifact_path: Optional[Path] = None
        artifact_dir = getattr(parsed, "track_artifacts", None)
        if artifact_dir:
            try:
                artifact_path = _archive_run_artifacts(out_dir, Path(artifact_dir), parsed.run_id)
            except Exception as exc:  # pragma: no cover - artifact packaging best-effort
                print(f"Failed to archive artifacts: {exc}", file=sys.stderr)
                artifact_path = None
        args_json = json.dumps(_serialize_args(parsed), sort_keys=True)
        tracker_record: Dict[str, Any] = {
            "run_id": parsed.run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "git_sha": _resolve_git_sha(),
            "out_dir": str(out_dir),
            "manifest": str(out_dir / "run_config.json"),
            "config": str(config_path) if config_path else "",
            "objective": str(parsed.objective),
            "artifact": str(artifact_path) if artifact_path else "",
            "args_json": args_json,
        }
        for key in RUN_TRACKER_METRICS:
            value = results.get(key)
            if value is None:
                tracker_record[key] = None
            else:
                try:
                    tracker_record[key] = float(value)
                except Exception:
                    tracker_record[key] = value
        try:
            _append_run_tracker(tracker_file, tracker_record)
        except Exception as exc:  # pragma: no cover - tracker best-effort
            print(f"Failed to append run tracker: {exc}", file=sys.stderr)

    if not parsed.skip_plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.figure()
            dates = results["dates"]
            gross_curve = np.cumprod(
                1.0 + np.asarray(results["gross_returns"], dtype=float_dtype), dtype=float_dtype
            )
            plt.plot(dates, gross_curve, label="gross")
            if isinstance(net_tx_returns, np.ndarray) and net_tx_returns.size:
                net_curve = np.cumprod(1.0 + net_tx_returns, dtype=float_dtype)
                if not np.allclose(net_curve, gross_curve):
                    plt.plot(dates, net_curve, label="net tx")
            if isinstance(slip_returns, np.ndarray) and slip_returns.size:
                slip_curve = np.cumprod(1.0 + slip_returns, dtype=float_dtype)
                if not np.allclose(slip_curve, gross_curve):
                    plt.plot(dates, slip_curve, label="net slippage")
            baseline_equity = results.get("baseline_equity")
            baseline_label_plot = results.get("baseline_label")
            if baseline_equity is not None and baseline_label_plot is not None:
                baseline_curve = np.asarray(baseline_equity, dtype=float_dtype)
                if baseline_curve.size == gross_curve.size:
                    plt.plot(dates, baseline_curve, label=f"baseline ({baseline_label_plot})")
            plt.title(f"Equity — {parsed.objective}")
            plt.xlabel("Date")
            plt.ylabel("Equity")
            if plt.gca().has_data():
                plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "equity.png", dpi=160)
            plt.close()
        except Exception:
            pass

    print(f"Wrote {out_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
