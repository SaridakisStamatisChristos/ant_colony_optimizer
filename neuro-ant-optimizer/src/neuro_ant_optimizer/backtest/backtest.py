"""Walk-forward backtesting utilities built around the neuro-ant optimizer."""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

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
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - minimal environments
    yaml = None  # type: ignore


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


@dataclass
class SlippageConfig:
    model: str
    param: float


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


def _write_run_manifest(
    out_dir: Path, args: argparse.Namespace, config_path: Optional[Path]
) -> None:
    manifest: Dict[str, Any] = {
        "args": _serialize_args(args),
        "package_version": __version__,
        "python_version": sys.version,
    }
    if config_path is not None:
        manifest["config_path"] = str(config_path)
    try:  # optional torch dependency
        import torch

        manifest["torch_version"] = torch.__version__
    except ModuleNotFoundError:  # pragma: no cover - environments without torch
        manifest["torch_version"] = None

    git_sha = _resolve_git_sha()
    if git_sha:
        manifest["git_sha"] = git_sha

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
        try:
            param = float(param_text)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid slippage parameter '{param_text}'") from exc
    else:
        model = text
        param = float("nan")
    defaults = {"proportional": 5.0, "square": 1.0, "vol_scaled": 1.0}
    if model not in defaults:
        raise ValueError(f"Unsupported slippage model '{model}'")
    if not np.isfinite(param):
        param = defaults[model]
    return SlippageConfig(model=model, param=param)


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
) -> Optional[np.ndarray]:
    if lookback <= 0:
        return None
    index_map = panel.index_map()
    rebalance_date = dates[start_idx]
    history_indices: List[int] = []
    for offset in range(max(0, start_idx - lookback), start_idx + 1):
        mapped = index_map.get(dates[offset])
        if mapped is not None:
            history_indices.append(mapped)
    if not history_indices:
        return None
    history = panel.loadings[history_indices]
    current_idx = index_map.get(rebalance_date)
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


def _compute_slippage_cost(
    cfg: Optional[SlippageConfig], turn: float, asset_block: np.ndarray
) -> float:
    if cfg is None or turn <= 0:
        return 0.0
    if cfg.model == "proportional":
        return (cfg.param / 1e4) * turn
    if cfg.model == "square":
        return cfg.param * (turn ** 2)
    if cfg.model == "vol_scaled":
        vol = _cross_sectional_volatility(asset_block)
        return cfg.param * turn * vol
    return 0.0


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


def turnover(previous: Optional[np.ndarray], current: np.ndarray) -> float:
    """Compute the L1 turnover between two weight vectors."""

    if previous is None:
        return float(np.abs(current).sum())
    return float(np.abs(current - previous).sum())


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


def _build_optimizer(n_assets: int, seed: int) -> NeuroAntPortfolioOptimizer:
    cfg = OptimizerConfig(
        n_ants=24,
        max_iter=25,
        topk_refine=6,
        topk_train=6,
        use_shrinkage=False,
        shrinkage_delta=0.15,
        cvar_alpha=0.05,
        seed=seed,
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


def _frame_to_numpy(frame: Any) -> np.ndarray:
    if hasattr(frame, "to_numpy"):
        return frame.to_numpy(dtype=float)  # type: ignore[no-any-return]
    return np.asarray(frame, dtype=float)


def _frame_index(frame: Any, length: int) -> List[Any]:
    if hasattr(frame, "index"):
        return list(frame.index)  # type: ignore[attr-defined]
    return list(range(length))


_OBJECTIVE_MAP: Dict[str, OptimizationObjective] = {
    "sharpe": OptimizationObjective.SHARPE_RATIO,
    "max_return": OptimizationObjective.MAX_RETURN,
    "min_variance": OptimizationObjective.MIN_VARIANCE,
    "risk_parity": OptimizationObjective.RISK_PARITY,
    "min_cvar": OptimizationObjective.MIN_CVAR,
    "tracking_error": OptimizationObjective.TRACKING_ERROR_MIN,
    "info_ratio": OptimizationObjective.INFO_RATIO_MAX,
}


_COV_MODELS = {"sample", "ewma", "lw", "oas"}


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
    slippage: Optional[SlippageConfig] = None,
    refine_every: int = 1,
    cov_model: str = "sample",
    benchmark: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run a rolling-window backtest on a return dataframe."""

    if lookback <= 0 or step <= 0:
        raise ValueError("lookback and step must be positive integers")
    if refine_every <= 0:
        raise ValueError("refine_every must be positive")
    if objective not in _OBJECTIVE_MAP:
        raise ValueError(f"Unknown objective '{objective}'")
    cov_model = str(cov_model).lower()
    if cov_model not in _COV_MODELS:
        raise ValueError(
            f"Unknown cov_model '{cov_model}' (choose from {sorted(_COV_MODELS)})"
        )

    returns = _frame_to_numpy(df)
    if returns.size == 0:
        raise ValueError("input dataframe must contain returns")

    set_seed(seed)
    n_periods, n_assets = returns.shape
    dates = _frame_index(df, n_periods)
    benchmark_series: Optional[np.ndarray] = None
    if benchmark is not None:
        bench_values = _frame_to_numpy(benchmark)
        bench_values = np.asarray(bench_values, dtype=float)
        if bench_values.ndim > 2:
            raise ValueError("Benchmark series must be one or two dimensional")
        if bench_values.ndim == 2:
            if bench_values.shape[1] == 0:
                raise ValueError("Benchmark series contains no columns")
            if bench_values.shape[1] > 1:
                raise ValueError("Benchmark series must contain exactly one column")
            bench_values = bench_values[:, 0]
        benchmark_series = np.asarray(bench_values, dtype=float).reshape(-1)
        if benchmark_series.size != n_periods:
            raise ValueError("Benchmark length must match returns length")

    objective_enum = _OBJECTIVE_MAP[objective]
    if (
        objective_enum
        in {
            OptimizationObjective.TRACKING_ERROR_MIN,
            OptimizationObjective.INFO_RATIO_MAX,
        }
        and benchmark_series is None
    ):
        raise ValueError("Benchmark series required for tracking_error/info_ratio objectives")

    asset_names = (
        list(getattr(df, "columns", []))
        if hasattr(df, "columns") and getattr(df, "columns") is not None
        else [f"A{i}" for i in range(n_assets)]
    )

    factor_panel: Optional[FactorPanel] = None
    factor_target_vec: Optional[np.ndarray] = None
    factor_names: List[str] = []
    if factors is not None:
        try:
            aligned = factors.align_assets(asset_names)
        except ValueError as exc:
            raise ValueError("No overlapping assets between returns and factor panel") from exc
        if len(aligned.assets) != len(asset_names):
            missing = [asset for asset in asset_names if asset not in aligned.assets]
            if missing:
                print(
                    "Warning: dropping assets without factor data: "
                    + ", ".join(missing)
                )
            keep_mask = [asset in aligned.assets for asset in asset_names]
            returns = returns[:, keep_mask]
            asset_names = [asset for asset, keep in zip(asset_names, keep_mask) if keep]
            n_periods, n_assets = returns.shape
            aligned = factors.align_assets(asset_names)
        factor_panel = aligned
        factor_names = list(factor_panel.factor_names)
        n_factors = factor_panel.loadings.shape[2]
        if n_factors == 0:
            factor_panel = None
        else:
            if factor_targets is not None:
                if factor_targets.shape[0] != n_factors:
                    raise ValueError("Factor targets dimension does not match factor loadings")
                factor_target_vec = factor_targets.astype(float)
            else:
                factor_target_vec = np.zeros(n_factors, dtype=float)

    if returns.shape[1] == 0:
        raise ValueError("No assets remain after aligning factors with returns")

    optimizer = _build_optimizer(returns.shape[1], seed)
    constraints = _build_constraints(returns.shape[1])
    constraints.factor_tolerance = factor_tolerance

    weights: List[np.ndarray] = []
    rebalance_dates: List[Any] = []
    realized_returns: List[float] = []
    realized_dates: List[Any] = []
    turnovers: List[float] = []
    prev_weights: Optional[np.ndarray] = None
    tc = float(tx_cost_bps) / 1e4
    slippage_costs: List[float] = []
    missing_factor_logged: set = set()
    factor_records: List[Dict[str, Any]] = []
    gross_returns: List[float] = []
    net_tx_returns: List[float] = []
    net_slip_returns: List[float] = []
    rebalance_records: List[Dict[str, Any]] = []
    benchmark_realized: List[float] = []
    cov_cache: "OrderedDict[bytes, np.ndarray]" = OrderedDict()

    for start in range(lookback, n_periods, step):
        end = min(start + step, n_periods)
        train = returns[start - lookback : start]
        test = returns[start:end]
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
        if cov_model == "ewma":
            span = int(ewma_span if ewma_span is not None else 60)
            cov_raw = ewma_cov(train, span=span)
        elif cov_model == "lw":
            cov_raw = _lw_cov(train)
        elif cov_model == "oas":
            cov_raw = _oas_cov(train)
        else:
            cov_raw = _sample_cov(train)
        if optimizer.cfg.use_shrinkage:
            cov_raw = shrink_covariance(cov_raw, delta=optimizer.cfg.shrinkage_delta)
        cov_key: Optional[bytes] = None
        cov: Optional[np.ndarray] = None
        if cov_model == "ewma":
            cov_key = cov_raw.tobytes()
            cached_cov = cov_cache.get(cov_key)
            if cached_cov is not None:
                cov = cached_cov
        if cov is None:
            cov = nearest_psd(cov_raw)
            if cov_key is not None:
                cov_cache[cov_key] = cov.copy()
                while len(cov_cache) > 4:
                    cov_cache.popitem(last=False)
        active_factors = False
        current_factor_snapshot: Optional[np.ndarray] = None
        if factor_panel is not None:
            snapshot = _compute_factor_snapshot(factor_panel, dates, start, lookback)
            if snapshot is None:
                rebalance_date = dates[start]
                if rebalance_date not in missing_factor_logged:
                    print(f"Skipping factor neutrality on {rebalance_date} (missing factor data)")
                    missing_factor_logged.add(rebalance_date)
                constraints.factor_loadings = None
                constraints.factor_targets = None
            else:
                constraints.factor_loadings = snapshot
                constraints.factor_targets = factor_target_vec
                active_factors = True
                current_factor_snapshot = snapshot
        else:
            constraints.factor_loadings = None
            constraints.factor_targets = None

        rebalance_idx = len(weights)
        should_refine = (rebalance_idx % refine_every) == 0
        result = optimizer.optimize(
            mu,
            cov,
            constraints,
            objective=objective_enum,
            refine=should_refine,
            benchmark=bench_stats,
        )
        w = result.weights
        weights.append(w)
        rebalance_date = dates[start]
        rebalance_dates.append(rebalance_date)
        gross_block_returns = test @ w
        gross_returns.extend(gross_block_returns.tolist())
        length = max(1, gross_block_returns.size)

        turn = turnover(prev_weights, w)
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

        slip_cost = _compute_slippage_cost(slippage, turn, test)
        slippage_costs.append(slip_cost)
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

        net_tx_returns.extend(tx_block_returns.tolist())
        net_slip_returns.extend(slip_block_returns.tolist())

        block_returns = block_returns_metrics
        realized_returns.extend(block_returns.tolist())
        turnovers.append(turn)
        prev_weights = w
        realized_dates.extend(dates[start:end])
        if benchmark_series is not None:
            benchmark_realized.extend(benchmark_series[start:end].tolist())

        sector_breaches = 0
        sector_exposures: Dict[str, float] = {}
        if constraints.sector_map is not None:
            sectors = np.asarray(constraints.sector_map)
            cap = float(constraints.max_sector_concentration)
            for sector in np.unique(sectors):
                mask = sectors == sector
                total = float(w[mask].sum())
                sector_exposures[f"sector_{sector}"] = total
                if total > cap + 1e-9:
                    sector_breaches += 1
        factor_inf_norm = 0.0
        exposures: Optional[np.ndarray] = None
        if active_factors and current_factor_snapshot is not None:
            exposures = current_factor_snapshot.T @ w
            targets = factor_target_vec if factor_target_vec is not None else None
            if targets is None:
                targets = np.zeros_like(exposures)
            factor_inf_norm = float(
                np.linalg.norm(exposures - targets, ord=np.inf)
            )
            factor_records.append(
                {
                    "date": rebalance_date,
                    "exposures": exposures,
                    "targets": factor_target_vec if factor_target_vec is not None else None,
                    "tolerance": factor_tolerance,
                    "sector_exposures": sector_exposures if sector_exposures else None,
                }
            )

        rebalance_records.append(
            {
                "date": rebalance_date,
                "gross_ret": float(np.prod(1.0 + gross_block_returns) - 1.0),
                "net_tx_ret": float(np.prod(1.0 + tx_block_returns) - 1.0),
                "net_slip_ret": float(np.prod(1.0 + slip_block_returns) - 1.0),
                "turnover": float(turn),
                "tx_cost": float(tx_cost_value),
                "slippage_cost": float(slip_cost),
                "sector_breaches": int(sector_breaches),
                "factor_inf_norm": float(factor_inf_norm),
            }
        )

    realized_returns_arr = np.asarray(realized_returns, dtype=float)
    gross_returns_arr = np.asarray(gross_returns, dtype=float)
    net_tx_returns_arr = np.asarray(net_tx_returns, dtype=float)
    net_slip_returns_arr = np.asarray(net_slip_returns, dtype=float)
    benchmark_returns_arr = (
        np.asarray(benchmark_realized, dtype=float)
        if benchmark_realized
        else np.array([])
    )
    equity = np.cumprod(1.0 + realized_returns_arr)
    slippage_costs_arr = np.asarray(slippage_costs, dtype=float) if slippage_costs else np.array([])
    avg_slippage_bps = (
        float(slippage_costs_arr.mean() * 1e4) if slippage_costs_arr.size else 0.0
    )
    slippage_net_returns: Optional[np.ndarray] = None
    if slippage is not None:
        slippage_net_returns = net_slip_returns_arr.copy()

    ann_vol = float(np.std(realized_returns_arr) * math.sqrt(252)) if realized_returns_arr.size else 0.0
    ann_return = float(np.mean(realized_returns_arr) * 252) if realized_returns_arr.size else 0.0
    sharpe = ann_return / ann_vol if ann_vol > 1e-12 else 0.0
    negatives = realized_returns_arr[realized_returns_arr < 0]
    downside_vol = float(negatives.std() * math.sqrt(252)) if negatives.size else 0.0
    sortino = ann_return / downside_vol if downside_vol > 1e-12 else 0.0
    alpha = float(np.clip(metric_alpha, 1e-4, 0.5))
    if realized_returns_arr.size:
        tail_len = max(1, int(math.floor(alpha * realized_returns_arr.size)))
        tail = np.sort(realized_returns_arr)[:tail_len]
        realized_cvar = float(-tail.mean()) if tail.size else 0.0
    else:
        realized_cvar = 0.0
    mdd = max_drawdown(equity)
    avg_turn = float(np.mean(turnovers)) if turnovers else 0.0

    tracking_error = None
    info_ratio = None
    if benchmark_returns_arr.size == realized_returns_arr.size and realized_returns_arr.size:
        active = realized_returns_arr - benchmark_returns_arr
        te = float(np.std(active))
        tracking_error = te
        if te <= 1e-12:
            info_ratio = 0.0 if abs(active.mean()) <= 1e-12 else math.copysign(1e6, active.mean())
        else:
            info_ratio = float(active.mean() / te)

    return {
        "dates": realized_dates,
        "returns": realized_returns_arr,
        "gross_returns": gross_returns_arr,
        "net_tx_returns": net_tx_returns_arr,
        "net_slip_returns": net_slip_returns_arr,
        "equity": equity,
        "weights": np.asarray(weights),
        "rebalance_dates": rebalance_dates,
        "asset_names": asset_names,
        "cov_model": cov_model,
        "benchmark_returns": benchmark_returns_arr if benchmark_returns_arr.size else None,
        "sharpe": sharpe,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_drawdown": mdd,
        "avg_turnover": avg_turn,
        "downside_vol": downside_vol,
        "sortino": sortino,
        "realized_cvar": realized_cvar,
        "tracking_error": tracking_error,
        "info_ratio": info_ratio,
        "factor_records": factor_records,
        "factor_names": factor_names,
        "factor_tolerance": factor_tolerance,
        "avg_slippage_bps": avg_slippage_bps,
        "slippage_costs": slippage_costs_arr,
        "slippage_net_returns": slippage_net_returns,
        "rebalance_records": rebalance_records,
    }


def _write_metrics(metrics_path: Path, results: Dict[str, Any]) -> None:
    with metrics_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for key in [
            "sharpe",
            "ann_return",
            "ann_vol",
            "max_drawdown",
            "avg_turnover",
            "avg_slippage_bps",
            "downside_vol",
            "sortino",
            "realized_cvar",
            "tracking_error",
            "info_ratio",
        ]:
            writer.writerow([key, results[key]])


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
    header.append("tolerance")

    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for record in records:
            date = record.get("date")
            exposures = np.asarray(record.get("exposures", []), dtype=float)
            targets = record.get("targets")
            if targets is None:
                targets_arr = np.zeros_like(exposures)
            else:
                targets_arr = np.asarray(targets, dtype=float)
            tolerance = float(record.get("tolerance", 0.0))
            row: List[Any] = [date]
            for idx, name in enumerate(factor_names):
                exp_val = float(exposures[idx]) if idx < exposures.size else 0.0
                tgt_val = float(targets_arr[idx]) if idx < targets_arr.size else 0.0
                diff_val = exp_val - tgt_val
                row.extend([exp_val, tgt_val, diff_val])
            row.append(tolerance)
            writer.writerow(row)


def _write_rebalance_report(path: Path, results: Dict[str, Any]) -> None:
    records: Sequence[Dict[str, Any]] = results.get("rebalance_records", [])  # type: ignore[assignment]
    if not records:
        return
    header = [
        "date",
        "gross_ret",
        "net_tx_ret",
        "net_slip_ret",
        "turnover",
        "tx_cost",
        "slippage_cost",
        "sector_breaches",
        "factor_inf_norm",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in header}
            writer.writerow(row)


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
            exposures = np.asarray(record.get("exposures", []), dtype=float)
            row: List[Any] = [record.get("date")]
            for idx in range(len(factor_names)):
                value = float(exposures[idx]) if idx < exposures.size else 0.0
                row.append(value)
            sectors = record.get("sector_exposures") or {}
            for name in sector_columns:
                row.append(float(sectors.get(name, 0.0)))
            writer.writerow(row)


def _read_csv(csv_path: Path):
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

    class _Frame:
        def __init__(
            self,
            arr: np.ndarray,
            idx: Sequence[str],
            cols: Optional[Sequence[str]] = None,
        ):
            self._arr = arr
            self._idx = [np.datetime64(d) for d in idx]
            self._cols = list(cols) if cols is not None else []

        def to_numpy(self, dtype=float):
            return self._arr.astype(dtype)

        @property
        def index(self):
            return self._idx

        @property
        def columns(self):
            return self._cols

    return _Frame(values, dates, header_cols)


def main(args: Optional[Iterable[str]] = None) -> None:
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
        choices=sorted(_COV_MODELS),
        default="sample",
        help="Covariance backend: sample|ewma|lw|oas",
    )
    parser.add_argument(
        "--objective",
        choices=sorted(_OBJECTIVE_MAP.keys()),
        default="sharpe",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="bt_out")
    parser.add_argument(
        "--save-weights",
        action="store_true",
        help="Write weights.csv with per-step allocations",
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
        "--slippage",
        type=str,
        default=None,
        help="Slippage model specification (e.g. proportional:5)",
    )
    parser.add_argument(
        "--refine-every",
        type=int,
        default=1,
        help="Run SLSQP refinement every k rebalances",
    )

    preliminary, _ = parser.parse_known_args(args=args)
    config_path: Optional[Path] = None
    if preliminary.config:
        config_path = Path(preliminary.config)
        config_overrides = _load_run_config(config_path)
        config_overrides.pop("config", None)
        parser.set_defaults(**config_overrides)

    parsed = parser.parse_args(args=args)
    if not parsed.csv:
        raise ValueError("--csv must be provided via CLI or config")

    df = _read_csv(Path(parsed.csv))
    benchmark_df = _read_csv(Path(parsed.benchmark_csv)) if parsed.benchmark_csv else None
    factor_panel = load_factor_panel(Path(parsed.factors)) if parsed.factors else None
    slippage_cfg = parse_slippage(parsed.slippage)
    factor_target_vec = None
    if parsed.factor_targets:
        if factor_panel is None:
            raise ValueError("Factor targets provided without factor loadings")
        factor_target_vec = load_factor_targets(Path(parsed.factor_targets), factor_panel.factor_names)

    tx_cost_bps = float(parsed.tx_cost_bps) if parsed.tx_cost_mode != "none" else 0.0
    results = backtest(
        df,
        lookback=parsed.lookback,
        step=parsed.step,
        ewma_span=parsed.ewma_span,
        objective=parsed.objective,
        seed=parsed.seed,
        tx_cost_bps=tx_cost_bps,
        tx_cost_mode=parsed.tx_cost_mode,
        metric_alpha=parsed.metric_alpha,
        factors=factor_panel,
        factor_targets=factor_target_vec,
        factor_tolerance=parsed.factor_tolerance,
        slippage=slippage_cfg,
        refine_every=parsed.refine_every,
        cov_model=parsed.cov_model,
        benchmark=benchmark_df,
    )

    out_dir = Path(parsed.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(out_dir / "metrics.csv", results)
    _write_rebalance_report(out_dir / "rebalance_report.csv", results)
    _write_equity(out_dir / "equity.csv", results)
    net_tx_returns = results.get("net_tx_returns")
    if isinstance(net_tx_returns, np.ndarray) and net_tx_returns.size:
        net_results = dict(results)
        net_results["returns"] = net_tx_returns
        net_results["equity"] = np.cumprod(1.0 + net_tx_returns)
        _write_equity(out_dir / "equity_net_of_tc.csv", net_results)
    slip_returns = results.get("slippage_net_returns")
    if isinstance(slip_returns, np.ndarray) and slip_returns.size:
        slip_results = dict(results)
        slip_results["returns"] = slip_returns
        slip_results["equity"] = np.cumprod(1.0 + slip_returns)
        _write_equity(out_dir / "equity_net_of_slippage.csv", slip_results)
    if parsed.factors:
        _write_factor_constraints(out_dir / "factor_constraints.csv", results)
        _write_exposures(out_dir / "exposures.csv", results)
    if parsed.save_weights:
        _write_weights(out_dir / "weights.csv", results)

    _write_run_manifest(out_dir, parsed, config_path)

    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure()
        dates = results["dates"]
        gross_curve = np.cumprod(1.0 + np.asarray(results["gross_returns"], dtype=float))
        plt.plot(dates, gross_curve, label="gross")
        if isinstance(net_tx_returns, np.ndarray) and net_tx_returns.size:
            net_curve = np.cumprod(1.0 + net_tx_returns)
            if not np.allclose(net_curve, gross_curve):
                plt.plot(dates, net_curve, label="net tx")
        if isinstance(slip_returns, np.ndarray) and slip_returns.size:
            slip_curve = np.cumprod(1.0 + slip_returns)
            if not np.allclose(slip_curve, gross_curve):
                plt.plot(dates, slip_curve, label="net slippage")
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
