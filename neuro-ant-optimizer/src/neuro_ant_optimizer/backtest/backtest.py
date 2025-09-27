"""Walk-forward backtesting utilities built around the neuro-ant optimizer."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from neuro_ant_optimizer.constraints import PortfolioConstraints
from neuro_ant_optimizer.optimizer import (
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
}


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
) -> Dict[str, Any]:
    """Run a rolling-window backtest on a return dataframe."""

    if lookback <= 0 or step <= 0:
        raise ValueError("lookback and step must be positive integers")
    if objective not in _OBJECTIVE_MAP:
        raise ValueError(f"Unknown objective '{objective}'")

    returns = _frame_to_numpy(df)
    if returns.size == 0:
        raise ValueError("input dataframe must contain returns")

    set_seed(seed)
    n_periods, n_assets = returns.shape
    dates = _frame_index(df, n_periods)
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

    for start in range(lookback, n_periods, step):
        end = min(start + step, n_periods)
        train = returns[start - lookback : start]
        test = returns[start:end]
        mu = train.mean(axis=0)
        if ewma_span is not None:
            cov_raw = ewma_cov(train, span=ewma_span)
        else:
            cov_raw = np.cov(train, rowvar=False)
        if optimizer.cfg.use_shrinkage:
            cov_raw = shrink_covariance(cov_raw, delta=optimizer.cfg.shrinkage_delta)
        cov = nearest_psd(cov_raw)
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

        result = optimizer.optimize(
            mu,
            cov,
            constraints,
            objective=_OBJECTIVE_MAP[objective],
        )
        w = result.weights
        weights.append(w)
        rebalance_dates.append(dates[start])
        block_returns = test @ w
        turn = turnover(prev_weights, w)
        slip_cost = _compute_slippage_cost(slippage, turn, test)
        slippage_costs.append(slip_cost)
        if tx_cost_mode in ("upfront", "amortized") and tc > 0.0:
            if tx_cost_mode == "upfront":
                if block_returns.size > 0:
                    block_returns = block_returns.copy()
                    block_returns[0] -= tc * turn
            else:
                length = max(1, block_returns.size)
                block_returns = block_returns - (tc * turn / length)
        if (
            slippage is not None
            and tx_cost_mode in ("upfront", "amortized")
            and slip_cost > 0.0
        ):
            block_returns = block_returns.copy()
            if tx_cost_mode == "upfront" and block_returns.size > 0:
                block_returns[0] -= slip_cost
            else:
                length = max(1, block_returns.size)
                block_returns = block_returns - (slip_cost / length)
        realized_returns.extend(block_returns.tolist())
        turnovers.append(turn)
        prev_weights = w
        realized_dates.extend(dates[start:end])
        if active_factors and current_factor_snapshot is not None:
            exposures = current_factor_snapshot.T @ w
            factor_records.append(
                {
                    "date": dates[start],
                    "exposures": exposures,
                    "targets": factor_target_vec if factor_target_vec is not None else None,
                    "tolerance": factor_tolerance,
                }
            )

    realized_returns_arr = np.asarray(realized_returns, dtype=float)
    equity = np.cumprod(1.0 + realized_returns_arr)
    slippage_costs_arr = np.asarray(slippage_costs, dtype=float) if slippage_costs else np.array([])
    avg_slippage_bps = (
        float(slippage_costs_arr.mean() * 1e4) if slippage_costs_arr.size else 0.0
    )
    slippage_net_returns: Optional[np.ndarray] = None
    if slippage is not None:
        slippage_net_returns = realized_returns_arr.copy()
        if (
            tx_cost_mode == "posthoc"
            and slippage_costs_arr.size
            and len(rebalance_dates) == slippage_costs_arr.size
        ):
            index_map = {date: idx for idx, date in enumerate(realized_dates)}
            starts = [index_map[date] for date in rebalance_dates if date in index_map]
            starts.append(len(realized_dates))
            for block_idx, cost in enumerate(slippage_costs_arr):
                if block_idx >= len(starts) - 1:
                    break
                i0, i1 = starts[block_idx], starts[block_idx + 1]
                length = max(1, i1 - i0)
                slippage_net_returns[i0:i1] = slippage_net_returns[i0:i1] - (cost / length)

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

    return {
        "dates": realized_dates,
        "returns": realized_returns_arr,
        "equity": equity,
        "weights": np.asarray(weights),
        "rebalance_dates": rebalance_dates,
        "asset_names": asset_names,
        "sharpe": sharpe,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "max_drawdown": mdd,
        "avg_turnover": avg_turn,
        "downside_vol": downside_vol,
        "sortino": sortino,
        "realized_cvar": realized_cvar,
        "factor_records": factor_records,
        "factor_names": factor_names,
        "factor_tolerance": factor_tolerance,
        "avg_slippage_bps": avg_slippage_bps,
        "slippage_costs": slippage_costs_arr,
        "slippage_net_returns": slippage_net_returns,
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
    parser.add_argument("--csv", required=True, help="CSV of daily returns with date index")
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--step", type=int, default=21)
    parser.add_argument("--ewma_span", type=int, default=60)
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
    parsed = parser.parse_args(args=args)

    df = _read_csv(Path(parsed.csv))
    factor_panel = load_factor_panel(Path(parsed.factors)) if parsed.factors else None
    slippage_cfg = parse_slippage(parsed.slippage)
    factor_target_vec = None
    if parsed.factor_targets:
        if factor_panel is None:
            raise ValueError("Factor targets provided without factor loadings")
        factor_target_vec = load_factor_targets(Path(parsed.factor_targets), factor_panel.factor_names)

    results = backtest(
        df,
        lookback=parsed.lookback,
        step=parsed.step,
        ewma_span=parsed.ewma_span,
        objective=parsed.objective,
        seed=parsed.seed,
        tx_cost_bps=parsed.tx_cost_bps if parsed.tx_cost_mode in ("upfront", "amortized") else 0.0,
        tx_cost_mode=parsed.tx_cost_mode,
        metric_alpha=parsed.metric_alpha,
        factors=factor_panel,
        factor_targets=factor_target_vec,
        factor_tolerance=parsed.factor_tolerance,
        slippage=slippage_cfg,
    )

    out_dir = Path(parsed.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(out_dir / "metrics.csv", results)
    if (
        parsed.tx_cost_mode == "posthoc"
        and parsed.tx_cost_bps
        and results["weights"].shape[0] > 0
    ):
        tc = float(parsed.tx_cost_bps) / 1e4
        rebalance_dates = results.get("rebalance_dates", [])
        all_dates = results["dates"]
        index_map = {date: idx for idx, date in enumerate(all_dates)}
        starts = [index_map[date] for date in rebalance_dates if date in index_map]
        starts.append(len(all_dates))
        net = results["returns"].copy()
        for block in range(len(starts) - 1):
            i0, i1 = starts[block], starts[block + 1]
            length = max(1, i1 - i0)
            if block == 0:
                turn = float(np.abs(results["weights"][block]).sum())
            else:
                turn = float(
                    np.abs(results["weights"][block] - results["weights"][block - 1]).sum()
                )
            net[i0:i1] = net[i0:i1] - (tc * turn / length)
        net_results = dict(results)
        net_results["returns"] = net
        net_results["equity"] = np.cumprod(1.0 + net)
        _write_equity(out_dir / "equity_net_of_tc.csv", net_results)
    _write_equity(out_dir / "equity.csv", results)
    if results.get("slippage_net_returns") is not None:
        slip_results = dict(results)
        slip_results["returns"] = results["slippage_net_returns"]
        slip_results["equity"] = np.cumprod(1.0 + results["slippage_net_returns"])
        _write_equity(out_dir / "equity_net_of_slippage.csv", slip_results)
    if parsed.factors:
        _write_factor_constraints(out_dir / "factor_constraints.csv", results)
    if parsed.save_weights:
        _write_weights(out_dir / "weights.csv", results)

    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure()
        plt.plot(results["dates"], results["equity"])
        plt.title(f"Equity — {parsed.objective}")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(out_dir / "equity.png", dpi=160)
        plt.close()
    except Exception:
        pass

    print(f"Wrote {out_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
