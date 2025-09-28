"""Walk-forward backtesting utilities built around the neuro-ant optimizer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Union,
)

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


SCHEMA_VERSION = "1.0.0"


ProgressCallback = Callable[[int, int], None]
RebalanceLogCallback = Callable[[Dict[str, Any]], None]


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


@dataclass
class SlippageConfig:
    model: str
    param: float


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
    out_dir: Path,
    args: argparse.Namespace,
    config_path: Optional[Path],
    results: Optional[Dict[str, Any]] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    manifest: Dict[str, Any] = {
        "args": _serialize_args(args),
        "schema_version": SCHEMA_VERSION,
        "package_version": __version__,
        "python_version": sys.version,
    }
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
    "min_cvar": OptimizationObjective.MIN_CVAR,
    "tracking_error": OptimizationObjective.TRACKING_ERROR_MIN,
    "min_tracking_error": OptimizationObjective.TRACKING_ERROR_MIN,
    "info_ratio": OptimizationObjective.INFO_RATIO_MAX,
    "te_target": OptimizationObjective.TRACKING_ERROR_TARGET,
    "multi_term": OptimizationObjective.MULTI_TERM,
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
    factor_align: str = "strict",
    factors_required: bool = False,
    slippage: Optional[SlippageConfig] = None,
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
    risk_free_rate: float = 0.0,
    trading_days: int = 252,
    progress_callback: Optional[ProgressCallback] = None,
    rebalance_callback: Optional[RebalanceLogCallback] = None,
    dtype: np.dtype = np.float64,
    cov_cache_size: int = 8,
    max_workers: Optional[int] = None,
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
    if cov_model != "ewma":
        ewma_span = None
    else:
        if ewma_span is None:
            ewma_span = 60
        try:
            ewma_span = int(ewma_span)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("ewma_span must be an integer") from exc
        if ewma_span < 2:
            raise ValueError("ewma_span must be >= 2")

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

    if trading_days <= 0:
        raise ValueError("trading_days must be positive")

    annual_rf = float(risk_free_rate)
    if annual_rf <= -1.0:
        raise ValueError("risk_free_rate must be greater than -100%")
    periodic_rf = float((1.0 + annual_rf) ** (1.0 / trading_days) - 1.0)

    float_dtype = np.dtype(dtype)
    if float_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        float_dtype = np.dtype(np.float64)
    max_workers_value = None if max_workers is None else int(max_workers)
    if max_workers_value is not None and max_workers_value <= 0:
        raise ValueError("max_workers must be positive when provided")
    returns = _frame_to_numpy(df, float_dtype)
    if returns.size == 0:
        raise ValueError("input dataframe must contain returns")

    set_seed(seed)
    n_periods, n_assets = returns.shape
    dates = _frame_index(df, n_periods)
    benchmark_series: Optional[np.ndarray] = None
    if benchmark is not None:
        bench_values = _frame_to_numpy(benchmark, float_dtype)
        bench_values = np.asarray(bench_values, dtype=float_dtype)
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

    objective_enum = _OBJECTIVE_MAP[objective]
    if (
        objective_enum
        in {
            OptimizationObjective.TRACKING_ERROR_MIN,
            OptimizationObjective.INFO_RATIO_MAX,
            OptimizationObjective.TRACKING_ERROR_TARGET,
            OptimizationObjective.MULTI_TERM,
        }
        and benchmark_series is None
    ):
        raise ValueError(
            "Benchmark series required for tracking_error/info_ratio/te_target/multi_term objectives"
        )

    asset_names = (
        list(getattr(df, "columns", []))
        if hasattr(df, "columns") and getattr(df, "columns") is not None
        else [f"A{i}" for i in range(n_assets)]
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
    tc = float(tx_cost_bps) / 1e4
    missing_factor_logged: Set[Any] = set()
    factor_records: List[Dict[str, Any]] = []
    rebalance_records: List[Dict[str, Any]] = []

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
    cursor = 0

    factor_index_map: Optional[Dict[Any, int]] = None
    if factor_panel is not None:
        factor_index_map = factor_panel.index_map()

    if not rebalance_points:
        if progress_callback is not None:
            progress_callback(0, 0)
        empty = np.array([], dtype=float_dtype)
        benchmark_returns = empty if benchmark_series is not None else None
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
            "cov_model": cov_model,
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
            "warnings": ["no_rebalances"],
            "cov_cache_size": int(max_cov_cache),
            "cov_cache_hits": 0,
            "cov_cache_misses": 0,
            "cov_cache_evictions": 0,
            "max_workers": max_workers_value,
            "dtype": float_dtype.name,
        }

    # include model + params in the cache key to avoid collisions across models/spans
    cov_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
    max_cov_cache = max(0, int(cov_cache_size))
    cov_cache_hits = 0
    cov_cache_misses = 0
    cov_cache_evictions = 0

    if progress_callback is not None:
        progress_callback(0, n_windows)

    for window_idx, start in enumerate(rebalance_points):
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
        span = ewma_span if cov_model == "ewma" and ewma_span is not None else None
        window_hash = _hash_returns_window(train)
        cov_key: Optional[Tuple[Any, ...]] = (cov_model, span, window_hash)
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
            if cov_model == "ewma":
                assert span is not None
                cov_raw = ewma_cov(train, span=span)
            elif cov_model == "lw":
                cov_raw = _lw_cov(train)
            elif cov_model == "oas":
                cov_raw = _oas_cov(train)
            else:
                cov_raw = _sample_cov(train)
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
        result = optimizer.optimize(
            mu,
            cov,
            constraints,
            objective=objective_enum,
            refine=should_refine,
            benchmark=bench_stats,
        )
        opt_elapsed_ms = (time.perf_counter() - opt_start) * 1000.0
        feasible_flags.append(bool(getattr(result, "feasible", True)))
        projection_iters.append(int(getattr(result, "projection_iterations", 0)))
        w = result.weights
        weights.append(w)
        rebalance_dates.append(rebalance_date)

        gross_block_returns = np.asarray(test @ w, dtype=float_dtype).reshape(-1)
        block_len = gross_block_returns.size
        idx_slice = slice(cursor, cursor + block_len)
        gross_returns_arr[idx_slice] = gross_block_returns
        length = max(1, block_len)

        turn = turnover(prev_weights, w)
        turnovers_arr[window_idx] = float(turn)
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

        slip_cost = _compute_slippage_cost(slippage, turn, test)
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
                sector_exposures = {
                    f"sector_{int(label)}": float(value)
                    for label, value in zip(labels, exposures_values)
                }
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
                block_tracking_error = float(
                    np.std(active_block) * math.sqrt(trading_days)
                )
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
            }
        )
        if rebalance_callback is not None:
            log_record: Dict[str, Any] = {
                "date": _stringify(rebalance_date),
                "seed": int(seed),
                "objective": str(objective),
                "cov_model": str(cov_model),
                "costs": {
                    "tx": float(tx_cost_value),
                    "slippage": float(slip_cost),
                },
                "turnover": float(turn),
                "feasible": bool(feasible_flags[-1]),
                "breaches": {
                    "active": int(active_breaches),
                    "group": int(group_breaches),
                    "factor": int(factor_bound_breaches),
                    "sector": int(sector_breaches),
                },
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
    slippage_costs_arr = slippage_costs[: len(weights)] if n_windows else np.array([])
    avg_slippage_bps = (
        float(slippage_costs_arr.mean() * 1e4) if slippage_costs_arr.size else 0.0
    )
    turnovers_used = turnovers_arr[: len(weights)] if n_windows else np.array([])
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

    tracking_error = None
    info_ratio = None
    if benchmark_returns_arr.size == realized_returns_arr.size and realized_returns_arr.size:
        active = realized_returns_arr - benchmark_returns_arr
        te = float(np.std(active) * ann_factor)
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
        "te_target": float(optimizer.cfg.te_target),
        "lambda_te": float(optimizer.cfg.lambda_te),
        "gamma_turnover": float(optimizer.cfg.gamma_turnover),
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
        "factor_diagnostics": factor_diagnostics.to_dict() if factor_diagnostics else None,
        "constraint_manifest": constraint_manifest,
        "warnings": [],
        "cov_cache_size": int(max_cov_cache),
        "cov_cache_hits": int(cov_cache_hits),
        "cov_cache_misses": int(cov_cache_misses),
        "cov_cache_evictions": int(cov_cache_evictions),
        "max_workers": max_workers_value,
        "dtype": float_dtype.name,
    }


def _write_metrics(metrics_path: Path, results: Dict[str, Any]) -> None:
    with metrics_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        standard_metrics = [
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
            "te_target",
            "lambda_te",
            "gamma_turnover",
            "cov_cache_size",
            "cov_cache_hits",
            "cov_cache_misses",
            "cov_cache_evictions",
        ]
        for key in standard_metrics:
            writer.writerow([key, results.get(key)])

        baseline_metrics = [
            "baseline_sharpe",
            "baseline_info_ratio",
            "alpha_vs_baseline",
            "hit_rate_vs_baseline",
        ]
        for key in baseline_metrics:
            if key in results:
                writer.writerow([key, results.get(key)])


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


def _write_rebalance_report(path: Path, results: Dict[str, Any]) -> None:
    records: Sequence[Dict[str, Any]] = results.get("rebalance_records", [])  # type: ignore[assignment]
    header = [
        "date",
        "gross_ret",
        "net_tx_ret",
        "net_slip_ret",
        "turnover",
        "tx_cost",
        "slippage_cost",
        "sector_breaches",
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
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for record in records or []:
            row = {key: record.get(key) for key in header}
            writer.writerow(row)


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


def _extract_asset_names(frame: Any, n_assets: int) -> List[str]:
    if hasattr(frame, "columns") and getattr(frame, "columns") is not None:
        cols = [str(col) for col in frame.columns]
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="bt_out")
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
        "--refine-every",
        type=int,
        default=1,
        help="Run SLSQP refinement every k rebalances",
    )

    return parser


def main(args: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()

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

    float_dtype = np.float32 if parsed.float32 else np.float64
    df = _read_csv(Path(parsed.csv))
    benchmark_df = _read_csv(Path(parsed.benchmark_csv)) if parsed.benchmark_csv else None
    factor_panel = load_factor_panel(Path(parsed.factors)) if parsed.factors else None
    slippage_cfg = parse_slippage(parsed.slippage)
    factor_target_vec = None
    if parsed.factor_targets:
        if factor_panel is None:
            raise ValueError("Factor targets provided without factor loadings")
        factor_target_vec = load_factor_targets(Path(parsed.factor_targets), factor_panel.factor_names)

    benchmark_weights_spec = getattr(parsed, "benchmark_weights", None)
    active_group_spec = parsed.active_group_caps or None
    factor_bounds_spec = parsed.factor_bounds or None

    tx_cost_bps = float(parsed.tx_cost_bps) if parsed.tx_cost_mode != "none" else 0.0
    returns_arr_for_baseline = _frame_to_numpy(df, float_dtype)
    returns_arr_for_baseline = np.atleast_2d(returns_arr_for_baseline)
    asset_names_for_baseline = _extract_asset_names(df, returns_arr_for_baseline.shape[1])
    all_dates = _frame_index(df, returns_arr_for_baseline.shape[0])

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
            factor_align=parsed.factor_align,
            factors_required=parsed.factors_required,
            slippage=slippage_cfg,
            refine_every=parsed.refine_every,
            cov_model=parsed.cov_model,
            benchmark=benchmark_df,
            benchmark_weights=benchmark_weights_spec,
            active_min=parsed.active_min,
            active_max=parsed.active_max,
            active_group_caps=active_group_spec,
            factor_bounds=factor_bounds_spec,
            te_target=parsed.te_target,
            lambda_te=parsed.lambda_te,
            gamma_turnover=parsed.gamma_turnover,
            risk_free_rate=float(parsed.rf_bps) / 1e4,
            trading_days=parsed.trading_days,
            progress_callback=progress_printer,
            rebalance_callback=jsonl_writer.write if jsonl_writer else None,
            dtype=float_dtype,
            cov_cache_size=parsed.cache_cov,
            max_workers=parsed.max_workers,
        )
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
            te = float(np.std(active_baseline) * ann_factor)
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
        )
        return

    _write_metrics(out_dir / "metrics.csv", results)
    _write_rebalance_report(out_dir / "rebalance_report.csv", results)
    _write_equity(out_dir / "equity.csv", results)
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
    if parsed.save_weights:
        _write_weights(out_dir / "weights.csv", results)

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
    )

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
