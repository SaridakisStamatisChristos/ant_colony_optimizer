"""Market data loaders with point-in-time safeguards."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, MutableSequence, Optional, Sequence, Tuple, Union

import numpy as np

from .calendars import CalendarError, align_calendar_index, normalize_timezone

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - minimal runtime
    pd = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import polars as pl  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - minimal runtime
    pl = None  # type: ignore

FrameLike = Union["_BasicFrame", "pd.DataFrame"]


@dataclass
class LoaderConfig:
    """Runtime options for :func:`load_returns`."""

    freq: str
    tz: Optional[str]
    pit: bool = True
    backend: str = "auto"
    dropna: str = "none"
    rename: Optional[Mapping[str, str]] = None


class LoaderError(RuntimeError):
    """Raised when the loader encounters malformed input."""


class PITViolationError(LoaderError):
    """Raised when point-in-time rules are violated."""


class CalendarAlignmentError(LoaderError):
    """Raised when the timestamp index fails to align with the requested calendar."""


class _BasicFrame:
    """Minimal frame wrapper mimicking pandas APIs used by the backtester."""

    def __init__(self, data: np.ndarray, index: Sequence[np.datetime64], columns: Sequence[str]):
        self._data = np.asarray(data, dtype=float)
        self._index = list(index)
        self._columns = list(columns)

    def to_numpy(self, dtype=float) -> np.ndarray:
        return np.asarray(self._data, dtype=dtype)

    @property
    def index(self) -> Sequence[np.datetime64]:
        return list(self._index)

    @property
    def columns(self) -> Sequence[str]:
        return list(self._columns)

    def __len__(self) -> int:  # pragma: no cover - trivial helper
        return len(self._index)


def _coerce_backend(config: LoaderConfig) -> str:
    backend = config.backend.lower().strip()
    if backend not in {"auto", "pandas", "polars"}:
        raise LoaderError(f"Unsupported IO backend '{config.backend}'")
    if backend == "polars" and pl is None:
        raise LoaderError("Polars backend requested but polars is not installed")
    if backend == "pandas" and pd is None:
        raise LoaderError("Pandas backend requested but pandas is not installed")
    if backend == "auto":
        if pl is not None:
            return "polars"
        if pd is not None:
            return "pandas"
        return "basic"
    return backend


def _read_with_polars(path: Path) -> Tuple[np.ndarray, Sequence, Sequence[str]]:
    if pl is None:  # pragma: no cover - defensive
        raise LoaderError("Polars backend not available")
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pl.read_parquet(path)
    else:
        frame = pl.read_csv(path, try_parse_dates=True)
    if not frame.width:
        return np.empty((0, 0), dtype=float), [], []
    lower_cols = [col.lower() for col in frame.columns]
    if "date" in lower_cols:
        date_col = frame.columns[lower_cols.index("date")]
    else:
        date_col = frame.columns[0]
    value_cols = [col for col in frame.columns if col != date_col]
    if not value_cols:
        return np.empty((0, 0), dtype=float), [], []
    arr = frame.select(value_cols).to_numpy().astype(float, copy=False)
    dates = frame.select(date_col).to_series().to_list()
    return arr, dates, value_cols


def _read_with_pandas(path: Path) -> Tuple[np.ndarray, Sequence, Sequence[str]]:
    if pd is None:  # pragma: no cover - defensive
        raise LoaderError("Pandas backend not available")
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, parse_dates=True)
    if frame.empty:
        return np.empty((0, 0), dtype=float), [], []
    if frame.index.name is None or frame.index.name.lower() not in {"date", "timestamp"}:
        if frame.columns.size and frame.columns[0].lower() in {"date", "timestamp"}:
            frame = frame.set_index(frame.columns[0])
        else:
            frame = frame.set_index(frame.columns[0])
    values = frame.to_numpy(dtype=float, copy=False)
    return values, list(frame.index), [str(col) for col in frame.columns]


def _read_basic(path: Path) -> Tuple[np.ndarray, Sequence, Sequence[str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = [row.strip() for row in fh]
    if not rows:
        return np.empty((0, 0), dtype=float), [], []
    header = rows[0].split(",")
    cols = [col.strip() for col in header[1:]]
    data: MutableSequence[np.ndarray] = []
    dates: MutableSequence[str] = []
    for line in rows[1:]:
        if not line:
            continue
        tokens = [token.strip() for token in line.split(",")]
        if len(tokens) <= 1:
            continue
        dates.append(tokens[0])
        row: MutableSequence[float] = []
        for tok in tokens[1:]:
            if tok == "":
                row.append(float("nan"))
            else:
                row.append(float(tok))
        data.append(np.array(row, dtype=float))
    if not data:
        return np.empty((0, 0), dtype=float), [], cols
    matrix = np.vstack(data)
    return matrix, dates, cols


def _filter_assets(matrix: np.ndarray, columns: Sequence[str], assets: Optional[Sequence[str]]):
    if not assets:
        return matrix, list(columns)
    wanted = [str(asset) for asset in assets]
    idx_map = {str(col): i for i, col in enumerate(columns)}
    missing = [name for name in wanted if name not in idx_map]
    if missing:
        raise LoaderError(f"Requested assets missing from file: {', '.join(missing)}")
    indices = [idx_map[name] for name in wanted]
    return matrix[:, indices], wanted


def _rename_columns(columns: Sequence[str], rename: Optional[Mapping[str, str]]) -> Sequence[str]:
    if not rename:
        return [str(col) for col in columns]
    renamed = []
    for col in columns:
        key = str(col)
        renamed.append(str(rename.get(key, key)))
    return renamed


def _apply_dropna(matrix: np.ndarray, dropna: str) -> Tuple[np.ndarray, np.ndarray]:
    mode = dropna.lower()
    if mode not in {"none", "any", "all"}:
        raise LoaderError("dropna must be 'none', 'any', or 'all'")
    if mode == "none":
        mask = np.ones(matrix.shape[0], dtype=bool)
        return matrix, mask
    if mode == "any":
        mask = ~np.any(np.isnan(matrix), axis=1)
    else:
        mask = ~np.all(np.isnan(matrix), axis=1)
    return matrix[mask], mask


def _assert_no_duplicates(index: Sequence[np.datetime64]) -> None:
    seen = set()
    for value in index:
        key = np.datetime64(value)
        if key in seen:
            raise LoaderError(f"Duplicate timestamp detected: {value}")
        seen.add(key)


def load_returns(
    path: Union[str, Path],
    freq: str,
    tz: Optional[str],
    pit: bool = True,
    columns: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
    *,
    dropna: str = "none",
    rename: Optional[Mapping[str, str]] = None,
    assets: Optional[Sequence[str]] = None,
    backend: str = "auto",
) -> FrameLike:
    """Load a return panel ensuring calendar alignment and PIT safety."""

    if not isinstance(path, (str, Path)):
        raise TypeError("path must be str or Path")
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Returns file not found: {source}")

    config = LoaderConfig(freq=freq, tz=tz, pit=pit, backend=backend, dropna=dropna)
    backend_mode = _coerce_backend(config)

    if isinstance(columns, Mapping):
        rename = dict(columns)
        columns_seq: Optional[Sequence[str]] = list(columns.keys())
    else:
        columns_seq = list(columns) if columns is not None else None

    if backend_mode == "polars":
        matrix, idx_raw, cols = _read_with_polars(source)
    elif backend_mode == "pandas":
        matrix, idx_raw, cols = _read_with_pandas(source)
    else:
        matrix, idx_raw, cols = _read_basic(source)

    if columns_seq:
        matrix, cols = _filter_assets(matrix, cols, columns_seq)
    elif assets:
        matrix, cols = _filter_assets(matrix, cols, assets)

    cols = _rename_columns(cols, rename)

    if matrix.ndim != 2:
        raise LoaderError("Loaded returns must be a 2D matrix")

    matrix = matrix.astype(float, copy=False)

    dropna_matrix, mask = _apply_dropna(matrix, dropna)
    idx_raw = [idx_raw[i] for i, flag in enumerate(mask) if flag]

    try:
        aligned_index = align_calendar_index(idx_raw, freq=freq, tz=tz)
    except CalendarError as exc:
        raise CalendarAlignmentError(str(exc)) from exc
    _assert_no_duplicates(aligned_index)
    if pit and aligned_index:
        last_dt = aligned_index[-1]
        tzinfo = normalize_timezone(tz)
        now_val = datetime.now(tzinfo)
        now_cmp = np.datetime64(now_val.astimezone(UTC))
        if np.datetime64(last_dt) > now_cmp:
            raise PITViolationError(
                f"Last observation {last_dt} is in the future relative to timezone {tz or 'UTC'}"
            )

    if backend_mode == "pandas" and pd is not None:
        index = pd.DatetimeIndex(aligned_index)
        frame = pd.DataFrame(dropna_matrix, index=index, columns=cols)
        return frame

    return _BasicFrame(dropna_matrix, aligned_index, cols)


__all__ = [
    "load_returns",
    "LoaderConfig",
    "LoaderError",
    "PITViolationError",
    "CalendarAlignmentError",
]
