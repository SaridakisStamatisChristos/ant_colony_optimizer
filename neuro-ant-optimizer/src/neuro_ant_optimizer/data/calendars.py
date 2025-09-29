"""Calendar utilities used by the data loader."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from typing import List, Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - minimal runtime
    pd = None  # type: ignore


class CalendarError(RuntimeError):
    """Raised when timestamps cannot be aligned to a calendar."""


@dataclass(frozen=True)
class CalendarSpec:
    freq: str

    def as_pandas(self) -> str:
        freq = self.freq.upper()
        if freq in {"D", "DAILY"}:
            return "D"
        if freq in {"B", "BUSINESS"}:
            return "B"
        if freq in {"W", "WEEKLY"}:
            return "W"
        if freq in {"M", "MONTHLY"}:
            return "M"
        raise CalendarError(f"Unsupported calendar frequency '{self.freq}'")


def normalize_timezone(tz: Optional[str]):
    if tz is None:
        return UTC
    if pd is not None:
        try:
            return pd.Timestamp.now(tz=tz).tzinfo  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - fallback path
            raise CalendarError(f"Unknown timezone '{tz}'") from exc
    import zoneinfo

    try:
        return zoneinfo.ZoneInfo(tz)
    except Exception as exc:  # pragma: no cover - fallback path
        raise CalendarError(f"Unknown timezone '{tz}'") from exc


def _ensure_datetime64(values: Sequence) -> List[np.datetime64]:
    converted: List[np.datetime64] = []
    for item in values:
        if isinstance(item, np.datetime64):
            converted.append(item)
            continue
        if pd is not None:
            converted.append(np.datetime64(pd.Timestamp(item)))
        else:
            converted.append(np.datetime64(str(item)))
    converted.sort()
    return converted


def align_calendar_index(
    index: Sequence,
    *,
    freq: str,
    tz: Optional[str],
) -> List[np.datetime64]:
    if not index:
        return []
    spec = CalendarSpec(freq=freq)
    pandas_freq = spec.as_pandas()
    if pd is None:
        aligned = _ensure_datetime64(index)
        day_array = np.array(aligned, dtype="datetime64[D]")
        if pandas_freq == "B":
            mask = np.is_busday(day_array)
            if not bool(np.all(mask)):
                bad = aligned[int(np.argmax(~mask))]
                raise CalendarError(f"Timestamp {bad} not aligned with {pandas_freq} calendar")
        elif pandas_freq == "W":
            diffs = np.diff(day_array)
            if diffs.size and not np.all(diffs == np.timedelta64(7, "D")):
                bad = aligned[int(np.argmax(diffs != np.timedelta64(7, "D")) + 1)]
                raise CalendarError(f"Timestamp {bad} not aligned with {pandas_freq} calendar")
        elif pandas_freq == "M":
            months = day_array.astype("datetime64[M]")
            diffs = np.diff(months)
            if diffs.size and not np.all(diffs == np.timedelta64(1, "M")):
                bad = aligned[int(np.argmax(diffs != np.timedelta64(1, "M")) + 1)]
                raise CalendarError(f"Timestamp {bad} not aligned with {pandas_freq} calendar")
        return aligned
    tzinfo = normalize_timezone(tz)
    dt_index = pd.DatetimeIndex(index)
    if dt_index.tzinfo is None:
        dt_index = dt_index.tz_localize(tzinfo)
    else:
        dt_index = dt_index.tz_convert(tzinfo)
    dt_index = dt_index.sort_values()
    expected = pd.date_range(dt_index[0], dt_index[-1], freq=pandas_freq, tz=tzinfo)
    if not dt_index.isin(expected).all():
        mismatch = dt_index[~dt_index.isin(expected)]
        first = mismatch[0] if len(mismatch) else dt_index[0]
        raise CalendarError(f"Timestamp {first} not aligned with {pandas_freq} calendar")
    return [np.datetime64(val.tz_convert(UTC)) for val in dt_index]


__all__ = ["align_calendar_index", "CalendarSpec", "CalendarError", "normalize_timezone"]
