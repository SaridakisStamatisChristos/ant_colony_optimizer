import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from neuro_ant_optimizer.data import (
    CalendarAlignmentError,
    LoaderError,
    PITViolationError,
    load_returns,
)


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def test_loader_rejects_future_dates(tmp_path: Path) -> None:
    tomorrow = datetime.now(tz=UTC) + timedelta(days=2)
    rows = [
        ["date", "A", "B"],
        [datetime.now(tz=UTC).strftime("%Y-%m-%d"), "0.01", "0.02"],
        [tomorrow.strftime("%Y-%m-%d"), "0.03", "0.01"],
    ]
    csv_path = tmp_path / "future.csv"
    _write_csv(csv_path, rows)

    with pytest.raises(PITViolationError):
        load_returns(csv_path, freq="B", tz="UTC")


def test_loader_duplicate_dates(tmp_path: Path) -> None:
    date_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    rows = [
        ["date", "A"],
        [date_str, "0.01"],
        [date_str, "0.02"],
    ]
    csv_path = tmp_path / "dup.csv"
    _write_csv(csv_path, rows)

    with pytest.raises(LoaderError):
        load_returns(csv_path, freq="B", tz="UTC")


def test_loader_calendar_alignment(tmp_path: Path) -> None:
    # Sunday should violate a business-day calendar
    rows = [
        ["date", "A"],
        ["2024-03-01", "0.01"],  # Friday
        ["2024-03-02", "0.02"],  # Saturday
    ]
    csv_path = tmp_path / "calendar.csv"
    _write_csv(csv_path, rows)

    with pytest.raises(CalendarAlignmentError):
        load_returns(csv_path, freq="B", tz="UTC")


def test_loader_dropna_policy(tmp_path: Path) -> None:
    rows = [
        ["date", "A", "B"],
        ["2024-03-01", "0.01", ""],
        ["2024-03-04", "0.02", "0.03"],
    ]
    csv_path = tmp_path / "dropna.csv"
    _write_csv(csv_path, rows)

    frame = load_returns(csv_path, freq="B", tz="UTC", dropna="any")
    data = frame.to_numpy(dtype=float)
    assert data.shape == (1, 2)
    np.testing.assert_allclose(data[0], [0.02, 0.03])
