from __future__ import annotations

import gc
import time

import numpy as np
import psutil
import pytest

from neuro_ant_optimizer.backtest.backtest import _turnover_penalty_components


def _baseline_penalty(prev: np.ndarray, current: np.ndarray, gamma: np.ndarray):
    diffs: list[list[float]] = []
    total = 0.0
    penalty = 0.0
    for value, base, weight in zip(current, prev, gamma):
        diff = abs(float(value) - float(base))
        total += diff
        penalty += diff * float(weight)
    diffs.append([diff for diff in np.abs(current - prev)])
    return total, penalty, diffs


def _optimized_penalty(
    prev: np.ndarray, current: np.ndarray, gamma: np.ndarray, buffer: np.ndarray
) -> tuple[float, float]:
    total, penalty, _ = _turnover_penalty_components(prev, current, gamma, out=buffer)
    return float(total), float(penalty)


def test_turnover_penalty_speed_and_memory() -> None:
    rng = np.random.default_rng(7)
    n_rows = 100_000
    n_assets = 200
    prev = rng.random(n_assets)
    gamma = rng.random(n_assets) * 0.5
    curr_matrix = rng.random((n_rows, n_assets))

    process = psutil.Process()

    gc.collect()
    rss_before = process.memory_info().rss
    baseline_total = 0.0
    baseline_penalty = 0.0
    cached_diffs = []
    start = time.perf_counter()
    for row in curr_matrix:
        total, penalty, diffs = _baseline_penalty(prev, row, gamma)
        baseline_total += total
        baseline_penalty += penalty
        cached_diffs.extend(diffs)
    baseline_time = time.perf_counter() - start
    rss_after_baseline = process.memory_info().rss

    buffer = np.empty_like(prev, dtype=float)
    gc.collect()
    rss_mid = process.memory_info().rss
    opt_total = 0.0
    opt_penalty = 0.0
    start = time.perf_counter()
    for row in curr_matrix:
        total, penalty = _optimized_penalty(prev, row, gamma, buffer)
        opt_total += total
        opt_penalty += penalty
    opt_time = time.perf_counter() - start
    rss_after_opt = process.memory_info().rss

    baseline_mem = max(rss_after_baseline - rss_before, 0)
    opt_mem = max(rss_after_opt - rss_mid, 0)

    assert opt_total == pytest.approx(baseline_total)
    assert opt_penalty == pytest.approx(baseline_penalty)
    assert opt_time <= baseline_time * 0.75
    assert opt_mem <= baseline_mem

    cached_diffs.clear()
