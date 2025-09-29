"""Prometheus metrics helpers for the FastAPI service."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Final

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

_LOGGER = logging.getLogger(__name__)


def _should_enable_otel() -> bool:
    flag = os.getenv("SERVICE_ENABLE_OTEL", "0").strip().lower()
    return flag in {"1", "true", "yes"}


def configure_optional_tracing(app: object) -> None:
    """Enable FastAPI OpenTelemetry instrumentation when available.

    The dependency is optional; failures to import simply log and continue.
    """

    if not _should_enable_otel():
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - best effort instrumentation
        _LOGGER.exception("Failed to enable OpenTelemetry instrumentation")


@dataclass
class _MetricsState:
    registry: CollectorRegistry
    runs_started: Counter
    runs_succeeded: Counter
    runs_failed: Counter
    active_workers: Gauge
    queue_depth: Gauge
    window_duration_seconds: Histogram
    total_runtime_seconds: Histogram


def _build_state() -> _MetricsState:
    registry = CollectorRegistry()
    runs_started = Counter(
        "runs_started_total",
        "Number of backtest runs accepted for processing.",
        registry=registry,
    )
    runs_succeeded = Counter(
        "runs_succeeded_total",
        "Number of backtest runs with successfully fetched manifests.",
        registry=registry,
    )
    runs_failed = Counter(
        "runs_failed_total",
        "Number of backtest runs that could not be retrieved.",
        registry=registry,
    )
    active_workers = Gauge(
        "active_workers",
        "Count of active worker processes handling backtests.",
        registry=registry,
    )
    queue_depth = Gauge(
        "queue_depth",
        "Depth of the backtest submission queue.",
        registry=registry,
    )
    window_duration_seconds = Histogram(
        "window_duration_seconds",
        "Observed optimization window durations from completed runs.",
        registry=registry,
    )
    total_runtime_seconds = Histogram(
        "total_runtime_seconds",
        "Observed end-to-end runtime for completed runs.",
        registry=registry,
    )

    # Ensure gauges have an explicit starting point.
    active_workers.set(0)
    queue_depth.set(0)

    return _MetricsState(
        registry=registry,
        runs_started=runs_started,
        runs_succeeded=runs_succeeded,
        runs_failed=runs_failed,
        active_workers=active_workers,
        queue_depth=queue_depth,
        window_duration_seconds=window_duration_seconds,
        total_runtime_seconds=total_runtime_seconds,
    )


_STATE: _MetricsState = _build_state()


def reset_metrics() -> None:
    """Reset the metrics registry for deterministic testing."""

    global _STATE
    _STATE = _build_state()


def registry() -> CollectorRegistry:
    return _STATE.registry


def mark_run_started() -> None:
    _STATE.runs_started.inc()


def mark_run_succeeded() -> None:
    _STATE.runs_succeeded.inc()


def mark_run_failed() -> None:
    _STATE.runs_failed.inc()


def set_active_workers(value: float) -> None:
    _STATE.active_workers.set(value)


def set_queue_depth(value: float) -> None:
    _STATE.queue_depth.set(value)


def observe_window_duration(duration: float) -> None:
    if duration >= 0:
        _STATE.window_duration_seconds.observe(duration)


def observe_total_runtime(duration: float) -> None:
    if duration >= 0:
        _STATE.total_runtime_seconds.observe(duration)


def render_latest() -> bytes:
    return generate_latest(_STATE.registry)


__all__: Final = [
    "CONTENT_TYPE_LATEST",
    "configure_optional_tracing",
    "mark_run_failed",
    "mark_run_started",
    "mark_run_succeeded",
    "observe_total_runtime",
    "observe_window_duration",
    "registry",
    "render_latest",
    "reset_metrics",
    "set_active_workers",
    "set_queue_depth",
]
