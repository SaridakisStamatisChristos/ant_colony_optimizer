"""Minimal Prometheus client implementation for offline testing."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


@dataclass(frozen=True)
class Sample:
    name: str
    labels: Mapping[str, str]
    value: float


@dataclass(frozen=True)
class Metric:
    name: str
    documentation: str
    type: str
    samples: List[Sample]


class CollectorRegistry:
    """Registry maintaining metric instances."""

    def __init__(self) -> None:
        self._metrics: "OrderedDict[str, _MetricBase]" = OrderedDict()

    def register(self, metric: "_MetricBase") -> None:
        self._metrics[metric.name] = metric

    def collect(self) -> Iterable[Metric]:
        for metric in self._metrics.values():
            yield metric.as_metric()


class _MetricBase:
    def __init__(self, name: str, documentation: str, metric_type: str, registry: CollectorRegistry) -> None:
        self.name = name
        self.documentation = documentation
        self.metric_type = metric_type
        registry.register(self)

    def as_metric(self) -> Metric:
        return Metric(self.name, self.documentation, self.metric_type, self._samples())

    def _samples(self) -> List[Sample]:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError


class Counter(_MetricBase):
    def __init__(self, name: str, documentation: str, registry: CollectorRegistry) -> None:
        self._value = 0.0
        super().__init__(name, documentation, "counter", registry)

    def inc(self, amount: float = 1.0) -> None:
        if amount < 0:
            raise ValueError("Counters can only increase")
        self._value += amount

    def _samples(self) -> List[Sample]:
        return [Sample(self.name, {}, float(self._value))]


class Gauge(_MetricBase):
    def __init__(self, name: str, documentation: str, registry: CollectorRegistry) -> None:
        self._value = 0.0
        super().__init__(name, documentation, "gauge", registry)

    def set(self, value: float) -> None:
        self._value = float(value)

    def _samples(self) -> List[Sample]:
        return [Sample(self.name, {}, float(self._value))]


class Histogram(_MetricBase):
    DEFAULT_BUCKETS: Sequence[float] = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
        30.0,
        60.0,
        120.0,
        300.0,
        600.0,
        900.0,
        1200.0,
        1800.0,
        3600.0,
        7200.0,
        14400.0,
    )

    def __init__(
        self,
        name: str,
        documentation: str,
        registry: CollectorRegistry,
        buckets: Sequence[float] | None = None,
    ) -> None:
        self._sum = 0.0
        self._count = 0
        bucket_bounds = sorted(set(buckets or self.DEFAULT_BUCKETS))
        self._buckets: List[float] = list(bucket_bounds)
        self._bucket_counts: List[int] = [0 for _ in self._buckets]
        super().__init__(name, documentation, "histogram", registry)

    def observe(self, amount: float) -> None:
        if amount < 0:
            return
        value = float(amount)
        self._sum += value
        self._count += 1
        for index, bound in enumerate(self._buckets):
            if value <= bound:
                self._bucket_counts[index] += 1

    def _samples(self) -> List[Sample]:
        samples: List[Sample] = []
        for bound, count in zip(self._buckets, self._bucket_counts):
            samples.append(
                Sample(
                    f"{self.name}_bucket",
                    {"le": _format_bound(bound)},
                    float(count),
                )
            )
        samples.append(Sample(f"{self.name}_bucket", {"le": "+Inf"}, float(self._count)))
        samples.append(Sample(f"{self.name}_sum", {}, float(self._sum)))
        samples.append(Sample(f"{self.name}_count", {}, float(self._count)))
        return samples


def _format_bound(bound: float) -> str:
    return format(bound, "g")


def _format_value(value: float) -> str:
    return format(value, ".10g")


def generate_latest(registry: CollectorRegistry) -> bytes:
    lines: List[str] = []
    for metric in registry.collect():
        lines.append(f"# HELP {metric.name} {metric.documentation}")
        lines.append(f"# TYPE {metric.name} {metric.type}")
        for sample in metric.samples:
            label_str = ""
            if sample.labels:
                label_parts = ",".join(
                    f"{key}=\"{value}\"" for key, value in sorted(sample.labels.items())
                )
                label_str = f"{{{label_parts}}}"
            lines.append(f"{sample.name}{label_str} {_format_value(sample.value)}")
    lines.append("")
    return "\n".join(lines).encode("utf-8")


__all__ = [
    "CollectorRegistry",
    "Counter",
    "Gauge",
    "Histogram",
    "Metric",
    "Sample",
    "CONTENT_TYPE_LATEST",
    "generate_latest",
]
