"""Very small parser for Prometheus text format used in tests."""

from __future__ import annotations

from typing import Dict, Iterator, List

from . import Metric, Sample


def text_string_to_metric_families(data: str) -> Iterator[Metric]:
    current_name: str | None = None
    documentation = ""
    metric_type = ""
    samples: List[Sample] = []
    for line in data.splitlines():
        if not line:
            continue
        if line.startswith("# HELP "):
            if current_name is not None:
                yield Metric(current_name, documentation, metric_type, samples)
                samples = []
            parts = line.split(" ", 3)
            current_name = parts[2]
            documentation = parts[3] if len(parts) > 3 else ""
        elif line.startswith("# TYPE "):
            parts = line.split(" ", 3)
            metric_type = parts[3] if len(parts) > 3 else ""
        else:
            name_and_labels, value_str = line.split(" ", 1)
            name, labels = _parse_name_and_labels(name_and_labels)
            samples.append(Sample(name, labels, float(value_str.strip())))
    if current_name is not None:
        yield Metric(current_name, documentation, metric_type, samples)


def _parse_name_and_labels(fragment: str) -> tuple[str, Dict[str, str]]:
    if "{" not in fragment:
        return fragment, {}
    name, remainder = fragment.split("{", 1)
    label_body = remainder.rstrip("}")
    labels: Dict[str, str] = {}
    if label_body:
        for pair in label_body.split(","):
            key, value = pair.split("=", 1)
            labels[key] = value.strip('"')
    return name, labels


__all__ = ["text_string_to_metric_families"]
