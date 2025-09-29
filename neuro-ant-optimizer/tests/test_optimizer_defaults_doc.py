from __future__ import annotations

import re
from importlib import import_module
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "field",
    ["n_ants", "max_iter", "topk_refine"],
)
def test_optimizer_defaults_synced_with_docs(field: str) -> None:
    optimizer_mod = import_module("neuro_ant_optimizer.optimizer")
    config = optimizer_mod.OptimizerConfig()
    expected = getattr(config, field)

    doc_path = Path("docs/optimizer_behavior_report.md")
    text = doc_path.read_text(encoding="utf-8")
    match = re.search(rf"\|\s+`{field}`\s+\|\s+([^|]+)\|", text)
    assert match, f"Documentation missing row for {field}"
    documented = match.group(1).strip()
    assert str(expected) == documented, (
        f"Documentation drift for {field}: expected {expected}, found {documented}"
    )
