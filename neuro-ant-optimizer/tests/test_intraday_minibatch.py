from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Mapping

import pytest

from neuro_ant_optimizer.intraday.engine import IntradayEngine
from neuro_ant_optimizer.state.positions import PositionsStore


@dataclass
class DummyFeed:
    payloads: Iterable[tuple[str, Mapping[str, float]]]

    def stream(self) -> Iterable[tuple[str, Mapping[str, float]]]:
        return iter(self.payloads)


def test_intraday_engine_uses_warm_start(tmp_path):
    store = PositionsStore(tmp_path / "positions.json")
    store.save({"AAPL": 0.5})

    batches = DummyFeed([
        ("09:30", {"AAPL": 0.1, "MSFT": 0.2}),
        ("09:31", {"AAPL": -0.05, "MSFT": 0.1}),
    ])

    warm_starts: list[Mapping[str, float]] = []

    def objective(minibatch: Mapping[str, float], warm_start: Mapping[str, float]):
        warm_starts.append(dict(warm_start))
        updated = dict(warm_start)
        for symbol, delta in minibatch.items():
            updated[symbol] = warm_start.get(symbol, 0.0) + delta
        return updated

    engine = IntradayEngine(batches, objective, state_store=store)
    result = engine.run()

    assert warm_starts[0]["AAPL"] == pytest.approx(0.5)
    assert result["AAPL"] == pytest.approx(0.55)
    assert result["MSFT"] == pytest.approx(0.3)
    persisted = store.load()
    assert persisted["AAPL"] == pytest.approx(result["AAPL"])
    assert persisted["MSFT"] == pytest.approx(result["MSFT"])


def test_intraday_engine_drop_overrun(tmp_path):
    store = PositionsStore(tmp_path / "positions.json")
    store.save({"AAPL": 1.0})

    batches = DummyFeed([("09:30", {"AAPL": -0.1})])

    def slow_objective(minibatch: Mapping[str, float], warm_start: Mapping[str, float]):
        time.sleep(0.005)
        return {"AAPL": warm_start.get("AAPL", 0.0) + minibatch["AAPL"]}

    engine = IntradayEngine(
        batches,
        slow_objective,
        state_store=store,
        latency_budget_ms=0.1,
        drop_overrun=True,
    )
    result = engine.run()

    assert result == {"AAPL": 1.0}
    assert engine.latency_events[0].dropped is True
    assert store.load() == {"AAPL": 1.0}
