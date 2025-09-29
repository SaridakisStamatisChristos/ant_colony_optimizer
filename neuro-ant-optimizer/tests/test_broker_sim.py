from __future__ import annotations

import time

import pytest

from neuro_ant_optimizer.live.broker import SimulatedBroker, ThrottleError
from neuro_ant_optimizer.state.positions import PositionsStore


def test_simulated_broker_idempotent_orders(tmp_path):
    store = PositionsStore(tmp_path / "positions.json")
    broker = SimulatedBroker(positions_store=store, throttle_window=0.0)

    first = broker.submit_target_weights(
        {"AAPL": 0.6, "MSFT": 0.4}, account_value=1_000_000, client_order_id="order-1"
    )
    second = broker.submit_target_weights(
        {"AAPL": 0.1}, account_value=1_000_000, client_order_id="order-1"
    )

    assert first == second
    positions = broker.load_positions()
    assert positions["AAPL"] == pytest.approx(600_000.0)
    assert positions["MSFT"] == pytest.approx(400_000.0)


def test_simulated_broker_throttles(tmp_path):
    store = PositionsStore(tmp_path / "positions.json")
    broker = SimulatedBroker(positions_store=store, throttle_window=0.01)

    broker.submit_target_weights({"AAPL": 1.0}, account_value=100.0, client_order_id="order-a")

    with pytest.raises(ThrottleError):
        broker.submit_target_weights({"AAPL": 0.5}, account_value=100.0, client_order_id="order-b")

    time.sleep(0.02)
    broker.submit_target_weights({"AAPL": 0.5}, account_value=100.0, client_order_id="order-c")
    assert broker.load_positions()["AAPL"] == pytest.approx(50.0)
