from typing import Tuple

import pytest
from fastapi.testclient import TestClient

from service import metrics
from service.app import create_app


@pytest.fixture()
def client_fixture(tmp_path, monkeypatch) -> Tuple[TestClient, pytest.MonkeyPatch]:
    metrics.reset_metrics()
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("RUNS_DIR", str(runs_root))
    monkeypatch.setenv("SERVICE_AUTH_TOKEN", "health-token")
    app = create_app()
    with TestClient(app) as client:
        yield client, monkeypatch


def test_healthz_returns_ok(client_fixture):
    client, _ = client_fixture
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_success_and_failure(client_fixture):
    client, monkeypatch = client_fixture

    monkeypatch.setenv("REGISTRY_URL", "s3://registry")
    ready_response = client.get("/readyz")
    assert ready_response.status_code == 200
    ready_payload = ready_response.json()
    assert ready_payload["status"] == "ok"
    assert ready_payload["checks"]["runs_dir"]["ok"] is True
    assert ready_payload["checks"]["registry_url"]["ok"] is True

    monkeypatch.delenv("REGISTRY_URL", raising=False)
    not_ready = client.get("/readyz")
    assert not_ready.status_code == 503
    not_ready_payload = not_ready.json()
    assert not_ready_payload["status"] == "unavailable"
    assert not_ready_payload["checks"]["registry_url"]["ok"] is False
    assert "error" in not_ready_payload["checks"]["registry_url"]
