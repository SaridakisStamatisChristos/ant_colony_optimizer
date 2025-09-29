import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from service.app import create_app

pytestmark = pytest.mark.slow


@pytest.fixture()
def client(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("RUNS_DIR", str(runs_root))
    monkeypatch.setenv("SERVICE_AUTH_TOKEN", "secret-token")
    app = create_app()
    with TestClient(app) as client:
        yield client


def _prepare_run(runs_root: Path, run_id: str) -> None:
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_dir / "output.txt"
    artifact.write_text("hello", encoding="utf-8")
    manifest = {"run_id": run_id, "artifact_index": "artifact_index.json", "signatures": {}}
    (run_dir / "run_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    artifact_index = [{"name": "output.txt", "sha256": "dummy", "size": artifact.stat().st_size}]
    (run_dir / "artifact_index.json").write_text(
        json.dumps(artifact_index, indent=2), encoding="utf-8"
    )
    urls = {"remote.bin": "s3://bucket/remote.bin"}
    (run_dir / "urls.json").write_text(json.dumps(urls), encoding="utf-8")


def test_service_end_to_end(client, tmp_path, monkeypatch):
    runs_root = Path(os.environ["RUNS_DIR"])
    run_id = "run-service"
    _prepare_run(runs_root, run_id)

    # Unauthorized access fails
    response = client.get(f"/runs/{run_id}")
    assert response.status_code == 401

    headers = {"Authorization": "Bearer secret-token"}
    response = client.post("/backtest", json={"csv": "data.csv"}, headers=headers)
    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "accepted"
    assert "run_id" in payload

    response = client.get(f"/runs/{run_id}", headers=headers)
    assert response.status_code == 200
    manifest = response.json()
    assert manifest["run_id"] == run_id
    assert manifest["artifact_index_entries"][0]["name"] == "output.txt"

    response = client.get(f"/artifacts/{run_id}/output.txt", headers=headers)
    assert response.status_code == 200
    assert response.content == b"hello"

    response = client.get(f"/artifacts/{run_id}/remote.bin", headers=headers)
    assert response.status_code == 200
    assert response.json()["location"] == "s3://bucket/remote.bin"
