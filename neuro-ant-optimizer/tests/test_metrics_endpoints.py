import json
from pathlib import Path
from typing import Dict

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from service import metrics
from service.app import create_app


@pytest.fixture()
def client_with_runs(tmp_path, monkeypatch):
    metrics.reset_metrics()
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("RUNS_DIR", str(runs_root))
    monkeypatch.setenv("SERVICE_AUTH_TOKEN", "metrics-token")
    monkeypatch.setenv("REGISTRY_URL", "s3://dummy-bucket/registry")
    app = create_app()
    with TestClient(app) as client:
        yield client, runs_root


def _prepare_run(runs_root: Path, run_id: str) -> None:
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "window_duration_seconds": 7200,
        "total_runtime_seconds": 3600,
    }
    (run_dir / "run_config.json").write_text(json.dumps(manifest), encoding="utf-8")


def _scrape_metrics(client: TestClient) -> Dict[str, Dict[str, float]]:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(metrics.CONTENT_TYPE_LATEST)
    families = {}
    for family in text_string_to_metric_families(response.text):
        samples: Dict[str, float] = {}
        for sample in family.samples:
            samples[sample.name] = sample.value
        families[family.name] = samples
    return families


def test_metrics_increment_and_histograms(client_with_runs):
    client, runs_root = client_with_runs
    headers = {"Authorization": "Bearer metrics-token"}
    run_id = "metrics-run"
    _prepare_run(runs_root, run_id)

    # Initial scrape verifies metrics endpoint is wired.
    pre_metrics = _scrape_metrics(client)
    assert pre_metrics["runs_started_total"]["runs_started_total"] == 0
    assert pre_metrics["runs_succeeded_total"]["runs_succeeded_total"] == 0
    assert pre_metrics["runs_failed_total"]["runs_failed_total"] == 0

    response = client.post("/backtest", json={"csv": "data.csv"}, headers=headers)
    assert response.status_code == 202

    response = client.get(f"/runs/{run_id}", headers=headers)
    assert response.status_code == 200

    missing_response = client.get("/runs/does-not-exist", headers=headers)
    assert missing_response.status_code == 404

    post_metrics = _scrape_metrics(client)
    assert post_metrics["runs_started_total"]["runs_started_total"] == 1
    assert post_metrics["runs_succeeded_total"]["runs_succeeded_total"] == 1
    assert post_metrics["runs_failed_total"]["runs_failed_total"] == 1

    assert post_metrics["active_workers"]["active_workers"] == 0
    assert post_metrics["queue_depth"]["queue_depth"] == 0

    histogram = post_metrics["window_duration_seconds"]
    assert histogram["window_duration_seconds_count"] == 1
    assert histogram["window_duration_seconds_sum"] == pytest.approx(7200.0)

    runtime_histogram = post_metrics["total_runtime_seconds"]
    assert runtime_histogram["total_runtime_seconds_count"] == 1
    assert runtime_histogram["total_runtime_seconds_sum"] == pytest.approx(3600.0)
