import hashlib

from neuro_ant_optimizer.runid import compute_run_id


def test_run_id_deterministic() -> None:
    manifest = {"args": {"csv": "data.csv"}, "timestamp": "2024-05-01T00:00:00Z", "git_sha": "abc123"}
    files = [
        {"name": "results.csv", "sha256": hashlib.sha256(b"data").hexdigest(), "size": 4},
        {"name": "metrics.json", "sha256": hashlib.sha256(b"{}").hexdigest(), "size": 2},
    ]
    first = compute_run_id(manifest, files)
    second = compute_run_id({**manifest, "run_id": "ignored"}, list(reversed(files)))
    assert first == second
    assert len(first) == 64


def test_run_id_changes_when_artifacts_change() -> None:
    manifest = {"args": {"csv": "data.csv"}, "timestamp": "2024-05-01T00:00:00Z"}
    files = [{"name": "results.csv", "sha256": "1", "size": 4}]
    alt_files = [{"name": "results.csv", "sha256": "2", "size": 4}]
    assert compute_run_id(manifest, files) != compute_run_id(manifest, alt_files)
