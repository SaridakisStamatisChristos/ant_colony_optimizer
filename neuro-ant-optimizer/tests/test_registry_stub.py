import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import pytest

from neuro_ant_optimizer.registry import RegistryError, upload_artifacts


def _uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    return Path(parsed.path)


def test_local_registry_upload(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sample_path = run_dir / "metrics.csv"
    sample_bytes = b"metric,value\nalpha,1\n"
    sample_path.write_bytes(sample_bytes)

    manifest = {"run_id": "run-001"}
    manifest_path = run_dir / "run_config.json"
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)

    artifact_index = [
        {
            "name": "metrics.csv",
            "sha256": hashlib.sha256(sample_bytes).hexdigest(),
            "size": len(sample_bytes),
        },
        {
            "name": "run_config.json",
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "size": len(manifest_bytes),
        },
    ]
    (run_dir / "artifact_index.json").write_text(
        json.dumps(artifact_index, indent=2, sort_keys=True), encoding="utf-8"
    )

    registry_root = tmp_path / "registry"
    monkeypatch.setenv("REGISTRY_URL", str(registry_root))

    urls_path = upload_artifacts(run_dir)
    urls = json.loads(urls_path.read_text(encoding="utf-8"))
    assert "metrics.csv" in urls

    copied_metric = _uri_to_path(urls["metrics.csv"])
    assert copied_metric.read_bytes() == sample_bytes
    assert copied_metric.parent.parent == registry_root.resolve()


def test_upload_requires_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("REGISTRY_URL", raising=False)
    with pytest.raises(RegistryError):
        upload_artifacts(tmp_path)
