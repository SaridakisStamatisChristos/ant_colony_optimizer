"""S3-compatible artifact registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from . import RegistryAdapter, RegistryCredentials, RegistryError


class S3Registry(RegistryAdapter):
    """Upload artifacts to an S3 or MinIO bucket."""

    def __init__(self, credentials: RegistryCredentials) -> None:
        try:
            import boto3
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RegistryError("boto3 is required for S3 uploads") from exc

        session = boto3.session.Session(
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            region_name=credentials.region,
        )
        self._client = session.client("s3", endpoint_url=credentials.endpoint_url)
        self._bucket = credentials.bucket
        self._prefix = credentials.prefix.strip("/")

    def _object_key(self, run_id: str, name: str) -> str:
        segments = [segment for segment in [self._prefix, run_id, name] if segment]
        return "/".join(segments)

    def upload_artifacts(self, run_dir: Path) -> Dict[str, str]:
        run_dir = run_dir.resolve()
        manifest = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        run_id = manifest.get("run_id")
        if not run_id:
            raise RegistryError("run_config.json must contain run_id for uploads")
        index = json.loads((run_dir / "artifact_index.json").read_text(encoding="utf-8"))
        if not isinstance(index, list):
            raise RegistryError("artifact_index.json must contain a list of artifacts")

        files = {"run_config.json", "artifact_index.json"}
        for entry in index:
            if isinstance(entry, dict) and "name" in entry:
                files.add(str(entry["name"]))

        urls: Dict[str, str] = {}
        for name in sorted(files):
            source = run_dir / name
            if not source.exists():
                continue
            key = self._object_key(str(run_id), name)
            self._client.upload_file(str(source), self._bucket, key)
            urls[name] = f"s3://{self._bucket}/{key}"
        return urls


__all__ = ["S3Registry"]
