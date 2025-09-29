"""Artifact registry adapters."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional
from urllib.parse import urlparse


class RegistryError(RuntimeError):
    """Raised when registry operations fail."""


@dataclass
class RegistryCredentials:
    endpoint_url: Optional[str]
    bucket: str
    prefix: str
    access_key: Optional[str]
    secret_key: Optional[str]
    region: Optional[str]


class RegistryAdapter:
    """Abstract base class for artifact registries."""

    def upload_artifacts(self, run_dir: Path) -> Dict[str, str]:  # pragma: no cover - interface
        raise NotImplementedError


class LocalRegistry(RegistryAdapter):
    """Registry implementation that copies artifacts to a local directory."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def upload_artifacts(self, run_dir: Path) -> Dict[str, str]:
        run_dir = run_dir.resolve()
        manifest_path = run_dir / "run_config.json"
        if not manifest_path.exists():
            raise RegistryError("run_config.json is required to upload artifacts")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = manifest.get("run_id")
        if not run_id:
            raise RegistryError("run_config.json is missing a run_id field")

        index_path = run_dir / "artifact_index.json"
        if not index_path.exists():
            raise RegistryError("artifact_index.json is required for uploads")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(index, list):
            raise RegistryError("artifact_index.json must contain a list of artifacts")

        dest_dir = self._root / str(run_id)
        dest_dir.mkdir(parents=True, exist_ok=True)

        files = {"run_config.json", "artifact_index.json"}
        for entry in index:
            if isinstance(entry, Mapping):
                files.add(str(entry.get("name")))

        urls: Dict[str, str] = {}
        for name in sorted(files):
            source = run_dir / name
            if not source.exists():
                continue
            target = dest_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            urls[name] = target.as_uri()

        return urls


def _parse_s3_credentials(url: str) -> RegistryCredentials:
    parsed = urlparse(url)
    scheme = parsed.scheme or "file"
    path = parsed.path.lstrip("/")
    if scheme == "s3":
        bucket = parsed.netloc or os.getenv("REGISTRY_BUCKET")
        if not bucket:
            raise RegistryError("REGISTRY_BUCKET must be set for s3 registries")
        return RegistryCredentials(
            endpoint_url=None,
            bucket=bucket,
            prefix=path,
            access_key=os.getenv("REGISTRY_ACCESS_KEY"),
            secret_key=os.getenv("REGISTRY_SECRET_KEY"),
            region=os.getenv("REGISTRY_REGION"),
        )

    if scheme == "minio":
        if not parsed.netloc:
            raise RegistryError("minio registry URLs must include host:port")
        components = path.split("/", 1)
        if not components or not components[0]:
            raise RegistryError("minio registry URLs must include a bucket name")
        bucket = components[0]
        prefix = components[1] if len(components) > 1 else ""
        endpoint = f"http://{parsed.netloc}"
        return RegistryCredentials(
            endpoint_url=endpoint,
            bucket=bucket,
            prefix=prefix,
            access_key=os.getenv("REGISTRY_ACCESS_KEY"),
            secret_key=os.getenv("REGISTRY_SECRET_KEY"),
            region=os.getenv("REGISTRY_REGION"),
        )

    raise RegistryError(f"Unsupported registry scheme: {scheme}")


def _load_s3_registry(url: str) -> RegistryAdapter:
    credentials = _parse_s3_credentials(url)
    from .s3 import S3Registry  # imported lazily to avoid optional dependency at import time

    return S3Registry(credentials)


def _resolve_registry(url: str) -> RegistryAdapter:
    parsed = urlparse(url)
    scheme = parsed.scheme or "file"
    if scheme in {"", "file"}:
        path = Path(parsed.path or url)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return LocalRegistry(path)
    if scheme in {"s3", "minio"}:
        return _load_s3_registry(url)
    raise RegistryError(f"Unsupported registry scheme: {scheme}")


def get_registry() -> Optional[RegistryAdapter]:
    url = os.getenv("REGISTRY_URL")
    if not url:
        return None
    return _resolve_registry(url)


def upload_artifacts(run_dir: Path) -> Path:
    registry = get_registry()
    if registry is None:
        raise RegistryError("REGISTRY_URL is not configured")

    run_dir = run_dir.resolve()
    urls = registry.upload_artifacts(run_dir)
    urls_path = run_dir / "urls.json"
    urls_path.write_text(json.dumps(urls, indent=2, sort_keys=True), encoding="utf-8")
    return urls_path


__all__ = [
    "LocalRegistry",
    "RegistryAdapter",
    "RegistryError",
    "get_registry",
    "upload_artifacts",
]
