"""Utilities for computing deterministic run identifiers."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, MutableMapping, Sequence


def _default_serializer(value: Any) -> Any:
    """Best-effort conversion for non-JSON-native values."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - fallback for exotic objects
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover - fallback for exotic objects
            pass
    return str(value)


def _normalize_manifest(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    data: MutableMapping[str, Any] = dict(manifest)
    data.pop("run_id", None)
    return data


def _normalize_files(files: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    normalized = []
    for entry in files:
        name = str(entry.get("name"))
        sha256 = str(entry.get("sha256"))
        size = int(entry.get("size", 0))
        normalized.append({"name": name, "sha256": sha256, "size": size})
    normalized.sort(key=lambda item: item["name"])
    return normalized


def compute_run_id(manifest: Mapping[str, Any], files: Sequence[Mapping[str, Any]]) -> str:
    """Compute a stable hex digest for a run.

    Parameters
    ----------
    manifest:
        Mapping describing the manifest (the ``run_id`` key, if present, is ignored).
    files:
        Sequence describing produced artifacts with ``name``, ``sha256``, and ``size`` fields.
    """

    payload = {
        "manifest": _normalize_manifest(manifest),
        "files": _normalize_files(files),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_default_serializer)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


__all__ = ["compute_run_id"]
