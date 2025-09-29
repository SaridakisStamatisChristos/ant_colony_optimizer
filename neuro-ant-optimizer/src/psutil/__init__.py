"""Minimal psutil stub used for testing in offline environments."""
from __future__ import annotations

import os
import resource
import sys
from dataclasses import dataclass


@dataclass
class _MemoryInfo:
    rss: int


class Process:
    """Subset of :mod:`psutil.Process` used in tests."""

    def __init__(self, pid: int | None = None) -> None:
        if pid is not None and pid != os.getpid():
            raise NotImplementedError("psutil stub only supports the current process")
        self._pid = os.getpid()

    def memory_info(self) -> _MemoryInfo:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_kb = usage.ru_maxrss
        if sys.platform == "darwin":
            rss_bytes = int(rss_kb)
        else:
            rss_bytes = int(rss_kb * 1024)
        return _MemoryInfo(rss=rss_bytes)

    def oneshot(self):  # pragma: no cover - simple helper for compatibility
        class _DummyContext:
            def __enter__(self_inner):
                return self

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _DummyContext()


__all__ = ["Process"]
