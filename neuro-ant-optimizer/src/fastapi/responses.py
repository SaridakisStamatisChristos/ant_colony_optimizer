"""Minimal response helpers for the FastAPI stub."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class Response:
    def __init__(self, content: bytes = b"", status_code: int = 200, headers: Optional[Dict[str, str]] = None) -> None:
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")

    def json(self) -> Any:
        return json.loads(self.text)


class JSONResponse(Response):
    def __init__(self, data: Any, status_code: int = 200) -> None:
        content = json.dumps(data).encode("utf-8")
        super().__init__(content=content, status_code=status_code, headers={"content-type": "application/json"})


class FileResponse(Response):
    def __init__(self, path: Path, status_code: int = 200) -> None:
        data = Path(path).read_bytes()
        super().__init__(content=data, status_code=status_code)


__all__ = ["Response", "JSONResponse", "FileResponse"]
