"""Test client for the FastAPI stub."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from . import FastAPI
from .responses import Response


class _ResponseWrapper:
    def __init__(self, response: Response) -> None:
        self._response = response
        self.status_code = response.status_code
        self.content = response.content

    def json(self) -> Any:
        return self._response.json()

    @property
    def text(self) -> str:
        return self._response.text


class TestClient:
    def __init__(self, app: FastAPI) -> None:
        self._app = app

    def __enter__(self) -> "TestClient":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def request(
        self,
        method: str,
        url: str,
        *,
        json: Any = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> _ResponseWrapper:
        response = self._app.handle_request(method.upper(), url, json_body=json, headers=headers)
        return _ResponseWrapper(response)

    def get(self, url: str, *, headers: Optional[Mapping[str, str]] = None) -> _ResponseWrapper:
        return self.request("GET", url, headers=headers)

    def post(
        self,
        url: str,
        *,
        json: Any = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> _ResponseWrapper:
        return self.request("POST", url, json=json, headers=headers)


__all__ = ["TestClient"]
