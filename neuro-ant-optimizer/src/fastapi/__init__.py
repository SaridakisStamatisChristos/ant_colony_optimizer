"""Minimal FastAPI-compatible interface for tests."""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

from .responses import JSONResponse, Response

__all__ = [
    "Depends",
    "FastAPI",
    "HTTPException",
    "Request",
    "Response",
    "status",
    "JSONResponse",
]


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: Any = None) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class status:
    HTTP_200_OK = 200
    HTTP_202_ACCEPTED = 202
    HTTP_401_UNAUTHORIZED = 401
    HTTP_403_FORBIDDEN = 403
    HTTP_404_NOT_FOUND = 404


class Request:
    def __init__(self, headers: Optional[Mapping[str, str]] = None) -> None:
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Request(headers={self.headers!r})"


@dataclass
class Depends:
    dependency: Callable[..., Any]


@dataclass
class _Route:
    path: str
    methods: Iterable[str]
    endpoint: Callable[..., Any]


def _split_path(path: str) -> List[str]:
    return [segment for segment in path.strip("/").split("/") if segment or path == "/"]


def _match_route(pattern: str, path: str) -> Optional[Dict[str, str]]:
    pattern_parts = _split_path(pattern)
    path_parts = _split_path(path)
    params: Dict[str, str] = {}
    i = j = 0
    while i < len(pattern_parts) and j < len(path_parts):
        token = pattern_parts[i]
        if token.startswith("{") and token.endswith("}"):
            inner = token[1:-1]
            if inner.endswith(":path"):
                name = inner[:-5]
                params[name] = "/".join(path_parts[j:])
                return params
            name = inner
            params[name] = path_parts[j]
        elif token != path_parts[j]:
            return None
        i += 1
        j += 1
    if i == len(pattern_parts) and j == len(path_parts):
        return params
    if i == len(pattern_parts) - 1 and pattern_parts[i] == "{path:path}" and j == len(path_parts):
        params["path"] = ""
        return params
    return None


class FastAPI:
    def __init__(self) -> None:
        self._routes: List[_Route] = []

    def add_api_route(
        self, path: str, endpoint: Callable[..., Any], methods: Iterable[str]
    ) -> None:
        self._routes.append(_Route(path=path, endpoint=endpoint, methods=[m.upper() for m in methods]))

    def get(self, path: str, *, status_code: int = status.HTTP_200_OK) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            func._status_code = status_code  # type: ignore[attr-defined]
            self.add_api_route(path, func, ["GET"])
            return func

        return decorator

    def post(self, path: str, *, status_code: int = status.HTTP_200_OK) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            func._status_code = status_code  # type: ignore[attr-defined]
            self.add_api_route(path, func, ["POST"])
            return func

        return decorator

    def _resolve_route(self, method: str, path: str) -> Optional[tuple[_Route, Dict[str, str]]]:
        for route in self._routes:
            if method.upper() not in route.methods:
                continue
            params = _match_route(route.path, path)
            if params is not None:
                return route, params
        return None

    def handle_request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Response:
        match = self._resolve_route(method, path)
        if match is None:
            return Response(status_code=status.HTTP_404_NOT_FOUND, content=b"")
        route, params = match
        request = Request(headers=headers)
        try:
            result = _call_endpoint(route.endpoint, request, params, json_body)
            if isinstance(result, Response):
                return result
            if isinstance(result, dict):
                return JSONResponse(result, status_code=getattr(route.endpoint, "_status_code", status.HTTP_200_OK))
            if isinstance(result, (bytes, bytearray)):
                return Response(content=bytes(result), status_code=getattr(route.endpoint, "_status_code", status.HTTP_200_OK))
            if isinstance(result, str):
                return Response(content=result.encode("utf-8"), status_code=getattr(route.endpoint, "_status_code", status.HTTP_200_OK))
            return Response(status_code=getattr(route.endpoint, "_status_code", status.HTTP_200_OK))
        except HTTPException as exc:
            detail = exc.detail if exc.detail is not None else b""
            if isinstance(detail, str):
                payload = detail.encode("utf-8")
            elif isinstance(detail, bytes):
                payload = detail
            else:
                payload = json.dumps(detail).encode("utf-8")
            return Response(status_code=exc.status_code, content=payload)


def _call_endpoint(
    endpoint: Callable[..., Any],
    request: Request,
    path_params: Mapping[str, str],
    body: Any,
) -> Any:
    sig = inspect.signature(endpoint)
    kwargs: MutableMapping[str, Any] = {}
    for name, param in sig.parameters.items():
        if isinstance(param.default, Depends):
            dependency = param.default.dependency
            kwargs[name] = dependency(request)
        elif name in path_params:
            kwargs[name] = path_params[name]
        elif name == "request":
            kwargs[name] = request
        elif body is not None and param.default is inspect._empty and name not in kwargs:
            kwargs[name] = body
    result = endpoint(**kwargs)
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result
