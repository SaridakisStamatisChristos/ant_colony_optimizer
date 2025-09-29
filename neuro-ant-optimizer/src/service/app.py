"""FastAPI application exposing backtest run metadata."""

from __future__ import annotations

import hmac
import json
import os
from pathlib import Path
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse

from neuro_ant_optimizer.runid import compute_run_id


def _runs_root() -> Path:
    env_path = os.getenv("RUNS_DIR")
    if env_path:
        root = Path(env_path)
    else:
        root = Path.cwd() / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _require_token(request: Request) -> None:
    token = os.getenv("SERVICE_AUTH_TOKEN")
    if not token:
        return
    header = request.headers.get("authorization")
    if not header or not header.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    provided = header.split(" ", 1)[1].strip()
    if not hmac.compare_digest(provided, token):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token")


def _safe_run_dir(run_id: str, root: Path) -> Path:
    candidate = (root / run_id).resolve()
    if not str(candidate).startswith(str(root)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return candidate


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Manifest not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_urls(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(key): str(value) for key, value in data.items()}
    return {}


def create_app() -> FastAPI:
    app = FastAPI()

    @app.post("/backtest", status_code=status.HTTP_202_ACCEPTED)
    async def submit_backtest(
        params: Dict[str, Any],
        _: None = Depends(_require_token),
    ) -> Dict[str, str]:
        manifest_seed: Dict[str, Any] = {"args": params}
        if "timestamp" in params:
            manifest_seed["timestamp"] = params["timestamp"]
        if "git_sha" in params:
            manifest_seed["git_sha"] = params["git_sha"]
        run_id = compute_run_id(manifest_seed, [])
        return {"run_id": run_id, "status": "accepted"}

    @app.get("/runs/{run_id}")
    async def get_run(
        run_id: str,
        _: None = Depends(_require_token),
    ) -> Dict[str, Any]:
        root = _runs_root()
        run_dir = _safe_run_dir(run_id, root)
        manifest = _load_manifest(run_dir / "run_config.json")
        index_path = run_dir / "artifact_index.json"
        if index_path.exists():
            manifest["artifact_index_entries"] = json.loads(
                index_path.read_text(encoding="utf-8")
            )
        return manifest

    @app.get("/artifacts/{run_id}/{artifact_path:path}")
    async def get_artifact(
        run_id: str,
        artifact_path: str,
        _: None = Depends(_require_token),
    ) -> Response:
        root = _runs_root()
        run_dir = _safe_run_dir(run_id, root)
        target = (run_dir / artifact_path).resolve()
        if target.exists() and target.is_file() and str(target).startswith(str(run_dir)):
            return FileResponse(target)

        urls = _load_urls(run_dir / "urls.json")
        if artifact_path in urls:
            return JSONResponse({"location": urls[artifact_path]})

        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")

    return app


app = create_app()


__all__ = ["app", "create_app"]
