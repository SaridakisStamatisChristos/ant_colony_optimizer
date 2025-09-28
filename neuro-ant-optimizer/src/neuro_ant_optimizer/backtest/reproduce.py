"""Utilities for replaying a backtest run from a saved manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Mapping, MutableMapping, Sequence

from .backtest import build_parser, main as backtest_main


def _ensure_mapping(data: Mapping[str, object]) -> MutableMapping[str, object]:
    return dict(data)


def _load_manifest(path: Path) -> MutableMapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Manifest must be a JSON object")
    if "args" not in payload:
        raise ValueError("Manifest is missing 'args' block")
    args_blob = payload["args"]
    if not isinstance(args_blob, Mapping):
        raise ValueError("Manifest 'args' section must be a mapping")
    manifest = _ensure_mapping(payload)
    manifest["args"] = _ensure_mapping(args_blob)
    return manifest


def _stringify(value: object) -> str:
    if isinstance(value, (Path,)):
        return str(value)
    return str(value)


def _build_cli_from_manifest(
    parser: argparse.ArgumentParser,
    manifest_args: Mapping[str, object],
    out_path: Path,
) -> List[str]:
    args_dict = dict(manifest_args)
    args_dict["out"] = str(out_path)

    cli: List[str] = []
    store_true = getattr(argparse, "_StoreTrueAction")
    store_false = getattr(argparse, "_StoreFalseAction")

    for action in parser._actions:
        if not action.option_strings:
            continue
        dest = action.dest
        if dest is None or dest == argparse.SUPPRESS:
            continue
        if dest not in args_dict:
            continue
        value = args_dict[dest]
        if isinstance(action, store_true):
            if value:
                cli.append(action.option_strings[-1])
            continue
        if isinstance(action, store_false):
            if not value:
                cli.append(action.option_strings[-1])
            continue
        if value is None:
            continue
        option = action.option_strings[-1]
        cli.extend([option, _stringify(value)])
    return cli


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay a backtest run from its manifest")
    parser.add_argument("--manifest", required=True, type=Path, help="Path to run_config.json")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for the replay")
    parsed = parser.parse_args(argv)

    manifest = _load_manifest(parsed.manifest)
    manifest_args = manifest["args"]  # type: ignore[index]
    assert isinstance(manifest_args, Mapping)

    backtest_parser = build_parser()
    cli_args = _build_cli_from_manifest(backtest_parser, manifest_args, parsed.out)
    backtest_main(cli_args)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
