"""Utilities for replaying a backtest run from a saved manifest."""

from __future__ import annotations

import argparse
import csv
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


def _resolve_manifest_from_tracker(run_id: str, tracker: Path) -> Path:
    if not tracker.exists():
        raise FileNotFoundError(f"Runs tracker not found: {tracker}")
    with tracker.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("run_id") != run_id:
                continue
            manifest_ref = row.get("manifest") or ""
            if not manifest_ref:
                raise ValueError(f"Run '{run_id}' is missing a manifest path in {tracker}")
            manifest_path = Path(manifest_ref)
            if not manifest_path.is_absolute():
                manifest_path = (tracker.parent / manifest_path).resolve()
            return manifest_path
    raise ValueError(f"Run '{run_id}' not present in {tracker}")


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
    parser.add_argument("--manifest", type=Path, help="Path to run_config.json")
    parser.add_argument("--run-id", type=str, help="Run identifier stored in runs.csv")
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=Path("runs.csv"),
        help="CSV tracker used for resolving run ids",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output directory for the replay")
    parsed = parser.parse_args(argv)

    manifest_path = parsed.manifest
    if parsed.run_id:
        manifest_path = _resolve_manifest_from_tracker(parsed.run_id, parsed.runs_csv)
    if manifest_path is None:
        raise SystemExit("Either --manifest or --run-id must be provided")

    manifest_path = Path(manifest_path)

    manifest = _load_manifest(manifest_path)

    manifest_args = manifest["args"]  # type: ignore[index]
    assert isinstance(manifest_args, Mapping)

    backtest_parser = build_parser()
    cli_args = _build_cli_from_manifest(backtest_parser, manifest_args, parsed.out)
    backtest_main(cli_args)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
