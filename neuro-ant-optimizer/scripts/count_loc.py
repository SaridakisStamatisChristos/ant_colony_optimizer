"""Utility to report line counts for Python files in the repository.

The script reports:

* Total number of lines tracked by git (all file types).
* Number of logical lines of Python code (non-blank, non-comment, non-docstring).
* The subset of those Python code lines that live in test files.

The output is printed as JSON for easy downstream consumption.
"""

from __future__ import annotations

import ast
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Set

import tokenize


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class LineCounts:
    """Container for the collected line counts."""

    total_lines: int = 0
    python_code_lines: int = 0
    test_code_lines: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "total_lines": self.total_lines,
                "python_code_lines": self.python_code_lines,
                "test_code_lines": self.test_code_lines,
            },
            indent=2,
        )


def iter_git_tracked_files() -> Iterable[Path]:
    """Yield all git tracked files relative to the repository root."""

    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    for line in completed.stdout.splitlines():
        yield REPO_ROOT / line


def is_test_file(path: Path) -> bool:
    """Return True if the given python file is considered a test file."""

    parts = {part.lower() for part in path.parts}
    if "tests" in parts:
        return True
    stem = path.stem.lower()
    return stem.startswith("test_") or stem.endswith("_test")


def total_line_count(path: Path) -> int:
    """Count the total lines in a text file."""

    with path.open("rb") as f:
        data = f.read()

    # Count newlines; add one if the last line has no trailing newline.
    total = data.count(b"\n")
    if data and not data.endswith(b"\n"):
        total += 1
    return total


def docstring_lines(node: ast.AST) -> Set[int]:
    """Collect line numbers that correspond to docstrings within the node."""

    doc_lines: Set[int] = set()

    def add_docstring(expr: ast.expr | None) -> None:
        if not isinstance(expr, ast.Expr):
            return
        value = expr.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            start = value.lineno
            end = getattr(value, "end_lineno", start)
            doc_lines.update(range(start, end + 1))

    if isinstance(node, ast.Module):
        body = node.body
    else:
        body = getattr(node, "body", [])

    if isinstance(body, list) and body:
        add_docstring(body[0])

    for child in ast.iter_child_nodes(node):
        doc_lines.update(docstring_lines(child))

    return doc_lines


def python_code_line_count(path: Path) -> int:
    """Count non-comment, non-docstring, non-blank Python lines."""

    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Skip files with encoding issues.
        return 0

    try:
        tree = ast.parse(source)
        doc_lines = docstring_lines(tree)
    except SyntaxError:
        doc_lines = set()

    code_lines: Set[int] = set()

    try:
        tokens = tokenize.generate_tokens(iter(source.splitlines(keepends=True)).__next__)
    except tokenize.TokenError:
        return 0

    for token in tokens:
        if token.type in {
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
        }:
            continue
        if token.type == tokenize.COMMENT:
            continue
        if token.type == tokenize.STRING and token.start[0] in doc_lines:
            continue
        code_lines.add(token.start[0])

    return len(code_lines)


def collect_counts() -> LineCounts:
    counts = LineCounts()

    for path in iter_git_tracked_files():
        counts.total_lines += total_line_count(path)

        if path.suffix != ".py":
            continue

        code_lines = python_code_line_count(path)
        counts.python_code_lines += code_lines
        if is_test_file(path.relative_to(REPO_ROOT)):
            counts.test_code_lines += code_lines

    return counts


def main() -> None:
    counts = collect_counts()
    print(counts.to_json())


if __name__ == "__main__":
    main()
