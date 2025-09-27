"""Development helper to expose the package without installation."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_src_pkg = _repo_root / "src" / "neuro_ant_optimizer"

if not _src_pkg.exists():  # pragma: no cover - safety guard
    raise ImportError("neuro_ant_optimizer source package not found")

spec = importlib.util.spec_from_file_location(
    __name__,
    _src_pkg / "__init__.py",
    submodule_search_locations=[str(_src_pkg)],
)
if spec is None or spec.loader is None:  # pragma: no cover - defensive
    raise ImportError("Unable to load neuro_ant_optimizer package")

module = importlib.util.module_from_spec(spec)
sys.modules[__name__] = module
spec.loader.exec_module(module)

# Mirror attributes into this module's namespace so direct imports work.
globals().update(module.__dict__)
