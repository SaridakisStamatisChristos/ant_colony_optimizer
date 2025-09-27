"""Development helper to expose the package without installation."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src"
_SRC_PKG = _SRC_ROOT / "neuro_ant_optimizer"

if _SRC_ROOT.is_dir():
    p = str(_SRC_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

if not _SRC_PKG.exists():  # pragma: no cover - safety guard
    raise ImportError("neuro_ant_optimizer source package not found")

spec = importlib.util.spec_from_file_location(
    __name__,
    _SRC_PKG / "__init__.py",
    submodule_search_locations=[str(_SRC_PKG)],
)
if spec is None or spec.loader is None:  # pragma: no cover - defensive
    raise ImportError("Unable to load neuro_ant_optimizer package")

module = importlib.util.module_from_spec(spec)
sys.modules[__name__] = module
spec.loader.exec_module(module)

globals().update(module.__dict__)
