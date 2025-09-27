"""
Ensure `neuro_ant_optimizer` is importable when running pytest from the repo root
without setting PYTHONPATH or doing an editable install.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_PKG_ROOT = _HERE.parents[1]   # neuro-ant-optimizer/
_SRC = _PKG_ROOT / "src"

if _SRC.is_dir():
    p = str(_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)
