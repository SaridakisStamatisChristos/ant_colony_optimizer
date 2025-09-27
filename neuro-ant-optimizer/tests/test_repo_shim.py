import importlib
import pathlib
import sys


def test_repo_shim_prefers_src():
    here = pathlib.Path(__file__).resolve()
    project_root = here.parents[1]
    src_root = project_root / "src"
    assert src_root.is_dir()

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    sys.modules.pop("neuro_ant_optimizer", None)
    pkg = importlib.import_module("neuro_ant_optimizer")
    pkg_path = pathlib.Path(pkg.__file__).resolve()
    assert src_root == pkg_path.parents[1]
