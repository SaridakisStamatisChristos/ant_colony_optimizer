"""Bundled example configuration files for Neuro Ant Optimizer."""
from importlib import resources as _resources
from importlib.resources.abc import Traversable as _Traversable

__all__ = ["configs_path", "iter_configs"]


def configs_path() -> _Traversable:
    """Return the traversable pointing to the packaged configuration templates."""
    return _resources.files(__name__).joinpath("configs")


def iter_configs(pattern: str = "*.yaml"):
    """Yield traversables for configs matching ``pattern`` inside the package."""
    base = configs_path()
    yield from base.glob(pattern)
