"""Name -> class registries backing ``get_backbone()`` and ``get_probe()``.

Keeping registration decoupled from the public API means adding a backbone or
task is a one-decorator change in that module, with no edits to __init__.py.
"""

from typing import Any, Callable

__all__ = [
    "register_backbone",
    "register_task",
    "get_backbone_class",
    "get_task_class",
    "list_backbones",
    "list_tasks",
]

_BACKBONES: dict[str, type] = {}
_TASKS: dict[str, type] = {}


def register_backbone(name: str) -> Callable[[type], type]:
    """Class decorator registering a :class:`BaseBackbone` subclass under ``name``.

    Raises on duplicate names rather than silently overwriting, so a typo in a
    new backbone cannot shadow an existing one.
    """
    raise NotImplementedError


def register_task(name: str) -> Callable[[type], type]:
    """Class decorator registering a :class:`BaseTask` subclass under ``name``."""
    raise NotImplementedError


def get_backbone_class(name: str) -> type:
    """Look up a registered backbone class, with a helpful error listing known names."""
    raise NotImplementedError


def get_task_class(name: str) -> type:
    """Look up a registered task class, with a helpful error listing known names."""
    raise NotImplementedError


def list_backbones() -> list[str]:
    """Return the sorted names of all registered backbones."""
    raise NotImplementedError


def list_tasks() -> list[str]:
    """Return the sorted names of all registered tasks."""
    raise NotImplementedError


def _ensure_imported() -> Any:
    """Import the backbone/task subpackages so their decorators have run.

    Called by the lookup helpers; keeps registration lazy enough that importing
    ``visbench`` does not pull in every optional dependency.
    """
    raise NotImplementedError
