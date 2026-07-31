"""Name -> class registries backing ``get_backbone()`` and ``get_probe()``.

Keeping registration decoupled from the public API means adding a backbone or
task is a one-decorator change in that module, with no edits to __init__.py.

One class may claim several names. DINOv2 registers both ``dinov2_vits14`` and
``dinov2_vitb14``, differing only by constructor kwargs, so an entry is
``(class, default_kwargs)`` rather than a bare class. That is also why
registration does **not** write ``cls.name``: with two names on one class the
last decorator would win and every instance would misreport itself — and
``name`` feeds the cache key, so the damage would be silently-wrong features,
not a crash. Instances set their own ``name`` in ``__init__``.
"""

from collections.abc import Callable
from typing import Any

__all__ = [
    "register_backbone",
    "register_task",
    "get_backbone_class",
    "get_task_class",
    "list_backbones",
    "list_tasks",
    "build_backbone",
    "build_task",
    "missing_extra",
]

#: Maps an optional dependency to the extra that installs it, for error text.
_EXTRA_FOR: dict[str, str] = {"open_clip": "clip", "timm": "timm"}

#: name -> (class, default constructor kwargs)
_BACKBONES: dict[str, tuple[type, dict]] = {}
_TASKS: dict[str, tuple[type, dict]] = {}


def _register(
    registry: dict[str, tuple[type, dict]],
    kind: str,
    name: str,
    defaults: dict,
) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        if name in registry:
            existing = registry[name][0]
            raise ValueError(
                f"{kind} name {name!r} is already registered to "
                f"{existing.__module__}.{existing.__qualname__}; "
                f"cannot re-register it for {cls.__module__}.{cls.__qualname__}"
            )
        registry[name] = (cls, defaults)
        return cls

    return decorator


def register_backbone(name: str, **default_kwargs: Any) -> Callable[[type], type]:
    """Class decorator registering a :class:`BaseBackbone` subclass under ``name``.

    ``default_kwargs`` are passed to the constructor when the backbone is built
    through :func:`visbench.get_backbone`, and are overridden by anything the
    caller passes explicitly.

    Raises on duplicate names rather than silently overwriting, so a typo in a
    new backbone cannot shadow an existing one.
    """
    return _register(_BACKBONES, "Backbone", name, default_kwargs)


def register_task(name: str, **default_kwargs: Any) -> Callable[[type], type]:
    """Class decorator registering a :class:`BaseTask` subclass under ``name``."""
    return _register(_TASKS, "Task", name, default_kwargs)


def _lookup(
    registry: dict[str, tuple[type, dict]],
    kind: str,
    name: str,
) -> tuple[type, dict]:
    _ensure_imported()
    if name not in registry:
        known = ", ".join(sorted(registry)) or "(none registered)"
        raise KeyError(f"Unknown {kind} {name!r}. Available: {known}")
    return registry[name]


def get_backbone_class(name: str) -> type:
    """Look up a registered backbone class, with a helpful error listing known names."""
    return _lookup(_BACKBONES, "backbone", name)[0]


def get_task_class(name: str) -> type:
    """Look up a registered task class, with a helpful error listing known names."""
    return _lookup(_TASKS, "task", name)[0]


def build_backbone(name: str, **kwargs: Any) -> Any:
    """Instantiate a registered backbone, merging registered defaults with ``kwargs``."""
    cls, defaults = _lookup(_BACKBONES, "backbone", name)
    return cls(**{**defaults, **kwargs})


def build_task(name: str, **kwargs: Any) -> Any:
    """Instantiate a registered task, merging registered defaults with ``kwargs``."""
    cls, defaults = _lookup(_TASKS, "task", name)
    return cls(**{**defaults, **kwargs})


def list_backbones() -> list[str]:
    """Return the sorted names of all registered backbones."""
    _ensure_imported()
    return sorted(_BACKBONES)


def list_tasks() -> list[str]:
    """Return the sorted names of all registered tasks."""
    _ensure_imported()
    return sorted(_TASKS)


#: module -> the optional third-party package it needs, or ``None`` if it has
#: no optional dependency and must therefore always import cleanly.
_REGISTRATION_MODULES: dict[str, str | None] = {
    "visbench.backbones.dinov2": None,
    "visbench.backbones.clip": "open_clip",
    "visbench.backbones.timm_backbone": "timm",
    "visbench.tasks.high_level.classification": None,
    "visbench.tasks.high_level.retrieval": None,
    "visbench.tasks.high_level.semantic_segmentation": None,
    "visbench.tasks.high_level.detection": None,
    "visbench.tasks.mid_level.correspondence": None,
    "visbench.tasks.mid_level.depth": None,
    "visbench.tasks.mid_level.surface_normal": None,
    "visbench.tasks.mid_level.generic_segmentation": None,
    "visbench.tasks.mid_level.similarity": None,
    "visbench.tasks.low_level.edge": None,
}

_IMPORTED = False


def _ensure_imported() -> None:
    """Import the backbone/task subpackages so their decorators have run.

    Called by the lookup helpers; keeps registration lazy enough that importing
    ``visbench`` does not pull in every optional dependency.

    A module whose *declared* optional dependency is missing (e.g. CLIP without
    ``open_clip_torch``) is skipped rather than fatal — the cost of a missing
    extra should be that one name is absent from :func:`list_backbones`, not
    that the whole library fails to import.

    Any other ImportError propagates. A blanket ``except ImportError`` here
    would turn a typo'd import inside a backbone module into "Unknown backbone
    'dinov2_vitb14'", sending the reader hunting through the registry for a
    registration bug that does not exist.
    """
    global _IMPORTED
    if _IMPORTED:
        return
    # Set first: a module that imports the registry during its own import must
    # not re-enter this function.
    _IMPORTED = True

    import importlib

    for module, optional_dependency in _REGISTRATION_MODULES.items():
        try:
            importlib.import_module(module)
        except ImportError as exc:
            missing = getattr(exc, "name", None)
            if optional_dependency is not None and _is_missing(missing, optional_dependency):
                continue
            raise ImportError(
                f"Failed to import {module}, so its backbones/tasks are not registered. "
                f"This is a real import error, not a missing optional dependency."
            ) from exc


def missing_extra(name: str) -> str | None:
    """The **extra** ``name`` needs and that is not installed, e.g. ``"clip"``.

    ``None`` when the backbone is ready to use. Returns the extra rather than
    the module because the extra is what a caller puts in an install command,
    and ``open_clip`` is imported from a distribution called
    ``open_clip_torch`` — a name no user should have to know to read an error
    message. Checked with ``find_spec``, so this never imports open_clip or
    timm: asking "what can I run" must not cost what running it would.

    This exists because a missing extra does *not* remove a name from
    :func:`list_backbones`, contrary to what the registry's skip logic looks
    like it does. Both CLIP and timm import their dependency lazily, inside
    ``__init__``, so the registration module imports cleanly and the name
    registers either way. That is the better arrangement — ``get_backbone``
    then raises "install visbench[clip]" instead of the registry raising
    "Unknown backbone 'clip_vitb16'" — but it means a listing has to say so
    itself rather than by omission.
    """
    import importlib.util

    _ensure_imported()
    if name not in _BACKBONES:
        return None
    module = _BACKBONES[name][0].__module__
    dependency = _REGISTRATION_MODULES.get(module)
    if dependency is None or importlib.util.find_spec(dependency) is not None:
        return None
    return _EXTRA_FOR.get(dependency, dependency)


def _is_missing(missing: str | None, optional_dependency: str) -> bool:
    """Whether ``missing`` is ``optional_dependency`` or a submodule of it."""
    if missing is None:
        return False
    return missing == optional_dependency or missing.startswith(f"{optional_dependency}.")
